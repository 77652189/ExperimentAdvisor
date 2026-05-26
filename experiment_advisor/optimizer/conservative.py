from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from experiment_advisor.model.predictor import predict_candidates
from experiment_advisor.model.trainer import ModelBundle
from experiment_advisor.optimizer.constraints import boundary_risk, filter_constraints
from experiment_advisor.optimizer.search_space import SearchSpace, generate_candidates


@dataclass(frozen=True)
class ConservativeWeights:
    exploration_weight: float = 0.15
    distance_penalty: float = 0.35
    boundary_penalty: float = 0.20

    def as_dict(self) -> dict[str, float]:
        return {
            "exploration_weight": self.exploration_weight,
            "distance_penalty": self.distance_penalty,
            "boundary_penalty": self.boundary_penalty,
        }


DEFAULT_CONSERVATIVE_WEIGHTS = ConservativeWeights()


def _history_distance(candidates: pd.DataFrame, history_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise ImportError("History distance requires numpy.") from exc
    history = history_df[feature_cols].dropna()
    if history.empty:
        return pd.Series(1.0, index=candidates.index)
    means = history.mean()
    stds = history.std(ddof=0).replace(0, 1.0)
    h = ((history - means) / stds).to_numpy()
    c = ((candidates[feature_cols] - means) / stds).to_numpy()
    distances = []
    for row in c:
        distances.append(float(np.sqrt(((h - row) ** 2).sum(axis=1)).min()))
    series = pd.Series(distances, index=candidates.index)
    max_distance = max(float(series.quantile(0.95)), 1e-9)
    return (series / max_distance).clip(0, 1)


def recommend_conservative(
    model_bundle: ModelBundle,
    history_df: pd.DataFrame,
    search_space: SearchSpace,
    top_k: int = 5,
    candidates: pd.DataFrame | None = None,
    weights: ConservativeWeights | dict[str, float] | None = None,
    exploration_weight: float | None = None,
    distance_penalty: float | None = None,
    boundary_penalty: float | None = None,
) -> list[dict]:
    """使用 surrogate ensemble 和显式风险惩罚推荐候选点。"""

    if weights is None:
        weight_obj = DEFAULT_CONSERVATIVE_WEIGHTS
    elif isinstance(weights, dict):
        weight_obj = ConservativeWeights(**{**DEFAULT_CONSERVATIVE_WEIGHTS.as_dict(), **weights})
    else:
        weight_obj = weights
    if exploration_weight is not None or distance_penalty is not None or boundary_penalty is not None:
        weight_obj = ConservativeWeights(
            exploration_weight=DEFAULT_CONSERVATIVE_WEIGHTS.exploration_weight if exploration_weight is None else exploration_weight,
            distance_penalty=DEFAULT_CONSERVATIVE_WEIGHTS.distance_penalty if distance_penalty is None else distance_penalty,
            boundary_penalty=DEFAULT_CONSERVATIVE_WEIGHTS.boundary_penalty if boundary_penalty is None else boundary_penalty,
        )

    pool = candidates if candidates is not None else generate_candidates(search_space, n=5000, seed=42)
    pool = filter_constraints(pool, search_space)
    scored = predict_candidates(model_bundle, pool)
    features = model_bundle.feature_columns
    scored["history_distance"] = _history_distance(scored, history_df, features)
    scored["boundary_risk"] = boundary_risk(scored, search_space)
    y_std = max(float(scored["predicted_yield"].std(ddof=0)), 1e-9)
    u_std = max(float(scored["model_uncertainty"].std(ddof=0)), 1e-9)
    scored["acquisition_score"] = (
        scored["predicted_yield"] / y_std
        + weight_obj.exploration_weight * scored["model_uncertainty"] / u_std
        - weight_obj.distance_penalty * scored["history_distance"]
        - weight_obj.boundary_penalty * scored["boundary_risk"]
    )
    top = scored.sort_values("acquisition_score", ascending=False).head(top_k)
    return [
        {
            "method": "conservative_ensemble",
            "rank": int(rank),
            "params": {feature: float(row[feature]) for feature in features},
            "predicted_yield": float(row["predicted_yield"]),
            "model_uncertainty": float(row["model_uncertainty"]),
            "uncertainty_type": "ensemble_disagreement_std",
            "model_predictions": {
                column.removeprefix("pred_"): float(row[column])
                for column in scored.columns
                if column.startswith("pred_")
            },
            "history_distance": float(row["history_distance"]),
            "boundary_risk": float(row["boundary_risk"]),
            "acquisition_score": float(row["acquisition_score"]),
            "scoring_weights": weight_obj.as_dict(),
        }
        for rank, (_, row) in enumerate(top.iterrows(), start=1)
    ]
