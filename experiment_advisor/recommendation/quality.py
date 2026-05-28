from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from experiment_advisor.optimizer.constraints import boundary_risk
from experiment_advisor.optimizer.search_space import SearchSpace


def _recommendations_frame(recommendations: list[dict[str, Any]], feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for item in recommendations:
        params = item.get("params", {})
        row = {feature: float(params[feature]) for feature in feature_cols if feature in params}
        row["rank"] = item.get("rank")
        row["predicted_yield"] = item.get("predicted_yield")
        row["model_uncertainty"] = item.get("model_uncertainty")
        rows.append(row)
    return pd.DataFrame(rows)


def _normalized(frame: pd.DataFrame, search_space: SearchSpace, feature_cols: list[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for feature in feature_cols:
        low, high = search_space.bounds[feature]
        result[feature] = (frame[feature] - low) / max(high - low, 1e-12)
    return result


def _pairwise_distances(norm: pd.DataFrame) -> np.ndarray:
    values = norm.to_numpy(dtype=float)
    if len(values) < 2:
        return np.array([], dtype=float)
    distances = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            distances.append(float(np.linalg.norm(values[i] - values[j])))
    return np.asarray(distances, dtype=float)


def _cluster_count(norm: pd.DataFrame, threshold: float = 0.10) -> int:
    values = norm.to_numpy(dtype=float)
    clusters: list[np.ndarray] = []
    for row in values:
        if not any(float(np.linalg.norm(row - center)) <= threshold for center in clusters):
            clusters.append(row)
    return len(clusters)


def _nearest_history_distances(
    recommendations: pd.DataFrame,
    history: pd.DataFrame,
    feature_cols: list[str],
) -> list[float]:
    if recommendations.empty or history.empty:
        return []

    means = history[feature_cols].mean()
    stds = history[feature_cols].std(ddof=0).replace(0, 1.0)
    norm_history = (history[feature_cols] - means) / stds
    distances = []
    for _, row in recommendations.iterrows():
        norm_candidate = (row[feature_cols] - means) / stds
        distance = ((norm_history - norm_candidate) ** 2).sum(axis=1) ** 0.5
        distances.append(float(distance.min()))
    return distances


def _nearest_history_info(
    recommendations: pd.DataFrame,
    history: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "yield_g_per_l",
) -> list[dict]:
    """Return distance, run_id, and yield of the nearest historical neighbour for each recommendation."""
    if recommendations.empty or history.empty:
        return []

    means = history[feature_cols].mean()
    stds = history[feature_cols].std(ddof=0).replace(0, 1.0)
    norm_history = (history[feature_cols] - means) / stds

    results = []
    for _, row in recommendations.iterrows():
        norm_candidate = (row[feature_cols] - means) / stds
        distances = ((norm_history - norm_candidate) ** 2).sum(axis=1) ** 0.5
        nearest_idx = distances.idxmin()
        run_id = (
            str(history.loc[nearest_idx, "fermenter_run_id"])
            if "fermenter_run_id" in history.columns
            else str(nearest_idx)
        )
        yield_val = (
            float(history.loc[nearest_idx, target_col])
            if target_col in history.columns
            else None
        )
        results.append(
            {
                "distance": float(distances[nearest_idx]),
                "run_id": run_id,
                "yield": yield_val,
            }
        )
    return results


def _classify_recommendation_types(
    predicted_yields: list[float],
    model_uncertainties: list[float],
) -> list[str]:
    """Classify each recommendation as exploitation-dominant or exploration-dominant.

    Within the batch, normalise predicted yield and model uncertainty to z-scores.
    A point whose yield z-score exceeds its uncertainty z-score is labelled
    "利用型" (exploitation); otherwise "探索型" (exploration).
    """
    yields = np.array(predicted_yields, dtype=float)
    uncertainties = np.array(model_uncertainties, dtype=float)

    def _zscore(arr: np.ndarray) -> np.ndarray:
        std = arr.std()
        return (arr - arr.mean()) / std if std > 1e-12 else np.zeros_like(arr)

    yield_z = _zscore(yields)
    uncertainty_z = _zscore(uncertainties)

    return [
        "利用型" if yz >= uz else "探索型"
        for yz, uz in zip(yield_z, uncertainty_z)
    ]


def evaluate_recommendation_quality(
    recommendations: list[dict[str, Any]],
    history: pd.DataFrame,
    search_space: SearchSpace,
    feature_cols: list[str] | None = None,
    target_col: str = "yield_g_per_l",
) -> dict[str, Any]:
    """Compute recommendation-strategy diagnostics independent of model LOO-CV."""

    features = feature_cols or list(search_space.bounds)
    features = [feature for feature in features if feature in search_space.bounds]
    frame = _recommendations_frame(recommendations, features)
    if frame.empty or not features:
        return {
            "batch_diversity": {},
            "history_support": {},
            "boundary_risk": {},
            "prediction_profile": {},
            "per_recommendation": [],
        }

    complete_history = history[[*features, target_col]].dropna()
    norm = _normalized(frame, search_space, features)
    pairwise = _pairwise_distances(norm[features])
    feature_coverage = {
        feature: float(norm[feature].max() - norm[feature].min())
        for feature in features
    }
    neighbour_info = _nearest_history_info(frame, complete_history, features, target_col)
    nearest_distances = [info["distance"] for info in neighbour_info]
    risks = boundary_risk(frame[features], search_space)

    predicted_yields = [
        float(frame.loc[i, "predicted_yield"]) if pd.notna(frame.loc[i, "predicted_yield"]) else 0.0
        for i in frame.index
    ]
    model_uncertainties = [
        float(frame.loc[i, "model_uncertainty"]) if pd.notna(frame.loc[i, "model_uncertainty"]) else 0.0
        for i in frame.index
    ]
    rec_types = _classify_recommendation_types(predicted_yields, model_uncertainties)

    per_recommendation = []
    for list_idx, (idx, row) in enumerate(frame.iterrows()):
        info = neighbour_info[list_idx] if list_idx < len(neighbour_info) else {}
        per_recommendation.append(
            {
                "rank": int(row["rank"]) if pd.notna(row.get("rank")) else int(idx + 1),
                "recommendation_type": rec_types[list_idx],
                "predicted_yield": float(row["predicted_yield"]) if pd.notna(row.get("predicted_yield")) else None,
                "model_uncertainty": float(row["model_uncertainty"]) if pd.notna(row.get("model_uncertainty")) else None,
                "nearest_history_distance": info.get("distance"),
                "nearest_run_id": info.get("run_id"),
                "nearest_run_yield": info.get("yield"),
                "boundary_risk": float(risks.iloc[list_idx]),
            }
        )

    return {
        "batch_diversity": {
            "n_recommendations": int(len(frame)),
            "min_pairwise_distance": float(pairwise.min()) if len(pairwise) else None,
            "mean_pairwise_distance": float(pairwise.mean()) if len(pairwise) else None,
            "max_pairwise_distance": float(pairwise.max()) if len(pairwise) else None,
            "cluster_count_threshold_0_10": int(_cluster_count(norm[features], threshold=0.10)),
            "mean_feature_range_coverage": float(np.mean(list(feature_coverage.values()))) if feature_coverage else None,
            "min_feature_range_coverage": float(np.min(list(feature_coverage.values()))) if feature_coverage else None,
            "feature_range_coverage": feature_coverage,
        },
        "history_support": {
            "mean_nearest_history_distance": float(np.mean(nearest_distances)) if nearest_distances else None,
            "min_nearest_history_distance": float(np.min(nearest_distances)) if nearest_distances else None,
            "max_nearest_history_distance": float(np.max(nearest_distances)) if nearest_distances else None,
            "n_far_from_history_gt_2": int(sum(distance > 2.0 for distance in nearest_distances)),
        },
        "boundary_risk": {
            "mean_boundary_risk": float(risks.mean()),
            "max_boundary_risk": float(risks.max()),
            "n_near_boundary_gt_0_8": int((risks > 0.8).sum()),
        },
        "prediction_profile": {
            "mean_predicted_yield": float(pd.to_numeric(frame["predicted_yield"], errors="coerce").mean()),
            "std_predicted_yield": float(pd.to_numeric(frame["predicted_yield"], errors="coerce").std(ddof=0)),
            "mean_model_uncertainty": float(pd.to_numeric(frame["model_uncertainty"], errors="coerce").mean()),
            "std_model_uncertainty": float(pd.to_numeric(frame["model_uncertainty"], errors="coerce").std(ddof=0)),
        },
        "per_recommendation": per_recommendation,
    }
