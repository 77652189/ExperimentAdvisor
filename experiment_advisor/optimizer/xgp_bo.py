from __future__ import annotations

import math
import warnings
from typing import Any

import pandas as pd

from experiment_advisor.optimizer.constraints import boundary_risk, filter_constraints
from experiment_advisor.optimizer.search_space import SearchSpace, generate_candidates

DEFAULT_XGB_PARAMS = {
    "max_depth": 2,
    "n_estimators": 50,
    "subsample": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 1,
    "reg_lambda": 2,
    "random_state": 42,
    "objective": "reg:squarederror",
}
GP_LENGTH_SCALE_BOUNDS = (1e-2, 1e3)
GP_NOISE_LEVEL_BOUNDS = (1e-5, 1e2)


def _normal_pdf(x: pd.Series) -> pd.Series:
    return (-0.5 * x * x).map(math.exp) / math.sqrt(2 * math.pi)


def _normal_cdf(x: pd.Series) -> pd.Series:
    return x.map(lambda value: 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))))


def _history_distance(candidates: pd.DataFrame, history_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """Return normalized nearest-history distance for candidate quality diagnostics."""

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
    distances = [float(np.sqrt(((h - row) ** 2).sum(axis=1)).min()) for row in c]
    series = pd.Series(distances, index=candidates.index)
    max_distance = max(float(series.quantile(0.95)), 1e-9)
    return (series / max_distance).clip(0, 1)


def _risk_level(history_distance: float, edge_risk: float, uncertainty: float, uncertainty_high: float) -> str:
    """Map candidate diagnostics to a compact human-readable risk level."""

    if history_distance >= 0.85 or edge_risk >= 0.90 or uncertainty >= uncertainty_high:
        return "high"
    if history_distance >= 0.60 or edge_risk >= 0.75:
        return "medium"
    return "low"


def _quality_flags(history_distance: float, edge_risk: float, uncertainty: float, uncertainty_high: float) -> list[str]:
    flags = []
    if history_distance >= 0.85:
        flags.append("far_from_history")
    if edge_risk >= 0.90:
        flags.append("near_search_boundary")
    if uncertainty >= uncertainty_high:
        flags.append("high_residual_uncertainty")
    return flags


def _select_diverse_top(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Select high-acquisition candidates while avoiding identical uncertainty plateaus."""

    ranked = scored.sort_values("acquisition_score", ascending=False)
    if top_k <= 0 or len(ranked) <= top_k:
        return ranked.head(top_k)

    selected_indices: list[Any] = []
    used_uncertainty_bins: set[float] = set()
    for index, row in ranked.iterrows():
        uncertainty_bin = round(float(row["model_uncertainty"]), 6)
        if uncertainty_bin in used_uncertainty_bins:
            continue
        selected_indices.append(index)
        used_uncertainty_bins.add(uncertainty_bin)
        if len(selected_indices) == top_k:
            break

    if len(selected_indices) < top_k:
        for index in ranked.index:
            if index in selected_indices:
                continue
            selected_indices.append(index)
            if len(selected_indices) == top_k:
                break

    return ranked.loc[selected_indices]


def recommend_xgp_bo(
    df: pd.DataFrame,
    search_space: SearchSpace,
    acquisition: str = "ei",
    top_k: int = 5,
    target_col: str = "yield_g_per_l",
    feature_cols: list[str] | None = None,
    candidates: pd.DataFrame | None = None,
    xgb_params: dict | None = None,
    n_gp_features: int = 4,
) -> list[dict]:
    """Use XGBoost for mean prediction and a GP over residuals for BO uncertainty."""

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.exceptions import ConvergenceWarning
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError("xgp_bo requires scikit-learn and xgboost. Install dependencies with: pip install -r requirements.txt") from exc

    features = feature_cols or list(search_space.bounds)
    train = df[[*features, target_col]].dropna()
    if len(train) < 5:
        raise ValueError("At least 5 complete training rows required for xgp_bo")

    x_train = train[features]
    y_train = train[target_col].astype(float)
    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    xgb_pipeline = make_pipeline(StandardScaler(), XGBRegressor(**params))
    xgb_pipeline.fit(x_train, y_train)

    residuals = y_train - xgb_pipeline.predict(x_train)
    n_gp = min(n_gp_features, len(features))
    if n_gp < len(features):
        importances = xgb_pipeline.named_steps["xgbregressor"].feature_importances_
        top_indices = sorted(range(len(features)), key=lambda i: importances[i], reverse=True)[:n_gp]
        gp_features = [features[i] for i in sorted(top_indices)]
    else:
        gp_features = features

    kernel = (
        ConstantKernel(1.0)
        * Matern(
            length_scale=[1.0] * len(gp_features),
            length_scale_bounds=GP_LENGTH_SCALE_BOUNDS,
            nu=2.5,
        )
        + WhiteKernel(noise_level=1.0, noise_level_bounds=GP_NOISE_LEVEL_BOUNDS)
    )
    gp = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            random_state=42,
            n_restarts_optimizer=3,
        ),
    )
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(x_train[gp_features], residuals)

    pool = candidates if candidates is not None else generate_candidates(search_space, n=5000, seed=11)
    pool = filter_constraints(pool, search_space)
    scored = pool.copy()
    scored["xgb_prediction"] = xgb_pipeline.predict(scored[features])
    residual_mean, residual_std = gp.predict(scored[gp_features], return_std=True)
    scored["gp_residual_mean"] = residual_mean
    scored["predicted_yield"] = scored["xgb_prediction"] + scored["gp_residual_mean"]
    scored["model_uncertainty"] = residual_std
    scored["history_distance"] = _history_distance(scored, train, features)
    scored["boundary_risk"] = boundary_risk(scored, search_space)

    best = float(y_train.max())
    if acquisition.lower() == "ucb":
        scored["acquisition_score"] = scored["predicted_yield"] + 2.0 * scored["model_uncertainty"]
    elif acquisition.lower() == "ei":
        sigma = scored["model_uncertainty"].clip(lower=1e-9)
        z = (scored["predicted_yield"] - best) / sigma
        scored["acquisition_score"] = (
            (scored["predicted_yield"] - best) * _normal_cdf(z)
            + sigma * _normal_pdf(z)
        )
    else:
        raise ValueError("acquisition must be 'ei' or 'ucb'")

    uncertainty_unique = int(scored["model_uncertainty"].round(6).nunique())
    uncertainty_min = float(scored["model_uncertainty"].min())
    uncertainty_max = float(scored["model_uncertainty"].max())
    residual_std_value = float(residuals.std(ddof=0))
    gp_model = gp.named_steps["gaussianprocessregressor"]
    gp_health: dict[str, Any] = {
        "gp_feature_cols": gp_features,
        "residual_mean": float(residuals.mean()),
        "residual_std": residual_std_value,
        "residual_max_abs": float(residuals.abs().max()),
        "candidate_uncertainty_min": uncertainty_min,
        "candidate_uncertainty_max": uncertainty_max,
        "candidate_uncertainty_unique_rounded": uncertainty_unique,
        "candidate_uncertainty_degenerate": uncertainty_unique <= 1 or abs(uncertainty_max - uncertainty_min) <= 1e-9,
        "kernel": str(gp_model.kernel_),
        "warnings": [str(item.message) for item in captured_warnings],
    }
    uncertainty_high = max(float(scored["model_uncertainty"].quantile(0.85)), 1e-9)
    top = _select_diverse_top(scored, top_k)
    return [
        {
            "method": f"xgp_bo_{acquisition.lower()}",
            "rank": int(rank),
            "params": {feature: float(row[feature]) for feature in features},
            "predicted_yield": float(row["predicted_yield"]),
            "model_uncertainty": float(row["model_uncertainty"]),
            "uncertainty_type": "xgp_gp_residual_std",
            "history_distance": float(row["history_distance"]),
            "boundary_risk": float(row["boundary_risk"]),
            "risk_level": _risk_level(
                float(row["history_distance"]),
                float(row["boundary_risk"]),
                float(row["model_uncertainty"]),
                uncertainty_high,
            ),
            "quality_flags": _quality_flags(
                float(row["history_distance"]),
                float(row["boundary_risk"]),
                float(row["model_uncertainty"]),
                uncertainty_high,
            ),
            "acquisition_score": float(row["acquisition_score"]),
            "xgb_prediction": float(row["xgb_prediction"]),
            "gp_residual_mean": float(row["gp_residual_mean"]),
            "gp_feature_cols": gp_features,
            "gp_health": gp_health,
        }
        for rank, (_, row) in enumerate(top.iterrows(), start=1)
    ]
