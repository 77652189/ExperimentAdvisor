from __future__ import annotations

import math

import numpy as np
import pandas as pd

from experiment_advisor.optimizer.constraints import filter_constraints
from experiment_advisor.optimizer.search_space import SearchSpace, generate_candidates


def _normal_pdf(x: pd.Series) -> pd.Series:
    return (-0.5 * x * x).map(math.exp) / math.sqrt(2 * math.pi)


def _normal_cdf(x: pd.Series) -> pd.Series:
    return x.map(lambda value: 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))))


def recommend_standard_bo(
    df: pd.DataFrame,
    search_space: SearchSpace,
    acquisition: str = "ei",
    top_k: int = 5,
    target_col: str = "yield_g_per_l",
    feature_cols: list[str] | None = None,
    candidates: pd.DataFrame | None = None,
) -> dict:
    """标准 GP Bayesian Optimization baseline，支持 EI 和 UCB。"""

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError("Standard BO requires scikit-learn. Install dependencies with: pip install -r requirements.txt") from exc

    features = feature_cols or list(search_space.bounds)
    train = df[[*features, target_col]].dropna()
    if len(train) < 5:
        raise ValueError("At least 5 complete training rows are required for standard BO")
    x = train[features]
    y = train[target_col].astype(float)
    n_features = len(features)
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(
            nu=2.5,
            length_scale=np.ones(n_features).tolist(),
            length_scale_bounds=[(1e-2, 1e2)] * n_features,
        )
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    )
    gp = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42),
    )
    gp.fit(x, y)

    pool = candidates if candidates is not None else generate_candidates(search_space, n=5000, seed=7)
    pool = filter_constraints(pool, search_space)
    mean, std = gp.predict(pool[features], return_std=True)
    scored = pool.copy()
    scored["predicted_yield"] = mean
    scored["model_uncertainty"] = std
    best = float(y.max())
    if acquisition.lower() == "ucb":
        scored["acquisition_score"] = scored["predicted_yield"] + 2.0 * scored["model_uncertainty"]
    elif acquisition.lower() == "ei":
        sigma = scored["model_uncertainty"].clip(lower=1e-9)
        z = (scored["predicted_yield"] - best) / sigma
        scored["acquisition_score"] = (scored["predicted_yield"] - best) * _normal_cdf(z) + sigma * _normal_pdf(z)
    else:
        raise ValueError("acquisition must be 'ei' or 'ucb'")

    top = scored.sort_values("acquisition_score", ascending=False).head(top_k)
    return {
        "recommendations": [
            {
                "method": f"standard_bo_{acquisition.lower()}",
                "rank": int(rank),
                "params": {feature: float(row[feature]) for feature in features},
                "predicted_yield": float(row["predicted_yield"]),
                "model_uncertainty": float(row["model_uncertainty"]),
                "uncertainty_type": "gp_posterior_std",
                "acquisition_score": float(row["acquisition_score"]),
            }
            for rank, (_, row) in enumerate(top.iterrows(), start=1)
        ],
        "fitted_gp": gp,
        "feature_cols": features,
    }
