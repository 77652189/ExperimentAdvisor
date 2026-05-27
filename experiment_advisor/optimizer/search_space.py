from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from experiment_advisor.ingestion.run_level import CONTROL_FEATURES, MODEL_FEATURES


@dataclass(frozen=True)
class SearchSpace:
    bounds: dict[str, tuple[float, float]]


def build_search_space_from_history(
    df: pd.DataFrame,
    overrides: dict[str, tuple[float, float]] | None = None,
    feature_cols: list[str] | None = None,
) -> SearchSpace:
    """从历史数据分位数构建保守搜索空间，可用 overrides 覆盖边界。"""

    features = feature_cols or [column for column in MODEL_FEATURES if column in df.columns]
    bounds: dict[str, tuple[float, float]] = {}
    for column in features:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        low = float(values.quantile(0.05))
        high = float(values.quantile(0.95))
        if low == high:
            low -= 0.5
            high += 0.5
        padding = (high - low) * 0.05
        bounds[column] = (low - padding, high + padding)
    if overrides:
        bounds.update({name: (float(low), float(high)) for name, (low, high) in overrides.items()})
    if not bounds:
        raise ValueError("No numeric features available to build search space")
    return SearchSpace(bounds=bounds)


def generate_candidates(search_space: SearchSpace, n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """在搜索空间内均匀采样候选点。"""

    if n <= 0:
        raise ValueError("n must be positive")
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise ImportError("Candidate generation requires numpy.") from exc
    rng = np.random.default_rng(seed)
    data = {}
    for name, (low, high) in search_space.bounds.items():
        data[name] = rng.uniform(low, high, size=n)
    return pd.DataFrame(data)
