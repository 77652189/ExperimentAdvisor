from __future__ import annotations

import pandas as pd

from experiment_advisor.optimizer.search_space import SearchSpace


def boundary_risk(candidates: pd.DataFrame, search_space: SearchSpace) -> pd.Series:
    """返回 0-1 边界风险，越靠近任意边界风险越高。"""

    risks = []
    for name, (low, high) in search_space.bounds.items():
        span = max(high - low, 1e-9)
        normalized = (candidates[name] - low) / span
        risk = 1.0 - (normalized - 0.5).abs() * 2.0
        risks.append(1.0 - risk.clip(0, 1))
    return pd.concat(risks, axis=1).max(axis=1) if risks else pd.Series(0.0, index=candidates.index)


def filter_constraints(candidates: pd.DataFrame, search_space: SearchSpace) -> pd.DataFrame:
    """过滤出搜索空间内候选点。后续文献/研究员约束可接入这里。"""

    mask = pd.Series(True, index=candidates.index)
    for name, (low, high) in search_space.bounds.items():
        mask &= candidates[name].between(low, high)
    return candidates.loc[mask].copy()
