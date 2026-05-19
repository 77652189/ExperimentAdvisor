from __future__ import annotations

from typing import Any, Optional, TypedDict


class AdvisorState(TypedDict, total=False):
    researcher_config: Optional[dict[str, Any]]
    optimization_mode: str
    primary_objective: str
    objective_weights: dict[str, float]
    space: Optional[dict[str, Any]]
    merge_log: Optional[dict[str, str]]
    doe_design: Optional[Any]
    effect_report: Optional[dict[str, Any]]
    current_trial: Optional[dict[str, Any]]
    report: Optional[str]
    phase: str
    doe_batch_limit: int
    completed_count: int
    best_outcomes: Optional[dict[str, Any]]
    error: Optional[str]
