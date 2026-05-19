from __future__ import annotations

from typing import Any

VALID_MODES = {"maximize_yield", "minimize_cost", "minimize_duration", "weighted_custom"}
OBJECTIVE_KEYS = ("yield", "cost", "duration")


def normalize_weights(weights: dict[str, float] | None, mode: str) -> dict[str, float]:
    if mode == "maximize_yield":
        return {"yield": 1.0, "cost": 0.0, "duration": 0.0}
    if mode == "minimize_cost":
        return {"yield": 0.0, "cost": 1.0, "duration": 0.0}
    if mode == "minimize_duration":
        return {"yield": 0.0, "cost": 0.0, "duration": 1.0}
    weights = weights or {}
    cleaned = {key: max(0.0, float(weights.get(key, 0.0))) for key in OBJECTIVE_KEYS}
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("objective_weights total must be > 0 for weighted_custom")
    return {key: value / total for key, value in cleaned.items()}


def primary_objective_for(mode: str) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported optimization_mode: {mode}")
    return {
        "maximize_yield": "yield",
        "minimize_cost": "cost",
        "minimize_duration": "duration",
        "weighted_custom": "advisor_score",
    }[mode]


def required_outcomes(mode: str, weights: dict[str, float]) -> list[str]:
    return list(OBJECTIVE_KEYS)


def _range_for(trials: list[dict[str, Any]], key: str) -> tuple[float, float] | None:
    values = [float(trial["outcomes"][key]) for trial in trials if key in trial.get("outcomes", {})]
    if not values:
        return None
    return min(values), max(values)


def advisor_score(
    outcomes: dict[str, float],
    weights: dict[str, float],
    completed_trials: list[dict[str, Any]],
) -> float:
    score = 0.0
    for key, weight in weights.items():
        if weight <= 0:
            continue
        if key not in outcomes:
            raise ValueError(f"missing weighted outcome: {key}")
        known = completed_trials + [{"outcomes": outcomes}]
        span = _range_for(known, key)
        if span is None:
            normalized = 0.5
        else:
            low, high = span
            if low == high:
                normalized = 0.5
            elif key == "yield":
                normalized = (float(outcomes[key]) - low) / (high - low)
            else:
                normalized = (high - float(outcomes[key])) / (high - low)
        score += weight * normalized
    return score


def is_better(objective: str, candidate: float, current: float) -> bool:
    if objective in {"cost", "duration"}:
        return candidate < current
    return candidate > current
