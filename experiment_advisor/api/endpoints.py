from __future__ import annotations

from typing import Any

from experiment_advisor.bayes.optimizer import ExperimentOptimizer
from experiment_advisor.bayes.scoring import (
    advisor_score,
    is_better,
    normalize_weights,
    primary_objective_for,
    required_outcomes,
)
from experiment_advisor.data_access import (
    load_constraints,
    load_design,
    load_pending,
    load_state,
    load_trials,
    save_pending,
    save_state,
    save_trials,
)
from experiment_advisor.doe.design_generator import generate_design
from experiment_advisor.doe.effect_analyzer import analyze_effects
from experiment_advisor.doe.space_builder import build_space
from experiment_advisor.storage import now_iso


def _as_dataframe(rows: list[dict[str, Any]]):
    flattened = [{"batch_index": row["batch_index"], **row["parameters"]} for row in rows]
    try:
        import pandas as pd

        return pd.DataFrame(flattened)
    except Exception:
        return flattened


def _space_from_design() -> dict[str, dict[str, Any]]:
    design = load_design().get("design", [])
    values: dict[str, list[float]] = {}
    for row in design:
        for name, value in row.get("parameters", {}).items():
            if value is not None:
                values.setdefault(name, []).append(float(value))
    space: dict[str, dict[str, Any]] = {}
    for name, series in values.items():
        lower, upper = min(series), max(series)
        if lower == upper:
            lower -= 0.5
            upper += 0.5
        padding = (upper - lower) * 0.15
        space[name] = {
            "bounds": [round(lower - padding, 6), round(upper + padding, 6)],
            "focus": [round(lower, 6), round(upper, 6)],
            "unit": "",
        }
    return space


def initialize(
    researcher_config: dict | None = None,
    optimization_mode: str = "maximize_yield",
    objective_weights: dict | None = None,
):
    weights = normalize_weights(objective_weights, optimization_mode)
    primary = primary_objective_for(optimization_mode)
    space, _ = build_space(researcher_config)
    design = generate_design(space, n_trials=8)
    now = now_iso()
    save_trials([])
    save_pending([])
    save_state(
        {
            "phase": "doe",
            "doe_batch_limit": len(design),
            "completed_count": 0,
            "next_doe_index": 0,
            "optimization_mode": optimization_mode,
            "primary_objective": primary,
            "objective_weights": weights,
            "space": space,
            "effect_report": None,
            "best_outcomes": {},
            "initialized_at": now,
            "last_updated": now,
        }
    )
    return _as_dataframe(design)


def _validate_outcomes(outcomes: dict[str, Any], state: dict[str, Any]) -> dict[str, float]:
    weights = state.get("objective_weights", {})
    needed = required_outcomes(state.get("optimization_mode", "maximize_yield"), weights)
    missing = [key for key in needed if key not in outcomes]
    if missing:
        raise ValueError(f"missing outcomes: {', '.join(missing)}")
    return {key: float(value) for key, value in outcomes.items() if key in {"yield", "cost", "duration"}}


def _update_best_outcomes(state: dict[str, Any], trial: dict[str, Any]) -> None:
    best = dict(state.get("best_outcomes", {}))
    for key, value in trial["outcomes"].items():
        numeric = float(value)
        current = best.get(key)
        if current is None or is_better(key, numeric, float(current["value"])):
            best[key] = {"value": numeric, "trial_index": trial["trial_index"]}
    state["best_outcomes"] = best


def complete_trial(trial_index: int, outcomes: dict, notes: str = "") -> None:
    pending = load_pending()
    match = next((item for item in pending if item.get("trial_index") == trial_index), None)
    if match is None:
        raise ValueError(f"trial_index {trial_index} is not pending")
    state = load_state()
    trials = load_trials()
    cleaned_outcomes = _validate_outcomes(outcomes, state)
    if state.get("optimization_mode") == "weighted_custom":
        cleaned_outcomes["advisor_score"] = advisor_score(cleaned_outcomes, state["objective_weights"], trials)
    trial = {
        "trial_index": trial_index,
        "phase": match["phase"],
        "parameters": match["parameters"],
        "outcomes": cleaned_outcomes,
        "notes": notes,
        "recorded_at": now_iso(),
    }
    trials.append(trial)
    save_trials(trials)
    save_pending([])

    state["completed_count"] = int(state.get("completed_count", 0)) + 1
    if match["phase"] == "doe":
        state["next_doe_index"] = int(state.get("next_doe_index", 0)) + 1
    _update_best_outcomes(state, trial)
    save_state(state)


def _pending_response(pending: dict[str, Any]) -> dict[str, Any]:
    return dict(pending)


def get_next_trial() -> dict[str, Any]:
    pending = load_pending()
    if pending:
        return _pending_response(pending[0])

    state = load_state()
    space = state.get("space")
    if not space:
        space = _space_from_design()
        if space:
            state["space"] = space
            save_state(state)
        else:
            space, _ = build_space()
    phase = state.get("phase", "doe")
    if phase == "doe" and int(state.get("completed_count", 0)) >= int(state.get("doe_batch_limit", 8)):
        effect_report = analyze_effects(space)
        if effect_report.get("ready_for_bayes"):
            state = load_state()
            state["phase"] = "bayes"
            save_state(state)
            phase = "bayes"

    if phase == "doe":
        design = load_design().get("design", [])
        index = int(state.get("next_doe_index", 0))
        if index >= len(design):
            raise ValueError("DOE design exhausted but phase has not switched to bayes")
        row = design[index]
        result = {
            "trial_index": row["batch_index"],
            "phase": "doe",
            "parameters": row["parameters"],
            "warnings": row.get("warnings", []),
        }
    else:
        optimizer = ExperimentOptimizer(
            space=space,
            constraints=load_constraints(),
            optimization_mode=state["optimization_mode"],
            objective_weights=state.get("objective_weights"),
        )
        result = optimizer.get_next_trial()

    result["suggested_at"] = now_iso()
    save_pending([result])
    return result
