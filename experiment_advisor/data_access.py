from __future__ import annotations

from typing import Any

from experiment_advisor.paths import (
    DOE_DESIGN_PATH,
    EXPERIMENT_STATE_PATH,
    KNOWLEDGE_RULES_PATH,
    PENDING_TRIALS_PATH,
    TRIAL_RESULTS_PATH,
)
from experiment_advisor.storage import now_iso, read_json, write_json


def load_knowledge_rules() -> dict[str, Any] | None:
    payload = read_json(KNOWLEDGE_RULES_PATH, None)
    if not payload:
        return None
    if not payload.get("variables") and not payload.get("hard_constraints") and not payload.get("warm_start_points"):
        return None
    return payload


def load_constraints() -> list[dict[str, Any]]:
    knowledge_rules = load_knowledge_rules()
    return list((knowledge_rules or {}).get("hard_constraints", []))


def load_state() -> dict[str, Any]:
    return read_json(
        EXPERIMENT_STATE_PATH,
        {
            "phase": "doe",
            "doe_batch_limit": 8,
            "completed_count": 0,
            "next_doe_index": 0,
            "optimization_mode": "maximize_yield",
            "primary_objective": "yield",
            "objective_weights": {"yield": 1.0, "cost": 0.0, "duration": 0.0},
            "effect_report": None,
            "best_outcomes": {},
            "initialized_at": None,
            "last_updated": None,
        },
    )


def save_state(state: dict[str, Any]) -> None:
    state["last_updated"] = now_iso()
    write_json(EXPERIMENT_STATE_PATH, state)


def load_design() -> dict[str, Any]:
    return read_json(DOE_DESIGN_PATH, {"generated_at": None, "batch_limit": 8, "design": []})


def save_design(batch_limit: int, design: list[dict[str, Any]]) -> None:
    write_json(DOE_DESIGN_PATH, {"generated_at": now_iso(), "batch_limit": batch_limit, "design": design})


def load_trials() -> list[dict[str, Any]]:
    return list(read_json(TRIAL_RESULTS_PATH, {"trials": []}).get("trials", []))


def save_trials(trials: list[dict[str, Any]]) -> None:
    write_json(TRIAL_RESULTS_PATH, {"trials": trials})


def load_pending() -> list[dict[str, Any]]:
    return list(read_json(PENDING_TRIALS_PATH, {"pending": []}).get("pending", []))


def save_pending(pending: list[dict[str, Any]]) -> None:
    write_json(PENDING_TRIALS_PATH, {"pending": pending[:1]})
