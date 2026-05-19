from __future__ import annotations

from experiment_advisor.agent.state import AdvisorState
from experiment_advisor.api.endpoints import get_next_trial, initialize


def start_doe(state: AdvisorState | None = None) -> AdvisorState:
    state = state or {}
    try:
        state["doe_design"] = initialize(
            researcher_config=state.get("researcher_config"),
            optimization_mode=state.get("optimization_mode", "maximize_yield"),
            objective_weights=state.get("objective_weights"),
        )
        state["phase"] = "doe"
        state["error"] = None
    except Exception as exc:
        state["error"] = str(exc)
    return state


def recommend_next(state: AdvisorState | None = None) -> AdvisorState:
    state = state or {}
    try:
        state["current_trial"] = get_next_trial()
        state["phase"] = state["current_trial"]["phase"]
        state["error"] = None
    except Exception as exc:
        state["error"] = str(exc)
    return state
