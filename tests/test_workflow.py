import pytest

from experiment_advisor import complete_trial, get_next_trial, initialize
from experiment_advisor.data_access import load_state, save_design, save_pending, save_state, save_trials


def _three_variable_config():
    return {
        "variables": [
            {"name": "lactose_flow", "unit": "g/L/h", "bounds": [1, 8], "focus_range": [2, 6]},
            {"name": "temperature", "unit": "C", "bounds": [25, 37], "focus_range": [28, 35]},
            {"name": "ph", "unit": "", "bounds": [6.5, 7.5], "focus_range": [6.8, 7.2]},
        ]
    }


def test_end_to_end_yield_workflow():
    design = initialize()
    assert len(design) == 8
    for index in range(8):
        trial = get_next_trial()
        assert trial["phase"] == "doe"
        assert trial["trial_index"] == index
        complete_trial(trial["trial_index"], {"yield": 80 + index})
    next_trial = get_next_trial()
    assert next_trial["phase"] == "bayes"
    assert "parameters" in next_trial
    assert next_trial["parameters"] != trial["parameters"]
    assert load_state()["phase"] == "bayes"


def test_bayes_keeps_initialized_variable_space():
    initialize(researcher_config=_three_variable_config())
    for index in range(8):
        trial = get_next_trial()
        assert set(trial["parameters"]) == {"lactose_flow", "temperature", "ph"}
        complete_trial(trial["trial_index"], {"yield": 80 + index})
    next_trial = get_next_trial()
    assert next_trial["phase"] == "bayes"
    assert set(next_trial["parameters"]) == {"lactose_flow", "temperature", "ph"}


def test_bayes_recovers_variable_space_from_existing_doe_design():
    save_design(
        2,
        [
            {"batch_index": 0, "phase": "doe", "parameters": {"x": 1.0, "y": 10.0, "PH": 7.0}, "warnings": []},
            {"batch_index": 1, "phase": "doe", "parameters": {"x": 2.0, "y": 20.0, "PH": 8.0}, "warnings": []},
        ],
    )
    save_trials(
        [
            {"trial_index": 0, "phase": "doe", "parameters": {"x": 1.0, "y": 10.0, "PH": 7.0}, "outcomes": {"yield": 1.0}},
            {"trial_index": 1, "phase": "doe", "parameters": {"x": 2.0, "y": 20.0, "PH": 8.0}, "outcomes": {"yield": 2.0}},
        ]
    )
    save_pending([])
    save_state(
        {
            "phase": "bayes",
            "doe_batch_limit": 2,
            "completed_count": 2,
            "next_doe_index": 2,
            "optimization_mode": "maximize_yield",
            "primary_objective": "yield",
            "objective_weights": {"yield": 1.0, "cost": 0.0, "duration": 0.0},
            "effect_report": None,
            "best_outcomes": {"yield": {"value": 2.0, "trial_index": 1}},
            "initialized_at": None,
            "last_updated": None,
        }
    )
    assert "space" not in load_state()
    next_trial = get_next_trial()
    assert set(next_trial["parameters"]) == {"x", "y", "PH"}
    assert set(load_state()["space"]) == {"x", "y", "PH"}


def test_custom_doe_batch_limit_switches_after_configured_count():
    design = initialize(doe_batch_limit=3)
    assert len(design) == 3
    for index in range(3):
        trial = get_next_trial()
        assert trial["phase"] == "doe"
        complete_trial(trial["trial_index"], {"yield": 80 + index})
    assert get_next_trial()["phase"] == "bayes"


def test_bayes_trial_limit_stops_recommendations():
    initialize(doe_batch_limit=2, bayes_trial_limit=1)
    for index in range(2):
        trial = get_next_trial()
        complete_trial(trial["trial_index"], {"yield": 80 + index})
    bayes_trial = get_next_trial()
    assert bayes_trial["phase"] == "bayes"
    complete_trial(bayes_trial["trial_index"], {"yield": 90})
    with pytest.raises(ValueError, match="Bayes trial limit reached"):
        get_next_trial()


def test_cost_mode_records_cost_as_primary_objective():
    initialize(optimization_mode="minimize_cost", doe_batch_limit=1)
    trial = get_next_trial()
    complete_trial(trial["trial_index"], {"cost": 1.2})
    state = load_state()
    assert state["primary_objective"] == "cost"
    assert state["best_outcomes"]["cost"]["value"] == 1.2


def test_duration_mode_records_duration_as_primary_objective():
    initialize(optimization_mode="minimize_duration", doe_batch_limit=1)
    trial = get_next_trial()
    complete_trial(trial["trial_index"], {"duration": 48})
    state = load_state()
    assert state["primary_objective"] == "duration"
    assert state["best_outcomes"]["duration"]["value"] == 48


def test_weighted_mode_requires_selected_weighted_outcomes():
    initialize(
        optimization_mode="weighted_custom",
        objective_weights={"yield": 0.7, "cost": 0.3, "duration": 0.0},
        doe_batch_limit=1,
    )
    trial = get_next_trial()
    with pytest.raises(ValueError, match="missing outcomes: cost"):
        complete_trial(trial["trial_index"], {"yield": 10})
