from experiment_advisor import complete_trial, get_next_trial, initialize
from experiment_advisor.data_access import load_state


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
