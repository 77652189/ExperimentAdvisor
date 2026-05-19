from experiment_advisor import complete_trial, get_next_trial, initialize
from experiment_advisor.data_access import load_trials


def test_weighted_custom_records_advisor_score():
    initialize(optimization_mode="weighted_custom", objective_weights={"yield": 0.5, "cost": 0.5, "duration": 0.0})
    trial = get_next_trial()
    complete_trial(trial["trial_index"], {"yield": 90.0, "cost": 1.2})
    recorded = load_trials()[0]
    assert "advisor_score" in recorded["outcomes"]
