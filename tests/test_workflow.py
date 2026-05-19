from experiment_advisor import complete_trial, get_next_trial, initialize
from experiment_advisor.data_access import load_state


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
    assert load_state()["phase"] == "bayes"
