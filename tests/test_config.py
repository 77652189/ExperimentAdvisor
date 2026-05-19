from experiment_advisor.config.space_merger import merge_space


def test_merge_space_priority():
    defaults = {"variables": [{"name": "x", "unit": "u", "bounds": [0, 10], "focus_range": [2, 8]}]}
    literature = {"variables": [{"name": "x", "unit": "u", "bounds": [1, 9], "focus_range": [3, 7]}]}
    researcher = {"variables": [{"name": "x", "unit": "u", "bounds": [2, 6], "focus_range": [3, 5]}]}
    space, log = merge_space(defaults, literature, researcher)
    assert space["x"]["bounds"] == [2.0, 6.0]
    assert log["x"] == "researcher"
