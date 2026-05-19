from experiment_advisor.bayes.constraint_handler import is_valid


def test_single_constraint_hits_forbidden_region():
    assert not is_valid({"x": 2.0}, [{"var": "x", "op": ">", "value": 1.0}])
    assert is_valid({"x": 0.5}, [{"var": "x", "op": ">", "value": 1.0}])


def test_compound_constraint():
    constraint = {
        "conditions": [{"var": "x", "op": ">", "value": 1.0}, {"var": "y", "op": "<", "value": 3.0}],
        "logic": "and",
    }
    assert not is_valid({"x": 2.0, "y": 2.0}, [constraint])
    assert is_valid({"x": 2.0, "y": 4.0}, [constraint])
