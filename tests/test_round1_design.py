from __future__ import annotations

import pytest

from experiment_advisor.recommendation.round1_design import (
    BASELINE,
    OFAT_LEVELS,
    generate_round1_design,
    round1_template_columns,
)
from experiment_advisor.recommendation.round2_design import FIXED_LEVELS


def test_generate_round1_design_default_shape():
    df = generate_round1_design()

    assert len(df) == 18
    counts = df["run_type"].value_counts().to_dict()
    assert counts == {"ofat": 11, "combo": 4, "baseline": 3}
    assert list(df.columns) == round1_template_columns()


def test_baseline_rows_match_baseline_dict_exactly():
    df = generate_round1_design()
    baseline_rows = df[df["run_type"] == "baseline"]

    assert len(baseline_rows) == 3
    for variable, value in BASELINE.items():
        assert (baseline_rows[variable] == value).all()
    assert baseline_rows["changed_variable"].isna().all()


def test_ofat_rows_change_exactly_one_variable_from_baseline():
    df = generate_round1_design()
    ofat_rows = df[df["run_type"] == "ofat"]

    for _, row in ofat_rows.iterrows():
        changed = row["changed_variable"]
        assert changed is not None
        for variable in BASELINE:
            if variable == changed:
                assert row[variable] != BASELINE[variable]
            else:
                assert row[variable] == BASELINE[variable]

    # every non-baseline OFAT_LEVELS entry is covered exactly once
    expected = sum(
        1 for variable, levels in OFAT_LEVELS.items() for level in levels if level != BASELINE[variable]
    )
    assert len(ofat_rows) == expected == 11


def test_combo_rows_only_use_allowed_discrete_temp_and_interval_levels():
    # Regression test for the v1 bug where combo rows were sampled continuously
    # across temp_c/interval_h and produced infeasible values like 22C or 18h.
    df = generate_round1_design()
    combo_rows = df[df["run_type"] == "combo"]

    assert len(combo_rows) == 4
    assert set(combo_rows["temp_c"]) <= set(FIXED_LEVELS["temp_c"])
    assert set(combo_rows["interval_h"]) <= set(FIXED_LEVELS["interval_h"])

    # continuous combo values fall inside each variable's real range
    from experiment_advisor.recommendation.round2_design import CONTINUOUS_BOUNDS

    for variable, (lower, upper) in CONTINUOUS_BOUNDS.items():
        assert combo_rows[variable].between(lower, upper).all()


def test_combo_rows_prioritise_temp_interval_pairs_ofat_never_tests_jointly():
    df = generate_round1_design()
    combo_pairs = set(zip(df.loc[df["run_type"] == "combo", "temp_c"], df.loc[df["run_type"] == "combo", "interval_h"]))

    # OFAT only ever varies temp_c or interval_h while holding the other at its
    # baseline level, so (20, 12) and (25, 12) are never tested jointly there.
    untested_by_ofat = {(20.0, 12.0), (25.0, 12.0)}
    assert untested_by_ofat <= combo_pairs


def test_generate_round1_design_is_reproducible_with_same_seed():
    first = generate_round1_design(seed=7)
    second = generate_round1_design(seed=7)
    pd_equal = (first[["seed_od", "glucose_pct", "ph", "volume_ml"]] == second[["seed_od", "glucose_pct", "ph", "volume_ml"]]).all().all()
    assert pd_equal


def test_generate_round1_design_respects_custom_run_id_prefix():
    df = generate_round1_design(run_id_prefix="Y103-R1")
    assert df["run_id"].iloc[0] == "Y103-R1-01"
    assert df["run_id"].iloc[-1] == "Y103-R1-18"
    assert df["run_id"].is_unique
