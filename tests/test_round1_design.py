from __future__ import annotations

import pytest

from experiment_advisor.recommendation.round1_design import (
    BASELINE,
    OFAT_LEVELS,
    count_round1_rows,
    generate_round1_design,
    round1_template_columns,
)
from experiment_advisor.recommendation.round2_design import CONTINUOUS_BOUNDS, FIXED_LEVELS


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


def test_zero_baseline_replicates_are_skipped():
    df = generate_round1_design(n_baseline_replicates=0)
    assert (df["run_type"] == "baseline").sum() == 0
    assert len(df) == 11 + 4  # ofat + combo unaffected


def test_ofat_disabled_entirely_skips_ofat_rows():
    df = generate_round1_design(ofat_levels=None)
    assert (df["run_type"] == "ofat").sum() == 0
    assert len(df) == 3 + 4  # baseline + combo unaffected


def test_ofat_partial_variable_coverage():
    # Only pH gets OFAT rows; every other variable stays untouched.
    df = generate_round1_design(ofat_levels={"ph": [5.0, 7.0]})
    ofat_rows = df[df["run_type"] == "ofat"]
    assert len(ofat_rows) == 2
    assert set(ofat_rows["changed_variable"]) == {"ph"}
    assert set(ofat_rows["ph"]) == {5.0, 7.0}


def test_zero_combo_points_skips_combo_rows():
    df = generate_round1_design(n_combo_points=0)
    assert (df["run_type"] == "combo").sum() == 0
    assert len(df) == 3 + 11


def test_combo_variables_subset_holds_others_at_baseline():
    df = generate_round1_design(combo_variables=["ph", "volume_ml"])
    combo_rows = df[df["run_type"] == "combo"]
    assert len(combo_rows) == 4
    # seed_od and glucose_pct were excluded from the LHS sweep -> stay at baseline
    assert (combo_rows["seed_od"] == BASELINE["seed_od"]).all()
    assert (combo_rows["glucose_pct"] == BASELINE["glucose_pct"]).all()
    # ph and volume_ml were included -> at least some rows should differ from baseline
    assert not (combo_rows["ph"] == BASELINE["ph"]).all()
    assert not (combo_rows["volume_ml"] == BASELINE["volume_ml"]).all()


def test_pure_lhs_design_disables_baseline_and_ofat():
    df = generate_round1_design(n_baseline_replicates=0, ofat_levels=None, n_combo_points=18)
    assert len(df) == 18
    assert (df["run_type"] == "combo").all()
    for variable, (lower, upper) in CONTINUOUS_BOUNDS.items():
        assert df[variable].between(lower, upper).all()
    assert set(df["temp_c"]) <= set(FIXED_LEVELS["temp_c"])
    assert set(df["interval_h"]) <= set(FIXED_LEVELS["interval_h"])


def test_pure_ofat_design_disables_combo():
    df = generate_round1_design(n_combo_points=0)
    assert len(df) == 3 + 11
    assert (df["run_type"] != "combo").all()


def test_combo_pairs_treat_disabled_ofat_variable_as_fully_untested():
    # If temp_c OFAT is turned off, every temp x interval pair is "untested" by
    # OFAT (there are no temp_c OFAT rows at all), so combo rows should be free
    # to explore the full grid, not just the two pairs OFAT-with-temp normally
    # misses.
    partial_levels = {k: v for k, v in OFAT_LEVELS.items() if k != "temp_c"}
    df = generate_round1_design(ofat_levels=partial_levels, n_combo_points=6)
    combo_rows = df[df["run_type"] == "combo"]
    pairs_seen = set(zip(combo_rows["temp_c"], combo_rows["interval_h"]))
    all_pairs = {(t, i) for t in FIXED_LEVELS["temp_c"] for i in FIXED_LEVELS["interval_h"]}
    # with temp_c OFAT off, only the interval_h-baseline-paired combos count as
    # "covered" -- everything else, including baseline-temp pairs, should be
    # prioritised, so with 6 combo points we should see more than the 2-pair
    # minimum guaranteed when temp_c OFAT is active.
    assert len(pairs_seen) >= 4
    assert pairs_seen <= all_pairs


def test_count_round1_rows_matches_actual_generation():
    configs = [
        (OFAT_LEVELS, 3, 4),
        (None, 0, 18),
        ({"ph": [5.0, 7.0]}, 3, 0),
        (OFAT_LEVELS, 0, 0),
    ]
    for ofat_levels, n_baseline, n_combo in configs:
        counts = count_round1_rows(ofat_levels, n_baseline, n_combo)
        actual = generate_round1_design(
            ofat_levels=ofat_levels, n_baseline_replicates=n_baseline, n_combo_points=n_combo
        )
        assert counts["total"] == len(actual), (ofat_levels, n_baseline, n_combo)
