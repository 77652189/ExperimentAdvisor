from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiment_advisor.recommendation.round2_design import (
    ALL_VARIABLES,
    FIXED_LEVELS,
    TARGET_COL,
    FactorEffect,
    analyze_interval_interaction,
    analyze_round1_effects,
    assemble_round2_design,
    canonical_analysis,
    classify_response_surface_case,
    effect_confidence_interval,
    estimate_baseline_noise,
    evaluate_response_surface,
    find_reusable_round1_row,
    fit_ccd_response_surface,
    generate_ccd,
    generate_round2_extension_design,
    gp_leave_one_out_cv,
    od600_threshold,
    optimize_joint_desirability,
    plan_round2,
    predict_with_confidence_interval,
    resolve_round2_variables,
    response_surface_grid,
    sensitivity_analysis,
    summarize_bo_recommendation,
    summarize_response_surface,
)

BASELINE = {
    "seed_od": 3.0,
    "glucose_pct": 1.0,
    "interval_h": 24.0,
    "ph": 6.0,
    "temp_c": 30.0,
    "volume_ml": 50.0,
}


def _row(run_id: str, yield_g_per_l: float, od600: float, **overrides) -> dict:
    row = {**BASELINE, "run_id": run_id, "yield_g_per_l": yield_g_per_l, "od600": od600}
    row.update(overrides)
    return row


def _round1_df() -> pd.DataFrame:
    """Synthetic round-1 data: temp_c and ph carry real signal, everything else
    is within noise. ph peaks AT baseline (interior optimum, both flanks worse) --
    this is the case that must not be mis-ranked as "no effect". temp_c is
    significant but fixed-level, so it must never enter a CCD. One combo row
    (R1-16) is deliberately way off the additive prediction to exercise the
    interaction flag; the others are additive-consistent by construction (built
    from exact OFAT-tested levels so the expected prediction is exact arithmetic,
    not interpolation)."""

    rows = [
        _row("R1-01", 0.585, 43),
        _row("R1-02", 0.620, 45),
        _row("R1-03", 0.655, 47),
        _row("R1-04", 0.65, 44, seed_od=9.0),
        _row("R1-05", 0.60, 44, seed_od=15.0),
        _row("R1-06", 0.64, 44, glucose_pct=0.5),
        _row("R1-07", 0.59, 44, glucose_pct=1.5),
        _row("R1-08", 0.63, 44, interval_h=12.0),
        _row("R1-09", 0.50, 44, ph=5.0),
        _row("R1-10", 0.49, 44, ph=7.0),
        _row("R1-11", 0.81, 38, temp_c=20.0),
        _row("R1-12", 0.71, 41, temp_c=25.0),
        _row("R1-13", 0.61, 44, volume_ml=25.0),
        _row("R1-14", 0.66, 44, volume_ml=75.0),
        # combo A: seed_od + volume_ml, both at exact OFAT levels -> predicted
        # 0.62 + (0.65-0.62) + (0.66-0.62) = 0.69, observed close -> no interaction
        _row("R1-15", 0.695, 44, seed_od=9.0, volume_ml=75.0),
        # combo B: ph + temp_c at exact OFAT levels -> predicted
        # 0.62 + (0.50-0.62) + (0.81-0.62) = 0.69, observed way off -> interaction
        _row("R1-16", 1.10, 30, ph=5.0, temp_c=20.0),
        # combo C: glucose_pct + volume_ml -> predicted 0.62-0.03-0.01=0.58, close
        _row("R1-17", 0.585, 44, glucose_pct=1.5, volume_ml=25.0),
        # combo D: seed_od + interval_h -> predicted 0.62-0.02+0.01=0.61, close
        _row("R1-18", 0.615, 44, seed_od=15.0, interval_h=12.0),
    ]
    return pd.DataFrame(rows)


def test_estimate_baseline_noise():
    noise = estimate_baseline_noise(_round1_df(), BASELINE)
    assert noise["n_replicates"] == 3
    assert noise["baseline_mean"] == pytest.approx(0.620, abs=1e-9)
    assert noise["baseline_sd"] == pytest.approx(0.035, abs=1e-6)
    assert noise["threshold"] == pytest.approx(0.07, abs=1e-6)


def test_analyze_round1_effects_classifies_variables_correctly():
    result = analyze_round1_effects(_round1_df(), BASELINE)
    effects = result["effects"]

    # Monotonic, significant, fixed-level: temp_c must never be a CCD candidate
    # even though it clearly matters -- round 1 already tested every level it can take.
    assert effects["temp_c"].kind == "fixed_level"
    assert effects["temp_c"].significant is True
    assert effects["temp_c"].best_value == pytest.approx(20.0)

    # Interior-peak, significant, continuous: best sits AT baseline, both flanks
    # worse. This is exactly the case the effect_magnitude fix targets.
    assert effects["ph"].kind == "continuous"
    assert effects["ph"].significant is True
    assert effects["ph"].best_value == pytest.approx(6.0)
    assert effects["ph"].effect_magnitude == pytest.approx(0.13, abs=1e-6)
    assert effects["ph"].at_lower_bound is False
    assert effects["ph"].at_upper_bound is False

    for variable in ["seed_od", "glucose_pct", "interval_h", "volume_ml"]:
        assert effects[variable].significant is False, variable

    # every variable gets a confidence interval around its effect_magnitude,
    # widened by the pooled baseline variance (3 replicates -> df=2)
    for variable, effect in effects.items():
        assert effect.ci_low <= effect.effect_magnitude <= effect.ci_high, variable
        assert effect.ci_low >= 0.0, variable

    # combo interactions: 4 rows changed 2+ variables; only R1-16 should be flagged
    combos = {item["row_id"]: item for item in result["combo_interactions"]}
    assert set(combos) == {"R1-15", "R1-16", "R1-17", "R1-18"}
    assert combos["R1-16"]["possible_interaction"] is True
    assert combos["R1-15"]["possible_interaction"] is False
    assert combos["R1-17"]["possible_interaction"] is False
    assert combos["R1-18"]["possible_interaction"] is False


def test_effect_confidence_interval_matches_hand_derived_t_margin():
    # t.ppf(0.975, df=2) ~= 4.303 (standard table value)
    low, high = effect_confidence_interval(effect=0.19, baseline_sd=0.035, n_baseline_replicates=3)
    expected_margin = 0.035 * 4.303 * (4.0 / 3.0) ** 0.5
    assert high - 0.19 == pytest.approx(expected_margin, rel=1e-3)
    assert 0.19 - low == pytest.approx(expected_margin, rel=1e-3)


def test_effect_confidence_interval_widens_with_fewer_replicates():
    _, high_3 = effect_confidence_interval(effect=0.1, baseline_sd=0.02, n_baseline_replicates=3)
    _, high_5 = effect_confidence_interval(effect=0.1, baseline_sd=0.02, n_baseline_replicates=5)
    assert (high_3 - 0.1) > (high_5 - 0.1)


def test_effect_confidence_interval_undefined_below_two_replicates():
    low, high = effect_confidence_interval(effect=0.1, baseline_sd=0.02, n_baseline_replicates=1)
    assert pd.isna(low) and pd.isna(high)


def test_resolve_round2_variables_keeps_interior_peak_active():
    result = analyze_round1_effects(_round1_df(), BASELINE)
    resolved = resolve_round2_variables(result["effects"], max_active_variables=3)

    assert resolved["ph"].active is True
    assert resolved["temp_c"].active is False
    assert resolved["temp_c"].fixed_value == pytest.approx(20.0)
    assert resolved["interval_h"].active is False
    assert resolved["interval_h"].fixed_value == pytest.approx(12.0)
    for variable, expected in [("seed_od", 9.0), ("glucose_pct", 0.5), ("volume_ml", 75.0)]:
        assert resolved[variable].active is False
        assert resolved[variable].fixed_value == pytest.approx(expected)


def test_resolve_round2_variables_ranks_by_effect_magnitude_not_best_minus_baseline():
    # Two significant continuous variables: "peak" has best==baseline (magnitude 0.13
    # from either flank) but a naive best-minus-baseline score of 0; "edge" has a
    # smaller true magnitude (0.08) but a nonzero best-minus-baseline. The fix must
    # keep "peak" ranked first.
    peak = FactorEffect(
        variable="ph", kind="continuous", baseline_value=6.0,
        tested_values={5.0: 0.49, 6.0: 0.62, 7.0: 0.50}, significant=True,
        best_value=6.0, best_target=0.62, effect_magnitude=0.13,
    )
    edge = FactorEffect(
        variable="seed_od", kind="continuous", baseline_value=3.0,
        tested_values={3.0: 0.62, 9.0: 0.70, 15.0: 0.66}, significant=True,
        best_value=9.0, best_target=0.70, effect_magnitude=0.08,
    )
    resolved = resolve_round2_variables({"ph": peak, "seed_od": edge}, max_active_variables=1)
    assert resolved["ph"].active is True
    assert resolved["seed_od"].active is False
    assert resolved["seed_od"].note_kind == "overflow"


def test_generate_ccd_k1_matches_ofat_step_half():
    result = analyze_round1_effects(_round1_df(), BASELINE)
    ph_effect = result["effects"]["ph"]

    rows = generate_ccd([ph_effect], step_fraction=0.5)

    assert len(rows) == 2 + 3  # low/high + 3 default centers for k=1
    values = sorted(row["ph"] for row in rows)
    assert values[0] == pytest.approx(5.5)  # baseline(6.0) - half of the 1.0 OFAT step
    assert values[-1] == pytest.approx(6.5)
    assert all(v == pytest.approx(5.5) or v == pytest.approx(6.5) or v == pytest.approx(6.0) for v in values)


def test_generate_ccd_k2_face_centered_shape():
    ph = FactorEffect(
        variable="ph", kind="continuous", baseline_value=6.0,
        tested_values={5.0: 0.5, 6.0: 0.62, 7.0: 0.49}, significant=True,
        best_value=6.0, best_target=0.62, effect_magnitude=0.13,
    )
    temp = FactorEffect(
        variable="seed_od", kind="continuous", baseline_value=3.0,
        tested_values={3.0: 0.62, 9.0: 0.7, 15.0: 0.6}, significant=True,
        best_value=9.0, best_target=0.70, effect_magnitude=0.08,
    )
    rows = generate_ccd([ph, temp], step_fraction=0.5, n_center=5)
    # 2^2 factorial + 2*2 axial + 5 centers = 13
    assert len(rows) == 4 + 4 + 5
    for row in rows:
        assert 5.0 <= row["ph"] <= 7.0
        assert 3.0 <= row["seed_od"] <= 15.0


def test_generate_ccd_shifts_band_inward_at_a_hard_boundary():
    # best_value pinned at the upper bound (75) -- a naive +-step around it would
    # reach 87.5, which is outside CONTINUOUS_BOUNDS["volume_ml"]. The design must
    # shift the whole band inward rather than collapsing one corner onto the center.
    pinned = FactorEffect(
        variable="volume_ml", kind="continuous", baseline_value=50.0,
        tested_values={25.0: 0.61, 50.0: 0.62, 75.0: 0.80}, significant=True,
        best_value=75.0, best_target=0.80, effect_magnitude=0.18, at_upper_bound=True,
    )
    rows = generate_ccd([pinned], step_fraction=0.5, n_center=2)
    values = {row["volume_ml"] for row in rows}
    assert max(values) <= 75.0
    assert min(values) >= 25.0
    assert len(values) > 1  # not degenerate: low != high


def test_find_reusable_round1_row():
    df = _round1_df()
    reused = find_reusable_round1_row(dict(BASELINE), df, list(BASELINE))
    assert reused in {"R1-01", "R1-02", "R1-03"}

    novel = dict(BASELINE)
    novel["ph"] = 5.5
    novel["seed_od"] = 9.0
    assert find_reusable_round1_row(novel, df, list(BASELINE)) is None


def test_od600_threshold_uses_baseline_mean_times_fraction():
    result = od600_threshold(_round1_df(), BASELINE, fraction=0.7)
    assert result["baseline_od_mean"] == pytest.approx(45.0)
    assert result["threshold"] == pytest.approx(31.5)


def test_plan_round2_end_to_end():
    plan = plan_round2(_round1_df(), BASELINE)

    assert plan.active_variables == ["ph"]
    assert plan.fixed_values["seed_od"] == pytest.approx(9.0)
    assert plan.fixed_values["glucose_pct"] == pytest.approx(0.5)
    assert plan.fixed_values["interval_h"] == pytest.approx(12.0)
    assert plan.fixed_values["temp_c"] == pytest.approx(20.0)
    assert plan.fixed_values["volume_ml"] == pytest.approx(75.0)

    assert len(plan.design_rows) == 5  # k=1 CCD: 2 + 3 centers
    for row in plan.design_rows:
        assert 5.0 <= row["ph"] <= 7.0
        assert row["temp_c"] == pytest.approx(20.0)

    assert plan.od_threshold["threshold"] == pytest.approx(45.0 * 0.7)
    assert any(item["possible_interaction"] for item in plan.combo_interactions)

    # plan.effects exposes the same FactorEffect objects analyze_round1_effects
    # computes internally, so callers (e.g. the effect-magnitude chart) don't
    # have to re-run the analysis themselves.
    assert set(plan.effects) == set(BASELINE)
    assert plan.effects["ph"].significant is True
    assert plan.effects["ph"].ci_low <= plan.effects["ph"].effect_magnitude <= plan.effects["ph"].ci_high


def test_analyze_round1_effects_flags_untested_fixed_level_variable():
    # Drop the two temp_c OFAT rows (R1-11 at 20, R1-12 at 25) and the one combo
    # row that also moves temp_c (R1-16) -- every remaining row sits at temp_c's
    # baseline (30), so temp_c was never actually tested this round. That must
    # read as "untested", not get folded into "tested and found insignificant"
    # (real 2026-08 Y103 round 1 data skipped temp_c's OFAT block entirely).
    df = _round1_df()
    df = df[df["temp_c"] == BASELINE["temp_c"]].reset_index(drop=True)

    result = analyze_round1_effects(df, BASELINE)
    effects = result["effects"]

    assert effects["temp_c"].untested is True
    assert effects["temp_c"].significant is False
    assert effects["temp_c"].effect_magnitude == pytest.approx(0.0)
    # ph's own OFAT rows (R1-09/10) don't touch temp_c, so it's still fully tested
    assert effects["ph"].untested is False


def test_resolve_round2_variables_notes_untested_variable_distinctly():
    df = _round1_df()
    df = df[df["temp_c"] == BASELINE["temp_c"]].reset_index(drop=True)

    plan = plan_round2(df, BASELINE)

    assert "temp_c" in plan.untested_notes
    assert "temp_c" not in plan.boundary_notes
    assert "temp_c" not in plan.overflow_notes
    assert plan.fixed_values["temp_c"] == pytest.approx(BASELINE["temp_c"])


def test_recommend_round2_bo_batch():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")
    from experiment_advisor.recommendation.round2_design import recommend_round2_bo_batch

    plan = plan_round2(_round1_df(), BASELINE)
    result = recommend_round2_bo_batch(
        _round1_df(),
        fixed_values=plan.fixed_values,
        active_variables=plan.active_variables,
        od_threshold=plan.od_threshold["threshold"],
        n_batch=5,
        n_candidates=300,
        seed=0,
    )

    recs = result["recommendations"]
    assert 0 < len(recs) <= 5
    for rec in recs:
        assert rec["predicted_od600"] >= plan.od_threshold["threshold"] - 1e-9
        assert 5.0 <= rec["ph"] <= 7.0
        assert rec["temp_c"] in {20.0, 25.0, 30.0}
        assert rec["interval_h"] in {12.0, 24.0}
    # sorted by predicted yield, descending
    yields = [rec["predicted_yield"] for rec in recs]
    assert yields == sorted(yields, reverse=True)


def test_recommend_round2_bo_batch_exposes_fitted_models_for_pdp():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")
    from experiment_advisor.recommendation.round2_design import gp_partial_dependence, recommend_round2_bo_batch

    plan = plan_round2(_round1_df(), BASELINE)
    result = recommend_round2_bo_batch(
        _round1_df(),
        fixed_values=plan.fixed_values,
        active_variables=plan.active_variables,
        od_threshold=plan.od_threshold["threshold"],
        n_batch=5,
        n_candidates=300,
        seed=0,
    )

    assert result["yield_model"] is not None
    assert result["od_model"] is not None
    assert result["feature_cols"] == list(ALL_VARIABLES)

    anchor = result["recommendations"][0]

    # degenerate 1-point "sweep" (lower == upper == anchor's own value) must
    # reproduce the anchor's own predicted_yield almost exactly -- same
    # model, same input point, proving this queries the model batch already
    # fit rather than a fresh refit.
    at_anchor = gp_partial_dependence(
        result["yield_model"], result["feature_cols"], "ph", anchor, lower=anchor["ph"], upper=anchor["ph"], resolution=1
    )
    assert at_anchor["mean"][0] == pytest.approx(anchor["predicted_yield"], abs=1e-4)

    # a real sweep across ph's full original range, holding everything else
    # at the anchor -- shape/type checks, plus std must be non-negative
    # everywhere (a variance can't be negative).
    swept = gp_partial_dependence(result["yield_model"], result["feature_cols"], "ph", anchor, lower=5.0, upper=7.0, resolution=9)
    assert swept["x"][0] == pytest.approx(5.0)
    assert swept["x"][-1] == pytest.approx(7.0)
    assert len(swept["mean"]) == 9
    assert (swept["std"] >= 0).all()


def test_generate_round2_extension_design_block_sizes_and_values():
    plan = plan_round2(_round1_df(), BASELINE)  # active_variables == ["ph"]

    extension = generate_round2_extension_design(
        plan,
        BASELINE,
        extra_interval_levels=[36.0],
        interaction_variable="glucose_pct",
        interaction_levels=[0.5, 1.5],
        n_noise_reference=2,
        n_lhs=10,
        seed=0,
    )

    counts = extension["run_type"].value_counts()
    assert counts["interval_interaction"] == 2  # 1 new interval level x 2 interaction levels
    assert counts["noise_reference"] == 2  # 1 new interval level x n_noise_reference
    assert counts["lhs"] == 10
    assert len(extension) == 14

    interaction_rows = extension[extension["run_type"] == "interval_interaction"]
    assert set(interaction_rows["glucose_pct"]) == {0.5, 1.5}
    assert (interaction_rows["interval_h"] == 36.0).all()
    # everything not being probed this block stays at plan.fixed_values / baseline
    assert (interaction_rows["temp_c"] == plan.fixed_values["temp_c"]).all()

    noise_rows = extension[extension["run_type"] == "noise_reference"]
    assert (noise_rows["interval_h"] == 36.0).all()
    assert (noise_rows["glucose_pct"] == BASELINE["glucose_pct"]).all()

    lhs_rows = extension[extension["run_type"] == "lhs"]
    assert set(lhs_rows["interval_h"].unique()) <= {24.0, 36.0}
    for variable in ("seed_od", "glucose_pct", "ph", "volume_ml"):
        lower, upper = {"seed_od": (3.0, 15.0), "glucose_pct": (0.5, 1.5), "ph": (5.0, 7.0), "volume_ml": (25.0, 75.0)}[
            variable
        ]
        assert lhs_rows[variable].between(lower, upper).all()

    # run_ids are contiguous starting at 1 by default
    assert list(extension["run_id"]) == [f"R2-{index:02d}" for index in range(1, len(extension) + 1)]


def test_generate_round2_extension_design_is_reproducible():
    plan = plan_round2(_round1_df(), BASELINE)

    first = generate_round2_extension_design(plan, BASELINE, extra_interval_levels=[36.0], n_lhs=10, seed=7)
    second = generate_round2_extension_design(plan, BASELINE, extra_interval_levels=[36.0], n_lhs=10, seed=7)
    pd.testing.assert_frame_equal(first, second)

    different_seed = generate_round2_extension_design(plan, BASELINE, extra_interval_levels=[36.0], n_lhs=10, seed=99)
    lhs_first = first[first["run_type"] == "lhs"][ALL_VARIABLES].reset_index(drop=True)
    lhs_different = different_seed[different_seed["run_type"] == "lhs"][ALL_VARIABLES].reset_index(drop=True)
    assert not lhs_first.equals(lhs_different)


def test_assemble_round2_design_combines_ccd_and_extension_with_contiguous_ids():
    round1_df = _round1_df()
    plan = plan_round2(round1_df, BASELINE)

    combined = assemble_round2_design(
        plan,
        BASELINE,
        round1_df,
        extra_interval_levels=[36.0],
        n_noise_reference=2,
        n_lhs=10,
        seed=0,
    )

    n_ccd = len(plan.design_rows)
    assert (combined["run_type"] == "ccd").sum() == n_ccd
    assert list(combined["run_id"]) == [f"R2-{index:02d}" for index in range(1, len(combined) + 1)]

    # the one CCD point plan_round2 found reusable from round 1 keeps that marker
    # AND is pre-filled with that round-1 run's actual measured values, not left blank
    reused_in_plan = [row.get("reused_from_round1") for row in plan.design_rows if row.get("reused_from_round1")]
    reused_rows = combined.loc[combined["run_type"] == "ccd"].dropna(subset=["reused_from_round1"])
    assert reused_rows["reused_from_round1"].tolist() == reused_in_plan
    for _, row in reused_rows.iterrows():
        source = round1_df.set_index("run_id").loc[row["reused_from_round1"]]
        assert row["yield_g_per_l"] == pytest.approx(source["yield_g_per_l"])
        assert row["od600"] == pytest.approx(source["od600"])

    # combining twice with the same arguments reproduces the same sheet
    combined_again = assemble_round2_design(
        plan, BASELINE, round1_df, extra_interval_levels=[36.0], n_noise_reference=2, n_lhs=10, seed=0
    )
    pd.testing.assert_frame_equal(combined, combined_again)


def test_fit_ccd_response_surface_recovers_a_noiseless_quadratic_k1():
    # y = 10 - (ph-6)^2 is exactly representable by the fitted model (which
    # includes a ph^2 term) -- R^2 must come out to 1, and the grid-search
    # optimum must land on the true peak (ph=6, y=10).
    ph_effect = FactorEffect(
        variable="ph", kind="continuous", baseline_value=6.0,
        tested_values={5.0: 9.75, 6.0: 10.0, 7.0: 9.75}, significant=True,
        best_value=6.0, best_target=10.0, effect_magnitude=0.25,
    )
    df = pd.DataFrame(generate_ccd([ph_effect], step_fraction=0.5))  # ph: 5.5, 6.5, 6.0, 6.0, 6.0
    df["yield_g_per_l"] = 10.0 - (df["ph"] - 6.0) ** 2

    fit = fit_ccd_response_surface(df, ["ph"])

    assert fit.n_points == 5
    assert fit.n_params == 3  # intercept, ph, ph^2
    assert fit.r_squared == pytest.approx(1.0, abs=1e-9)
    # k=1's default design (2 non-center + 3 center reps) has residual df == pure
    # error df once the 3 parameters are fit -- there's no df left to test lack
    # of fit at all, and that must read as "can't tell", not a false "0% lack of fit".
    assert fit.lack_of_fit is None
    assert fit.optimum["ph"] == pytest.approx(6.0, abs=0.05)
    assert fit.predicted_optimum == pytest.approx(10.0, abs=1e-6)


def _k2_ccd_df() -> pd.DataFrame:
    ph_effect = FactorEffect(
        variable="ph", kind="continuous", baseline_value=6.0,
        tested_values={5.0: 0.5, 6.0: 0.62, 7.0: 0.5}, significant=True,
        best_value=6.0, best_target=0.62, effect_magnitude=0.12,
    )
    glucose_effect = FactorEffect(
        variable="glucose_pct", kind="continuous", baseline_value=1.0,
        tested_values={0.5: 0.55, 1.0: 0.62, 1.5: 0.55}, significant=True,
        best_value=1.0, best_target=0.62, effect_magnitude=0.07,
    )
    rows = generate_ccd([ph_effect, glucose_effect], step_fraction=0.5, n_center=5)
    return pd.DataFrame(rows)


def test_fit_ccd_response_surface_lack_of_fit_stays_insignificant_for_a_clean_fit():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    center_mask = (df["ph"] == 6.0) & (df["glucose_pct"] == 1.0)
    assert center_mask.sum() == 5
    # small, non-zero pure-error noise on the 5 center replicates only
    df.loc[center_mask, "yield_g_per_l"] += [0.0, 0.01, -0.01, 0.02, -0.02]

    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    assert fit.n_points == 13
    assert fit.n_params == 6  # intercept, ph, glucose_pct, ph^2, glucose_pct^2, ph*glucose_pct
    assert fit.r_squared > 0.999
    assert fit.lack_of_fit is not None
    assert fit.lack_of_fit["df_pure_error"] == 4
    assert fit.lack_of_fit["df_lack_of_fit"] == 3  # (13-6) residual df - 4 pure-error df
    assert fit.lack_of_fit["significant_lack_of_fit"] is False
    assert fit.optimum["ph"] == pytest.approx(6.0, abs=0.05)
    assert fit.optimum["glucose_pct"] == pytest.approx(1.0, abs=0.05)


def test_fit_ccd_response_surface_lack_of_fit_flags_one_inconsistent_point():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    center_mask = (df["ph"] == 6.0) & (df["glucose_pct"] == 1.0)
    df.loc[center_mask, "yield_g_per_l"] += [0.0, 0.01, -0.01, 0.02, -0.02]

    clean_fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    corrupted = df.copy()
    corner_index = corrupted.index[~center_mask][0]
    corrupted.loc[corner_index, "yield_g_per_l"] += 50.0  # one wildly inconsistent point

    corrupted_fit = fit_ccd_response_surface(corrupted, ["ph", "glucose_pct"])

    assert corrupted_fit.lack_of_fit is not None
    assert corrupted_fit.lack_of_fit["significant_lack_of_fit"] is True
    assert corrupted_fit.lack_of_fit["p_value"] < 0.01
    assert corrupted_fit.lack_of_fit["ss_lack_of_fit"] > clean_fit.lack_of_fit["ss_lack_of_fit"] * 100


def test_fit_ccd_response_surface_optimum_stays_within_tested_range_not_original_bounds():
    # glucose_pct's original CONTINUOUS_BOUNDS is (0.5, 1.5), but this CCD's
    # own axial points only reach 0.75/1.25 (step_fraction=0.5 around a
    # baseline-centered effect) -- the optimum search must respect that
    # narrower tested range, not extrapolate back out to 0.5/1.5.
    df = _k2_ccd_df()
    assert df["glucose_pct"].min() == pytest.approx(0.75)
    assert df["glucose_pct"].max() == pytest.approx(1.25)
    # a function whose unconstrained max over glucose_pct sits at 1.5 (outside
    # what was tested) -- if the search searched the full original bounds
    # instead of the tested range, it would wrongly report an optimum there.
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 + 5.0 * (df["glucose_pct"] - 0.5)

    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    assert fit.optimum["glucose_pct"] <= 1.25 + 1e-9
    assert fit.optimum["glucose_pct"] == pytest.approx(1.25, abs=0.02)


def test_fit_ccd_response_surface_requires_more_points_than_parameters():
    df = _k2_ccd_df().head(4)  # 4 points, 6 params for a k=2 quadratic model
    df["yield_g_per_l"] = 1.0

    with pytest.raises(ValueError):
        fit_ccd_response_surface(df, ["ph", "glucose_pct"])


def _interval_interaction_round2_df(low_yield: float, high_yield: float, noise_values: list[float]) -> pd.DataFrame:
    plan = plan_round2(_round1_df(), BASELINE)
    df = generate_round2_extension_design(
        plan,
        BASELINE,
        extra_interval_levels=[36.0],
        interaction_variable="glucose_pct",
        interaction_levels=[0.5, 1.5],
        n_noise_reference=len(noise_values),
        n_lhs=0,
        seed=0,
    ).set_index("run_id")

    interaction_rows = df[df["run_type"] == "interval_interaction"]
    low_id = interaction_rows[interaction_rows["glucose_pct"] == 0.5].index[0]
    high_id = interaction_rows[interaction_rows["glucose_pct"] == 1.5].index[0]
    df.loc[low_id, "yield_g_per_l"] = low_yield
    df.loc[high_id, "yield_g_per_l"] = high_yield

    noise_ids = df[df["run_type"] == "noise_reference"].index
    for run_id, value in zip(noise_ids, noise_values):
        df.loc[run_id, "yield_g_per_l"] = value
    return df.reset_index()


def test_analyze_interval_interaction_detects_uniform_effect_no_interaction():
    # round 1's glucose_pct OFAT rows: 0.64 at 0.5%, 0.59 at 1.5% (see _round1_df).
    # Both drop by exactly 0.20 at the new interval -- same-size effect at both
    # levels, so there's a real interval effect but no interaction with glucose.
    round1_df = _round1_df()
    round2_df = _interval_interaction_round2_df(low_yield=0.44, high_yield=0.39, noise_values=[0.50, 0.52, 0.48])

    result = analyze_interval_interaction(round2_df, round1_df, BASELINE, "glucose_pct", 36.0, interaction_levels=[0.5, 1.5])

    assert result["effect_at_low"] == pytest.approx(0.44 - 0.64)
    assert result["effect_at_high"] == pytest.approx(0.39 - 0.59)
    assert result["interaction_effect"] == pytest.approx(0.0, abs=1e-9)
    assert result["noise_n"] == 3
    assert result["interval_effect_significant_at_low"] is True
    assert result["interval_effect_significant_at_high"] is True
    assert result["interaction_significant"] is False


def test_analyze_interval_interaction_detects_a_real_interaction():
    round1_df = _round1_df()
    # +0.10 at low glucose, -0.30 at high glucose -- a big swing between the
    # two levels the tiny noise-reference spread can't explain away.
    round2_df = _interval_interaction_round2_df(low_yield=0.74, high_yield=0.29, noise_values=[0.50, 0.52, 0.48])

    result = analyze_interval_interaction(round2_df, round1_df, BASELINE, "glucose_pct", 36.0, interaction_levels=[0.5, 1.5])

    assert result["interaction_significant"] is True


def test_analyze_interval_interaction_reports_none_when_noise_reference_insufficient():
    round1_df = _round1_df()
    round2_df = _interval_interaction_round2_df(low_yield=0.44, high_yield=0.39, noise_values=[0.50])  # n=1, no SD possible

    result = analyze_interval_interaction(round2_df, round1_df, BASELINE, "glucose_pct", 36.0, interaction_levels=[0.5, 1.5])

    assert result["noise_sd"] is None
    assert result["threshold"] is None
    assert result["interval_effect_significant_at_low"] is None
    assert result["interaction_significant"] is None
    # the effect sizes themselves are still reported even without a noise verdict
    assert result["effect_at_low"] == pytest.approx(0.44 - 0.64)


def test_analyze_interval_interaction_missing_backfill_returns_none_not_error():
    round1_df = _round1_df()
    plan = plan_round2(round1_df, BASELINE)
    round2_df = generate_round2_extension_design(
        plan, BASELINE, extra_interval_levels=[36.0], interaction_variable="glucose_pct",
        interaction_levels=[0.5, 1.5], n_noise_reference=2, n_lhs=0, seed=0,
    )  # yield_g_per_l left blank, as it is right after generation and before backfill

    result = analyze_interval_interaction(round2_df, round1_df, BASELINE, "glucose_pct", 36.0, interaction_levels=[0.5, 1.5])

    assert result["effect_at_low"] is None
    assert result["effect_at_high"] is None
    assert result["interaction_effect"] is None


def test_fit_ccd_response_surface_coefficient_significance_distinguishes_real_terms_from_noise():
    # true model has real ph^2/glucose_pct^2 curvature (coefficients -2, -3)
    # and a true-zero ph*glucose_pct interaction -- small noise on every
    # point (not just the center replicates) so there's a genuine, non-zero
    # residual variance to test coefficients against.
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    rng = np.random.default_rng(0)
    df["yield_g_per_l"] += rng.normal(0.0, 0.01, len(df))

    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    assert set(fit.coefficient_significance) == set(fit.term_names)
    assert fit.coefficient_significance["ph^2"]["significant"] is True
    assert fit.coefficient_significance["glucose_pct^2"]["significant"] is True
    assert fit.coefficient_significance["ph*glucose_pct"]["significant"] is False
    from scipy import stats

    for term, stats_dict in fit.coefficient_significance.items():
        assert stats_dict["se"] > 0
        expected_p = 2.0 * stats.t.sf(abs(stats_dict["t_statistic"]), fit.n_points - fit.n_params)
        assert stats_dict["p_value"] == pytest.approx(expected_p)


def test_evaluate_response_surface_reproduces_exact_fit_on_noiseless_data():
    ph_effect = FactorEffect(
        variable="ph", kind="continuous", baseline_value=6.0,
        tested_values={5.0: 9.75, 6.0: 10.0, 7.0: 9.75}, significant=True,
        best_value=6.0, best_target=10.0, effect_magnitude=0.25,
    )
    df = pd.DataFrame(generate_ccd([ph_effect], step_fraction=0.5))
    df["yield_g_per_l"] = 10.0 - (df["ph"] - 6.0) ** 2
    fit = fit_ccd_response_surface(df, ["ph"])

    predicted = evaluate_response_surface(fit, df)

    assert np.allclose(predicted, df["yield_g_per_l"].to_numpy(), atol=1e-9)


def test_response_surface_grid_matches_the_fits_own_optimum_for_k2():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    grid = response_surface_grid(fit, df, "ph", "glucose_pct")

    assert grid["z"].shape == (41, 41)
    # same variables, same range, same default resolution as the fit's own
    # optimum search -- the grid's max must land on the identical value.
    assert float(grid["z"].max()) == pytest.approx(fit.predicted_optimum, abs=1e-6)
    assert grid["fixed_at"] == {}  # nothing left over once both active variables are the requested x/y


def _k3_ccd_df() -> pd.DataFrame:
    ph_effect = FactorEffect(
        variable="ph", kind="continuous", baseline_value=6.0,
        tested_values={5.0: 0.5, 6.0: 0.62, 7.0: 0.5}, significant=True,
        best_value=6.0, best_target=0.62, effect_magnitude=0.12,
    )
    glucose_effect = FactorEffect(
        variable="glucose_pct", kind="continuous", baseline_value=1.0,
        tested_values={0.5: 0.55, 1.0: 0.62, 1.5: 0.55}, significant=True,
        best_value=1.0, best_target=0.62, effect_magnitude=0.07,
    )
    volume_effect = FactorEffect(
        variable="volume_ml", kind="continuous", baseline_value=50.0,
        tested_values={25.0: 0.5, 50.0: 0.62, 75.0: 0.7}, significant=True,
        best_value=75.0, best_target=0.7, effect_magnitude=0.12, at_upper_bound=True,
    )
    rows = generate_ccd([ph_effect, glucose_effect, volume_effect], step_fraction=0.5, n_center=4)
    return pd.DataFrame(rows)


def test_response_surface_grid_holds_the_third_active_variable_at_optimum_by_default():
    df = _k3_ccd_df()
    df["yield_g_per_l"] = (
        10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2 + 0.01 * (df["volume_ml"] - 50.0)
    )
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct", "volume_ml"])

    default_grid = response_surface_grid(fit, df, "ph", "glucose_pct")
    assert default_grid["fixed_at"]["volume_ml"] == pytest.approx(fit.optimum["volume_ml"])

    overridden_grid = response_surface_grid(fit, df, "ph", "glucose_pct", fixed_at={"volume_ml": 60.0})
    assert overridden_grid["fixed_at"]["volume_ml"] == pytest.approx(60.0)
    # yield is increasing in volume_ml here, so pinning it below the optimum's
    # volume_ml must lower every predicted value on the grid, including the max.
    assert overridden_grid["z"].max() < default_grid["z"].max()


def test_response_surface_grid_rejects_a_variable_outside_the_fit():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    with pytest.raises(ValueError):
        response_surface_grid(fit, df, "ph", "volume_ml")


def test_interval_h_fixed_levels_excludes_untested_36h():
    # ADR-0012: a new interval level explored via generate_round2_extension_design's
    # extra_interval_levels is deliberately NOT added to this shared constant --
    # recommend_round2_bo_batch's candidate grid crosses every FIXED_LEVELS value
    # regardless of whether the current training data has any row at that value,
    # so adding an untested level here would let BO start recommending it with zero
    # supporting data. This must stay [12.0, 24.0] until real 36h data exists.
    assert FIXED_LEVELS["interval_h"] == [12.0, 24.0]


def _last_message(verdicts) -> str:
    return verdicts[-1].message


def test_summarize_response_surface_clean_case_recommends_a_confirmation_run():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    verdicts = summarize_response_surface(fit, df)

    assert verdicts[-1].severity == "success"
    assert "验证批次" in _last_message(verdicts)
    assert not any(v.severity == "warning" for v in verdicts)


def test_summarize_response_surface_flags_significant_lack_of_fit_first():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    center_mask = (df["ph"] == 6.0) & (df["glucose_pct"] == 1.0)
    df.loc[center_mask, "yield_g_per_l"] += [0.0, 0.01, -0.01, 0.02, -0.02]
    corrupted = df.copy()
    corner_index = corrupted.index[~center_mask][0]
    corrupted.loc[corner_index, "yield_g_per_l"] += 50.0
    fit = fit_ccd_response_surface(corrupted, ["ph", "glucose_pct"])
    assert fit.lack_of_fit["significant_lack_of_fit"] is True

    verdicts = summarize_response_surface(fit, corrupted)

    assert verdicts[-1].severity == "warning"
    assert "先不要用" in _last_message(verdicts)
    assert any("失拟" in v.message for v in verdicts)


def test_summarize_response_surface_flags_boundary_pinned_optimum():
    df = _k2_ccd_df()
    # glucose_pct's own tested range here is [0.75, 1.25] -- a function whose
    # max over that range sits exactly at the upper edge (1.25).
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 + 5.0 * (df["glucose_pct"] - 0.5)
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])
    assert fit.optimum["glucose_pct"] == pytest.approx(1.25, abs=0.02)

    verdicts = summarize_response_surface(fit, df)

    assert verdicts[-1].severity == "warning"
    assert "扩大测试范围" in _last_message(verdicts)
    assert any("glucose_pct" in v.message and "上限" in v.message for v in verdicts)


def test_summarize_response_surface_flags_od600_infeasible_optimum():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    # od600 is highest far from the yield optimum (peaks at the low corner
    # instead of at ph=6/glucose=1) -- so the yield-optimum point is predicted
    # to have low od600.
    df["od600"] = 20.0 + 2.0 * (df["ph"] - 6.0) ** 2 + 3.0 * (df["glucose_pct"] - 1.0) ** 2
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])
    od_fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"], target_col="od600")

    verdicts = summarize_response_surface(fit, df, od_fit=od_fit, od_threshold=25.0)

    assert verdicts[-1].severity == "warning"
    assert "折中点" in _last_message(verdicts)
    assert any("OD600" in v.message and "可行阈值" in v.message for v in verdicts)


def test_classify_response_surface_case_matches_each_summarize_branch():
    # same four fixtures as the four test_summarize_response_surface_* tests
    # above -- cross-checks that the extracted classifier agrees with
    # summarize_response_surface's own (message-driven) priority order.
    clean_df = _k2_ccd_df()
    clean_df["yield_g_per_l"] = 10.0 - 2.0 * (clean_df["ph"] - 6.0) ** 2 - 3.0 * (clean_df["glucose_pct"] - 1.0) ** 2
    clean_fit = fit_ccd_response_surface(clean_df, ["ph", "glucose_pct"])
    assert classify_response_surface_case(clean_fit, clean_df) == "normal"

    center_mask = (clean_df["ph"] == 6.0) & (clean_df["glucose_pct"] == 1.0)
    # tiny non-zero pure-error noise on the center replicates (same as
    # test_summarize_response_surface_flags_significant_lack_of_fit_first) --
    # without it ss_pure_error is exactly 0 and the lack-of-fit F-test can't
    # be computed at all (falls back to "not significant" by construction,
    # not because the fit is actually clean).
    clean_df.loc[center_mask, "yield_g_per_l"] += [0.0, 0.01, -0.01, 0.02, -0.02]
    corrupted = clean_df.copy()
    corner_index = corrupted.index[~center_mask][0]
    corrupted.loc[corner_index, "yield_g_per_l"] += 50.0
    lof_fit = fit_ccd_response_surface(corrupted, ["ph", "glucose_pct"])
    assert classify_response_surface_case(lof_fit, corrupted) == "lack_of_fit"

    boundary_df = _k2_ccd_df()
    boundary_df["yield_g_per_l"] = 10.0 - 2.0 * (boundary_df["ph"] - 6.0) ** 2 + 5.0 * (boundary_df["glucose_pct"] - 0.5)
    boundary_fit = fit_ccd_response_surface(boundary_df, ["ph", "glucose_pct"])
    assert classify_response_surface_case(boundary_fit, boundary_df) == "boundary"

    od_df = _k2_ccd_df()
    od_df["yield_g_per_l"] = 10.0 - 2.0 * (od_df["ph"] - 6.0) ** 2 - 3.0 * (od_df["glucose_pct"] - 1.0) ** 2
    od_df["od600"] = 20.0 + 2.0 * (od_df["ph"] - 6.0) ** 2 + 3.0 * (od_df["glucose_pct"] - 1.0) ** 2
    od_fit = fit_ccd_response_surface(od_df, ["ph", "glucose_pct"])
    od_od_fit = fit_ccd_response_surface(od_df, ["ph", "glucose_pct"], target_col="od600")
    assert classify_response_surface_case(od_fit, od_df, od_fit=od_od_fit, od_threshold=25.0) == "od_infeasible"


def test_predict_with_confidence_interval_is_zero_width_for_a_noiseless_fit():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    result = predict_with_confidence_interval(fit, pd.DataFrame([fit.optimum]))

    assert result.loc[0, "predicted"] == pytest.approx(fit.predicted_optimum)
    assert result.loc[0, "se"] == pytest.approx(0.0, abs=1e-6)
    assert result.loc[0, "ci_low"] == pytest.approx(fit.predicted_optimum, abs=1e-6)
    assert result.loc[0, "ci_high"] == pytest.approx(fit.predicted_optimum, abs=1e-6)


def test_predict_with_confidence_interval_widens_with_noise_and_away_from_center():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    rng = np.random.default_rng(0)
    df["yield_g_per_l"] += rng.normal(0.0, 0.05, len(df))
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    at_optimum = predict_with_confidence_interval(fit, pd.DataFrame([fit.optimum]))
    assert at_optimum.loc[0, "se"] > 0
    margin_high = at_optimum.loc[0, "ci_high"] - at_optimum.loc[0, "predicted"]
    margin_low = at_optimum.loc[0, "predicted"] - at_optimum.loc[0, "ci_low"]
    assert margin_high == pytest.approx(margin_low)  # symmetric around the point estimate

    # standard OLS prediction variance grows away from the design's centroid
    # -- a factorial corner of this CCD must have a wider interval than the
    # (5-times-replicated) center point.
    center_point = pd.DataFrame([{"ph": 6.0, "glucose_pct": 1.0}])
    corner_point = pd.DataFrame([{"ph": 6.5, "glucose_pct": 1.25}])
    ci_center = predict_with_confidence_interval(fit, center_point)
    ci_corner = predict_with_confidence_interval(fit, corner_point)
    assert ci_center.loc[0, "se"] < ci_corner.loc[0, "se"]


def test_sensitivity_analysis_plateau_matches_analytic_width_for_k1():
    ph_effect = FactorEffect(
        variable="ph", kind="continuous", baseline_value=6.0,
        tested_values={5.0: 9.75, 6.0: 10.0, 7.0: 9.75}, significant=True,
        best_value=6.0, best_target=10.0, effect_magnitude=0.25,
    )
    df = pd.DataFrame(generate_ccd([ph_effect], step_fraction=0.5))  # ph tested range: [5.5, 6.5]
    # steep curvature (coefficient -8, not -1) so the true 5%-of-peak plateau
    # (half-width sqrt(0.5/8) = 0.25) sits strictly inside the tested range,
    # not clipped by it -- lets the plateau boundary itself be hand-checked.
    df["yield_g_per_l"] = 10.0 - 8.0 * (df["ph"] - 6.0) ** 2
    fit = fit_ccd_response_surface(df, ["ph"])

    result = sensitivity_analysis(fit, df, tolerance_fraction=0.05, resolution=401)
    ph_result = result["ph"]

    assert ph_result.peak_x == pytest.approx(6.0, abs=0.01)
    assert ph_result.peak_value == pytest.approx(10.0, abs=1e-6)
    assert ph_result.plateau_low == pytest.approx(5.75, abs=0.01)
    assert ph_result.plateau_high == pytest.approx(6.25, abs=0.01)
    assert ph_result.touches_lower_bound is False
    assert ph_result.touches_upper_bound is False
    assert ph_result.plateau_width_fraction == pytest.approx(0.5, abs=0.02)  # 0.5-wide plateau / 1.0-wide tested range


def test_sensitivity_analysis_flags_plateau_touching_tested_range_edge():
    ph_effect = FactorEffect(
        variable="ph", kind="continuous", baseline_value=6.0,
        tested_values={5.0: 9.75, 6.0: 10.0, 7.0: 9.75}, significant=True,
        best_value=6.0, best_target=10.0, effect_magnitude=0.25,
    )
    df = pd.DataFrame(generate_ccd([ph_effect], step_fraction=0.5))
    # shallow curvature (coefficient -1) -- true half-width sqrt(0.5) = 0.71
    # is wider than the tested range's own half-width (0.5), so the plateau
    # must be reported as clipped by the tested range on both sides.
    df["yield_g_per_l"] = 10.0 - (df["ph"] - 6.0) ** 2
    fit = fit_ccd_response_surface(df, ["ph"])

    result = sensitivity_analysis(fit, df, tolerance_fraction=0.05)
    ph_result = result["ph"]

    assert ph_result.touches_lower_bound is True
    assert ph_result.touches_upper_bound is True
    assert ph_result.plateau_low == pytest.approx(5.5, abs=0.01)
    assert ph_result.plateau_high == pytest.approx(6.5, abs=0.01)


def test_canonical_analysis_classifies_a_true_maximum():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    result = canonical_analysis(fit, df)

    assert result.classification == "maximum"
    assert all(value < 0 for value in result.eigenvalues)
    assert result.stationary_point["ph"] == pytest.approx(6.0, abs=1e-6)
    assert result.stationary_point["glucose_pct"] == pytest.approx(1.0, abs=1e-6)
    assert result.stationary_point_in_tested_range is True
    assert result.stationary_point_reliable is True
    assert result.predicted_at_stationary_point == pytest.approx(10.0, abs=1e-6)


def test_canonical_analysis_classifies_a_true_minimum():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = -10.0 + 2.0 * (df["ph"] - 6.0) ** 2 + 3.0 * (df["glucose_pct"] - 1.0) ** 2
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    result = canonical_analysis(fit, df)

    assert result.classification == "minimum"
    assert all(value > 0 for value in result.eigenvalues)
    assert result.stationary_point_reliable is True


def test_canonical_analysis_classifies_a_saddle_point():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    result = canonical_analysis(fit, df)

    assert result.classification == "saddle"
    assert any(value < 0 for value in result.eigenvalues)
    assert any(value > 0 for value in result.eigenvalues)
    assert result.stationary_point["ph"] == pytest.approx(6.0, abs=1e-6)
    assert result.stationary_point["glucose_pct"] == pytest.approx(1.0, abs=1e-6)
    assert result.stationary_point_reliable is True


def test_canonical_analysis_classifies_a_ridge_when_one_direction_is_flat():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2  # zero true dependence on glucose_pct
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    result = canonical_analysis(fit, df)

    assert result.classification == "ridge"
    assert result.stationary_point_reliable is False


def test_canonical_analysis_matches_boundary_case_with_a_rising_ridge():
    # same scenario as test_summarize_response_surface_flags_boundary_pinned_optimum:
    # linear-only in glucose_pct (no curvature at all in that direction) is
    # exactly why the grid-search optimum has nowhere to peak and ends up
    # pinned at whatever edge the tested range happens to have.
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 + 5.0 * (df["glucose_pct"] - 0.5)
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])

    result = canonical_analysis(fit, df)

    assert result.classification == "ridge"
    assert result.stationary_point_reliable is False


def test_optimize_joint_desirability_finds_a_compromise_when_yield_optimum_is_infeasible():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    # od600 decreases with both ph and glucose_pct -- the opposite direction
    # from yield's peak at center, so raising od600 costs some yield. At the
    # yield optimum (ph=6, glucose_pct=1) od600 is exactly 27 (95-60-8), so a
    # threshold of 28 is infeasible right at the yield peak but easily
    # reachable a short distance away (verified empirically before writing
    # this assertion, not hand-derived: the actual argmax lands at
    # ph=5.925/glucose_pct=0.9625 with od600=28.05, yield=9.985).
    df["od600"] = 95.0 - 10.0 * df["ph"] - 8.0 * df["glucose_pct"]
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])
    od_fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"], target_col="od600")
    at_yield_optimum = float(evaluate_response_surface(od_fit, pd.DataFrame([fit.optimum]))[0])
    assert at_yield_optimum == pytest.approx(27.0, abs=0.05)

    result = optimize_joint_desirability(fit, od_fit, df, od_threshold=28.0)

    assert result.pure_yield_optimum_value == pytest.approx(fit.predicted_optimum)
    # the pure-yield optimum itself is OD600-infeasible in this scenario, so
    # the joint-desirability point must give up some yield to actually clear
    # the threshold, rather than just re-reporting the infeasible point.
    assert result.predicted_yield < result.pure_yield_optimum_value
    assert result.predicted_yield >= result.pure_yield_optimum_value * 0.98  # but only barely
    assert result.predicted_od600 >= 28.0 - 1e-6
    assert result.od_desirability == pytest.approx(1.0, abs=1e-6)
    assert result.composite_desirability == pytest.approx(
        (result.yield_desirability * result.od_desirability) ** 0.5
    )


def test_optimize_joint_desirability_matches_pure_yield_optimum_when_already_feasible():
    df = _k2_ccd_df()
    df["yield_g_per_l"] = 10.0 - 2.0 * (df["ph"] - 6.0) ** 2 - 3.0 * (df["glucose_pct"] - 1.0) ** 2
    df["od600"] = 30.0  # constant and far above any threshold used below -- always feasible
    fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"])
    od_fit = fit_ccd_response_surface(df, ["ph", "glucose_pct"], target_col="od600")

    result = optimize_joint_desirability(fit, od_fit, df, od_threshold=10.0, resolution=41)

    assert result.od_desirability == pytest.approx(1.0)
    assert result.point["ph"] == pytest.approx(fit.optimum["ph"], abs=0.05)
    assert result.point["glucose_pct"] == pytest.approx(fit.optimum["glucose_pct"], abs=0.05)
    assert result.predicted_yield == pytest.approx(fit.predicted_optimum, abs=0.05)


def test_gp_leave_one_out_cv_requires_minimum_points():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")

    with pytest.raises(ValueError):
        gp_leave_one_out_cv(_round1_df().head(4), list(ALL_VARIABLES), TARGET_COL)


def test_gp_leave_one_out_cv_reports_internally_consistent_shapes():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")

    df = _round1_df()
    cv = gp_leave_one_out_cv(df, list(ALL_VARIABLES), TARGET_COL, seed=0)

    assert cv["n_points"] == 18
    for key in ("actual", "predicted", "predicted_std", "residuals"):
        assert len(cv[key]) == 18
    assert cv["residuals"] == pytest.approx(cv["actual"] - cv["predicted"])
    assert (cv["predicted_std"] >= 0).all()
    assert cv["rmse"] >= 0
    assert cv["q_squared"] <= 1.0 + 1e-9


def test_gp_leave_one_out_cv_q_squared_is_higher_for_learnable_signal_than_pure_noise():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")

    rng = np.random.default_rng(0)
    ph_values = np.linspace(5.0, 7.0, 14)
    # a smooth, low-noise quadratic in ph -- a GP should predict a held-out
    # point from its 13 neighbors quite well.
    learnable_df = pd.DataFrame(
        {"ph": ph_values, "yield_g_per_l": 10.0 - (ph_values - 6.0) ** 2 + rng.normal(0, 0.01, 14)}
    )
    # yield independent of ph entirely -- no held-out point is predictable
    # from its neighbors any better than guessing the training mean.
    noise_df = pd.DataFrame({"ph": ph_values, "yield_g_per_l": rng.normal(0, 1.0, 14)})

    learnable_cv = gp_leave_one_out_cv(learnable_df, ["ph"], "yield_g_per_l", seed=0)
    noise_cv = gp_leave_one_out_cv(noise_df, ["ph"], "yield_g_per_l", seed=0)

    assert learnable_cv["q_squared"] > 0.5
    assert learnable_cv["q_squared"] > noise_cv["q_squared"]


def _bo_result_stub(n_candidates: int = 1000, n_feasible: int = 500, top_yield: float = 0.02, top_std: float = 0.001) -> dict:
    return {
        "n_candidates": n_candidates,
        "n_feasible": n_feasible,
        "recommendations": [{"predicted_yield": top_yield, "predicted_yield_std": top_std}],
    }


def test_summarize_bo_recommendation_without_cv_still_reads_feasibility_and_uncertainty():
    verdicts = summarize_bo_recommendation(_bo_result_stub(n_candidates=1000, n_feasible=500, top_yield=0.02, top_std=0.001))

    assert not any("Q²" in v.message for v in verdicts)  # no cv supplied -> no Q^2 verdict at all
    assert any("利用" in v.message for v in verdicts)  # std is 5% of mean -> confident/exploit read


def test_summarize_bo_recommendation_flags_narrow_feasible_region():
    verdicts = summarize_bo_recommendation(_bo_result_stub(n_candidates=1000, n_feasible=20))

    assert any(v.severity == "warning" and "可行区域很窄" in v.message for v in verdicts)


def test_summarize_bo_recommendation_flags_high_uncertainty_top_pick_as_exploration():
    verdicts = summarize_bo_recommendation(_bo_result_stub(top_yield=0.02, top_std=0.01))  # std = 50% of mean

    assert any("探索" in v.message for v in verdicts)


def test_summarize_bo_recommendation_reads_q_squared_buckets():
    good = summarize_bo_recommendation(_bo_result_stub(), yield_cv={"q_squared": 0.8})
    weak = summarize_bo_recommendation(_bo_result_stub(), yield_cv={"q_squared": 0.2})
    bad = summarize_bo_recommendation(_bo_result_stub(), yield_cv={"q_squared": -0.5})

    assert any(v.severity == "success" and "产量" in v.message for v in good)
    assert any(v.severity == "info" and "产量" in v.message for v in weak)
    assert any(v.severity == "warning" and "产量" in v.message and "不如直接猜" in v.message for v in bad)


def test_summarize_bo_recommendation_reports_both_targets_independently():
    verdicts = summarize_bo_recommendation(
        _bo_result_stub(), yield_cv={"q_squared": 0.9}, od_cv={"q_squared": -0.2}
    )

    assert any(v.severity == "success" and "产量" in v.message for v in verdicts)
    assert any(v.severity == "warning" and "OD600" in v.message for v in verdicts)


def test_summarize_bo_recommendation_treats_nan_q_squared_as_undecided_not_negative():
    # ss_total == 0 (every training value identical) makes gp_leave_one_out_cv
    # return q_squared = nan -- `nan >= 0.5`/`nan >= 0.0` are both False in
    # Python, so without an explicit NaN branch this used to fall through to
    # the "worse than guessing the mean" warning, asserting something false
    # about a model this metric simply can't evaluate.
    verdicts = summarize_bo_recommendation(_bo_result_stub(), yield_cv={"q_squared": float("nan")})

    assert any(v.severity == "info" and "算不出" in v.message for v in verdicts)
    assert not any("负值" in v.message or "不如直接猜" in v.message for v in verdicts)


def test_summarize_bo_recommendation_handles_zero_predicted_yield_without_crashing():
    # predicted_yield == 0.0 exactly must not be treated as "missing" (a
    # plain `if mean:` truthiness check would silently drop this verdict)
    # nor divide by zero -- it gets its own absolute-value message instead
    # of a relative-uncertainty one.
    verdicts = summarize_bo_recommendation(_bo_result_stub(top_yield=0.0, top_std=0.001))

    assert any("恰好为 0" in v.message and "0.001" in v.message for v in verdicts)
