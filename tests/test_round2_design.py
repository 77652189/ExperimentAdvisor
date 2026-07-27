from __future__ import annotations

import pandas as pd
import pytest

from experiment_advisor.recommendation.round2_design import (
    FactorEffect,
    analyze_round1_effects,
    estimate_baseline_noise,
    find_reusable_round1_row,
    generate_ccd,
    od600_threshold,
    plan_round2,
    resolve_round2_variables,
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

    # combo interactions: 4 rows changed 2+ variables; only R1-16 should be flagged
    combos = {item["row_id"]: item for item in result["combo_interactions"]}
    assert set(combos) == {"R1-15", "R1-16", "R1-17", "R1-18"}
    assert combos["R1-16"]["possible_interaction"] is True
    assert combos["R1-15"]["possible_interaction"] is False
    assert combos["R1-17"]["possible_interaction"] is False
    assert combos["R1-18"]["possible_interaction"] is False


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
