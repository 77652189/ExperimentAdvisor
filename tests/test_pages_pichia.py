"""Guards for the Round 1 result-upload remapping in App/pichia_results_io.py.

Covers real-world variations seen in returned lab data (2026-08 Y103 round 1)
that the original exact-header-match remap didn't handle: a unit-annotated
yield header, a "not detected" token in a result cell, and 2+ raw rows
sharing one run_id (technical replicate measurements) -- averaged for the
analysis, but without discarding the raw per-replicate values or silently
resolving a design-variable mismatch between "replicate" rows.
"""

from __future__ import annotations

import pandas as pd
import pytest
import streamlit as st

from App.pichia_common import (
    PICHIA_OD_COL,
    PICHIA_TARGET_COL,
    _pichia_restore_persisted_dataset,
)
from App.pichia_results_io import (
    _pichia_pooled_technical_noise,
    _pichia_remap_uploaded_columns,
)
from App.pichia_round2_bo_views import _pichia_bo_cv_training_rows
from App.pichia_round2_surface_views import (
    _pichia_canonical_eigenvalue_chart,
    _pichia_canonical_table,
    _pichia_curvature_direction_label,
    _pichia_flat_direction_text,
    _pichia_short_variable_label,
)
from App.pichia_round2_sections import (
    _pichia_bo_run_signature,
    _pichia_bo_staleness_reasons,
    _pichia_simulate_round2_results,
)


def _row(run_id: str = "R1-01", yield_value=13.0, yield_header: str = "hLF产量（mg/L）(待填)", **extra) -> dict:
    row = {
        "编号": run_id,
        "类型": "基线重复",
        "种子初始 OD600": 3.0,
        "补料葡萄糖浓度 (%)": 1.0,
        "培养基 pH": 6.0,
        "初始装液量 (mL)": 50,
        "发酵温度 (℃)": 30,
        "补料时间间隔 (h)": 24,
        "收获时OD600(待填)": 34.0,
        yield_header: yield_value,
    }
    row.update(extra)
    return row


def test_remap_converts_mg_per_l_yield_header_to_g_per_l():
    df = pd.DataFrame([_row(yield_value=13.0)])
    cleaned, spread, warnings = _pichia_remap_uploaded_columns(df)

    assert cleaned.loc[0, PICHIA_TARGET_COL] == pytest.approx(0.013)
    assert spread.empty
    assert warnings == []


def test_remap_keeps_plain_unitless_header_unscaled():
    # this app's own template export header, no unit annotation -- must stay
    # backward compatible (no /1000).
    df = pd.DataFrame([_row(yield_header="hLF产量(待填)", yield_value=0.5)])
    cleaned, _, _ = _pichia_remap_uploaded_columns(df)

    assert cleaned.loc[0, PICHIA_TARGET_COL] == pytest.approx(0.5)


def test_remap_treats_not_detected_token_as_zero():
    df = pd.DataFrame([_row(yield_value="未检测到")])
    cleaned, _, _ = _pichia_remap_uploaded_columns(df)

    assert cleaned.loc[0, PICHIA_TARGET_COL] == 0.0


def test_remap_single_row_run_id_gets_n_one_and_no_spread():
    df = pd.DataFrame([_row(run_id="R1-01"), _row(run_id="R1-02")])
    cleaned, spread, _ = _pichia_remap_uploaded_columns(df)

    # single-row uploads don't even go through _average_duplicate_run_ids,
    # so no _n/_spread/_reps columns are added at all -- nothing to report.
    assert f"{PICHIA_TARGET_COL}_n" not in cleaned.columns
    assert spread.empty


def test_remap_averages_duplicate_run_ids_without_discarding_raw_values():
    df = pd.DataFrame(
        [_row(run_id="R1-01", yield_value=10.0), _row(run_id="R1-01", yield_value=20.0), _row(run_id="R1-02", yield_value=5.0)]
    )
    cleaned, spread, warnings = _pichia_remap_uploaded_columns(df)

    assert warnings == []
    assert len(cleaned) == 2

    dup = cleaned.set_index("run_id").loc["R1-01"]
    assert dup[PICHIA_TARGET_COL] == pytest.approx(0.015)  # mean(10, 20) mg/L -> g/L
    assert dup[f"{PICHIA_TARGET_COL}_n"] == 2
    assert dup[f"{PICHIA_TARGET_COL}_spread"] == pytest.approx(0.010)  # (20-10) mg/L -> g/L
    assert dup[f"{PICHIA_TARGET_COL}_reps"] == "0.01|0.02"

    # the run_id that was never duplicated still gets a uniform schema (n=1,
    # spread/reps blank) rather than a ragged/missing column.
    single = cleaned.set_index("run_id").loc["R1-02"]
    assert single[f"{PICHIA_TARGET_COL}_n"] == 1
    assert pd.isna(single[f"{PICHIA_TARGET_COL}_spread"])

    assert len(spread) == 1
    assert spread.loc[0, "run_id"] == "R1-01"
    assert spread.loc[0, "重复次数"] == 2


def test_remap_flags_inconsistent_design_variable_across_replicate_rows():
    # same run_id, but ph disagrees between the two rows -- a real data
    # problem (typo, or two different physical samples mislabeled with the
    # same run_id), not something to silently resolve by "first row wins".
    df = pd.DataFrame([_row(run_id="R1-01", **{"培养基 pH": 6.0}), _row(run_id="R1-01", **{"培养基 pH": 5.0})])
    cleaned, _, warnings = _pichia_remap_uploaded_columns(df)

    assert len(cleaned) == 1
    assert len(warnings) == 1
    assert "R1-01" in warnings[0]
    assert "培养基 pH" in warnings[0]


def test_remap_preserves_unknown_extra_column():
    df = pd.DataFrame([_row(**{"UPR（未折叠蛋白反应强度）": 160.0})])
    cleaned, _, _ = _pichia_remap_uploaded_columns(df)

    assert cleaned.loc[0, "UPR（未折叠蛋白反应强度）"] == pytest.approx(160.0)


def test_remap_averages_unknown_extra_column_across_duplicate_run_ids():
    df = pd.DataFrame(
        [
            _row(run_id="R1-01", **{"UPR（未折叠蛋白反应强度）": 150.0}),
            _row(run_id="R1-01", **{"UPR（未折叠蛋白反应强度）": 170.0}),
        ]
    )
    cleaned, _, _ = _pichia_remap_uploaded_columns(df)

    assert cleaned.loc[0, "UPR（未折叠蛋白反应强度）"] == pytest.approx(160.0)
    assert cleaned.loc[0, "UPR（未折叠蛋白反应强度）_reps"] == "150|170"


def test_pooled_technical_noise_flags_clear_outlier():
    # 5 "tight" runs (values 9/11, SS=2 each, 1 dof each) plus one run whose
    # pair disagrees far more than the others (0/20, SS=200) -- pooling across
    # all 6 groups (6 dof total) should not let the one outlier's own huge SS
    # dominate the *pooled* estimate the way a single 3-baseline-run estimate
    # would, and the outlier's own per-run SD should clear the 2x cutoff.
    rows = [{"run_id": f"R{i}", f"{PICHIA_TARGET_COL}_reps": "9|11"} for i in range(5)]
    rows.append({"run_id": "R-OUT", f"{PICHIA_TARGET_COL}_reps": "0|20"})
    df = pd.DataFrame(rows)

    result = _pichia_pooled_technical_noise(df)

    assert result is not None
    assert result["dof"] == 6
    assert result["n_runs"] == 6
    assert result["pooled_sd"] == pytest.approx(35**0.5, rel=1e-9)  # sqrt((5*2 + 200) / 6)
    assert [run_id for run_id, _ in result["outliers"]] == ["R-OUT"]
    assert result["outliers"][0][1] == pytest.approx(200**0.5, rel=1e-9)


def test_pooled_technical_noise_returns_none_without_reps_column():
    df = pd.DataFrame({"run_id": ["R1"], PICHIA_TARGET_COL: [0.5]})

    assert _pichia_pooled_technical_noise(df) is None


def test_pooled_technical_noise_ignores_single_replicate_rows():
    df = pd.DataFrame({"run_id": ["R1", "R2"], f"{PICHIA_TARGET_COL}_reps": ["9|11", None]})

    result = _pichia_pooled_technical_noise(df)

    assert result["n_runs"] == 1
    assert result["dof"] == 1


def test_restore_persisted_dataset_loads_when_session_key_absent(tmp_path):
    # a Streamlit process restart wipes st.session_state even though "save to
    # data/pichia/final/" already wrote the real data to disk -- this is what
    # is supposed to make that data reappear without re-uploading.
    path = tmp_path / "dataset.csv"
    pd.DataFrame({"run_id": ["R1-01"], "yield_g_per_l": [0.5]}).to_csv(path, index=False)
    st.session_state.pop("_test_restore_key", None)

    try:
        restored = _pichia_restore_persisted_dataset("_test_restore_key", path)

        assert restored is True
        assert list(st.session_state["_test_restore_key"]["run_id"]) == ["R1-01"]
    finally:
        st.session_state.pop("_test_restore_key", None)


def test_restore_persisted_dataset_skips_when_already_present(tmp_path):
    path = tmp_path / "dataset.csv"
    pd.DataFrame({"run_id": ["R1-01"]}).to_csv(path, index=False)
    st.session_state["_test_restore_key"] = "already here"

    try:
        restored = _pichia_restore_persisted_dataset("_test_restore_key", path)

        assert restored is False
        assert st.session_state["_test_restore_key"] == "already here"
    finally:
        st.session_state.pop("_test_restore_key", None)


def test_restore_persisted_dataset_skips_when_file_missing(tmp_path):
    st.session_state.pop("_test_restore_key", None)
    missing_path = tmp_path / "does_not_exist.csv"

    assert _pichia_restore_persisted_dataset("_test_restore_key", missing_path) is False
    assert "_test_restore_key" not in st.session_state


def _round2_design_stub(n_rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [f"R2-{i:02d}" for i in range(1, n_rows + 1)],
            "glucose_pct": [1.0, 0.75, 1.25, 1.0],
            "ph": [6.2, 5.8, 6.5, 6.2],
            "volume_ml": [50.0, 55.0, 75.0, 50.0],
            "interval_h": [24.0, 24.0, 24.0, 36.0],
            PICHIA_TARGET_COL: [None] * n_rows,
            PICHIA_OD_COL: [None] * n_rows,
        }
    )


def test_simulate_round2_results_only_fills_blank_cells():
    design = _round2_design_stub()
    design.loc[0, PICHIA_TARGET_COL] = 0.0123  # a real, already-backfilled row
    design.loc[0, PICHIA_OD_COL] = 31.0

    result = _pichia_simulate_round2_results(design)

    assert result.loc[0, PICHIA_TARGET_COL] == pytest.approx(0.0123)  # untouched
    assert result.loc[0, PICHIA_OD_COL] == pytest.approx(31.0)  # untouched
    assert result.loc[1:, PICHIA_TARGET_COL].notna().all()  # every other row got filled
    assert result.loc[1:, PICHIA_OD_COL].notna().all()
    assert (result[PICHIA_TARGET_COL] > 0).all()  # never negative/zero-floored to nothing


def test_simulate_round2_results_is_deterministic():
    design = _round2_design_stub()

    first = _pichia_simulate_round2_results(design, seed=42)
    second = _pichia_simulate_round2_results(design, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_simulate_round2_results_noop_when_nothing_blank():
    design = _round2_design_stub()
    design[PICHIA_TARGET_COL] = [0.01, 0.02, 0.03, 0.04]
    design[PICHIA_OD_COL] = [30.0, 31.0, 32.0, 33.0]

    result = _pichia_simulate_round2_results(design)

    pd.testing.assert_frame_equal(result, design)


def test_bo_cv_training_rows_requires_both_targets_present():
    # matches recommend_round2_bo_batch's own row selection (dropna on both
    # yield and od600) -- a row missing only one of the two targets must be
    # excluded from cross-validating EITHER model, not just the one it's
    # actually missing, so the CV always validates the same row set the
    # deployed yield_model/od_model were actually trained on.
    df = pd.DataFrame(
        {
            "run_id": ["R1", "R2", "R3", "R4"],
            PICHIA_TARGET_COL: [0.01, 0.02, None, 0.04],
            PICHIA_OD_COL: [30.0, None, 40.0, 41.0],
        }
    )

    result = _pichia_bo_cv_training_rows(df)

    assert list(result["run_id"]) == ["R1", "R4"]


def _bo_plan(active=("ph", "glucose_pct"), od_threshold=20.0, fixed=None):
    """Minimal stand-in for the plan fields the BO signature reads."""
    from experiment_advisor.recommendation.round2_design import Round2Plan

    return Round2Plan(
        fixed_values=dict(fixed or {"temp_c": 30.0}),
        active_variables=list(active),
        boundary_notes={},
        overflow_notes={},
        untested_notes={},
        design_rows=[],
        combo_interactions=[],
        od_threshold={"threshold": od_threshold},
        noise={},
        effects={},
    )


def _bo_training_frame(yields=(0.01, 0.02, 0.03)):
    return pd.DataFrame(
        {
            "run_id": [f"R{i}" for i in range(1, len(yields) + 1)],
            "ph": [6.0] * len(yields),
            PICHIA_TARGET_COL: list(yields),
            PICHIA_OD_COL: [30.0] * len(yields),
        }
    )


def test_bo_staleness_is_quiet_when_nothing_changed():
    plan = _bo_plan()
    signature = _pichia_bo_run_signature(_bo_training_frame(), plan, 9)

    # a rerun that changed nothing must not accuse the stored batch of being stale
    again = _pichia_bo_run_signature(_bo_training_frame(), plan, 9)

    assert _pichia_bo_staleness_reasons(signature, again) == []


def test_bo_staleness_catches_an_edited_value_at_unchanged_row_count():
    # the case a row count alone misses: correcting one already-filled yield.
    # This is the likelier way the training data moves -- the data_editor is the
    # main backfill path, so a typo fix never changes how many rows there are.
    plan = _bo_plan()
    before = _pichia_bo_run_signature(_bo_training_frame((0.01, 0.02, 0.03)), plan, 9)
    after = _pichia_bo_run_signature(_bo_training_frame((0.01, 0.02, 0.099)), plan, 9)

    assert before["n_rows"] == after["n_rows"]
    assert _pichia_bo_staleness_reasons(before, after) == ["已回填的数值"]


def test_bo_staleness_reports_row_count_without_also_blaming_the_values():
    plan = _bo_plan()
    before = _pichia_bo_run_signature(_bo_training_frame((0.01, 0.02, 0.03)), plan, 9)
    after = _pichia_bo_run_signature(_bo_training_frame((0.01, 0.02, 0.03, 0.04)), plan, 9)

    # more rows necessarily changes the content hash too; saying so twice reads
    # like two independent problems
    assert _pichia_bo_staleness_reasons(before, after) == ["合并数据的行数"]


@pytest.mark.parametrize(
    "changed_plan, changed_batch, expected",
    [
        (_bo_plan(active=("ph",)), 9, ["活跃变量"]),
        (_bo_plan(fixed={"temp_c": 25.0}), 9, ["固定变量取值"]),
        (_bo_plan(od_threshold=24.0), 9, ["OD600 约束阈值"]),
        (_bo_plan(), 12, ["建议批次大小"]),
    ],
)
def test_bo_staleness_catches_the_analysis_parameters_too(changed_plan, changed_batch, expected):
    # the 分析参数 / batch-size sliders sit above the stored batch and are far
    # easier to nudge than re-uploading data, but they feed the same BO call
    frame = _bo_training_frame()
    before = _pichia_bo_run_signature(frame, _bo_plan(), 9)
    after = _pichia_bo_run_signature(frame, changed_plan, changed_batch)

    assert _pichia_bo_staleness_reasons(before, after) == expected


def test_bo_staleness_says_nothing_when_no_batch_was_ever_stored():
    # first visit: session_state has no signature yet, and there are no
    # recommendations on screen to be stale
    current = _pichia_bo_run_signature(_bo_training_frame(), _bo_plan(), 9)

    assert _pichia_bo_staleness_reasons(None, current) == []
    assert _pichia_bo_staleness_reasons({}, current) == []


def test_curvature_direction_label_names_a_single_dominant_factor():
    # unit eigenvector lying essentially along one axis -> name that one factor
    label = _pichia_curvature_direction_label([0.99, 0.14], ["ph", "glucose_pct"])

    assert label == "培养基 pH↑"


def test_curvature_direction_label_names_both_halves_of_a_mixed_direction():
    # an even two-way mix is ~0.707 each; the point of the label is that neither
    # factor alone describes the direction
    label = _pichia_curvature_direction_label([0.707, -0.707], ["ph", "glucose_pct"])

    assert label == "培养基 pH↑+补料葡萄糖浓度↓"


def test_curvature_direction_label_orders_by_contribution_not_by_variable_order():
    label = _pichia_curvature_direction_label([0.5, 0.86], ["ph", "glucose_pct"])

    assert label.startswith("补料葡萄糖浓度↑")


def test_curvature_direction_label_never_returns_empty():
    # a direction spread evenly over many factors can leave every component
    # under the threshold; naming the biggest one beats naming nothing
    label = _pichia_curvature_direction_label([0.34, 0.33, 0.33], ["ph", "glucose_pct", "volume_ml"])

    assert label == "培养基 pH↑"


def test_short_variable_label_drops_the_unit_parenthetical():
    assert _pichia_short_variable_label("glucose_pct") == "补料葡萄糖浓度"
    assert _pichia_short_variable_label("volume_ml") == "初始装液量"
    # nothing to strip -- must stay intact rather than losing a trailing word
    assert _pichia_short_variable_label("ph") == "培养基 pH"


class _Canonical:
    def __init__(self, classification, eigenvalues, eigenvectors, variables):
        self.classification = classification
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.active_variables = variables
        self.stationary_point = dict.fromkeys(variables, 0.0)
        self.stationary_point_in_tested_range = True
        self.stationary_point_reliable = classification in ("maximum", "minimum", "saddle")
        self.predicted_at_stationary_point = 1.0


def test_flat_direction_is_reported_for_a_ridge():
    canonical = _Canonical(
        "ridge",
        eigenvalues=[-2.0, -0.01],
        eigenvectors=[[0.707, 0.707], [0.707, -0.707]],
        variables=["ph", "glucose_pct"],
    )

    assert _pichia_flat_direction_text(canonical) == "培养基 pH↑+补料葡萄糖浓度↓"


def test_flat_direction_is_blank_when_every_direction_has_curvature():
    # a true peak has no free direction to move along, so promising one would be
    # actively misleading
    canonical = _Canonical(
        "maximum",
        eigenvalues=[-2.0, -3.0],
        eigenvectors=[[1.0, 0.0], [0.0, 1.0]],
        variables=["ph", "glucose_pct"],
    )

    assert _pichia_flat_direction_text(canonical) == ""
    assert _pichia_canonical_table(canonical).loc[0, "无曲率方向"] == ""


def test_flat_direction_threshold_matches_the_classifier_that_produced_it():
    # 0.04 is under 5% of 1.0 and 0.06 is over it, so exactly one direction is
    # "no real curvature" -- reporting both would contradict the classification
    canonical = _Canonical(
        "ridge",
        eigenvalues=[-1.0, -0.06, -0.04],
        eigenvectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        variables=["ph", "glucose_pct", "volume_ml"],
    )

    assert _pichia_flat_direction_text(canonical, ridge_tolerance=0.05) == "初始装液量↑"


def test_flat_direction_handles_a_surface_with_no_curvature_at_all():
    canonical = _Canonical("flat", eigenvalues=[0.0, 0.0], eigenvectors=[[1.0, 0.0], [0.0, 1.0]], variables=["ph", "glucose_pct"])

    assert "全部方向" in _pichia_flat_direction_text(canonical)


def test_eigenvalue_chart_labels_each_bar_with_its_direction():
    canonical = _Canonical(
        "ridge",
        eigenvalues=[-2.0, -0.01],
        eigenvectors=[[0.707, 0.707], [0.707, -0.707]],
        variables=["ph", "glucose_pct"],
    )

    figure = _pichia_canonical_eigenvalue_chart(canonical)
    bar = figure.data[0]

    # the old chart said "曲率方向 1 / 2", which told the reader a direction was
    # flat but not which one
    assert list(bar.x) == ["培养基 pH↑+补料葡萄糖浓度↑", "培养基 pH↑+补料葡萄糖浓度↓"]
    assert "方向系数" in bar.hovertemplate
    assert list(bar.customdata) == ["培养基 pH +0.71、补料葡萄糖浓度 +0.71", "培养基 pH +0.71、补料葡萄糖浓度 -0.71"]


def test_eigenvalue_chart_keeps_the_plain_variable_name_when_k_is_one():
    canonical = _Canonical("maximum", eigenvalues=[-2.0], eigenvectors=[[1.0]], variables=["ph"])

    figure = _pichia_canonical_eigenvalue_chart(canonical)

    # with a single factor there is no "combination" to name, and an arrow would
    # only add noise
    assert list(figure.data[0].x) == ["培养基 pH"]
