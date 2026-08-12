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
from App.pichia_round2_sections import _pichia_simulate_round2_results


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
