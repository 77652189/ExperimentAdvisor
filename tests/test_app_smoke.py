"""End-to-end smoke test: does the Streamlit app actually run?

Every other test in this suite calls helper functions directly, so all 149 of
them can pass while the page itself raises on load -- a broken import, a widget
key collision, a Streamlit API that changed under a version bump. This drives
the real app through the Round 2 workflow with `streamlit.testing.v1.AppTest`,
which is the verification method docs/HANDOFF.md prescribes for UI changes.

Deliberately hermetic. The real datasets are gitignored (ADR-0002), so a fresh
clone has none and a test that depended on them would be dead weight exactly
where it is needed most; on a machine that *does* have them, the page would
silently restore them and the assertions would run against whatever that
developer happens to have on disk. Both dataset paths are therefore redirected
into tmp_path and Round 1 results are seeded synthetically.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from App import pichia_round1, pichia_round2_sections
from App.pichia_common import PROJECT_ROOT
from experiment_advisor.recommendation.round1_design import BASELINE

APP_SCRIPT = str(PROJECT_ROOT / "App" / "app.py")
# a GP fit on ~30 points is seconds, not milliseconds, and CI machines are slower
TIMEOUT = 300

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _row(run_id: str, yield_g_per_l: float, od600: float, **overrides) -> dict:
    """One backfilled Round 1 run. run_type/changed_variable are derived from
    which levels were moved off baseline, the same labelling the real uploaded
    sheet carries -- the Round 1 charts colour by run_type and the effect
    analysis reads changed_variable, so a frame without them isn't a stand-in
    for real data at all."""
    if not overrides:
        run_type, changed = "baseline", None
    elif len(overrides) == 1:
        run_type, changed = "ofat", next(iter(overrides))
    else:
        run_type, changed = "combo", None
    return {
        **BASELINE,
        "run_id": run_id,
        "run_type": run_type,
        "changed_variable": changed,
        "yield_g_per_l": yield_g_per_l,
        "od600": od600,
        **overrides,
    }


def _round1_results() -> pd.DataFrame:
    """Round 1 shaped like the real Y103 outcome: glucose_pct / ph / volume_ml
    all carry signal (K=3), seed_od and temp_c stay inside the noise.

    K matters here. At K=1 the page renders a single response curve; only K>=2
    reaches the 3D surface / contour / feasibility-overlay code paths, and K=3
    is what the real data actually produced -- a smoke test on K=1 would leave
    most of the Round 2 charts unexecuted.
    """
    return pd.DataFrame(
        [
            _row("R1-01", 0.585, 33.0),
            _row("R1-02", 0.620, 34.0),
            _row("R1-03", 0.655, 35.0),
            _row("R1-04", 0.610, 34.0, seed_od=9.0),
            _row("R1-05", 0.625, 34.0, seed_od=15.0),
            _row("R1-06", 0.400, 22.0, glucose_pct=0.5),
            _row("R1-07", 0.030, 40.0, glucose_pct=1.5),
            _row("R1-08", 0.050, 50.0, interval_h=12.0),
            _row("R1-09", 0.120, 31.0, ph=5.0),
            _row("R1-10", 0.520, 33.0, ph=7.0),
            _row("R1-11", 0.860, 32.0, volume_ml=75.0),
            _row("R1-12", 0.300, 39.0, volume_ml=25.0),
            _row("R1-13", 0.615, 34.0, temp_c=25.0),
            _row("R1-14", 0.600, 34.0, temp_c=20.0),
            _row("R1-15", 0.700, 30.0, glucose_pct=0.5, volume_ml=75.0),
            _row("R1-16", 0.250, 42.0, ph=5.0, glucose_pct=1.5),
        ]
    )


@pytest.fixture
def app(monkeypatch, tmp_path) -> AppTest:
    """An AppTest with Round 1 seeded and both persisted-dataset paths isolated."""
    monkeypatch.setattr(pichia_round1, "PICHIA_DEFAULT_DATASET_PATH", tmp_path / "round1.csv")
    monkeypatch.setattr(pichia_round2_sections, "PICHIA_ROUND2_DATASET_PATH", tmp_path / "round2.csv")

    at = AppTest.from_file(APP_SCRIPT, default_timeout=TIMEOUT)
    at.session_state["round1_results_df"] = _round1_results()
    at.run(timeout=TIMEOUT)
    _assert_clean(at, "initial load")
    return at


def _assert_clean(at: AppTest, stage: str) -> None:
    assert not at.exception, f"exception at {stage}: {[e.value for e in at.exception]}"


def _chart_titles(at: AppTest) -> list[str]:
    import json

    titles = []
    for chart in at.get("plotly_chart"):
        layout = json.loads(chart.proto.spec).get("layout", {})
        title = layout.get("title")
        titles.append(title.get("text") if isinstance(title, dict) else title)
    return [t for t in titles if t]


def test_app_loads_with_both_top_level_tabs(app: AppTest):
    labels = [tab.label for tab in app.tabs]

    assert "Round 1：实验设计" in labels
    assert any(label.startswith("Round 2") for label in labels)


def test_round2_has_the_four_subtabs_in_execution_order(app: AppTest):
    labels = [tab.label for tab in app.tabs]

    # order is load-bearing, not cosmetic: the design sub-tab writes the frame
    # the two after it read (see ADR-0017 and test_adr_invariants)
    subtabs = [label for label in labels if label[0] in "①②③④"]
    assert subtabs == ["① 显著性分析", "② 设计生成与回填", "③ 响应面结果", "④ 合并数据贝叶斯优化"]


def test_significance_subtab_renders_the_effect_chart(app: AppTest):
    assert any("效应量" in title for title in _chart_titles(app))


def test_analysis_subtabs_point_back_to_backfill_before_any_results_exist(app: AppTest):
    # a blank sub-tab reads as a broken page; both must say why they're empty
    hints = [info.value for info in app.info if "② 设计生成与回填" in info.value]

    assert len(hints) == 2


def test_round2_design_generates_and_then_analyses_end_to_end(app: AppTest):
    app.button(key="generate_round2_full_design").click().run(timeout=TIMEOUT)
    _assert_clean(app, "generate design")
    design = app.session_state["round2_full_design_df"]
    assert not design.empty
    assert {"ccd", "interval_interaction", "noise_reference", "lhs"} <= set(design["run_type"])

    app.checkbox(key="confirm_simulate_round2").check().run(timeout=TIMEOUT)
    app.button(key="simulate_round2_results").click().run(timeout=TIMEOUT)
    _assert_clean(app, "simulated backfill")

    titles = _chart_titles(app)
    # the response-surface sub-tab is now populated: residuals, a 3D surface, a
    # contour, sensitivity sweeps and the curvature read
    assert any("残差诊断" in t for t in titles)
    assert any("3D" in t for t in titles)
    assert any("灵敏度扫描" in t for t in titles)
    assert any("特征值" in t for t in titles)
    assert any("权衡" in t for t in titles)


def test_combined_bo_recommendation_runs(app: AppTest):
    app.button(key="generate_round2_full_design").click().run(timeout=TIMEOUT)
    app.checkbox(key="confirm_simulate_round2").check().run(timeout=TIMEOUT)
    app.button(key="simulate_round2_results").click().run(timeout=TIMEOUT)
    app.button(key="run_combined_bo").click().run(timeout=TIMEOUT)
    _assert_clean(app, "combined BO")

    if not TORCH_AVAILABLE:
        # the page must degrade to a warning rather than blowing up
        assert any("torch" in warning.value for warning in app.warning)
        return

    result = app.session_state["round2_combined_bo_result"]
    assert result["recommendations"]
    assert any("偏依赖" in title for title in _chart_titles(app))


def test_the_isolation_fixture_actually_redirects_the_dataset_path(monkeypatch, tmp_path):
    """Proof that the redirect above bites.

    Without it every other assertion here is only as trustworthy as whatever
    happens to sit in the developer's data/pichia/final/. A passing suite cannot
    tell a working monkeypatch from a machine that simply has no round 2 file
    yet, so this writes one at the *redirected* path and checks the app picks it
    up: if the patch were a no-op the app would look elsewhere and find nothing.
    """
    redirected = tmp_path / "round2.csv"
    planted = pd.DataFrame(
        [{"run_id": "PLANTED-01", "run_type": "ccd", "yield_g_per_l": 0.42, "od600": 30.0, **BASELINE}]
    )
    planted.to_csv(redirected, index=False, encoding="utf-8-sig")

    monkeypatch.setattr(pichia_round1, "PICHIA_DEFAULT_DATASET_PATH", tmp_path / "round1.csv")
    monkeypatch.setattr(pichia_round2_sections, "PICHIA_ROUND2_DATASET_PATH", redirected)

    at = AppTest.from_file(APP_SCRIPT, default_timeout=TIMEOUT)
    at.session_state["round1_results_df"] = _round1_results()
    at.run(timeout=TIMEOUT)
    _assert_clean(at, "restore from redirected path")

    assert list(at.session_state["round2_full_design_df"]["run_id"]) == ["PLANTED-01"]


def test_smoke_run_leaves_the_real_data_directory_untouched(app: AppTest, tmp_path):
    # the app writes to data/pichia/final/ only on an explicit save click, and
    # this test must never be the reason a developer's real dataset changes
    app.button(key="generate_round2_full_design").click().run(timeout=TIMEOUT)
    app.checkbox(key="confirm_simulate_round2").check().run(timeout=TIMEOUT)
    app.button(key="simulate_round2_results").click().run(timeout=TIMEOUT)

    assert not (tmp_path / "round1.csv").exists()
    assert not (tmp_path / "round2.csv").exists()
