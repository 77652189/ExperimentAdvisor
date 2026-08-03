"""Guards for ADR invariants that aren't already covered by a feature test.

ADR-0004, ADR-0006, and ADR-0007's boundary-shift rule already have dedicated
regression tests in tests/test_recommender_comparison.py, tests/test_app_helpers.py,
and tests/test_round2_design.py -- this file only covers the invariants that had
no existing guard.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_adr_0003_legacy_ecoli_path_still_present():
    """ADR-0003: the fermenter-based E.coli path is kept as historical
    reference, not deleted -- deleting it needs a new ADR to supersede this one.
    """
    for rel_path in (
        "App/pages_legacy_ecoli.py",
        "experiment_advisor/optimizer/standard_bo.py",
        "experiment_advisor/recommendation/service.py",
        "experiment_advisor/ingestion/run_level.py",
    ):
        path = REPO_ROOT / rel_path
        assert path.is_file(), f"ADR-0003 covers this path, but it's gone: {rel_path}"

    app_text = (REPO_ROOT / "App" / "app.py").read_text(encoding="utf-8")
    assert "大肠杆菌 BO（历史，数据已作废）" in app_text


def test_adr_0005_gp_visualization_reuses_fitted_model_not_refit():
    """ADR-0005: _standard_gp_plot / _gp_pdp must receive a fitted GP as a
    parameter and must never construct their own.
    """
    path = REPO_ROOT / "App" / "pages_legacy_ecoli.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in {"_standard_gp_plot", "_gp_pdp"}
    }
    assert set(functions) == {"_standard_gp_plot", "_gp_pdp"}

    for name, node in functions.items():
        param_names = {arg.arg for arg in node.args.args}
        assert "fitted_gp" in param_names, f"{name} must take fitted_gp as a parameter"
        body_source = ast.get_source_segment(text, node) or ""
        for banned in ("SingleTaskGP(", "GaussianProcessRegressor(", "_fit_gp("):
            assert banned not in body_source, f"{name} must not fit its own GP ({banned} found)"


def test_adr_0008_ofat_levels_are_user_extensible_in_ui():
    """ADR-0008: Round 1 builder UI must let users add OFAT levels beyond the
    protocol defaults, not just render fixed checkboxes.
    """
    path = REPO_ROOT / "App" / "pages_pichia.py"
    text = path.read_text(encoding="utf-8")
    assert "增加新水平" in text
