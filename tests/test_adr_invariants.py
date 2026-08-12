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
    path = REPO_ROOT / "App" / "pichia_round1.py"
    text = path.read_text(encoding="utf-8")
    assert "增加新水平" in text


def test_adr_0017_common_module_is_the_bottom_of_the_import_graph():
    """ADR-0017: App/pichia_common.py must not import any other App.pichia_*
    module -- everything else is allowed to depend on it, which only stays
    cycle-free if the dependency never points back.
    """
    tree = ast.parse((REPO_ROOT / "App" / "pichia_common.py").read_text(encoding="utf-8"))
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not [name for name in imported_modules if name.startswith("App.pichia_")]


def test_adr_0017_surface_views_own_no_page_state():
    """ADR-0017: the response-surface views are a pure view layer -- they take a
    fitted model plus data and render it. Reading/writing st.session_state or
    owning a widget key there is what would put page state in two places at
    once, so it's the boundary worth asserting rather than just documenting.
    """
    text = (REPO_ROOT / "App" / "pichia_round2_surface_views.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    # strip every docstring and comment: this module's own prose explains the
    # rule, and matching that prose would make the assertion pass for the wrong
    # reason (and keep passing after a real violation).
    docstrings = {
        ast.get_docstring(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    for doc in docstrings:
        if doc:
            body = body.replace(doc, "")
    assert "session_state" not in body
    assert "key=" not in body


def test_adr_0017_round2_backfill_subtab_precedes_the_subtabs_reading_it():
    """ADR-0017: st.tabs hides its bodies, it does not defer them -- all four
    Round 2 sub-tabs execute in source order on every rerun. The design/backfill
    sub-tab is what writes "round2_full_design_df" from its st.data_editor, and
    the response-surface / BO sub-tabs read it, so reordering these three `with`
    blocks would silently cost a freshly typed result one extra interaction
    before the analysis noticed it.
    """
    text = (REPO_ROOT / "App" / "pichia_round2_sections.py").read_text(encoding="utf-8")
    positions = [text.index(f"with {name}:") for name in ("design_tab", "surface_tab", "bo_tab")]
    assert positions == sorted(positions)
