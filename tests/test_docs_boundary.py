from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
ACTIVE_DOCS = {
    "REQUIREMENTS.md",
    "ARCHITECTURE.md",
    "EXECUTION_PLAN.md",
    "HANDOFF.md",
}


def test_active_docs_set_is_reviewed_and_complete():
    assert DOCS_ROOT.is_dir()
    assert {path.name for path in DOCS_ROOT.glob("*.md") if path.is_file()} == ACTIVE_DOCS


def test_adr_index_and_records_are_present():
    adr_root = DOCS_ROOT / "adr"
    assert adr_root.is_dir()
    assert {path.name for path in adr_root.glob("*.md") if path.is_file()} == {
        "README.md",
        "0001-pichia-hlf-shake-flask-is-active-route.md",
        "0002-fermentation-data-stays-out-of-version-control.md",
        "0003-legacy-ecoli-fermenter-path-retained-as-reference.md",
        "0004-standard-recommender-converges-on-gp-qnei.md",
        "0005-standard-gp-visualization-reuses-fitted-model.md",
        "0006-soft-filter-failures-grow-pool-not-backfill.md",
        "0007-round1-variable-set-and-ccd-boundary-rules.md",
        "0008-ofat-levels-are-user-extensible.md",
        "0009-round2-primary-method-is-ccd-not-pure-bo.md",
        "0010-round2-replication-is-structural-not-per-point.md",
        "0011-technical-noise-is-a-separate-diagnostic.md",
        "0012-new-interval-level-stays-out-of-fixed-levels.md",
        "0013-ccd-response-surface-fit-design.md",
        "0014-response-surface-deep-dive-four-analyses.md",
    }


def test_handoff_keeps_hard_boundaries_and_machine_readable_status():
    text = (DOCS_ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    headings = {line.strip() for line in text.splitlines()}
    assert {"## 当前目标", "## 下一步", "## 硬约束"}.issubset(headings)
    assert re.search(r"^slice_status: (in_progress|done|blocked)$", text, re.MULTILINE)
    assert "HMO/2FL 发酵罐数据已确认无效" in text
    assert "原始、处理后和用户上传的发酵数据不得提交到版本库" in text
    assert "软件不自动授权实验" in text


def test_data_boundary_remains_version_controlled_policy():
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/*" in text.splitlines()
    assert "!data/pichia/templates/pichia_run_level_template.csv" in text.splitlines()
