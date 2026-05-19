from __future__ import annotations

from pathlib import Path

import pytest


DATA_FILES = [
    Path("data/doe_design.json"),
    Path("data/experiment_state.json"),
    Path("data/pending_trials.json"),
    Path("data/trial_results.json"),
]


@pytest.fixture(autouse=True)
def restore_runtime_data():
    snapshots = {path: path.read_text(encoding="utf-8") for path in DATA_FILES}
    try:
        yield
    finally:
        for path, content in snapshots.items():
            path.write_text(content, encoding="utf-8")
