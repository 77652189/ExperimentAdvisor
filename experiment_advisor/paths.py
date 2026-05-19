from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = DATA_DIR / "parameter_configs"

KNOWLEDGE_RULES_PATH = DATA_DIR / "knowledge_rules.json"
PARAMETER_DEFAULTS_PATH = DATA_DIR / "parameter_defaults.json"
DOE_DESIGN_PATH = DATA_DIR / "doe_design.json"
TRIAL_RESULTS_PATH = DATA_DIR / "trial_results.json"
PENDING_TRIALS_PATH = DATA_DIR / "pending_trials.json"
EXPERIMENT_STATE_PATH = DATA_DIR / "experiment_state.json"
