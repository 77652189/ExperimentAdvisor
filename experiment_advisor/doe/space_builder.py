from __future__ import annotations

from experiment_advisor.config.config_manager import ConfigManager
from experiment_advisor.config.space_merger import merge_space
from experiment_advisor.data_access import load_knowledge_rules
from experiment_advisor.paths import PARAMETER_DEFAULTS_PATH
from experiment_advisor.storage import read_json


def build_space(researcher_config: dict | None = None) -> tuple[dict, dict]:
    defaults = read_json(PARAMETER_DEFAULTS_PATH, {"variables": []})
    knowledge_rules = load_knowledge_rules()
    active_config = researcher_config if researcher_config is not None else ConfigManager().get_active_config()
    return merge_space(defaults, knowledge_rules, active_config)
