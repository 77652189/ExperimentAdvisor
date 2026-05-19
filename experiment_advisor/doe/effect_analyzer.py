from __future__ import annotations

from statistics import median
from typing import Any

from experiment_advisor.data_access import load_state, load_trials, save_state


def analyze_effects(space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    trials = [trial for trial in load_trials() if trial.get("phase") == "doe"]
    effect_sizes: dict[str, float] = {}
    significant_vars: list[str] = []
    fixed_vars: list[str] = []

    if len(trials) < 2:
        report = {
            "significant_vars": list(space),
            "fixed_vars": [],
            "effect_sizes": {name: 0.0 for name in space},
            "ready_for_bayes": False,
        }
    else:
        primary = load_state().get("primary_objective", "yield")
        objective = "yield" if primary == "advisor_score" else primary
        for name in space:
            values = [float(trial["parameters"][name]) for trial in trials if name in trial.get("parameters", {})]
            if not values:
                effect_sizes[name] = 0.0
                fixed_vars.append(name)
                continue
            split = median(values)
            high = [
                float(trial["outcomes"][objective])
                for trial in trials
                if name in trial.get("parameters", {}) and objective in trial.get("outcomes", {}) and trial["parameters"][name] >= split
            ]
            low = [
                float(trial["outcomes"][objective])
                for trial in trials
                if name in trial.get("parameters", {}) and objective in trial.get("outcomes", {}) and trial["parameters"][name] < split
            ]
            effect = (sum(high) / len(high) - sum(low) / len(low)) if high and low else 0.0
            effect_sizes[name] = round(effect, 6)
        max_abs = max((abs(value) for value in effect_sizes.values()), default=0.0)
        threshold = max_abs * 0.2
        for name, effect in effect_sizes.items():
            if max_abs == 0.0 or abs(effect) >= threshold:
                significant_vars.append(name)
            else:
                fixed_vars.append(name)
        report = {
            "significant_vars": significant_vars or list(space),
            "fixed_vars": fixed_vars if significant_vars else [],
            "effect_sizes": effect_sizes,
            "ready_for_bayes": True,
        }

    state = load_state()
    state["effect_report"] = report
    save_state(state)
    return report
