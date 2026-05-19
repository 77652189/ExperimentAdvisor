from __future__ import annotations

from typing import Any

from experiment_advisor.bayes.scoring import primary_objective_for
from experiment_advisor.data_access import load_knowledge_rules, load_trials


def training_rows(optimization_mode: str) -> list[dict[str, Any]]:
    primary = primary_objective_for(optimization_mode)
    rows: list[dict[str, Any]] = []

    knowledge = load_knowledge_rules() or {}
    for point in knowledge.get("warm_start_points", []):
        if primary == "advisor_score":
            continue
        if primary not in point:
            continue
        params = {key: value for key, value in point.items() if key not in {"yield", "cost", "duration"}}
        rows.append(
            {
                "phase": "warm_start",
                "parameters": params,
                "outcomes": {primary: point[primary]},
                "objective_value": float(point[primary]),
                "noise": 0.2,
            }
        )

    for trial in load_trials():
        outcomes = trial.get("outcomes", {})
        if primary not in outcomes:
            continue
        rows.append(
            {
                "phase": trial.get("phase"),
                "parameters": trial.get("parameters", {}),
                "outcomes": outcomes,
                "objective_value": float(outcomes[primary]),
                "noise": 0.0,
            }
        )
    return rows


def build_ax_client(space: dict, constraints: list, optimization_mode: str, objective_weights: dict | None):
    try:
        from ax.service.ax_client import AxClient
    except Exception:
        return None

    # Kept intentionally conservative. Optimizer falls back if the installed Ax
    # version differs from this API, which is common across ax-platform releases.
    try:
        primary = primary_objective_for(optimization_mode)
        minimize = primary in {"cost", "duration"}
        ax_client = AxClient()
        parameters = [
            {
                "name": name,
                "type": "range",
                "bounds": item["bounds"],
                "value_type": "float",
            }
            for name, item in space.items()
        ]
        ax_client.create_experiment(
            name="hmo_fermentation",
            parameters=parameters,
            objective_name=primary,
            minimize=minimize,
        )
        for row in training_rows(optimization_mode):
            _, trial_index = ax_client.attach_trial(row["parameters"])
            ax_client.complete_trial(trial_index=trial_index, raw_data={primary: row["objective_value"]})
        return ax_client
    except Exception:
        return None
