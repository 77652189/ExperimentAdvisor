from __future__ import annotations

import random
from typing import Any

from experiment_advisor.bayes.constraint_handler import is_valid
from experiment_advisor.data_access import load_constraints, save_design


def _lhs_unit(n_trials: int, n_vars: int) -> list[list[float]]:
    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=n_vars, seed=42)
        return sampler.random(n=n_trials).tolist()
    except Exception:
        rng = random.Random(42)
        columns = []
        for _ in range(n_vars):
            values = [(index + rng.random()) / n_trials for index in range(n_trials)]
            rng.shuffle(values)
            columns.append(values)
        return [[columns[col][row] for col in range(n_vars)] for row in range(n_trials)]


def _center(space_item: dict[str, Any]) -> float:
    lower, upper = space_item.get("focus") or space_item["bounds"]
    return (float(lower) + float(upper)) / 2


def _candidate_from_unit(names: list[str], space: dict[str, dict], row: list[float]) -> dict[str, float]:
    candidate: dict[str, float] = {}
    for index, name in enumerate(names):
        lower, upper = space[name].get("focus") or space[name]["bounds"]
        candidate[name] = round(float(lower) + row[index] * (float(upper) - float(lower)), 6)
    return candidate


def generate_design(space: dict[str, dict], n_trials: int = 8) -> list[dict[str, Any]]:
    if n_trials < 0:
        raise ValueError("n_trials cannot be negative")
    if n_trials == 0:
        save_design(0, [])
        return []
    names = list(space)
    hard_constraints = load_constraints()
    rows = _lhs_unit(n_trials, len(names))
    design: list[dict[str, Any]] = []
    replacements = 0
    center = {name: round(_center(space[name]), 6) for name in names}

    for batch_index, unit_row in enumerate(rows):
        params = _candidate_from_unit(names, space, unit_row)
        warnings: list[str] = []
        if not is_valid(params, hard_constraints):
            replacements += 1
            params = dict(center)
            warnings.append("constraint_hit_replaced_by_focus_center")
        if not is_valid(params, hard_constraints):
            raise ValueError("focus_range center also violates hard constraints")
        design.append({"batch_index": batch_index, "phase": "doe", "parameters": params, "warnings": warnings})

    if replacements / n_trials > 0.2:
        raise ValueError("too many DOE points violate constraints; loosen constraints or increase space")
    save_design(n_trials, design)
    return design
