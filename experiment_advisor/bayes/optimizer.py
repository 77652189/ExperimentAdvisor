from __future__ import annotations

import math
import random
from typing import Any

from experiment_advisor.bayes.constraint_handler import is_valid
from experiment_advisor.bayes.initializer import build_ax_client, training_rows
from experiment_advisor.bayes.scoring import primary_objective_for
from experiment_advisor.data_access import load_state, load_trials


def _center(space_item: dict[str, Any]) -> float:
    lower, upper = space_item.get("focus") or space_item["bounds"]
    return round((float(lower) + float(upper)) / 2, 6)


def _best_trial_for(primary: str) -> dict[str, Any] | None:
    state = load_state()
    best = state.get("best_outcomes", {}).get(primary)
    if not best:
        return None
    for trial in load_trials():
        if trial.get("trial_index") == best.get("trial_index"):
            return trial
    return None


def _signature(params: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((name, round(float(value), 6)) for name, value in params.items()))


class ExperimentOptimizer:
    def __init__(
        self,
        space: dict[str, dict[str, Any]],
        constraints: list[dict[str, Any]],
        optimization_mode: str,
        objective_weights: dict[str, float] | None = None,
    ):
        self.space = space
        self.constraints = constraints
        self.optimization_mode = optimization_mode
        self.objective_weights = objective_weights or {}
        self.primary_objective = primary_objective_for(optimization_mode)

    def _fixed_values(self) -> dict[str, float]:
        state = load_state()
        effect_report = state.get("effect_report") or {}
        fixed_vars = effect_report.get("fixed_vars", [])
        best_trial = _best_trial_for(self.primary_objective)
        if not best_trial:
            return {name: _center(self.space[name]) for name in fixed_vars if name in self.space}
        return {
            name: float(best_trial.get("parameters", {}).get(name, _center(self.space[name])))
            for name in fixed_vars
            if name in self.space
        }

    def _fallback_candidate(self) -> dict[str, float]:
        fixed = self._fixed_values()
        best_trial = _best_trial_for(self.primary_objective)
        trials = load_trials()
        next_index = max([trial.get("trial_index", -1) for trial in trials] + [-1]) + 1
        seen = {_signature(trial.get("parameters", {})) for trial in trials}
        base = {
            name: float(best_trial.get("parameters", {}).get(name, _center(item))) if best_trial else _center(item)
            for name, item in self.space.items()
        }

        # Ax may be unavailable in a lightweight install. In that case, keep the
        # advisor useful by exploring near the current best point instead of
        # recommending the exact same parameters forever.
        variable_names = [name for name in self.space if name not in fixed]
        for attempt in range(30):
            rng = random.Random(next_index * 997 + attempt * 37)
            candidate: dict[str, float] = {}
            for name, item in self.space.items():
                if name in fixed:
                    candidate[name] = fixed[name]
                    continue
                focus_lower, focus_upper = item.get("focus") or item["bounds"]
                span = float(focus_upper) - float(focus_lower)
                if span <= 0:
                    candidate[name] = round(float(focus_lower), 6)
                    continue
                # Shrink slowly as data grows, but keep a minimum movement so
                # repeated Bayes trials continue to probe the local region.
                radius = max(span * 0.08, span * (0.35 / math.sqrt(max(len(trials), 1))))
                direction = -1 if (attempt + next_index + len(name)) % 2 else 1
                jitter = direction * radius * (0.35 + 0.65 * rng.random())
                value = min(float(focus_upper), max(float(focus_lower), base[name] + jitter))
                candidate[name] = round(value, 6)
            if not variable_names:
                candidate = dict(base)
                candidate.update(fixed)
            sig = _signature(candidate)
            if sig not in seen and is_valid(candidate, self.constraints):
                return candidate

        candidate = dict(base)
        candidate.update(fixed)
        if _signature(candidate) not in seen and is_valid(candidate, self.constraints):
            return candidate
        center = {name: _center(item) for name, item in self.space.items()}
        if _signature(center) not in seen and is_valid(center, self.constraints):
            return center
        raise ValueError("no valid bayesian candidate found")

    def _ax_candidate(self) -> dict[str, float] | None:
        ax_client = build_ax_client(self.space, self.constraints, self.optimization_mode, self.objective_weights)
        if ax_client is None:
            return None
        for _ in range(5):
            try:
                params, _ = ax_client.get_next_trial()
            except Exception:
                return None
            candidate = {name: float(value) for name, value in params.items()}
            candidate.update(self._fixed_values())
            seen = {_signature(trial.get("parameters", {})) for trial in load_trials()}
            if _signature(candidate) not in seen and is_valid(candidate, self.constraints):
                return candidate
        return None

    def get_next_trial(self) -> dict[str, Any]:
        candidate = self._ax_candidate() or self._fallback_candidate()
        state = load_state()
        trials = load_trials()
        next_index = max([trial.get("trial_index", -1) for trial in trials] + [-1]) + 1
        training_count = len(training_rows(self.optimization_mode))
        confidence = "low" if training_count < 10 else "medium" if training_count <= 20 else "high"
        return {
            "trial_index": next_index,
            "phase": "bayes",
            "parameters": candidate,
            "predicted_outcomes": self._predict_ranges(),
            "confidence": confidence,
            "best_outcomes_so_far": state.get("best_outcomes", {}),
        }

    def _predict_ranges(self) -> dict[str, dict[str, Any]]:
        ranges: dict[str, dict[str, Any]] = {}
        trials = load_trials()
        keys = ["yield", "cost", "duration"] if self.optimization_mode == "weighted_custom" else [self.primary_objective]
        for key in keys:
            values = [float(trial["outcomes"][key]) for trial in trials if key in trial.get("outcomes", {})]
            if values:
                low, high = min(values), max(values)
                pad = max((high - low) * 0.1, 0.01)
                value_range = [round(low - pad, 6), round(high + pad, 6)]
            else:
                value_range = [None, None]
            ranges[key] = {
                "range": value_range,
                "direction": "maximize" if key in {"yield", "advisor_score"} else "minimize",
            }
        return ranges
