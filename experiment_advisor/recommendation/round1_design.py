"""Round-1 experiment design for the Pichia hLF shake-flask workflow.

Hybrid design confirmed in project discussion: baseline replicates (for noise
estimation) + one-factor-at-a-time rows at each PDF-given non-baseline level
(safe, clean main-effect reads, never more than one variable away from the
known-working baseline) + a handful of jointly-varied combination rows (a bit
of interaction/joint-space signal within the same sample budget). See
round2_design.py for how round 1's results feed into round 2 planning.

Pure one-factor-at-a-time can't see interactions between variables (all points
sit on axes through the baseline); pure joint sampling covers interactions but
risks several samples failing at once if a joint combination turns out to be
non-viable. This hybrid spends most of the budget on the safe axis-aligned
reads and a few points on joint coverage, rather than committing fully to
either extreme.
"""

from __future__ import annotations

import itertools

import pandas as pd

from experiment_advisor.recommendation.round2_design import (
    ALL_VARIABLES,
    CONTINUOUS_BOUNDS,
    OD_COL,
    TARGET_COL,
)
from experiment_advisor.utils.lhs import latin_hypercube

BASELINE: dict[str, float] = {
    "seed_od": 3.0,
    "glucose_pct": 1.0,
    "interval_h": 24.0,
    "ph": 6.0,
    "temp_c": 30.0,
    "volume_ml": 50.0,
}

OFAT_LEVELS: dict[str, list[float]] = {
    "seed_od": [3.0, 9.0, 15.0],
    "glucose_pct": [0.5, 1.0, 1.5],
    "interval_h": [12.0, 24.0],
    "ph": [5.0, 6.0, 7.0],
    "temp_c": [20.0, 25.0, 30.0],
    "volume_ml": [25.0, 50.0, 75.0],
}


def _is_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def _combo_temp_interval_pairs(
    baseline: dict[str, float],
    ofat_levels: dict[str, list[float]],
    n_combo_points: int,
) -> list[tuple[float, float]]:
    """temp_c/interval_h values for combo rows, prioritising pairs round 1's
    OFAT rows never test jointly. OFAT only ever pairs one of the two against
    the other's baseline level, so e.g. (20, 12) is untested even though 20
    and 12 are each individually tested. Already-covered pairs are used to
    fill out any remaining combo rows once the untested pairs run out."""

    temp_levels = ofat_levels["temp_c"]
    interval_levels = ofat_levels["interval_h"]
    all_pairs = list(itertools.product(temp_levels, interval_levels))
    baseline_temp = baseline["temp_c"]
    baseline_interval = baseline["interval_h"]
    covered = {(t, baseline_interval) for t in temp_levels} | {(baseline_temp, i) for i in interval_levels}
    untested = [pair for pair in all_pairs if pair not in covered]
    ordered = untested + [pair for pair in all_pairs if pair in covered]
    return [ordered[i % len(ordered)] for i in range(n_combo_points)]


def _round_value(variable: str, value: float) -> float:
    decimals = 1 if variable in {"seed_od", "glucose_pct", "ph"} else 0
    return round(float(value), decimals)


def round1_template_columns() -> list[str]:
    return ["run_id", "run_type", "changed_variable", *ALL_VARIABLES, TARGET_COL, OD_COL]


def generate_round1_design(
    baseline: dict[str, float] = BASELINE,
    ofat_levels: dict[str, list[float]] = OFAT_LEVELS,
    n_baseline_replicates: int = 3,
    n_combo_points: int = 4,
    run_id_prefix: str = "R1",
    seed: int = 42,
) -> pd.DataFrame:
    """Baseline replicates + OFAT rows + LHS joint-exploration rows.

    OFAT rows change exactly one variable to a PDF-given non-baseline level,
    holding everything else at `baseline`. Combo rows vary the 4 continuous
    variables jointly via Latin Hypercube Sampling (`experiment_advisor.utils.lhs`)
    and assign temp_c/interval_h from `_combo_temp_interval_pairs` -- both are
    fixed-level (discrete) in practice, so they can never take an interpolated
    LHS value the way a continuous variable can.
    """

    continuous_variables = list(CONTINUOUS_BOUNDS)
    rows: list[dict] = []
    counter = 0

    def _next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{run_id_prefix}-{counter:02d}"

    for _ in range(n_baseline_replicates):
        rows.append({"run_id": _next_id(), "run_type": "baseline", "changed_variable": None, **baseline})

    for variable in ALL_VARIABLES:
        for level in ofat_levels[variable]:
            if _is_close(level, baseline[variable]):
                continue
            row = dict(baseline)
            row[variable] = level
            rows.append({"run_id": _next_id(), "run_type": "ofat", "changed_variable": variable, **row})

    lhs_values = latin_hypercube(n_combo_points, len(continuous_variables), seed=seed)
    temp_interval_pairs = _combo_temp_interval_pairs(baseline, ofat_levels, n_combo_points)
    for index in range(n_combo_points):
        row = dict(baseline)
        for position, variable in enumerate(continuous_variables):
            lower, upper = CONTINUOUS_BOUNDS[variable]
            raw_value = lower + lhs_values[index][position] * (upper - lower)
            row[variable] = _round_value(variable, raw_value)
        row["temp_c"], row["interval_h"] = temp_interval_pairs[index]
        rows.append({"run_id": _next_id(), "run_type": "combo", "changed_variable": None, **row})

    frame = pd.DataFrame(rows)
    frame[TARGET_COL] = float("nan")
    frame[OD_COL] = float("nan")
    return frame[round1_template_columns()]
