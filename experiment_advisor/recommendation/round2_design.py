"""Round-2 design planning for the Pichia hLF shake-flask workflow.

Round 1 is a fixed screening design (3 baseline replicates + 11 OFAT rows + 4 joint
LHS combination rows, see data/pichia templates). This module turns round 1's observed
yields/OD600 into a round 2 plan: which variables get a classical response-surface
refinement (CCD) and which get fixed, plus a constrained BO batch that explores the
same active variables adaptively. Both tracks run side by side on shared data rather
than splitting the round's sample budget by construction.

Two of the six variables are physically fixed-level, not continuous: temperature can
only be set to one of a few fixed incubator levels, and feed interval only to one of a
few fixed feeding-schedule slots. Round 1's OFAT rows already test every level of both,
so round 2 never refines them further -- it only ever fixes them at whichever tested
level scored best.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TARGET_COL = "yield_g_per_l"
OD_COL = "od600"

CONTINUOUS_BOUNDS: dict[str, tuple[float, float]] = {
    "seed_od": (3.0, 15.0),
    "glucose_pct": (0.5, 1.5),
    "ph": (5.0, 7.0),
    "volume_ml": (25.0, 75.0),
}

FIXED_LEVELS: dict[str, list[float]] = {
    "temp_c": [20.0, 25.0, 30.0],
    "interval_h": [12.0, 24.0],
}

ALL_VARIABLES: list[str] = [*CONTINUOUS_BOUNDS, *FIXED_LEVELS]


def _is_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def _round_key(value: float, ndigits: int = 6) -> float:
    """Round a float before using it as a dict key, so a baseline value written as
    6.0 always matches the "same" value re-derived via groupby from the dataframe."""

    return round(float(value), ndigits)


def estimate_baseline_noise(
    df: pd.DataFrame,
    baseline: dict[str, float],
    target_col: str = TARGET_COL,
    min_relative_noise: float = 0.10,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Estimate replicate noise from rows matching `baseline` on every variable.

    Floored at `min_relative_noise` of the baseline mean so a lucky (low-variance)
    handful of replicates can't make every later comparison look "significant".
    """

    mask = pd.Series(True, index=df.index)
    for variable in ALL_VARIABLES:
        if variable in df.columns:
            mask &= df[variable].apply(lambda value: _is_close(value, baseline[variable], tol))
    replicates = df.loc[mask, target_col].dropna()
    if len(replicates) < 2:
        raise ValueError("Need at least 2 baseline replicates to estimate noise")

    mean = float(replicates.mean())
    sd = float(replicates.std(ddof=1))
    threshold = max(2.0 * sd, min_relative_noise * abs(mean))
    return {
        "baseline_mean": mean,
        "baseline_sd": sd,
        "threshold": threshold,
        "n_replicates": int(len(replicates)),
    }


@dataclass(frozen=True)
class FactorEffect:
    variable: str
    kind: str  # "continuous" | "fixed_level"
    baseline_value: float
    tested_values: dict[float, float]  # level -> mean observed target at that level
    significant: bool
    best_value: float
    best_target: float
    effect_magnitude: float  # max |deviation from baseline| across non-baseline levels
    at_lower_bound: bool = False
    at_upper_bound: bool = False


def _levels_for(variable: str, df: pd.DataFrame, baseline: dict[str, float], other_vars: list[str], tol: float) -> pd.DataFrame:
    """Rows where every variable except `variable` sits at its baseline value."""

    mask = pd.Series(True, index=df.index)
    for other in other_vars:
        if other == variable or other not in df.columns:
            continue
        mask &= df[other].apply(lambda value: _is_close(value, baseline[other], tol))
    return df.loc[mask]


def analyze_round1_effects(
    df: pd.DataFrame,
    baseline: dict[str, float],
    target_col: str = TARGET_COL,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Classify each of the 6 variables from round-1's OFAT rows.

    A variable's "OFAT rows" are rows where every *other* variable sits at its
    baseline value -- this is derived structurally from the data rather than from a
    label column, so it stays correct even if round 1's row order or ids change.
    """

    noise = estimate_baseline_noise(df, baseline, target_col=target_col, tol=tol)
    threshold = noise["threshold"]
    baseline_mean = noise["baseline_mean"]

    effects: dict[str, FactorEffect] = {}
    for variable in ALL_VARIABLES:
        if variable not in df.columns:
            continue
        rows = _levels_for(variable, df, baseline, ALL_VARIABLES, tol)
        tested = (
            rows.groupby(variable)[target_col]
            .mean()
            .to_dict()
        )
        tested = {_round_key(level): float(value) for level, value in tested.items() if pd.notna(value)}
        baseline_key = _round_key(baseline[variable])
        tested.setdefault(baseline_key, baseline_mean)

        kind = "continuous" if variable in CONTINUOUS_BOUNDS else "fixed_level"
        best_value, best_target = max(tested.items(), key=lambda item: item[1])
        non_baseline_deviations = [
            abs(target - baseline_mean) for level, target in tested.items() if level != baseline_key
        ]
        effect_magnitude = max(non_baseline_deviations) if non_baseline_deviations else 0.0
        significant = effect_magnitude > threshold

        at_lower = at_upper = False
        if kind == "continuous":
            lower, upper = CONTINUOUS_BOUNDS[variable]
            at_lower = _is_close(best_value, lower, tol=1e-3)
            at_upper = _is_close(best_value, upper, tol=1e-3)

        effects[variable] = FactorEffect(
            variable=variable,
            kind=kind,
            baseline_value=baseline_key,
            tested_values=tested,
            significant=significant,
            best_value=best_value,
            best_target=best_target,
            effect_magnitude=effect_magnitude,
            at_lower_bound=at_lower,
            at_upper_bound=at_upper,
        )

    return {
        "noise": noise,
        "effects": effects,
        "combo_interactions": _check_combo_interactions(df, baseline, effects, target_col, noise, tol),
    }


def _predict_additive(row: pd.Series, baseline_mean: float, effects: dict[str, FactorEffect]) -> float:
    total = baseline_mean
    for variable, effect in effects.items():
        if variable not in row.index or pd.isna(row[variable]):
            continue
        xs = sorted(effect.tested_values)
        if len(xs) < 2:
            continue
        ys = [effect.tested_values[x] for x in xs]
        predicted_at_value = float(np.interp(float(row[variable]), xs, ys))
        total += predicted_at_value - baseline_mean
    return total


def _check_combo_interactions(
    df: pd.DataFrame,
    baseline: dict[str, float],
    effects: dict[str, FactorEffect],
    target_col: str,
    noise: dict[str, Any],
    tol: float,
) -> list[dict[str, Any]]:
    """Flag rows that change 2+ variables at once and deviate from the additive prediction.

    Not a rigorous interaction test (round 1 only has 4 such rows) -- a directional
    "does this combination behave like the sum of its parts" sanity check that feeds
    into whether round 2 should worry about interactions among the active variables.
    """

    threshold = noise["threshold"]
    baseline_mean = noise["baseline_mean"]

    flagged = []
    for idx, row in df.iterrows():
        if pd.isna(row.get(target_col)):
            continue
        changed = [
            variable
            for variable in ALL_VARIABLES
            if variable in row.index and not _is_close(row[variable], baseline[variable], tol)
        ]
        if len(changed) < 2:
            continue
        predicted = _predict_additive(row, baseline_mean, effects)
        residual = float(row[target_col]) - predicted
        flagged.append(
            {
                "row_id": row.get("run_id", idx),
                "changed_variables": changed,
                "observed": float(row[target_col]),
                "additive_prediction": predicted,
                "residual": residual,
                "possible_interaction": abs(residual) > threshold,
            }
        )
    return flagged


@dataclass(frozen=True)
class Round2Variable:
    variable: str
    active: bool
    fixed_value: float | None
    boundary_note: str | None = None
    note_kind: str | None = None  # "boundary" | "overflow", distinguishes the two note reasons


def resolve_round2_variables(
    effects: dict[str, FactorEffect],
    max_active_variables: int = 3,
) -> dict[str, Round2Variable]:
    """Decide which variables round 2 refines (CCD) vs. fixes at a single value.

    Fixed-level variables (temp_c, interval_h) are *always* fixed -- round 1's OFAT
    rows already cover every level either one can take, so there is nothing left to
    refine. Continuous variables are active only if significant, capped at
    `max_active_variables` (kept by |effect| if more come out significant than a CCD
    can afford within a 15-20 sample round).
    """

    resolved: dict[str, Round2Variable] = {}
    continuous_candidates = []
    for variable, effect in effects.items():
        if effect.kind == "fixed_level":
            resolved[variable] = Round2Variable(variable=variable, active=False, fixed_value=effect.best_value)
            continue
        if effect.significant:
            continuous_candidates.append(effect)
        else:
            resolved[variable] = Round2Variable(variable=variable, active=False, fixed_value=effect.best_value)

    # Sort by effect_magnitude (max |deviation from baseline|), not "best minus baseline":
    # a variable whose response peaks AT baseline (e.g. pH in the worked example) has
    # best_value == baseline_value, so "best minus baseline" would read as zero even
    # though both flanks are significantly worse -- that's a real, strong effect, not
    # a weak one, and must not be starved out by max_active_variables.
    continuous_candidates.sort(key=lambda e: e.effect_magnitude, reverse=True)
    active, overflow = continuous_candidates[:max_active_variables], continuous_candidates[max_active_variables:]

    for effect in active:
        note = None
        if effect.at_lower_bound:
            note = f"最优值贴着下限 {CONTINUOUS_BOUNDS[effect.variable][0]}，round 2 只能向上探索；如果 round 2 仍然是下限最优，建议和研发组确认能否放宽下限"
        elif effect.at_upper_bound:
            note = f"最优值贴着上限 {CONTINUOUS_BOUNDS[effect.variable][1]}，round 2 只能向下探索；如果 round 2 仍然是上限最优，建议和研发组确认能否放宽上限"
        resolved[effect.variable] = Round2Variable(
            variable=effect.variable,
            active=True,
            fixed_value=None,
            boundary_note=note,
            note_kind="boundary" if note else None,
        )

    for effect in overflow:
        resolved[effect.variable] = Round2Variable(
            variable=effect.variable,
            active=False,
            fixed_value=effect.best_value,
            boundary_note=f"效应显著但超出本轮 CCD 容量上限({max_active_variables})，暂固定在 round 1 最优水平，留待后续轮次",
            note_kind="overflow",
        )

    return resolved


def _ccd_levels(effect: FactorEffect, step_fraction: float) -> tuple[float, float, float]:
    """Low/center/high levels for one active variable's CCD axis.

    Centers on round 1's best observed level. If a +-step band around that best
    level would poke past a hard bound, the whole band is shifted inward (not just
    clipped on one side) so round 2 still gets a real spread instead of a corner
    that collapses onto the center -- the boundary_note surfaced by
    resolve_round2_variables is what tells a human this happened, not this shift.
    """

    lower, upper = CONTINUOUS_BOUNDS[effect.variable]
    xs = sorted(effect.tested_values)
    ofat_step = (max(xs) - min(xs)) / max(len(xs) - 1, 1)
    step = ofat_step * step_fraction
    center = effect.best_value
    if center - step < lower:
        center = lower + step
    elif center + step > upper:
        center = upper - step
    center = min(max(center, lower), upper)
    low_level = max(center - step, lower)
    high_level = min(center + step, upper)
    return low_level, center, high_level


def generate_ccd(
    active_effects: list[FactorEffect],
    step_fraction: float = 0.5,
    n_center: int | None = None,
) -> list[dict[str, float]]:
    """Face-centered CCD (alpha=1) for 0-3 active continuous variables.

    Face-centered (rather than rotatable) so axial points never extend past the
    factorial corners -- with hard PDF-given bounds, a rotatable design's
    farther-out axial points could fall outside the sanctioned range.
    """

    k = len(active_effects)
    if k == 0:
        return []
    default_centers = {1: 3, 2: 5, 3: 4}
    n_center = n_center if n_center is not None else default_centers.get(k, 3)

    names = [effect.variable for effect in active_effects]
    levels = {effect.variable: _ccd_levels(effect, step_fraction) for effect in active_effects}
    center_row = {name: levels[name][1] for name in names}

    if k == 1:
        (name,) = names
        low, center, high = levels[name]
        rows = [{name: low}, {name: high}] + [dict(center_row) for _ in range(n_center)]
        return rows

    rows = []
    for combo in itertools.product([-1, 1], repeat=k):
        rows.append({name: levels[name][1 + code] for name, code in zip(names, combo)})
    for name in names:
        for code in (-1, 1):
            row = dict(center_row)
            row[name] = levels[name][1 + code]
            rows.append(row)
    rows.extend(dict(center_row) for _ in range(n_center))
    return rows


def find_reusable_round1_row(
    candidate: dict[str, float],
    round1_df: pd.DataFrame,
    variables: list[str],
    tol: float = 1e-6,
) -> str | None:
    """Return the round-1 run_id whose full 6-variable point matches `candidate`, if any."""

    mask = pd.Series(True, index=round1_df.index)
    for variable in variables:
        if variable not in round1_df.columns:
            return None
        mask &= round1_df[variable].apply(lambda value: _is_close(value, candidate[variable], tol))
    matches = round1_df.loc[mask]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return str(row["run_id"]) if "run_id" in row.index else str(matches.index[0])


def od600_threshold(
    df: pd.DataFrame,
    baseline: dict[str, float],
    od_col: str = OD_COL,
    fraction: float = 0.7,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """OD600 floor = `fraction` x baseline OD600 mean.

    `fraction` is an engineering default (growth should be at least this share of
    what the known-working baseline achieves), not a biological judgement -- the R&D
    team should confirm or override it once round 1's OD600 spread is in hand.
    """

    mask = pd.Series(True, index=df.index)
    for variable in ALL_VARIABLES:
        if variable in df.columns:
            mask &= df[variable].apply(lambda value: _is_close(value, baseline[variable], tol))
    baseline_od = df.loc[mask, od_col].dropna()
    if baseline_od.empty:
        raise ValueError("No baseline rows with an od600 value to derive a threshold from")
    mean_od = float(baseline_od.mean())
    return {"baseline_od_mean": mean_od, "fraction": fraction, "threshold": mean_od * fraction}


@dataclass
class Round2Plan:
    fixed_values: dict[str, float]
    active_variables: list[str]
    boundary_notes: dict[str, str]
    overflow_notes: dict[str, str]
    design_rows: list[dict[str, Any]]
    combo_interactions: list[dict[str, Any]]
    od_threshold: dict[str, Any]
    noise: dict[str, Any]


def plan_round2(
    round1_df: pd.DataFrame,
    baseline: dict[str, float],
    target_col: str = TARGET_COL,
    od_col: str = OD_COL,
    od_threshold_fraction: float = 0.7,
    ccd_step_fraction: float = 0.5,
    max_active_variables: int = 3,
) -> Round2Plan:
    """End-to-end round-2 plan from round-1 data: significance -> fixed values -> CCD rows."""

    analysis = analyze_round1_effects(round1_df, baseline, target_col=target_col)
    effects = analysis["effects"]
    resolved = resolve_round2_variables(effects, max_active_variables=max_active_variables)

    fixed_values = {name: var.fixed_value for name, var in resolved.items() if not var.active}
    active_names = [name for name, var in resolved.items() if var.active]
    active_effects = [effects[name] for name in active_names]
    boundary_notes = {name: var.boundary_note for name, var in resolved.items() if var.note_kind == "boundary"}
    overflow_notes = {name: var.boundary_note for name, var in resolved.items() if var.note_kind == "overflow"}

    ccd_rows = generate_ccd(active_effects, step_fraction=ccd_step_fraction)
    design_rows = []
    for row in ccd_rows:
        full_row = {**fixed_values, **row}
        reused = find_reusable_round1_row(full_row, round1_df, ALL_VARIABLES)
        design_rows.append({**full_row, "reused_from_round1": reused})

    return Round2Plan(
        fixed_values=fixed_values,
        active_variables=active_names,
        boundary_notes=boundary_notes,
        overflow_notes=overflow_notes,
        design_rows=design_rows,
        combo_interactions=analysis["combo_interactions"],
        od_threshold=od600_threshold(round1_df, baseline, od_col=od_col, fraction=od_threshold_fraction),
        noise=analysis["noise"],
    )


def _sample_bo_candidates(
    fixed_continuous_values: dict[str, float],
    active_variables: list[str],
    n_candidates: int,
    seed: int,
) -> pd.DataFrame:
    """Candidate grid = every temp_c x interval_h combination (BO, unlike the CCD
    track, is free to pick among all fixed-level options, not just round 1's best)
    crossed with uniform-random draws of the active continuous variables; every
    other continuous variable stays at its resolved round-1 value."""

    rng = np.random.default_rng(seed)
    fixed_level_names = list(FIXED_LEVELS)
    grid_combos = list(itertools.product(*[FIXED_LEVELS[name] for name in fixed_level_names]))
    per_combo = max(n_candidates // len(grid_combos), 1)
    other_fixed = {k: v for k, v in fixed_continuous_values.items() if k not in FIXED_LEVELS}

    rows = []
    for combo in grid_combos:
        for _ in range(per_combo):
            row = dict(other_fixed)
            row.update(dict(zip(fixed_level_names, combo)))
            for variable in active_variables:
                lower, upper = CONTINUOUS_BOUNDS[variable]
                row[variable] = float(rng.uniform(lower, upper))
            rows.append(row)
    return pd.DataFrame(rows)


def recommend_round2_bo_batch(
    round1_df: pd.DataFrame,
    fixed_values: dict[str, float],
    active_variables: list[str],
    od_threshold: float,
    n_batch: int = 9,
    target_col: str = TARGET_COL,
    od_col: str = OD_COL,
    n_candidates: int = 3000,
    seed: int = 0,
) -> dict[str, Any]:
    """Constrained BO batch: maximize predicted yield subject to predicted OD600 >=
    od_threshold, using two independent SingleTaskGPs (yield, OD600) rather than a
    joint constrained acquisition function.

    That's a deliberate simplification for this data regime: with ~18-40 training
    points a hand-tuned ConstrainedMCObjective is harder to sanity-check than
    "fit two GPs, filter infeasible candidates, rank by predicted yield", and the
    filter step is easy to explain to a non-BO audience. Revisit if/when there's
    enough data that leaving qNEI's acquisition value on the table starts to matter.
    """

    try:
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError("recommend_round2_bo_batch requires torch, botorch, and gpytorch.") from exc

    feature_cols = [variable for variable in ALL_VARIABLES if variable in round1_df.columns]
    train = round1_df[[*feature_cols, target_col, od_col]].dropna()
    if len(train) < 5:
        raise ValueError("At least 5 complete round-1 rows (with both yield and od600) are required")

    def _fit(train_y: np.ndarray) -> Any:
        torch.manual_seed(seed)
        train_X_t = torch.tensor(train[feature_cols].to_numpy(dtype=float), dtype=torch.double)
        train_Y_t = torch.tensor(train_y, dtype=torch.double).unsqueeze(-1)
        model = SingleTaskGP(
            train_X=train_X_t,
            train_Y=train_Y_t,
            input_transform=Normalize(d=train_X_t.shape[-1]),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        return model

    yield_model = _fit(train[target_col].to_numpy(dtype=float))
    od_model = _fit(train[od_col].to_numpy(dtype=float))

    candidates = _sample_bo_candidates(fixed_values, active_variables, n_candidates, seed)
    candidates = candidates[feature_cols]
    X_cand = torch.tensor(candidates.to_numpy(dtype=float), dtype=torch.double)

    with torch.no_grad():
        yield_post = yield_model.posterior(X_cand)
        yield_mean = yield_post.mean.squeeze(-1).cpu().numpy()
        yield_std = yield_post.variance.sqrt().squeeze(-1).cpu().numpy()
        od_mean = od_model.posterior(X_cand).mean.squeeze(-1).cpu().numpy()

    result = candidates.copy()
    result["predicted_yield"] = yield_mean
    result["predicted_yield_std"] = yield_std
    result["predicted_od600"] = od_mean
    result["feasible"] = od_mean >= od_threshold

    feasible = result.loc[result["feasible"]].sort_values("predicted_yield", ascending=False)
    if feasible.empty:
        raise ValueError(
            "No sampled candidates satisfy the OD600 constraint; try a lower od_threshold, "
            "more n_candidates, or wider active-variable bounds"
        )

    top = feasible.head(n_batch).reset_index(drop=True)
    return {
        "recommendations": top.to_dict("records"),
        "n_feasible": int(len(feasible)),
        "n_candidates": int(len(result)),
        "od_threshold": float(od_threshold),
    }
