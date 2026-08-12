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

from experiment_advisor.utils.lhs import latin_hypercube

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
    ci_low: float = 0.0  # confidence interval on effect_magnitude, see effect_confidence_interval()
    ci_high: float = 0.0
    # True when tested_values holds only the baseline level -- round 1 skipped this
    # variable's OFAT block entirely, so "not significant" here means "never tried",
    # not "tried and found flat". Round 1 execution can legitimately drop a variable's
    # OFAT block (equipment/scheduling limits), so this must stay a flag round 2 can
    # act on rather than an error.
    untested: bool = False


def effect_confidence_interval(
    effect: float,
    baseline_sd: float,
    n_baseline_replicates: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Confidence interval for a single-replicate effect vs. a replicated baseline mean.

    Round 1 mostly has one observation per non-baseline level, so a per-level
    variance can't be estimated on its own. Standard DOE practice is to assume
    the baseline replicates' variance (pure error) applies uniformly across the
    design space -- that's the whole point of repeating the center/baseline
    point rather than every treatment. Under that assumption, a single new
    observation minus the baseline mean has variance baseline_sd^2 * (1 + 1/n),
    and the interval uses a t critical value at n_baseline_replicates - 1
    degrees of freedom. With only 3 replicates (df=2) the interval is wide --
    that width is honest information about how little replication round 1 has,
    not a defect to hide.
    """

    df = n_baseline_replicates - 1
    if df < 1:
        return (float("nan"), float("nan"))
    from scipy import stats

    se = baseline_sd * (1.0 + 1.0 / n_baseline_replicates) ** 0.5
    t_crit = float(stats.t.ppf(0.5 + confidence / 2.0, df))
    margin = t_crit * se
    return (effect - margin, effect + margin)


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
        untested = not non_baseline_deviations

        at_lower = at_upper = False
        if kind == "continuous":
            lower, upper = CONTINUOUS_BOUNDS[variable]
            at_lower = _is_close(best_value, lower, tol=1e-3)
            at_upper = _is_close(best_value, upper, tol=1e-3)

        ci_low, ci_high = effect_confidence_interval(
            effect_magnitude, noise["baseline_sd"], noise["n_replicates"]
        )
        ci_low = max(ci_low, 0.0) if pd.notna(ci_low) else 0.0

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
            ci_low=ci_low,
            ci_high=ci_high if pd.notna(ci_high) else effect_magnitude,
            untested=untested,
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
    note_kind: str | None = None  # "boundary" | "overflow" | "untested", distinguishes the note reasons
    untested: bool = False


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

    def _fixed(effect: FactorEffect) -> Round2Variable:
        note = None
        if effect.untested:
            note = (
                f"round 1 未对该变量做单变量测试（没有任何行只改它、其余维持基线），"
                f"固定在 {effect.best_value:g} 只是沿用基线取值，不代表已验证比其它水平更好"
            )
        return Round2Variable(
            variable=effect.variable,
            active=False,
            fixed_value=effect.best_value,
            boundary_note=note,
            note_kind="untested" if note else None,
            untested=effect.untested,
        )

    resolved: dict[str, Round2Variable] = {}
    continuous_candidates = []
    for variable, effect in effects.items():
        if effect.kind == "fixed_level":
            resolved[variable] = _fixed(effect)
            continue
        if effect.significant:
            continuous_candidates.append(effect)
        else:
            resolved[variable] = _fixed(effect)

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
    untested_notes: dict[str, str]
    design_rows: list[dict[str, Any]]
    combo_interactions: list[dict[str, Any]]
    od_threshold: dict[str, Any]
    noise: dict[str, Any]
    effects: dict[str, FactorEffect]


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
    untested_notes = {name: var.boundary_note for name, var in resolved.items() if var.note_kind == "untested"}

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
        untested_notes=untested_notes,
        design_rows=design_rows,
        combo_interactions=analysis["combo_interactions"],
        od_threshold=od600_threshold(round1_df, baseline, od_col=od_col, fraction=od_threshold_fraction),
        noise=analysis["noise"],
        effects=effects,
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
        # fitted models + the exact column order they were trained on, so a
        # caller can query .posterior() for visualization (e.g. gp_partial_
        # dependence below) without refitting -- same principle as ADR-0005
        # for the legacy E.coli GP plots.
        "yield_model": yield_model,
        "od_model": od_model,
        "feature_cols": feature_cols,
    }


def gp_partial_dependence(
    model: Any,
    feature_cols: list[str],
    variable: str,
    anchor: dict[str, float],
    lower: float,
    upper: float,
    resolution: int = 41,
) -> dict[str, Any]:
    """Sweeps `variable` across [lower, upper] holding every other feature in
    feature_cols at anchor's value, querying an already-fitted GP's posterior
    mean/std -- a partial-dependence read of what the model itself currently
    believes, not a refit (same principle as ADR-0005: reuse the fitted model
    for visualization).

    Unlike fit_ccd_response_surface's optimum search, this deliberately does
    NOT restrict the sweep to a narrow already-tested band: the GP's training
    data (round 1's OFAT rows, and any LHS points) already spans each
    variable's full original range, and the whole point of looking at this
    plot is to see where the model is confident (near training points) versus
    guessing (gaps between them) across that full range -- narrowing the
    sweep would hide exactly the signal this plot exists to show.
    """
    import torch

    x_values = np.linspace(lower, upper, resolution)
    grid = pd.DataFrame({column: [float(anchor[column])] * resolution for column in feature_cols})
    grid[variable] = x_values
    X = torch.tensor(grid[feature_cols].to_numpy(dtype=float), dtype=torch.double)
    with torch.no_grad():
        posterior = model.posterior(X)
        mean = posterior.mean.squeeze(-1).cpu().numpy()
        std = posterior.variance.sqrt().squeeze(-1).cpu().numpy()
    return {"variable": variable, "x": x_values, "mean": mean, "std": std}


ROUND2_TEMPLATE_COLUMNS: list[str] = [
    "run_id",
    "run_type",
    "changed_variable",
    "reused_from_round1",
    *ALL_VARIABLES,
    TARGET_COL,
    OD_COL,
]


def generate_round2_extension_design(
    plan: Round2Plan,
    baseline: dict[str, float],
    extra_interval_levels: list[float],
    interaction_variable: str = "glucose_pct",
    interaction_levels: list[float] | None = None,
    n_noise_reference: int = 2,
    n_lhs: int = 10,
    lhs_variables: list[str] | None = None,
    lhs_interval_levels: list[float] | None = None,
    run_id_start: int = 1,
    run_id_prefix: str = "R2",
    seed: int = 42,
) -> pd.DataFrame:
    """Three extra blocks beyond plan.design_rows (CCD), for testing a feed
    interval round 1 never tried (e.g. 36h, beyond round 1's 12/24h) --
    deliberately NOT added to FIXED_LEVELS["interval_h"]/generate_ccd, which
    stay untouched: generate_ccd only ever varies CONTINUOUS_BOUNDS
    variables, so a fixed-level variable's new level can't be folded into a
    response surface, and recommend_round2_bo_batch's candidate grid must not
    start proposing this interval before any real data at it exists (that
    would be the GP extrapolating into a total blind spot, not a supported
    prediction).

    - "interval_interaction": extra_interval_levels crossed with
      interaction_variable at interaction_levels (default: its round-1 OFAT
      low/high), everything else at plan.fixed_values -- the smallest design
      that can tell whether the new interval's effect depends on that
      variable, since generate_ccd has no mechanism to vary a fixed-level
      axis itself.
    - "noise_reference": n_noise_reference replicate rows at the new
      interval with interaction_variable held at its round-1 baseline value
      -- round 1's baseline noise says nothing about variability in a feed
      interval nobody has run yet, and this is the cheapest way to get a
      real (if rough) noise estimate for it before trusting anything else
      measured at that interval.
    - "lhs": n_lhs Latin Hypercube points across lhs_variables (default: the
      CCD's own active continuous variables), feed interval drawn uniformly
      from lhs_interval_levels (default: baseline's interval plus
      extra_interval_levels) -- CCD's corners/axial-arms/center leave gaps a
      GP has no training signal for; this targets those gaps for whichever
      BO round consumes the combined dataset next, rather than re-testing
      what the CCD already covers.

    Deterministic given (plan, baseline, extra_interval_levels, seed): every
    random draw (LHS point positions, per-LHS-row interval choice) goes
    through a generator seeded from `seed`, so regenerating with the same
    arguments reproduces byte-identical rows in the same order -- the design
    sheet handed to the wet lab must match whatever gets analyzed afterwards.
    """

    interaction_levels = list(interaction_levels) if interaction_levels is not None else [0.5, 1.5]
    lhs_variables = list(lhs_variables) if lhs_variables is not None else list(CONTINUOUS_BOUNDS)
    lhs_interval_levels = (
        list(lhs_interval_levels) if lhs_interval_levels is not None else [baseline["interval_h"], *extra_interval_levels]
    )

    counter = run_id_start - 1

    def _next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{run_id_prefix}-{counter:02d}"

    def _base_row() -> dict[str, float]:
        row = dict(baseline)
        row.update(plan.fixed_values)
        return row

    rows: list[dict[str, Any]] = []

    for interval_level in extra_interval_levels:
        for level in interaction_levels:
            row = _base_row()
            row[interaction_variable] = level
            row["interval_h"] = interval_level
            rows.append(
                {"run_id": _next_id(), "run_type": "interval_interaction", "changed_variable": interaction_variable, **row}
            )

    for interval_level in extra_interval_levels:
        for _ in range(max(int(n_noise_reference), 0)):
            row = _base_row()
            row["interval_h"] = interval_level
            # explicit, not left to _base_row()'s fallthrough: if
            # interaction_variable happens to already be in plan.fixed_values
            # (not active this round), _base_row() would otherwise hold it at
            # round 1's resolved-best value instead of baseline, making this
            # reference point's meaning depend on which variables happened to
            # be active -- pin it to baseline explicitly so it doesn't.
            row[interaction_variable] = baseline[interaction_variable]
            rows.append({"run_id": _next_id(), "run_type": "noise_reference", "changed_variable": None, **row})

    n_lhs = max(int(n_lhs), 0)
    if n_lhs > 0:
        lhs_unit_values = latin_hypercube(n_lhs, len(lhs_variables), seed=seed)
        rng = np.random.default_rng(seed)
        interval_choices = rng.choice(lhs_interval_levels, size=n_lhs)
        for index in range(n_lhs):
            row = _base_row()
            for position, variable in enumerate(lhs_variables):
                lower, upper = CONTINUOUS_BOUNDS[variable]
                row[variable] = lower + lhs_unit_values[index][position] * (upper - lower)
            row["interval_h"] = float(interval_choices[index])
            rows.append({"run_id": _next_id(), "run_type": "lhs", "changed_variable": None, **row})

    return pd.DataFrame(rows, columns=ROUND2_TEMPLATE_COLUMNS)


def assemble_round2_design(
    plan: Round2Plan,
    baseline: dict[str, float],
    round1_df: pd.DataFrame,
    extra_interval_levels: list[float],
    interaction_variable: str = "glucose_pct",
    interaction_levels: list[float] | None = None,
    n_noise_reference: int = 2,
    n_lhs: int = 10,
    lhs_variables: list[str] | None = None,
    lhs_interval_levels: list[float] | None = None,
    run_id_prefix: str = "R2",
    seed: int = 42,
) -> pd.DataFrame:
    """Combines plan.design_rows (CCD) with generate_round2_extension_design's
    three blocks into one flat, sequentially-numbered sheet ready for the wet
    lab -- a CCD point that already coincides with a round-1 sample
    ("reused_from_round1") is kept in the sheet, marked as such, AND
    pre-filled with that round-1 run's actual yield_g_per_l/od600 (looked up
    from round1_df), rather than left blank for someone to notice the marker
    and go find the value themselves -- the whole point of flagging a reused
    point is that nobody has to re-run that flask, so the sheet should say so
    with the number already in place, not just a note that a number exists
    somewhere else.
    """

    round1_by_run_id = round1_df.set_index("run_id") if "run_id" in round1_df.columns else round1_df

    rows: list[dict[str, Any]] = []
    for index, ccd_row in enumerate(plan.design_rows, start=1):
        reused = ccd_row.get("reused_from_round1")
        variables_only = {key: value for key, value in ccd_row.items() if key != "reused_from_round1"}
        row: dict[str, Any] = {
            "run_id": f"{run_id_prefix}-{index:02d}",
            "run_type": "ccd",
            "changed_variable": None,
            "reused_from_round1": reused,
            **variables_only,
        }
        if reused and reused in round1_by_run_id.index:
            source = round1_by_run_id.loc[reused]
            row[TARGET_COL] = source.get(TARGET_COL)
            row[OD_COL] = source.get(OD_COL)
        rows.append(row)
    ccd_frame = pd.DataFrame(rows, columns=ROUND2_TEMPLATE_COLUMNS)

    extension_frame = generate_round2_extension_design(
        plan,
        baseline,
        extra_interval_levels=extra_interval_levels,
        interaction_variable=interaction_variable,
        interaction_levels=interaction_levels,
        n_noise_reference=n_noise_reference,
        n_lhs=n_lhs,
        lhs_variables=lhs_variables,
        lhs_interval_levels=lhs_interval_levels,
        run_id_start=len(plan.design_rows) + 1,
        run_id_prefix=run_id_prefix,
        seed=seed,
    )

    return pd.concat([ccd_frame, extension_frame], ignore_index=True)


@dataclass(frozen=True)
class ResponseSurfaceFit:
    active_variables: list[str]
    term_names: list[str]
    coefficients: dict[str, float]
    coefficient_significance: dict[str, dict[str, float]]  # term -> {se, t_statistic, p_value, significant}
    r_squared: float
    n_points: int
    n_params: int
    lack_of_fit: dict[str, Any] | None  # None when the design has no replicated point to estimate pure error from
    optimum: dict[str, float]
    predicted_optimum: float


def _response_surface_design_matrix(df: pd.DataFrame, active_variables: list[str]) -> tuple[np.ndarray, list[str]]:
    """Full quadratic model: intercept + linear + quadratic + every pairwise
    interaction among active_variables -- the standard CCD analysis model,
    matched term-for-term to what generate_ccd's factorial/axial/center
    points are laid out to estimate."""
    term_names = ["intercept"]
    columns = [np.ones(len(df))]
    for variable in active_variables:
        term_names.append(variable)
        columns.append(df[variable].to_numpy(dtype=float))
    for variable in active_variables:
        term_names.append(f"{variable}^2")
        columns.append(df[variable].to_numpy(dtype=float) ** 2)
    for index, first in enumerate(active_variables):
        for second in active_variables[index + 1 :]:
            term_names.append(f"{first}*{second}")
            columns.append(df[first].to_numpy(dtype=float) * df[second].to_numpy(dtype=float))
    return np.column_stack(columns), term_names


def fit_ccd_response_surface(
    df: pd.DataFrame,
    active_variables: list[str],
    target_col: str = TARGET_COL,
    grid_resolution: int = 41,
    tol: float = 1e-6,
) -> ResponseSurfaceFit:
    """Fits the standard quadratic response-surface model to a CCD block's
    results and reports what that fit is actually good for: how well it
    describes the tested points (R^2), which individual terms are actually
    distinguishable from noise (coefficient_significance -- a t-test per
    term using the residual mean square, standard OLS practice, separate
    from the whole-model lack-of-fit test below), whether the model is
    missing real curvature (lack-of-fit F-test against the center points'
    pure error -- the reason generate_ccd always includes several center
    replicates), and where the fitted surface peaks.

    The optimum is found by grid search over each active variable's own
    *tested* range (min/max of that variable across df), not the original
    round-1 CONTINUOUS_BOUNDS -- the fitted quadratic is only trustworthy
    where the design actually has support; searching past that would be
    extrapolating a fitted curve into territory nobody measured.
    """

    rows = df.dropna(subset=[*active_variables, target_col])
    if len(rows) < 2:
        raise ValueError("Need at least 2 complete rows to fit a response surface")

    X, term_names = _response_surface_design_matrix(rows, active_variables)
    y = rows[target_col].to_numpy(dtype=float)
    n_points, n_params = X.shape
    if n_points <= n_params:
        raise ValueError(
            f"Not enough points ({n_points}) to fit {n_params} parameters for {len(active_variables)} active variable(s)"
        )

    coefficients_array, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coefficients_array
    residuals = y - fitted
    ss_residual = float(np.sum(residuals**2))
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 1.0
    df_residual = n_points - n_params

    lack_of_fit: dict[str, Any] | None = None
    location_keys = [tuple(round(float(rows.iloc[i][variable]), 6) for variable in active_variables) for i in range(len(rows))]
    groups: dict[tuple[float, ...], list[float]] = {}
    for key, value in zip(location_keys, y):
        groups.setdefault(key, []).append(value)
    replicated = {key: values for key, values in groups.items() if len(values) > 1}
    df_pure_error = sum(len(values) - 1 for values in replicated.values())
    if df_pure_error > 0 and df_residual > df_pure_error:
        ss_pure_error = sum(float(np.sum((np.array(values) - np.mean(values)) ** 2)) for values in replicated.values())
        ss_lack_of_fit = ss_residual - ss_pure_error
        df_lack_of_fit = df_residual - df_pure_error
        if ss_pure_error > 0 and df_lack_of_fit > 0:
            from scipy import stats

            f_statistic = (ss_lack_of_fit / df_lack_of_fit) / (ss_pure_error / df_pure_error)
            p_value = float(stats.f.sf(f_statistic, df_lack_of_fit, df_pure_error))
        else:
            f_statistic = float("nan")
            p_value = float("nan")
        lack_of_fit = {
            "ss_pure_error": ss_pure_error,
            "ss_lack_of_fit": ss_lack_of_fit,
            "df_pure_error": df_pure_error,
            "df_lack_of_fit": df_lack_of_fit,
            "f_statistic": f_statistic,
            "p_value": p_value,
            "significant_lack_of_fit": bool(pd.notna(p_value) and p_value < 0.05),
        }

    coefficients = {name: float(value) for name, value in zip(term_names, coefficients_array)}

    coefficient_significance: dict[str, dict[str, float]] = {}
    if df_residual > 0:
        from scipy import stats

        mse = ss_residual / df_residual
        xtx_inv = np.linalg.inv(X.T @ X)
        standard_errors = np.sqrt(np.clip(mse * np.diag(xtx_inv), 0.0, None))
        for name, coefficient, se in zip(term_names, coefficients_array, standard_errors):
            if se > 0:
                t_statistic = float(coefficient / se)
                p_value = float(2.0 * stats.t.sf(abs(t_statistic), df_residual))
            else:
                t_statistic = float("nan")
                p_value = float("nan")
            coefficient_significance[name] = {
                "se": float(se),
                "t_statistic": t_statistic,
                "p_value": p_value,
                "significant": bool(pd.notna(p_value) and p_value < 0.05),
            }

    grids = [np.linspace(float(rows[variable].min()), float(rows[variable].max()), grid_resolution) for variable in active_variables]
    mesh = np.meshgrid(*grids, indexing="ij") if len(grids) > 1 else [grids[0]]
    flat_columns = [component.ravel() for component in mesh]
    grid_points = pd.DataFrame({variable: flat_columns[index] for index, variable in enumerate(active_variables)})
    grid_X, _ = _response_surface_design_matrix(grid_points, active_variables)
    predicted_grid = grid_X @ coefficients_array
    best_index = int(np.argmax(predicted_grid))
    optimum = {variable: float(flat_columns[index][best_index]) for index, variable in enumerate(active_variables)}

    return ResponseSurfaceFit(
        active_variables=list(active_variables),
        term_names=term_names,
        coefficients=coefficients,
        coefficient_significance=coefficient_significance,
        r_squared=float(r_squared),
        n_points=n_points,
        n_params=n_params,
        lack_of_fit=lack_of_fit,
        optimum=optimum,
        predicted_optimum=float(predicted_grid[best_index]),
    )


def analyze_interval_interaction(
    round2_df: pd.DataFrame,
    round1_df: pd.DataFrame,
    baseline: dict[str, float],
    interaction_variable: str,
    extra_interval_level: float,
    interaction_levels: list[float] | None = None,
    target_col: str = TARGET_COL,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Compares generate_round2_extension_design's "interval_interaction"
    block (extra_interval_level x interaction_variable at 2 levels) against
    round 1's matching OFAT rows for interaction_variable (same 2 levels, at
    round 1's baseline interval) to answer two questions: (1) does the new
    interval level change the yield at all, and (2) does that change depend
    on interaction_variable's level (a real interaction) or is it roughly
    the same regardless (just a main effect of interval).

    The noise estimate is the "noise_reference" block's own replicate spread
    at extra_interval_level -- deliberately NOT round 1's baseline noise,
    since that says nothing about variability in a feed interval nobody had
    tried before this round. Any of the returned "*_significant" fields is
    None (not False) when there isn't enough data to judge -- e.g. a missing
    backfilled result, or n_noise_reference was 0/1 when the design was
    generated -- so "not enough data" is never silently read as "no effect".
    """

    interaction_levels = (
        list(interaction_levels) if interaction_levels is not None else list(CONTINUOUS_BOUNDS[interaction_variable])
    )
    if len(interaction_levels) != 2:
        raise ValueError("analyze_interval_interaction compares exactly 2 interaction_variable levels")
    low_level, high_level = sorted(interaction_levels)

    def _lookup(rows: pd.DataFrame, level: float) -> float | None:
        matches = rows[rows[interaction_variable].apply(lambda value: _is_close(value, level, tol))]
        if matches.empty or pd.isna(matches.iloc[0][target_col]):
            return None
        return float(matches.iloc[0][target_col])

    new_rows = round2_df[
        (round2_df["run_type"] == "interval_interaction")
        & round2_df["interval_h"].apply(lambda value: _is_close(value, extra_interval_level, tol))
    ]
    new_low = _lookup(new_rows, low_level)
    new_high = _lookup(new_rows, high_level)

    old_rows = _levels_for(interaction_variable, round1_df, baseline, ALL_VARIABLES, tol)
    old_low = _lookup(old_rows, low_level)
    old_high = _lookup(old_rows, high_level)

    noise_rows = round2_df[
        (round2_df["run_type"] == "noise_reference")
        & round2_df["interval_h"].apply(lambda value: _is_close(value, extra_interval_level, tol))
    ]
    noise_values = noise_rows[target_col].dropna().to_numpy(dtype=float)
    noise_sd = float(np.std(noise_values, ddof=1)) if len(noise_values) >= 2 else None
    threshold = 2.0 * noise_sd if noise_sd is not None else None

    effect_at_low = new_low - old_low if new_low is not None and old_low is not None else None
    effect_at_high = new_high - old_high if new_high is not None and old_high is not None else None
    interaction_effect = (
        effect_at_high - effect_at_low if effect_at_low is not None and effect_at_high is not None else None
    )

    def _significant(value: float | None) -> bool | None:
        if value is None or threshold is None:
            return None
        return abs(value) > threshold

    return {
        "extra_interval_level": extra_interval_level,
        "interaction_variable": interaction_variable,
        "low_level": low_level,
        "high_level": high_level,
        "effect_at_low": effect_at_low,
        "effect_at_high": effect_at_high,
        "interaction_effect": interaction_effect,
        "noise_sd": noise_sd,
        "noise_n": int(len(noise_values)),
        "threshold": threshold,
        "interval_effect_significant_at_low": _significant(effect_at_low),
        "interval_effect_significant_at_high": _significant(effect_at_high),
        "interaction_significant": _significant(interaction_effect),
    }


def evaluate_response_surface(fit: ResponseSurfaceFit, df: pd.DataFrame) -> np.ndarray:
    """Predicted target values for arbitrary rows, reusing fit's already-fit
    coefficients (not a refit). The two consumers are both diagnostic, not
    analytical: residual plots (predicted vs actual on the fitted points
    themselves) and response_surface_grid's contour slices."""
    X, term_names = _response_surface_design_matrix(df, fit.active_variables)
    coefficients_array = np.array([fit.coefficients[name] for name in term_names])
    return X @ coefficients_array


def response_surface_grid(
    fit: ResponseSurfaceFit,
    df: pd.DataFrame,
    x_variable: str,
    y_variable: str,
    fixed_at: dict[str, float] | None = None,
    resolution: int = 41,
) -> dict[str, Any]:
    """2D prediction grid for x_variable/y_variable, each spanning its own
    *tested* range within df (same reasoning as fit_ccd_response_surface's
    optimum search -- no extrapolating past what was actually measured),
    with every other active variable held at fixed_at (defaults to the
    fitted optimum's value for that variable). This is the standard
    CCD contour-plot slice: what the surface looks like across two
    variables at a time, with the rest pinned at a sensible point.
    """
    if x_variable not in fit.active_variables or y_variable not in fit.active_variables:
        raise ValueError("x_variable/y_variable must both be in fit.active_variables")

    fixed_at = dict(fixed_at) if fixed_at is not None else {}
    other_variables = [variable for variable in fit.active_variables if variable not in (x_variable, y_variable)]
    for variable in other_variables:
        fixed_at.setdefault(variable, fit.optimum[variable])

    x_values = np.linspace(float(df[x_variable].min()), float(df[x_variable].max()), resolution)
    y_values = np.linspace(float(df[y_variable].min()), float(df[y_variable].max()), resolution)
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)
    grid_points = pd.DataFrame({x_variable: mesh_x.ravel(), y_variable: mesh_y.ravel()})
    for variable in other_variables:
        grid_points[variable] = fixed_at[variable]
    predicted = evaluate_response_surface(fit, grid_points).reshape(mesh_x.shape)

    return {
        "x_variable": x_variable,
        "y_variable": y_variable,
        "x_values": x_values,
        "y_values": y_values,
        "z": predicted,
        "fixed_at": fixed_at,
    }


@dataclass(frozen=True)
class ResponseSurfaceVerdict:
    severity: str  # "success" | "info" | "warning"
    message: str


def summarize_response_surface(
    fit: ResponseSurfaceFit,
    df: pd.DataFrame,
    od_fit: ResponseSurfaceFit | None = None,
    od_threshold: float | None = None,
    boundary_tol: float = 0.02,
) -> list[ResponseSurfaceVerdict]:
    """Turns fit_ccd_response_surface's (and optionally an OD600 fit's)
    already-computed numbers into the plain-language read a domain-competent
    reviewer would do by hand, ending in one concrete next step -- R^2 and a
    lack-of-fit p-value on their own don't say what to actually do next, and
    without this function that reading only exists in whoever looks at the
    numbers that particular day.

    df must be the same rows fit was built from (only used here to read each
    active variable's tested min/max, to check whether the optimum landed on
    the edge of what was actually tested).
    """

    verdicts: list[ResponseSurfaceVerdict] = []

    if fit.r_squared >= 0.9:
        verdicts.append(ResponseSurfaceVerdict("success", f"拟合优度很高（R²={fit.r_squared:.3g}），二次模型能很好地描述这批数据。"))
    elif fit.r_squared >= 0.7:
        verdicts.append(ResponseSurfaceVerdict("info", f"拟合优度中等（R²={fit.r_squared:.3g}），二次模型大致能描述趋势，但还有一部分变化解释不了。"))
    else:
        verdicts.append(ResponseSurfaceVerdict("warning", f"拟合优度偏低（R²={fit.r_squared:.3g}），二次模型可能不足以描述这批数据，后面的结论要谨慎看待。"))

    lack_of_fit_bad = False
    if fit.lack_of_fit is None:
        verdicts.append(ResponseSurfaceVerdict("info", "样本不足以做失拟检验，模型形式对不对暂时无法判断。"))
    elif fit.lack_of_fit["significant_lack_of_fit"]:
        lack_of_fit_bad = True
        verdicts.append(
            ResponseSurfaceVerdict(
                "warning",
                f"存在显著失拟（p={fit.lack_of_fit['p_value']:.3g}）：真实响应面可能不是简单的二次曲面"
                "（比如有更高阶弯曲，或者漏看了某个变量/交互），预测最优点只能当参考，不能直接当结论。",
            )
        )
    else:
        verdicts.append(ResponseSurfaceVerdict("success", "未见显著失拟：目前数据不能拒绝「二次模型合适」这个假设。"))

    for variable in fit.active_variables:
        quad_stats = fit.coefficient_significance.get(f"{variable}^2")
        linear_stats = fit.coefficient_significance.get(variable)
        quad_coef = fit.coefficients.get(f"{variable}^2", 0.0)
        if quad_stats and quad_stats["significant"]:
            shape = "峰值曲率（存在真实的转折/封顶）" if quad_coef < 0 else "谷值曲率（如果目标是最大化产量，最优可能在边界而不是中间）"
            verdicts.append(ResponseSurfaceVerdict("info", f"「{variable}」呈现{shape}。"))
        elif linear_stats and linear_stats["significant"]:
            verdicts.append(ResponseSurfaceVerdict("info", f"「{variable}」在测试范围内大致是单调趋势，暂时没看到弯曲/封顶的迹象。"))

    any_boundary = False
    for variable in fit.active_variables:
        lower = float(df[variable].min())
        upper = float(df[variable].max())
        span = upper - lower
        if span <= 0:
            continue
        value = fit.optimum[variable]
        if (value - lower) / span < boundary_tol:
            any_boundary = True
            verdicts.append(
                ResponseSurfaceVerdict(
                    "warning",
                    f"「{variable}」的联合最优点贴着本轮测试范围下限（{lower:g}），实际最优可能在范围之外，"
                    "建议下一轮往更低的方向扩大范围，而不是直接验证当前点。",
                )
            )
        elif (upper - value) / span < boundary_tol:
            any_boundary = True
            verdicts.append(
                ResponseSurfaceVerdict(
                    "warning",
                    f"「{variable}」的联合最优点贴着本轮测试范围上限（{upper:g}），实际最优可能在范围之外，"
                    "建议下一轮往更高的方向扩大范围，而不是直接验证当前点。",
                )
            )

    od_feasible: bool | None = None
    if od_fit is not None and od_threshold is not None:
        predicted_od = float(evaluate_response_surface(od_fit, pd.DataFrame([fit.optimum]))[0])
        od_feasible = predicted_od >= od_threshold
        if not od_feasible:
            verdicts.append(
                ResponseSurfaceVerdict(
                    "warning",
                    f"产量最优点的预测 OD600（{predicted_od:.3g}）低于可行阈值（{od_threshold:.3g}）："
                    "这个点按产量看最优，但按生长可行性看不成立，不能直接当作推荐条件。",
                )
            )

    if lack_of_fit_bad:
        next_step = "先不要用这个模型的最优点做决策：建议重新检查是否漏看了变量/交互，或尝试更高阶的模型，而不是直接推进到验证实验。"
        severity = "warning"
    elif any_boundary:
        next_step = "建议先扩大测试范围，而不是直接安排验证批次——当前最优点很可能不是真正的最优，只是测试范围的边缘。"
        severity = "warning"
    elif od_feasible is False:
        next_step = "建议在产量和 OD600 可行性之间找折中点（参考「合并数据后的贝叶斯优化建议」），不要直接采用这个纯产量最优点。"
        severity = "warning"
    else:
        next_step = "模型和数据一致、最优点在测试范围内部且（若已检验）可行：建议安排 1-2 个验证批次，实测这个预测最优点附近的条件，确认与模型预测一致后再定为下一轮基准。"
        severity = "success"
    verdicts.append(ResponseSurfaceVerdict(severity, f"下一步建议：{next_step}"))

    return verdicts
