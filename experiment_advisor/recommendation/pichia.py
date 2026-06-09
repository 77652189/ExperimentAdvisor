from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import random

import pandas as pd

from experiment_advisor.utils.lhs import latin_hypercube

PICHIA_TARGET_COL = "yield_g_per_l"


@dataclass(frozen=True)
class PichiaParameterSpec:
    label: str
    unit: str
    lower: float
    upper: float
    step: float
    low_delta: float
    medium_delta: float
    high_delta: float


PICHIA_PARAMETER_SPECS: dict[str, PichiaParameterSpec] = {
    "growth_phase_ph": PichiaParameterSpec("生长期 pH", "", 5.0, 6.0, 0.05, 0.10, 0.20, 0.35),
    "production_phase_ph": PichiaParameterSpec("生产期 pH", "", 5.0, 6.0, 0.05, 0.10, 0.20, 0.35),
    "growth_phase_temp_c": PichiaParameterSpec("生长期温度", "℃", 20.0, 30.0, 0.5, 1.0, 2.0, 3.5),
    "production_phase_temp_c": PichiaParameterSpec("生产期温度", "℃", 20.0, 30.0, 0.5, 1.0, 2.0, 3.5),
    "glucose_start_time_h": PichiaParameterSpec("葡萄糖加入时间", "h", 0.0, 120.0, 1.0, 6.0, 12.0, 24.0),
    "glucose_concentration_g_l": PichiaParameterSpec("葡萄糖补料浓度", "g/L", 3.0, 18.0, 0.5, 1.5, 3.0, 5.0),
    "fan_speed_rpm": PichiaParameterSpec("风扇转速", "rpm", 0.0, 900.0, 50.0, 100.0, 200.0, 300.0),
}

INTENSITY_DELTAS = {
    "低": "low_delta",
    "中": "medium_delta",
    "高": "high_delta",
}

METHOD_COUPLING_DEFAULTS = {
    "序贯 DOE（2因子）": 0.5,
    "单变量验证": 0.0,
    "联合探索（LHS）": 1.0,
}


def pichia_template_columns() -> list[str]:
    return [
        "run_id",
        "experiment_date",
        "strain_id",
        "parent_strain_id",
        *PICHIA_PARAMETER_SPECS.keys(),
        PICHIA_TARGET_COL,
        "notes",
    ]


def empty_pichia_template() -> pd.DataFrame:
    return pd.DataFrame(columns=pichia_template_columns())


def _as_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    rounded = round(value / step) * step
    if step < 1:
        step_text = f"{step:.10f}".rstrip("0")
        decimals = len(step_text.split(".")[1]) if "." in step_text else 0
    else:
        decimals = 0
    return round(float(rounded), decimals)


def _clip_and_round(value: float, spec: PichiaParameterSpec) -> float:
    clipped = min(max(float(value), spec.lower), spec.upper)
    return _round_to_step(clipped, spec.step)


def _clean_history(df: pd.DataFrame | None, target_col: str) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    history = df.copy()
    for column in PICHIA_PARAMETER_SPECS:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    if target_col in history.columns:
        history[target_col] = pd.to_numeric(history[target_col], errors="coerce")
    if "experiment_date" in history.columns:
        history["experiment_date"] = pd.to_datetime(history["experiment_date"], errors="coerce")
    return history


def _row_to_baseline(row: pd.Series) -> dict[str, float]:
    baseline = {}
    for name, spec in PICHIA_PARAMETER_SPECS.items():
        value = _as_float(row.get(name))
        if value is not None:
            baseline[name] = _clip_and_round(value, spec)
    return baseline


def _manual_baseline(manual_baseline: dict[str, Any] | None) -> dict[str, float]:
    result = {}
    for name, spec in PICHIA_PARAMETER_SPECS.items():
        value = _as_float((manual_baseline or {}).get(name))
        if value is not None:
            result[name] = _clip_and_round(value, spec)
    return result


def select_pichia_baseline(
    history: pd.DataFrame,
    *,
    strain_id: str,
    parent_strain_id: str | None,
    baseline_source: str,
    manual_baseline: dict[str, Any] | None = None,
    target_col: str = PICHIA_TARGET_COL,
) -> tuple[dict[str, float], dict[str, Any]]:
    history = _clean_history(history, target_col)
    strain_id = str(strain_id or "").strip()
    parent_strain_id = str(parent_strain_id or "").strip() or None

    def complete_rows(frame: pd.DataFrame) -> pd.DataFrame:
        available = [column for column in PICHIA_PARAMETER_SPECS if column in frame.columns]
        if not available:
            return pd.DataFrame()
        return frame.dropna(subset=available, how="all")

    selected = pd.DataFrame()
    support_type = "manual"
    warning = ""
    if baseline_source == "手动输入":
        baseline = _manual_baseline(manual_baseline)
        if not baseline:
            raise ValueError("手动基准点至少需要填写一个推荐参数")
        return baseline, {"source": baseline_source, "support_type": support_type, "warning": warning}

    if "strain_id" in history.columns:
        same = complete_rows(history[history["strain_id"].astype(str) == strain_id])
    else:
        same = pd.DataFrame()

    if baseline_source == "同菌种最近成功实验":
        if "experiment_date" in same.columns:
            selected = same.sort_values("experiment_date", ascending=False, na_position="last").head(1)
        else:
            selected = same.tail(1)
        support_type = "same_strain"
    elif baseline_source == "同菌种历史最优":
        if target_col in same.columns:
            selected = same.dropna(subset=[target_col]).sort_values(target_col, ascending=False).head(1)
        support_type = "same_strain"
    elif baseline_source == "亲本菌种历史最优":
        if parent_strain_id and "strain_id" in history.columns:
            parent = complete_rows(history[history["strain_id"].astype(str) == parent_strain_id])
            if target_col in parent.columns:
                selected = parent.dropna(subset=[target_col]).sort_values(target_col, ascending=False).head(1)
            support_type = "parent_strain"

    if selected.empty:
        manual = _manual_baseline(manual_baseline)
        if manual:
            warning = f"未找到 `{baseline_source}` 可用基准点，已使用手动输入。"
            return manual, {"source": "手动输入", "support_type": "manual", "warning": warning}
        raise ValueError(f"未找到 `{baseline_source}` 可用基准点，请改用手动输入。")

    row = selected.iloc[0]
    baseline = _row_to_baseline(row)
    if not baseline:
        raise ValueError(f"`{baseline_source}` 基准点缺少可推荐参数，请改用手动输入。")
    return baseline, {
        "source": baseline_source,
        "support_type": support_type,
        "run_id": row.get("run_id") or row.get("fermenter_run_id") or "",
        "strain_id": row.get("strain_id", ""),
        "yield": _as_float(row.get(target_col)),
        "warning": warning,
    }


def changed_variable_count(coupling: float, n_variables: int) -> int:
    if n_variables <= 0:
        return 0
    coupling = min(max(float(coupling), 0.0), 1.0)
    return min(n_variables, max(1, 1 + round(coupling * (n_variables - 1))))


def _select_changed_variables(
    variables: list[str],
    count: int,
    rank: int,
    rng: random.Random,
    exploration_counts: dict[str, int],
) -> list[str]:
    if count >= len(variables):
        return list(variables)
    ordered = sorted(variables, key=lambda name: (exploration_counts.get(name, 0), rng.random()))
    start = (rank - 1) % len(variables)
    rotated = ordered[start:] + ordered[:start]
    return rotated[:count]


def _intensity_delta(spec: PichiaParameterSpec, intensity: str) -> float:
    attr = INTENSITY_DELTAS.get(intensity, "medium_delta")
    return float(getattr(spec, attr))


def _perturb_value(
    baseline: float,
    spec: PichiaParameterSpec,
    intensity: str,
    unit_value: float,
) -> float:
    max_delta = _intensity_delta(spec, intensity)
    signed = (float(unit_value) - 0.5) * 2.0 * max_delta
    if abs(signed) < spec.step:
        signed = spec.step if unit_value >= 0.5 else -spec.step
    proposed = _clip_and_round(baseline + signed, spec)
    if proposed != baseline:
        return proposed
    flipped = _clip_and_round(baseline - signed, spec)
    return flipped if flipped != baseline else proposed


def _boundary_risk(params: dict[str, float]) -> tuple[str, float]:
    scores = []
    for name, value in params.items():
        spec = PICHIA_PARAMETER_SPECS.get(name)
        if not spec or spec.upper <= spec.lower:
            continue
        normalized = (float(value) - spec.lower) / (spec.upper - spec.lower)
        edge_distance = min(normalized, 1.0 - normalized)
        scores.append(max(0.0, 1.0 - edge_distance / 0.2))
    score = max(scores) if scores else 0.0
    if score >= 0.8:
        return "高", score
    if score >= 0.4:
        return "中", score
    return "低", score


def _glucose_overlap_risk(params: dict[str, float]) -> str:
    start = params.get("glucose_start_time_h")
    if start is None:
        return "未知：未推荐葡萄糖加入时间"
    if start < 24:
        return "高：葡萄糖较早加入，甘油是否耗尽未知"
    if start <= 36:
        return "中：次日加入，可能存在甘油/葡萄糖重叠"
    return "未知：缺少甘油耗尽记录"


def _execution_difficulty(changed_count: int, total_variables: int) -> str:
    if total_variables <= 1 or changed_count <= 1:
        return "低"
    if changed_count <= max(2, math.ceil(total_variables * 0.5)):
        return "中"
    return "高"


def _recommendation_type(changed_count: int, total_variables: int) -> str:
    if changed_count <= 1:
        return "单变量验证"
    if changed_count >= total_variables:
        return "全变量探索"
    return "折中探索"


def _unique_values(values: list[float]) -> list[float]:
    result = []
    seen = set()
    for value in values:
        key = round(float(value), 8)
        if key in seen:
            continue
        seen.add(key)
        result.append(float(value))
    return result


def uniform_single_variable_levels(
    variable: str,
    n_recommendations: int,
    lower: float | None = None,
    upper: float | None = None,
) -> list[float]:
    spec = PICHIA_PARAMETER_SPECS[variable]
    n_recommendations = min(max(int(n_recommendations), 2), 4)
    low = spec.lower if lower is None else _clip_and_round(float(lower), spec)
    high = spec.upper if upper is None else _clip_and_round(float(upper), spec)
    if low > high:
        low, high = high, low
    if low == high:
        return [low]
    values = [
        low + index * (high - low) / (n_recommendations - 1)
        for index in range(n_recommendations)
    ]
    return _unique_values([_clip_and_round(value, spec) for value in values])


def two_factor_doe_levels(
    variable: str,
    lower: float | None = None,
    upper: float | None = None,
) -> dict[str, float]:
    spec = PICHIA_PARAMETER_SPECS[variable]
    low = spec.lower if lower is None else _clip_and_round(float(lower), spec)
    high = spec.upper if upper is None else _clip_and_round(float(upper), spec)
    if low > high:
        low, high = high, low
    if low == high:
        raise ValueError(f"{spec.label} 的 DOE 低/高水平不能相同")
    return {"low": low, "high": high}


def two_factor_design_matrix(n_recommendations: int) -> list[tuple[int, int]]:
    if n_recommendations >= 4:
        return [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    if n_recommendations == 3:
        return [(-1, -1), (1, -1), (-1, 1)]
    return [(-1, -1), (1, 1)]


def _two_factor_information_gain(
    active_variables: list[str],
    n_recommendations: int,
) -> dict[str, Any]:
    labels = [PICHIA_PARAMETER_SPECS[name].label for name in active_variables]
    can_estimate_main = n_recommendations >= 3
    can_estimate_interaction = n_recommendations >= 4
    estimable = []
    if can_estimate_main:
        estimable.extend([f"{labels[0]}主效应", f"{labels[1]}主效应"])
    else:
        estimable.append("两个变量的联合趋势")
    if can_estimate_interaction:
        estimable.append(f"{labels[0]}×{labels[1]}交互效应")
    return {
        "design_type": "2因子局部 DOE",
        "active_variables": active_variables,
        "active_variable_labels": labels,
        "n_recommendations": int(n_recommendations),
        "can_estimate_main_effects": can_estimate_main,
        "can_estimate_interaction": can_estimate_interaction,
        "design_matrix_rank": min(int(n_recommendations), 4),
        "estimable_terms": estimable,
        "vs_single_variable": (
            "同样 4 罐时，单变量法通常只能验证一个变量的响应曲线；"
            "2因子 DOE 可同时估计两个主效应和一次交互效应。"
            if can_estimate_interaction
            else "当前罐数不足以完整估计交互效应；建议 4 罐时使用完整 2×2 DOE。"
        ),
        "next_round_rule": "根据最高产组合移动基准点；主效应强则缩小该变量范围，交互强则继续围绕最佳组合做局部 DOE。",
    }


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _effect_direction(effect: float | None, threshold: float) -> str:
    if effect is None:
        return "信息不足"
    if abs(effect) < threshold:
        return "无明显信号"
    return "高水平更好" if effect > 0 else "低水平更好"


def _observed_yields_by_rank(observed_yields: dict[Any, Any] | list[Any]) -> dict[int, float]:
    if isinstance(observed_yields, dict):
        items = observed_yields.items()
    else:
        items = enumerate(observed_yields, start=1)
    result = {}
    for rank, value in items:
        numeric = _as_float(value)
        if numeric is None:
            continue
        try:
            result[int(rank)] = numeric
        except (TypeError, ValueError):
            continue
    return result


def analyze_pichia_doe_feedback(
    design_result: dict[str, Any],
    observed_yields: dict[Any, Any] | list[Any],
    *,
    practical_threshold: float = 0.05,
) -> dict[str, Any]:
    """Analyze observed yields from a two-factor sequential DOE result."""

    recommendations = design_result.get("recommendations") or []
    active_variables = (design_result.get("doe") or {}).get("active_variables") or design_result.get("variables") or []
    active_variables = [name for name in active_variables if name in PICHIA_PARAMETER_SPECS][:2]
    observed_by_rank = _observed_yields_by_rank(observed_yields)
    threshold = max(float(practical_threshold), 0.0)

    rows = []
    for item in recommendations:
        rank = int(item.get("rank", 0))
        observed = observed_by_rank.get(rank)
        codes = item.get("doe_factor_codes") or {}
        if observed is None or len(active_variables) != 2:
            continue
        if any(variable not in codes for variable in active_variables):
            continue
        rows.append(
            {
                "rank": rank,
                "observed_yield": observed,
                "params": item.get("params", {}),
                "codes": {variable: int(codes[variable]) for variable in active_variables},
            }
        )

    complete_count = len(rows)
    if len(active_variables) != 2 or complete_count == 0:
        return {
            "status": "insufficient",
            "completed_count": complete_count,
            "message": "需要先生成 2 因子 DOE，并回填至少 2 个实测产量。",
            "main_effects": [],
            "interaction_effect": None,
        }

    main_effects = []
    for variable in active_variables:
        low_values = [row["observed_yield"] for row in rows if row["codes"][variable] < 0]
        high_values = [row["observed_yield"] for row in rows if row["codes"][variable] > 0]
        low_mean = _mean(low_values)
        high_mean = _mean(high_values)
        effect = high_mean - low_mean if low_mean is not None and high_mean is not None else None
        main_effects.append(
            {
                "variable": variable,
                "label": PICHIA_PARAMETER_SPECS[variable].label,
                "low_mean": low_mean,
                "high_mean": high_mean,
                "effect": effect,
                "direction": _effect_direction(effect, threshold),
                "n_low": len(low_values),
                "n_high": len(high_values),
            }
        )

    by_codes = {
        (row["codes"][active_variables[0]], row["codes"][active_variables[1]]): row["observed_yield"]
        for row in rows
    }
    required_codes = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    interaction = None
    if all(code in by_codes for code in required_codes):
        interaction_effect = (
            by_codes[(1, 1)]
            + by_codes[(-1, -1)]
            - by_codes[(1, -1)]
            - by_codes[(-1, 1)]
        ) / 2.0
        interaction = {
            "effect": float(interaction_effect),
            "direction": (
                "同向组合更好"
                if interaction_effect >= threshold
                else "反向组合更好"
                if interaction_effect <= -threshold
                else "无明显交互"
            ),
            "can_estimate": True,
        }
    else:
        interaction = {
            "effect": None,
            "direction": "信息不足",
            "can_estimate": False,
        }

    best = max(rows, key=lambda row: row["observed_yield"])
    significant_main = [
        item
        for item in main_effects
        if item["effect"] is not None and abs(float(item["effect"])) >= threshold
    ]
    interaction_significant = (
        interaction.get("effect") is not None
        and abs(float(interaction["effect"])) >= threshold
    )

    if interaction_significant:
        suggestion = "交互效应明显：下一轮以最高产组合为新基准，继续围绕这两个变量做更窄范围的局部 DOE。"
    elif significant_main:
        labels = "、".join(item["label"] for item in significant_main)
        suggestion = f"{labels}存在主效应信号：下一轮把基准点往更好水平移动，并缩小该变量探索范围。"
    else:
        suggestion = "当前差异低于阈值：建议固定这两个变量，换另一组变量，或增加重复罐确认噪声。"

    return {
        "status": "complete" if complete_count == len(recommendations) else "partial",
        "completed_count": complete_count,
        "total_count": len(recommendations),
        "practical_threshold": threshold,
        "active_variables": active_variables,
        "active_variable_labels": [PICHIA_PARAMETER_SPECS[name].label for name in active_variables],
        "best_rank": best["rank"],
        "best_yield": best["observed_yield"],
        "suggested_baseline": best["params"],
        "main_effects": main_effects,
        "interaction_effect": interaction,
        "suggestion": suggestion,
    }


def _nearest_history(history: pd.DataFrame, params: dict[str, float]) -> dict[str, Any]:
    features = [name for name in params if name in history.columns]
    if not features:
        return {}
    clean = history.dropna(subset=features).copy()
    if clean.empty:
        return {}
    means = clean[features].mean()
    stds = clean[features].std(ddof=0).replace(0, 1.0)
    candidate = pd.Series(params)[features]
    distances = (((clean[features] - means) / stds - ((candidate - means) / stds)) ** 2).sum(axis=1) ** 0.5
    row = clean.assign(_distance=distances).sort_values("_distance").iloc[0]
    return {
        "nearest_run_id": row.get("run_id") or row.get("fermenter_run_id") or "",
        "nearest_strain_id": row.get("strain_id", ""),
        "nearest_distance": float(row["_distance"]),
        "nearest_yield": _as_float(row.get(PICHIA_TARGET_COL)),
    }


def _support_summary(
    baseline_meta: dict[str, Any],
    *,
    strain_id: str,
    parent_strain_id: str | None,
) -> tuple[str, str]:
    support_type = baseline_meta.get("support_type")
    if support_type == "same_strain":
        return "同菌种支持", "低"
    if support_type == "parent_strain":
        return "亲本菌种借鉴", "中"
    if support_type == "manual":
        return "手动基准点", "未知"
    nearest_strain = str(baseline_meta.get("strain_id", ""))
    if nearest_strain == strain_id:
        return "同菌种支持", "低"
    if parent_strain_id and nearest_strain == parent_strain_id:
        return "亲本菌种借鉴", "中"
    return "跨菌种信息不足", "高"


def _exploration_metrics(
    recommendations: list[dict[str, Any]],
    variables: list[str],
) -> dict[str, Any]:
    variable_rows = []
    for variable in variables:
        spec = PICHIA_PARAMETER_SPECS[variable]
        values = [rec["params"][variable] for rec in recommendations if variable in rec["changed_variables"]]
        all_values = [rec["params"][variable] for rec in recommendations]
        if values:
            rec_min = min(values)
            rec_max = max(values)
            coverage = (max(all_values) - min(all_values)) / (spec.upper - spec.lower) if spec.upper > spec.lower else 0.0
        else:
            rec_min = rec_max = None
            coverage = 0.0
        variable_rows.append(
            {
                "variable": variable,
                "label": spec.label,
                "changed_count": len(values),
                "n_recommendations": len(recommendations),
                "min": rec_min,
                "max": rec_max,
                "allowed_lower": spec.lower,
                "allowed_upper": spec.upper,
                "coverage": float(coverage),
            }
        )

    changed_counts = [len(rec["changed_variables"]) for rec in recommendations]
    avg_changed = float(sum(changed_counts) / len(changed_counts)) if changed_counts else 0.0
    denominator = max(len(variables) - 1, 1)
    joint = min(max((avg_changed - 1.0) / denominator, 0.0), 1.0)
    pairwise = []
    for i, left in enumerate(recommendations):
        for right in recommendations[i + 1 :]:
            dist = 0.0
            for variable in variables:
                spec = PICHIA_PARAMETER_SPECS[variable]
                scale = spec.upper - spec.lower or 1.0
                dist += ((left["params"][variable] - right["params"][variable]) / scale) ** 2
            pairwise.append(dist ** 0.5)
    mean_pairwise = float(sum(pairwise) / len(pairwise)) if pairwise else 0.0
    if mean_pairwise >= 0.45:
        diversity = "高"
    elif mean_pairwise >= 0.2:
        diversity = "中"
    else:
        diversity = "低"

    return {
        "variables": variable_rows,
        "average_changed_variables": avg_changed,
        "univariate_score": 1.0 - joint,
        "joint_exploration_score": joint,
        "mean_pairwise_distance": mean_pairwise,
        "batch_diversity": diversity,
    }


def recommend_pichia_design(
    df: pd.DataFrame | None,
    *,
    strain_id: str,
    parent_strain_id: str | None = None,
    baseline_source: str = "同菌种历史最优",
    manual_baseline: dict[str, Any] | None = None,
    variables: list[str] | None = None,
    intensities: dict[str, str] | None = None,
    coupling: float = 0.5,
    doe_variables: list[str] | None = None,
    doe_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    single_variable: str | None = None,
    single_variable_bounds: tuple[float | None, float | None] | None = None,
    n_recommendations: int = 4,
    seed: int = 0,
    target_col: str = PICHIA_TARGET_COL,
) -> dict[str, Any]:
    history = _clean_history(df, target_col)
    variables = [name for name in (variables or list(PICHIA_PARAMETER_SPECS)) if name in PICHIA_PARAMETER_SPECS]
    if not variables:
        raise ValueError("至少需要选择一个可探索变量")
    n_recommendations = min(max(int(n_recommendations), 2), 4)

    baseline, baseline_meta = select_pichia_baseline(
        history,
        strain_id=strain_id,
        parent_strain_id=parent_strain_id,
        baseline_source=baseline_source,
        manual_baseline=manual_baseline,
        target_col=target_col,
    )

    for name, spec in PICHIA_PARAMETER_SPECS.items():
        if name not in baseline:
            baseline[name] = (spec.lower + spec.upper) / 2.0
            baseline[name] = _clip_and_round(baseline[name], spec)

    rng = random.Random(seed)
    lhs_values = latin_hypercube(n_recommendations, len(variables), seed=seed)
    change_count = changed_variable_count(coupling, len(variables))
    exploration_counts = {name: 0 for name in variables}
    support_label, cross_strain_risk = _support_summary(
        baseline_meta,
        strain_id=strain_id,
        parent_strain_id=parent_strain_id,
    )

    if doe_variables:
        active_variables = [name for name in doe_variables if name in PICHIA_PARAMETER_SPECS]
        if len(active_variables) != 2:
            raise ValueError("序贯 DOE 需要且只需要选择 2 个主动探索变量")

        factor_levels = {}
        for variable in active_variables:
            lower, upper = (doe_bounds or {}).get(variable, (None, None))
            factor_levels[variable] = two_factor_doe_levels(variable, lower=lower, upper=upper)

        matrix = two_factor_design_matrix(n_recommendations)
        recommendations = []
        for index, (level_a, level_b) in enumerate(matrix):
            rank = index + 1
            params = dict(baseline)
            variable_changes = []
            factor_codes = {
                active_variables[0]: level_a,
                active_variables[1]: level_b,
            }
            for variable, code in factor_codes.items():
                spec = PICHIA_PARAMETER_SPECS[variable]
                old_value = params[variable]
                level_name = "high" if code > 0 else "low"
                new_value = factor_levels[variable][level_name]
                params[variable] = new_value
                variable_changes.append(
                    {
                        "variable": variable,
                        "label": spec.label,
                        "from": old_value,
                        "to": new_value,
                        "delta": _round_to_step(new_value - old_value, spec.step),
                        "intensity": "高水平" if code > 0 else "低水平",
                        "factor_code": code,
                    }
                )

            boundary_label, boundary_score = _boundary_risk(params)
            nearest = _nearest_history(history, params)
            recommendations.append(
                {
                    "rank": rank,
                    "recommendation_type": "序贯 DOE",
                    "params": params,
                    "changed_variables": active_variables,
                    "variable_changes": variable_changes,
                    "doe_factor_codes": factor_codes,
                    "strain_support": support_label,
                    "cross_strain_risk": cross_strain_risk,
                    "boundary_risk": boundary_label,
                    "boundary_risk_score": boundary_score,
                    "glucose_glycerol_overlap_risk": _glucose_overlap_risk(params),
                    "execution_difficulty": "中",
                    "nearest_history": nearest,
                    "reason": (
                        f"以{baseline_meta.get('source', baseline_source)}为基准，执行"
                        f"{PICHIA_PARAMETER_SPECS[active_variables[0]].label}×"
                        f"{PICHIA_PARAMETER_SPECS[active_variables[1]].label} 的局部 DOE。"
                    ),
                }
            )

        return {
            "mode": "pichia_sequential_doe",
            "strain_id": strain_id,
            "parent_strain_id": parent_strain_id,
            "baseline": baseline,
            "baseline_meta": baseline_meta,
            "variables": active_variables,
            "coupling": 0.5,
            "changed_variable_count": 2,
            "doe": {
                "active_variables": active_variables,
                "factor_levels": factor_levels,
                "design_matrix": matrix,
            },
            "information_gain": _two_factor_information_gain(active_variables, len(matrix)),
            "recommendations": recommendations,
            "exploration_metrics": _exploration_metrics(recommendations, active_variables),
        }

    if single_variable:
        if single_variable not in PICHIA_PARAMETER_SPECS:
            raise ValueError(f"未知单变量验证参数：{single_variable}")
        variables = [single_variable]
        level_lower, level_upper = single_variable_bounds or (None, None)
        levels = uniform_single_variable_levels(
            single_variable,
            n_recommendations,
            lower=level_lower,
            upper=level_upper,
        )
        recommendations = []
        spec = PICHIA_PARAMETER_SPECS[single_variable]
        for index, new_value in enumerate(levels):
            rank = index + 1
            params = dict(baseline)
            old_value = params[single_variable]
            params[single_variable] = new_value
            variable_changes = [
                {
                    "variable": single_variable,
                    "label": spec.label,
                    "from": old_value,
                    "to": new_value,
                    "delta": _round_to_step(new_value - old_value, spec.step),
                    "intensity": "水平值",
                }
            ]
            boundary_label, boundary_score = _boundary_risk(params)
            nearest = _nearest_history(history, params)
            recommendations.append(
                {
                    "rank": rank,
                    "recommendation_type": "单变量验证",
                    "params": params,
                    "changed_variables": [single_variable],
                    "variable_changes": variable_changes,
                    "strain_support": support_label,
                    "cross_strain_risk": cross_strain_risk,
                    "boundary_risk": boundary_label,
                    "boundary_risk_score": boundary_score,
                    "glucose_glycerol_overlap_risk": _glucose_overlap_risk(params),
                    "execution_difficulty": "低",
                    "nearest_history": nearest,
                    "reason": (
                        f"以{baseline_meta.get('source', baseline_source)}为基准，仅改变"
                        f"{spec.label}到 {new_value}，其他变量保持不变。"
                    ),
                }
            )

        return {
            "mode": "pichia_early_design",
            "strain_id": strain_id,
            "parent_strain_id": parent_strain_id,
            "baseline": baseline,
            "baseline_meta": baseline_meta,
            "variables": variables,
            "coupling": 0.0,
            "single_variable": single_variable,
            "single_variable_levels": levels,
            "changed_variable_count": 1,
            "recommendations": recommendations,
            "exploration_metrics": _exploration_metrics(recommendations, variables),
        }

    recommendations: list[dict[str, Any]] = []
    for index in range(n_recommendations):
        rank = index + 1
        params = dict(baseline)
        changed = _select_changed_variables(variables, change_count, rank, rng, exploration_counts)
        lhs_by_variable = {variable: lhs_values[index][position] for position, variable in enumerate(variables)}
        variable_changes = []
        for variable in changed:
            spec = PICHIA_PARAMETER_SPECS[variable]
            old_value = params[variable]
            intensity = (intensities or {}).get(variable, "中")
            new_value = _perturb_value(old_value, spec, intensity, lhs_by_variable[variable])
            params[variable] = new_value
            exploration_counts[variable] += 1
            variable_changes.append(
                {
                    "variable": variable,
                    "label": spec.label,
                    "from": old_value,
                    "to": new_value,
                    "delta": _round_to_step(new_value - old_value, spec.step),
                    "intensity": intensity,
                }
            )

        boundary_label, boundary_score = _boundary_risk(params)
        nearest = _nearest_history(history, params)
        rec_type = _recommendation_type(len(changed), len(variables))
        recommendations.append(
            {
                "rank": rank,
                "recommendation_type": rec_type,
                "params": params,
                "changed_variables": changed,
                "variable_changes": variable_changes,
                "strain_support": support_label,
                "cross_strain_risk": cross_strain_risk,
                "boundary_risk": boundary_label,
                "boundary_risk_score": boundary_score,
                "glucose_glycerol_overlap_risk": _glucose_overlap_risk(params),
                "execution_difficulty": _execution_difficulty(len(changed), len(variables)),
                "nearest_history": nearest,
                "reason": f"以{baseline_meta.get('source', baseline_source)}为基准，进行{rec_type}，改变 {len(changed)} 个变量。",
            }
        )

    return {
        "mode": "pichia_early_design",
        "strain_id": strain_id,
        "parent_strain_id": parent_strain_id,
        "baseline": baseline,
        "baseline_meta": baseline_meta,
        "variables": variables,
        "coupling": min(max(float(coupling), 0.0), 1.0),
        "changed_variable_count": change_count,
        "recommendations": recommendations,
        "exploration_metrics": _exploration_metrics(recommendations, variables),
    }
