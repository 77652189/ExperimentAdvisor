from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TARGET_COL = "yield_g_per_l"
DEFAULT_SOURCE_DIR = "data/csv_from_excel"

CONTROL_FEATURES = [
    "temperature_growth_phase_c",
    "temperature_shift_time_h",
    "temperature_production_phase_c",
    "ph_mean",
    "feed1_total_ml",
    "feed1_start_time_h",
    "feed1_end_time_h",
    "feed2_total_ml",
    "feed2_start_time_h",
    "feed2_end_time_h",
    "lactose_total_ml",
    "lactose_first_add_time_h",
    "lactose_last_add_time_h",
    "lactose_after_48h_ml",
    "fermentation_duration_h",
]

# 贝叶斯优化特征白名单（基于 EDA 筛选）。
# 依据详见 summary/supporting_reports/eda_report.md
#
# 被排除的特征及原因：
#   temperature_growth_phase_c  — CV=0.003，高低产组 Mann-Whitney p=0.981，无区分信号
#   ph_mean                     — CV=0.003，与 temperature_shift_time_h 强耦合（r=-0.89），
#                                  不是独立可控变量
#   feed1_start_time_h          — 与 feed2_start_time_h 完全相同（r=1.00），CV=0.012
#   feed2_start_time_h          — 与 feed1_start_time_h 完全相同（r=1.00），CV=0.012
#   feed1_end_time_h            — 与 fermentation_duration_h 高度冗余（r=0.96）
#   feed2_end_time_h            — 与 feed2_total_ml 高度冗余（r=0.89）
#   lactose_last_add_time_h     — 高低产组无显著差异（p=0.258），与 lactose_total_ml 语义重叠
#   fermentation_duration_h     — Spearman r=-0.29（最弱），与 feed2_total_ml/temperature_shift_time_h
#                                  高度混淆（r≈0.53）；短周期实验产量不低于长周期，
#                                  时长是实验代次的代理变量而非独立工艺参数
#   lactose_after_48h_ml        — Spearman r=-0.70，但与 temperature_shift_time_h 高度共线
#                                  （两者均为实验年代代理变量）。加入后 BoTorch GP ARD 将
#                                  temperature_shift_time_h 压平（length scale → ∞），
#                                  LOO-CV 无改善（MAE 10.71 vs 10.74），净效果为负，回退。
MODEL_FEATURES: list[str] = [
    "temperature_shift_time_h",        # 产量最强预测因子：Spearman r=-0.55，高产组比低产组早 6.7 h
    "temperature_production_phase_c",  # Spearman r=0.51，独立可调的生产相温度
    "lactose_total_ml",                # Spearman r=-0.52，HMO 底物总量
    "feed1_total_ml",                  # Spearman r=0.38，主碳源补料量
    "feed2_total_ml",                  # Spearman r=-0.42，与乳糖竞争，与产量负相关
    "lactose_first_add_time_h",        # Pearson r=-0.39，乳糖首次添加时机，与时长独立性好
]


def _read_csv(source_dir: Path, name: str) -> pd.DataFrame:
    path = source_dir / f"{name}.csv"
    if not path.exists():
        raise ValueError(f"Missing required source table: {path}")
    return pd.read_csv(path)


def _series(group: pd.DataFrame, column: str) -> pd.Series:
    if column not in group:
        return pd.Series(dtype=float)
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    if column == "od600":
        values = _correct_od600_outliers(values)
    return values


def _correct_od600_outliers(values: pd.Series) -> pd.Series:
    corrected = values.copy()
    for position, index in enumerate(values.index):
        value = values.loc[index]
        if pd.isna(value) or value < 1000:
            continue
        candidate = value / 10
        previous_value = corrected.iloc[position - 1] if position > 0 else None
        next_value = values.iloc[position + 1] if position + 1 < len(values) else None
        neighbors = [item for item in [previous_value, next_value] if item is not None and not pd.isna(item) and item < 1000]
        if neighbors and min(neighbors) * 0.5 <= candidate <= max(neighbors) * 1.5:
            corrected.loc[index] = candidate
    return corrected


def _od600_outlier_count(group: pd.DataFrame) -> int:
    values = pd.to_numeric(group["od600"], errors="coerce").dropna() if "od600" in group else pd.Series(dtype=float)
    return int((values >= 1000).sum())


def _last(group: pd.DataFrame, column: str) -> float | None:
    values = _series(group, column)
    return float(values.iloc[-1]) if not values.empty else None


def _max(group: pd.DataFrame, column: str) -> float | None:
    values = _series(group, column)
    return float(values.max()) if not values.empty else None


def _min(group: pd.DataFrame, column: str) -> float | None:
    values = _series(group, column)
    return float(values.min()) if not values.empty else None


def _mean(group: pd.DataFrame, column: str) -> float | None:
    values = _series(group, column)
    return float(values.mean()) if not values.empty else None


def _std(group: pd.DataFrame, column: str) -> float:
    values = _series(group, column)
    return float(values.std(ddof=0)) if len(values) > 1 else 0.0


def _delta(group: pd.DataFrame, column: str) -> float | None:
    high = _max(group, column)
    low = _min(group, column)
    return high - low if high is not None and low is not None else None


def _cumulative_features(group: pd.DataFrame, column: str, prefix: str) -> dict[str, float | None]:
    values = group[["fermentation_time_h", column]].copy() if column in group else pd.DataFrame()
    if values.empty:
        return {
            f"{prefix}_total_ml": None,
            f"{prefix}_start_time_h": None,
            f"{prefix}_end_time_h": None,
            f"{prefix}_before_24h_ml": None,
            f"{prefix}_24_48h_ml": None,
            f"{prefix}_after_48h_ml": None,
            f"{prefix}_peak_delta_ml": None,
            f"{prefix}_mean_rate_ml_per_h": None,
        }

    values["time"] = pd.to_numeric(values["fermentation_time_h"], errors="coerce")
    values["value"] = pd.to_numeric(values[column], errors="coerce")
    values = values.dropna(subset=["time", "value"]).sort_values("time")
    if values.empty:
        return {
            f"{prefix}_total_ml": None,
            f"{prefix}_start_time_h": None,
            f"{prefix}_end_time_h": None,
            f"{prefix}_before_24h_ml": None,
            f"{prefix}_24_48h_ml": None,
            f"{prefix}_after_48h_ml": None,
            f"{prefix}_peak_delta_ml": None,
            f"{prefix}_mean_rate_ml_per_h": None,
        }

    cumulative = values["value"].clip(lower=0)
    deltas = cumulative.diff()
    deltas.iloc[0] = cumulative.iloc[0]
    deltas = deltas.clip(lower=0)
    values = values.assign(delta=deltas)

    positive_cumulative = values[values["value"] > 0]
    positive_delta = values[values["delta"] > 0]
    total = float(cumulative.max()) if not cumulative.empty else None
    start_time = float(positive_cumulative["time"].iloc[0]) if not positive_cumulative.empty else None
    end_time = float(positive_delta["time"].iloc[-1]) if not positive_delta.empty else start_time
    if total is not None and start_time is not None and end_time is not None and end_time > start_time:
        mean_rate = total / (end_time - start_time)
    elif total is not None and total > 0:
        mean_rate = total
    else:
        mean_rate = None

    return {
        f"{prefix}_total_ml": total,
        f"{prefix}_start_time_h": start_time,
        f"{prefix}_end_time_h": end_time,
        f"{prefix}_before_24h_ml": float(values.loc[values["time"] < 24, "delta"].sum()),
        f"{prefix}_24_48h_ml": float(values.loc[(values["time"] >= 24) & (values["time"] < 48), "delta"].sum()),
        f"{prefix}_after_48h_ml": float(values.loc[values["time"] >= 48, "delta"].sum()),
        f"{prefix}_peak_delta_ml": float(values["delta"].max()) if not values.empty else None,
        f"{prefix}_mean_rate_ml_per_h": mean_rate,
    }


def _temperature_phase_features(group: pd.DataFrame, threshold_c: float = 1.0) -> dict[str, float | None]:
    values = group[["fermentation_time_h", "temperature_c"]].copy() if "temperature_c" in group else pd.DataFrame()
    result = {
        "temperature_growth_phase_c": None,
        "temperature_shift_time_h": None,
        "temperature_production_phase_c": None,
    }
    if values.empty:
        return result

    values["time"] = pd.to_numeric(values["fermentation_time_h"], errors="coerce")
    values["temperature"] = pd.to_numeric(values["temperature_c"], errors="coerce")
    values = values.dropna(subset=["time", "temperature"]).sort_values("time")
    if values.empty:
        return result

    result["temperature_growth_phase_c"] = float(values["temperature"].iloc[0])
    diffs = values["temperature"].diff().abs()
    shift_positions = diffs[diffs >= threshold_c].index
    if len(shift_positions) == 0:
        result["temperature_production_phase_c"] = float(values["temperature"].iloc[-1])
        return result

    shift_idx = shift_positions[0]
    shift_loc = values.index.get_loc(shift_idx)
    before = values.iloc[:shift_loc]
    after = values.iloc[shift_loc:]
    result["temperature_shift_time_h"] = float(values.loc[shift_idx, "time"])
    result["temperature_growth_phase_c"] = float(before["temperature"].median()) if not before.empty else float(values["temperature"].iloc[0])
    result["temperature_production_phase_c"] = float(after["temperature"].median()) if not after.empty else float(values["temperature"].iloc[-1])
    return result


def _event_flags(text: str) -> dict[str, bool]:
    lowered = text.lower()
    abnormal = any(token in lowered for token in ["contamination", "abnormal", "fail"])
    abnormal = abnormal or any(token in text for token in ["污染", "异常", "失败", "杂菌"])
    return {
        "event_has_iptg": "iptg" in lowered,
        "event_has_lactose_addition": "乳糖" in text or "lactose" in lowered,
        "event_has_abnormal_note": abnormal,
        "event_has_magnesium_reduction": "降镁" in text or "镁离子" in text,
        "event_has_lactose_fed_batch": "乳糖流加" in text or "流加乳糖" in text,
    }


def _training_exclusion(row: dict[str, Any]) -> tuple[bool, str]:
    if pd.isna(row.get(TARGET_COL)):
        return True, "missing_yield"
    if row.get("event_has_abnormal_note"):
        return True, "abnormal_or_contamination_note"
    return False, ""


def _optional_csv(source_dir: Path, name: str) -> pd.DataFrame:
    path = source_dir / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _label_prefix(value: Any) -> str:
    import re

    text = "" if pd.isna(value) else str(value)
    match = re.search(r"F\d+", text, flags=re.IGNORECASE)
    return match.group(0).upper() if match else ""


def _liquid_run_map(runs: pd.DataFrame, liquid: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    label_to_run: dict[str, str] = {}
    label_to_method: dict[str, str] = {}
    if liquid.empty or "sample_label" not in liquid.columns:
        return label_to_run, label_to_method

    for experiment_id, liquid_group in liquid.groupby("experiment_id", dropna=False):
        run_group = runs[runs["experiment_id"] == experiment_id].copy()
        if run_group.empty:
            continue
        labels = liquid_group["sample_label"].dropna().astype(str).drop_duplicates().tolist()
        prefixes = {label: _label_prefix(label) for label in labels}
        label_counts = run_group["fermenter_label"].astype(str).str.upper().value_counts().to_dict()
        all_exact = bool(labels) and all(prefixes[label] and label_counts.get(prefixes[label], 0) == 1 for label in labels)
        if all_exact:
            for label in labels:
                run_id = run_group.loc[run_group["fermenter_label"].astype(str).str.upper() == prefixes[label], "id"].iloc[0]
                label_to_run[f"{experiment_id}::{label}"] = run_id
                label_to_method[f"{experiment_id}::{label}"] = "liquid_label_to_fermenter_label"
            continue

        used_run_ids: set[str] = set()
        for label in labels:
            prefix = prefixes[label]
            if prefix and label_counts.get(prefix, 0) == 1:
                run_id = run_group.loc[run_group["fermenter_label"].astype(str).str.upper() == prefix, "id"].iloc[0]
                label_to_run[f"{experiment_id}::{label}"] = run_id
                label_to_method[f"{experiment_id}::{label}"] = "liquid_label_partial_fermenter_label"
                used_run_ids.add(run_id)

        unmatched_labels = [label for label in labels if f"{experiment_id}::{label}" not in label_to_run]
        unmatched_runs = [run for _, run in run_group.iterrows() if run["id"] not in used_run_ids]
        for label, run in zip(unmatched_labels, unmatched_runs):
            label_to_run[f"{experiment_id}::{label}"] = run["id"]
            label_to_method[f"{experiment_id}::{label}"] = "liquid_label_excel_order_inferred"

        if len(unmatched_labels) == len(unmatched_runs):
            continue

        for label, (_, run) in zip(labels, run_group.iterrows()):
            key = f"{experiment_id}::{label}"
            if key not in label_to_run:
                label_to_run[key] = run["id"]
                label_to_method[key] = "liquid_label_sequence_fallback"
    return label_to_run, label_to_method


def _liquid_targets(runs: pd.DataFrame, liquid: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if liquid.empty:
        return {}

    label_to_run, label_to_method = _liquid_run_map(runs, liquid)
    liquid = liquid.copy()
    liquid["fermenter_run_id"] = liquid.apply(
        lambda row: label_to_run.get(f"{row.get('experiment_id')}::{row.get('sample_label')}"),
        axis=1,
    )
    liquid["target_match_method"] = liquid.apply(
        lambda row: label_to_method.get(f"{row.get('experiment_id')}::{row.get('sample_label')}", ""),
        axis=1,
    )
    liquid = liquid[liquid["fermenter_run_id"].notna()].copy()
    if liquid.empty:
        return {}

    liquid["value"] = pd.to_numeric(liquid["value"], errors="coerce")
    liquid["sample_time_h"] = pd.to_numeric(liquid["sample_time_h"], errors="coerce")
    targets: dict[str, dict[str, Any]] = {}
    for run_id, group in liquid.groupby("fermenter_run_id"):
        row: dict[str, Any] = {
            "target_source": "",
            "target_match_method": group["target_match_method"].dropna().astype(str).iloc[0],
        }
        yield_rows = group[(group["section"] == "extracellular_yield_g_per_l") & group["value"].notna()].sort_values("sample_time_h")
        if not yield_rows.empty:
            max_idx = yield_rows["value"].idxmax()
            final = yield_rows.iloc[-1]
            row.update(
                {
                    "target_yield_g_per_l": float(yield_rows.loc[max_idx, "value"]),
                    "target_yield_time_h": float(yield_rows.loc[max_idx, "sample_time_h"]),
                    "final_yield_g_per_l": float(final["value"]),
                    "final_yield_time_h": float(final["sample_time_h"]),
                    "target_source": "liquid_long_data.extracellular_yield_g_per_l",
                }
            )

        lactose_rows = group[group["section"].astype(str).str.contains("lactose", case=False, na=False) & group["value"].notna()].sort_values("sample_time_h")
        if not lactose_rows.empty:
            final_lactose = lactose_rows.iloc[-1]
            row["final_lactose_g_per_l"] = float(final_lactose["value"])
            row["final_lactose_time_h"] = float(final_lactose["sample_time_h"])
            row["min_lactose_g_per_l"] = float(lactose_rows["value"].min())
        targets[run_id] = row
    return targets


def build_run_level_dataset(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_path: str | Path = "data/final/run_level_modeling_dataset.csv",
) -> pd.DataFrame:
    """
    将整理后的关系型 CSV 聚合为每个 fermenter_run 一行的建模数据集。

    无产量或异常 run 会保留在输出中用于审计，但通过 exclude_from_training 标记排除训练。
    """

    source = Path(source_dir)
    experiments = _read_csv(source, "experiment")
    runs = _read_csv(source, "fermenter_run")
    time_series = _read_csv(source, "time_series_data")
    liquid = _optional_csv(source, "liquid_long_data")
    experiment_meta = experiments.set_index("id").to_dict(orient="index")
    liquid_targets = _liquid_targets(runs, liquid)

    rows: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        run_id = run["id"]
        group = time_series[time_series["fermenter_run_id"] == run_id].sort_values("fermentation_time_h")
        remarks = group["remarks"].dropna().astype(str).tolist() if "remarks" in group else []
        note_parts = [str(run.get("condition_notes", "")), *remarks]
        notes = " ".join(part for part in note_parts if part and part != "nan")
        experiment = experiment_meta.get(run.get("experiment_id"), {})

        row: dict[str, Any] = {
            "fermenter_run_id": run_id,
            "experiment_id": run.get("experiment_id"),
            "experiment_date": experiment.get("experiment_date"),
            "file_name": experiment.get("file_name"),
            "sheet_name": run.get("sheet_name"),
            "fermenter_label": run.get("fermenter_label"),
            "strain_name": run.get("strain_name"),
            "batch_number": run.get("batch_number"),
            "condition_notes": run.get("condition_notes"),
            "remarks_text": notes,
            "n_timepoints": int(len(group)),
            "fermentation_duration_h": _max(group, "fermentation_time_h"),
            "final_lactose_g_per_l": _last(group, "lactose_g_per_l"),
            "min_lactose_g_per_l": _min(group, "lactose_g_per_l"),
        }
        row.update(
            {
                "target_yield_g_per_l": _max(group, "yield_g_per_l"),
                "target_yield_time_h": None,
                "final_yield_g_per_l": _last(group, "yield_g_per_l"),
                "final_yield_time_h": _max(group, "fermentation_time_h") if _last(group, "yield_g_per_l") is not None else None,
                "target_source": "time_series_data.yield_g_per_l" if _max(group, "yield_g_per_l") is not None else "",
                "target_match_method": "same_fermenter_run",
            }
        )
        if row["target_yield_g_per_l"] is not None and "yield_g_per_l" in group:
            yields = group[["fermentation_time_h", "yield_g_per_l"]].copy()
            yields["yield_g_per_l"] = pd.to_numeric(yields["yield_g_per_l"], errors="coerce")
            yields = yields.dropna(subset=["yield_g_per_l"])
            if not yields.empty:
                max_idx = yields["yield_g_per_l"].idxmax()
                row["target_yield_time_h"] = float(yields.loc[max_idx, "fermentation_time_h"])

        row.update(liquid_targets.get(run_id, {}))
        row["yield_g_per_l"] = row.get("target_yield_g_per_l")
        row["max_yield_g_per_l"] = row.get("target_yield_g_per_l")

        for column in ["temperature_c", "ph", "od600"]:
            row[f"{column}_mean"] = _mean(group, column)
            row[f"{column}_std"] = _std(group, column)
            row[f"{column}_min"] = _min(group, column)
            row[f"{column}_max"] = _max(group, column)
            row[f"{column}_final"] = _last(group, column)
        row["od600_outlier_corrected_count"] = _od600_outlier_count(group)
        row.update(_temperature_phase_features(group))
        for column in ["feed1_ml", "feed2_ml", "base_ml", "lactose_ml", "volume_ml"]:
            row[f"{column}_final"] = _last(group, column)
            row[f"{column}_max"] = _max(group, column)
            row[f"{column}_delta"] = _delta(group, column)
        row.update(_cumulative_features(group, "feed1_ml", "feed1"))
        row.update(_cumulative_features(group, "feed2_ml", "feed2"))
        row.update(_cumulative_features(group, "base_ml", "base"))
        lactose_features = _cumulative_features(group, "lactose_ml", "lactose")
        lactose_features["lactose_first_add_time_h"] = lactose_features.pop("lactose_start_time_h")
        lactose_features["lactose_last_add_time_h"] = lactose_features.pop("lactose_end_time_h")
        row.update(lactose_features)
        row.update(_event_flags(notes))
        excluded, reason = _training_exclusion(row)
        row["exclude_from_training"] = excluded
        row["exclusion_reason"] = reason
        rows.append(row)

    result = pd.DataFrame(rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding="utf-8")
    return result


def training_view(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """返回可用于监督学习训练的 run-level 子集。"""

    excluded = df.get("exclude_from_training", False)
    return df.loc[(~excluded.astype(bool)) & df[target_col].notna()].copy()
