"""Shared Pichia hLF page pieces: data-area paths, Chinese labels, chart colors
and the row-level display helpers both rounds render design tables with.

Bottom of the Pichia UI import graph (see docs/adr/0017) -- this module must not
import any other App.pichia_* module. The PICHIA_-prefixed aliases for
experiment_advisor constants are all re-exported from here, so there is exactly
one place where the UI's vocabulary maps onto the algorithm layer's names;
algorithm *functions* are imported straight from experiment_advisor by whoever
calls them, not funnelled through this module.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_advisor.recommendation.round1_design import (
    BASELINE as PICHIA_ROUND1_BASELINE,
    OFAT_LEVELS as PICHIA_OFAT_LEVELS,
    round1_template_columns,
)
from experiment_advisor.recommendation.round2_design import (
    ALL_VARIABLES as PICHIA_VARIABLES,
    CONTINUOUS_BOUNDS as PICHIA_CONTINUOUS_BOUNDS,
    FIXED_LEVELS as PICHIA_FIXED_LEVELS,
    OD_COL as PICHIA_OD_COL,
    TARGET_COL as PICHIA_TARGET_COL,
)
from App.ui_shared import _num

PICHIA_DATA_DIR = PROJECT_ROOT / "data" / "pichia"
PICHIA_UPLOAD_DIR = PICHIA_DATA_DIR / "uploads"
PICHIA_FINAL_DIR = PICHIA_DATA_DIR / "final"
PICHIA_TEMPLATE_DIR = PICHIA_DATA_DIR / "templates"
PICHIA_DEFAULT_DATASET_PATH = PICHIA_FINAL_DIR / "pichia_run_level_dataset.csv"
PICHIA_ROUND2_DATASET_PATH = PICHIA_FINAL_DIR / "pichia_round2_dataset.csv"
PICHIA_TEMPLATE_PATH = PICHIA_TEMPLATE_DIR / "pichia_run_level_template.csv"
PICHIA_VARIABLE_LABELS = {
    "seed_od": "种子初始 OD600",
    "glucose_pct": "补料葡萄糖浓度 (%)",
    "ph": "培养基 pH",
    "volume_ml": "初始装液量 (mL)",
    "temp_c": "发酵温度 (℃)",
    "interval_h": "补料时间间隔 (h)",
}

# dataviz palette (dark-mode slots) -- see .claude/plans note on chart conventions:
# fixed categorical order for run_type identity, one diverging pair for correlation,
# accent-vs-muted "emphasis" pair for the effect-magnitude significance cutoff.
PICHIA_RUN_TYPE_COLORS = {
    "baseline": "#3987e5",
    "ofat": "#d95926",
    "combo": "#199e70",
    "ccd": "#8e6fce",
    "interval_interaction": "#c9a227",
    "noise_reference": "#e0607e",
    "lhs": "#4fb8af",
}
PICHIA_RUN_TYPE_LABELS = {
    "baseline": "基线重复",
    "ofat": "单变量(OFAT)",
    "combo": "联合探索",
    "ccd": "响应面(CCD)",
    "interval_interaction": "补料间隔交互",
    "noise_reference": "新区域噪声参考",
    "lhs": "拉丁超立方(LHS)",
}
PICHIA_DIVERGING_COLORSCALE = [[0, "#3987e5"], [0.5, "#383835"], [1, "#e66767"]]
# sequential (dark->bright) for magnitude data (predicted yield on a 3D
# surface/contour) -- several user-selectable options rather than one fixed
# scale: a 3-stop single hue (the original choice) reads "on brand" but
# compresses most of the value range into similar-looking mid blues, which
# made adjacent CCD contour bands hard to tell apart in practice. The default
# widens that same blue hue's lightness/chroma span instead of changing hue
# family; the other options trade "on brand" for more perceptual travel
# (still never a full hue-wheel rainbow -- see dataviz skill's sequential
# palette guidance).
PICHIA_SEQUENTIAL_COLORSCALE_OPTIONS: dict[str, list[list[Any]]] = {
    "扩展单色蓝（推荐）": [[0, "#0a1420"], [0.2, "#163a6b"], [0.4, "#2c5fa8"], [0.6, "#5ba0e0"], [0.8, "#a8d4f5"], [1, "#eaf4fd"]],
    "单色蓝（原配色）": [[0, "#1b2838"], [0.5, "#2c5fa8"], [1, "#7fb8f5"]],
    "蓝→青双色": [[0, "#0d1b2e"], [0.25, "#1a5276"], [0.5, "#17a398"], [0.75, "#7ee8d8"], [1, "#eafff9"]],
    "Viridis（多色，辨识度最高）": [[0, "#440154"], [0.25, "#3b528b"], [0.5, "#21918c"], [0.75, "#5ec962"], [1, "#fde725"]],
}
PICHIA_SEQUENTIAL_COLORSCALE_DEFAULT = "扩展单色蓝（推荐）"
PICHIA_ACCENT_COLOR = "#3987e5"
PICHIA_MUTED_COLOR = "#6b6a64"

def _ensure_pichia_data_area() -> None:
    for directory in [PICHIA_UPLOAD_DIR, PICHIA_FINAL_DIR, PICHIA_TEMPLATE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    if not PICHIA_TEMPLATE_PATH.exists():
        pd.DataFrame(columns=round1_template_columns()).to_csv(PICHIA_TEMPLATE_PATH, index=False, encoding="utf-8-sig")

def _pichia_restore_persisted_dataset(session_key: str, path: Path) -> bool:
    """Loads path into st.session_state[session_key] if that key isn't
    already set and the file exists -- st.session_state is per Streamlit
    process, so a server restart (not just a browser refresh) wipes it even
    though "save to data/pichia/final/" already wrote the real data to disk.
    Both save buttons literally promise "供...下次打开使用" (usable next time
    the app opens); without this, that promise only holds across a browser
    refresh, not a process restart, which is the more likely reason someone
    actually reopens the app later. Returns whether a restore happened, so
    the caller can show a one-time notice instead of staying silent about it."""
    if session_key in st.session_state:
        return False
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty:
        return False
    st.session_state[session_key] = df
    return True

def _pichia_numeric_results(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.copy()
    for column in (PICHIA_TARGET_COL, PICHIA_OD_COL):
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric

def _pichia_variable_display(row: dict[str, Any]) -> dict[str, Any]:
    return {PICHIA_VARIABLE_LABELS.get(name, name): _num(row.get(name)) for name in PICHIA_VARIABLES if name in row}

def _pichia_baseline_lookup(df: pd.DataFrame) -> dict[str, Any]:
    """Most-common value per variable column. Stands in for "the baseline"
    even when baseline-replicate rows are disabled (n_baseline_replicates=0),
    since every row that doesn't deliberately vary a given variable leaves it
    at that value -- so the mode across the whole design is still correct."""
    lookup: dict[str, Any] = {}
    for variable in PICHIA_VARIABLES:
        if variable in df.columns:
            mode = df[variable].mode()
            if not mode.empty:
                lookup[variable] = mode.iloc[0]
    return lookup

def _pichia_format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)

def _pichia_type_label(run_type: str, changed_variable: str | None) -> str:
    """"类型" column text. OFAT rows name the specific variable being swept
    ("单变量-种子初始OD600") rather than a generic "单变量(OFAT)" label shared
    by all 11 OFAT rows -- otherwise the type column alone can't tell them
    apart at a glance. Baseline/combo stay generic since every row of that
    type really is the same thing."""
    if run_type == "ofat":
        label = PICHIA_VARIABLE_LABELS.get(changed_variable, changed_variable or "")
        return f"单变量-{label}"
    if run_type == "interval_interaction" and changed_variable:
        label = PICHIA_VARIABLE_LABELS.get(changed_variable, changed_variable)
        return f"补料间隔交互-{label}"
    return PICHIA_RUN_TYPE_LABELS.get(run_type, run_type)

def _pichia_row_note(
    run_type: str,
    changed_variable: str | None,
    row: pd.Series | None = None,
    baseline_lookup: dict[str, Any] | None = None,
) -> str:
    if run_type == "baseline":
        return "当前摇瓶标准条件重复，用于估计批次噪声（纯误差）"
    if run_type == "ofat":
        label = PICHIA_VARIABLE_LABELS.get(changed_variable, changed_variable or "")
        if row is not None and baseline_lookup and changed_variable in baseline_lookup:
            old = _pichia_format_value(baseline_lookup[changed_variable])
            new = _pichia_format_value(row.get(changed_variable))
            return f"单变量法：「{label}」由基线 {old} 改为 {new}，其余变量维持基线不变"
        return f"单变量法：只改变「{label}」，其余变量维持基线不变"
    if run_type == "combo":
        return "联合探索：连续变量按拉丁超立方(LHS)联合取值，温度/补料间隔取固定档位组合"
    if run_type == "ccd":
        reused = row.get("reused_from_round1") if row is not None else None
        if reused:
            return f"响应面(CCD)：条件和 round 1 的 {reused} 完全重合，产量/OD600 已按该样本回填，不需要重新做"
        return "响应面(CCD)：round 1 显著变量的角点/轴点/中心点，用于拟合响应面方程"
    if run_type == "interval_interaction":
        label = PICHIA_VARIABLE_LABELS.get(changed_variable, changed_variable or "")
        return f"补料间隔交互：新补料间隔 x 「{label}」，检验间隔的效应是否随「{label}」变化"
    if run_type == "noise_reference":
        return "新区域噪声参考：新补料间隔下的重复测量，用于估计该区域此前没有的噪声水平"
    if run_type == "lhs":
        return "拉丁超立方(LHS)：随机空间填充点，喂给贝叶斯优化的GP模型，不用于响应面拟合"
    return ""

def _pichia_design_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Chinese-labeled, note-annotated preview of the design -- used for the
    plain in-app st.dataframe. No row coloring here: st.dataframe's Styler
    support renders inconsistently for this table in practice, so the actual
    "pretty, colored" output is the Excel workbook below instead of trying to
    force parity inside Streamlit's canvas-rendered grid."""
    run_type = df["run_type"] if "run_type" in df.columns else pd.Series("", index=df.index)
    changed_variable = df["changed_variable"] if "changed_variable" in df.columns else pd.Series(None, index=df.index)
    baseline_lookup = _pichia_baseline_lookup(df)

    display = pd.DataFrame(index=df.index)
    display["编号"] = df["run_id"]
    display["类型"] = [
        _pichia_type_label(run_type.loc[idx], changed_variable.loc[idx]) for idx in df.index
    ]
    for variable in PICHIA_VARIABLES:
        if variable in df.columns:
            display[PICHIA_VARIABLE_LABELS.get(variable, variable)] = df[variable]
    display["收获时OD600"] = df.get(PICHIA_OD_COL)
    display["hLF产量"] = df.get(PICHIA_TARGET_COL)
    # only present after a multi-replicate upload (_average_duplicate_run_ids) --
    # kept visible here, not just in the one-time upload toast, since this table
    # is also what "保存到 final" persists and what gets looked at again later.
    if f"{PICHIA_TARGET_COL}_n" in df.columns:
        display["重复次数"] = df.get(f"{PICHIA_TARGET_COL}_n")
        display["产量重复间差值"] = df.get(f"{PICHIA_TARGET_COL}_spread")
    if f"{PICHIA_OD_COL}_spread" in df.columns:
        display["OD600重复间差值"] = df.get(f"{PICHIA_OD_COL}_spread")
    display["备注/目的"] = [
        _pichia_row_note(run_type.loc[idx], changed_variable.loc[idx], df.loc[idx], baseline_lookup)
        for idx in df.index
    ]
    return display
