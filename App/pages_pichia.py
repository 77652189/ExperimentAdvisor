"""Pichia hLF shake-flask page: Round 1 / Round 2 / History tabs.

Split out of App/app.py.
"""
from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_advisor.recommendation.round1_design import (
    BASELINE as PICHIA_ROUND1_BASELINE,
    OFAT_LEVELS as PICHIA_OFAT_LEVELS,
    count_round1_rows,
    generate_round1_design,
    round1_template_columns,
)
from experiment_advisor.recommendation.round2_design import (
    ALL_VARIABLES as PICHIA_VARIABLES,
    CONTINUOUS_BOUNDS as PICHIA_CONTINUOUS_BOUNDS,
    FIXED_LEVELS as PICHIA_FIXED_LEVELS,
    OD_COL as PICHIA_OD_COL,
    TARGET_COL as PICHIA_TARGET_COL,
    FactorEffect,
    plan_round2,
    recommend_round2_bo_batch,
)
from App.ui_shared import _num, _clear_ui_cache, _remember_ui_cache

PICHIA_DATA_DIR = PROJECT_ROOT / "data" / "pichia"
PICHIA_UPLOAD_DIR = PICHIA_DATA_DIR / "uploads"
PICHIA_FINAL_DIR = PICHIA_DATA_DIR / "final"
PICHIA_TEMPLATE_DIR = PICHIA_DATA_DIR / "templates"
PICHIA_DEFAULT_DATASET_PATH = PICHIA_FINAL_DIR / "pichia_run_level_dataset.csv"
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
PICHIA_RUN_TYPE_COLORS = {"baseline": "#3987e5", "ofat": "#d95926", "combo": "#199e70"}
PICHIA_RUN_TYPE_LABELS = {"baseline": "基线重复", "ofat": "单变量(OFAT)", "combo": "联合探索"}
PICHIA_DIVERGING_COLORSCALE = [[0, "#3987e5"], [0.5, "#383835"], [1, "#e66767"]]
PICHIA_ACCENT_COLOR = "#3987e5"
PICHIA_MUTED_COLOR = "#6b6a64"

def _ensure_pichia_data_area() -> None:
    for directory in [PICHIA_UPLOAD_DIR, PICHIA_FINAL_DIR, PICHIA_TEMPLATE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    if not PICHIA_TEMPLATE_PATH.exists():
        pd.DataFrame(columns=round1_template_columns()).to_csv(PICHIA_TEMPLATE_PATH, index=False, encoding="utf-8-sig")

def _pichia_numeric_results(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.copy()
    for column in (PICHIA_TARGET_COL, PICHIA_OD_COL):
        if column in numeric.columns:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric

def _pichia_ui_records() -> list[dict[str, Any]]:
    records = st.session_state.setdefault("pichia_ui_design_records", [])
    if not isinstance(records, list):
        records = []
        st.session_state["pichia_ui_design_records"] = records
    return records

def _pichia_variable_display(row: dict[str, Any]) -> dict[str, Any]:
    return {PICHIA_VARIABLE_LABELS.get(name, name): _num(row.get(name)) for name in PICHIA_VARIABLES if name in row}

def _pichia_round1_builder() -> None:
    st.markdown("#### 方案配置")
    st.caption(
        "基线重复、单变量法(OFAT)、联合探索(LHS) 三个模块可以独立开关和配置——某个数量填 0，"
        "或不选任何变量，就相当于关掉那个模块，可以拼出纯 LHS、纯 OFAT 或任意组合。"
        "表单打开时的默认值就是已经和研发组确认过的方案（基线×3 + OFAT×11 + 联合探索×4，共 18 个样本）。"
    )

    st.markdown("##### 基线取值")
    baseline_config: dict[str, float] = {}
    continuous_vars = list(PICHIA_CONTINUOUS_BOUNDS)
    baseline_cols = st.columns(len(continuous_vars))
    for index, variable in enumerate(continuous_vars):
        lower, upper = PICHIA_CONTINUOUS_BOUNDS[variable]
        with baseline_cols[index]:
            baseline_config[variable] = float(
                st.number_input(
                    PICHIA_VARIABLE_LABELS.get(variable, variable),
                    min_value=float(lower),
                    max_value=float(upper),
                    value=float(PICHIA_ROUND1_BASELINE[variable]),
                    key=f"round1_builder_baseline_{variable}",
                )
            )
    fixed_vars = list(PICHIA_FIXED_LEVELS)
    fixed_cols = st.columns(len(fixed_vars))
    for index, variable in enumerate(fixed_vars):
        options = PICHIA_FIXED_LEVELS[variable]
        default_value = PICHIA_ROUND1_BASELINE[variable]
        with fixed_cols[index]:
            baseline_config[variable] = float(
                st.selectbox(
                    PICHIA_VARIABLE_LABELS.get(variable, variable),
                    options,
                    index=options.index(default_value) if default_value in options else 0,
                    key=f"round1_builder_baseline_{variable}",
                )
            )

    st.markdown("##### 基线重复")
    n_baseline = st.number_input(
        "基线重复次数（0 = 不生成基线重复行）",
        min_value=0,
        max_value=10,
        value=3,
        key="round1_builder_n_baseline",
    )

    st.markdown("##### 单变量法 (OFAT)")
    st.caption(
        "连续变量的测试水平不是固定选项——推荐值只是预填，勾掉即可移除，下面还有专门的输入框+按钮可以增加新水平；"
        "温度和补料间隔受设备限制，只能在 20/25/30℃、12/24h 这几个物理档位里选，不能新增。"
    )
    ofat_vars = st.multiselect(
        "参与 OFAT 的变量（不选 = 不生成 OFAT 行）",
        list(PICHIA_VARIABLES),
        default=list(PICHIA_VARIABLES),
        format_func=lambda v: PICHIA_VARIABLE_LABELS.get(v, v),
        key="round1_builder_ofat_vars",
    )
    ofat_levels_config: dict[str, list[float]] = {}
    ofat_errors: list[str] = []
    for variable in ofat_vars:
        default_levels = PICHIA_OFAT_LEVELS.get(variable, [])
        label = PICHIA_VARIABLE_LABELS.get(variable, variable)
        is_fixed_level = variable in PICHIA_FIXED_LEVELS
        if is_fixed_level:
            selected = st.multiselect(
                f"{label} 的测试水平",
                options=PICHIA_FIXED_LEVELS[variable],
                default=default_levels,
                help="设备只支持这几个固定档位，不能输入自定义数值。",
                key=f"round1_builder_ofat_levels_{variable}",
            )
        else:
            # explicit pool + add-button instead of relying on multiselect's
            # accept_new_options=True: that only reveals its "type to add" UI
            # once the box is clicked/focused, so it looked identical to a
            # plain fixed-choice dropdown in a screenshot -- not discoverable.
            # The add-row must run BEFORE the multiselect below: Streamlit
            # forbids writing to a widget's session_state key after that
            # widget has already been instantiated in the same script run, so
            # mutating pool/levels has to happen ahead of the multiselect call
            # that owns `levels_key`, not in a button handler placed after it.
            pool_key = f"round1_builder_ofat_pool_{variable}"
            levels_key = f"round1_builder_ofat_levels_{variable}"
            st.session_state.setdefault(pool_key, list(default_levels))
            st.session_state.setdefault(levels_key, list(default_levels))

            lower, upper = PICHIA_CONTINUOUS_BOUNDS[variable]
            st.caption(f"新增「{label}」测试水平")
            add_cols = st.columns([3, 1])
            with add_cols[0]:
                new_level = st.number_input(
                    f"新增{label}水平",
                    min_value=float(lower),
                    max_value=float(upper),
                    value=None,
                    step=0.1,
                    placeholder="输入数值",
                    label_visibility="collapsed",
                    key=f"round1_builder_ofat_newval_{variable}",
                )
            with add_cols[1]:
                add_clicked = st.button(
                    "+ 添加", width="stretch", key=f"round1_builder_ofat_addbtn_{variable}"
                )
            if add_clicked:
                if new_level is None:
                    st.warning("请先在左侧输入一个数值。")
                else:
                    pool = st.session_state[pool_key]
                    if new_level not in pool:
                        st.session_state[pool_key] = sorted(pool + [new_level])
                    levels = st.session_state[levels_key]
                    if new_level not in levels:
                        st.session_state[levels_key] = sorted(levels + [new_level])

            selected = st.multiselect(
                f"{label} 的测试水平（勾掉即可移除，上方可新增）",
                options=st.session_state[pool_key],
                key=levels_key,
            )
        try:
            ofat_levels_config[variable] = sorted({float(value) for value in selected})
        except (TypeError, ValueError):
            ofat_errors.append(label)
    if ofat_errors:
        st.error(f"以下变量输入了无法识别成数字的自定义水平，请检查：{', '.join(ofat_errors)}")

    st.markdown("##### 联合探索 (LHS)")
    n_combo = st.number_input(
        "联合探索点数（0 = 不生成联合探索行）",
        min_value=0,
        max_value=50,
        value=4,
        key="round1_builder_n_combo",
    )
    combo_vars = st.multiselect(
        "参与联合探索的连续变量",
        continuous_vars,
        default=continuous_vars,
        format_func=lambda v: PICHIA_VARIABLE_LABELS.get(v, v),
        key="round1_builder_combo_vars",
    )

    counts = count_round1_rows(ofat_levels_config, int(n_baseline), int(n_combo), baseline=baseline_config)
    st.caption(
        f"预计生成：基线 {counts['baseline']} + OFAT {counts['ofat']} + 联合探索 {counts['combo']} "
        f"= 共 {counts['total']} 行"
    )

    generate_col, reset_col = st.columns(2)
    with generate_col:
        if st.button(
            "生成 Round 1 设计",
            type="primary",
            width="stretch",
            disabled=bool(ofat_errors),
            key="generate_round1_design_button",
        ):
            st.session_state["round1_results_df"] = generate_round1_design(
                baseline=baseline_config,
                ofat_levels=ofat_levels_config or None,
                n_baseline_replicates=int(n_baseline),
                n_combo_points=int(n_combo),
                combo_variables=combo_vars or None,
            )
            st.session_state.pop("round2_bo_result", None)
            st.success(f"已生成 {counts['total']} 行 Round 1 设计。")
    with reset_col:
        if st.button("直接使用已验证方案（18 样本，不改上面的表单）", width="stretch", key="round1_use_preset"):
            st.session_state["round1_results_df"] = generate_round1_design()
            st.session_state.pop("round2_bo_result", None)
            st.success("已生成已验证的 18 样本方案。")

def _pichia_yield_scatter_chart(df: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    """Box+jitter chart of `value_col` grouped by run_type: shows both the
    per-group distribution/spread and every individual point (run_id on hover),
    covering the "distribution" and "error/noise" asks in one figure."""

    numeric = df.copy()
    numeric[value_col] = pd.to_numeric(numeric[value_col], errors="coerce")
    plot_df = numeric.dropna(subset=[value_col])

    fig = go.Figure()
    for run_type in ["baseline", "ofat", "combo"]:
        subset = plot_df[plot_df["run_type"] == run_type]
        if subset.empty:
            continue
        fig.add_trace(
            go.Box(
                y=subset[value_col],
                x=[PICHIA_RUN_TYPE_LABELS[run_type]] * len(subset),
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                marker_color=PICHIA_RUN_TYPE_COLORS[run_type],
                line_color=PICHIA_RUN_TYPE_COLORS[run_type],
                text=subset["run_id"],
                hovertemplate="%{text}<br>" + value_col + "=%{y}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis_title=value_col,
        margin=dict(t=40, b=30, l=50, r=20),
        height=380,
    )
    return fig

def _pichia_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    columns = [*PICHIA_VARIABLES, PICHIA_TARGET_COL, PICHIA_OD_COL]
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman", min_periods=3)
    labels = [PICHIA_VARIABLE_LABELS.get(column, column) for column in columns]
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.to_numpy(),
            x=labels,
            y=labels,
            colorscale=PICHIA_DIVERGING_COLORSCALE,
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="Spearman r"),
        )
    )
    fig.update_layout(
        title="变量与产量/OD600 相关性（Spearman，仅 18 个样本，供参考，非结论性）",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=40, l=110, r=20),
        height=450,
    )
    return fig

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
    display["备注/目的"] = [
        _pichia_row_note(run_type.loc[idx], changed_variable.loc[idx], df.loc[idx], baseline_lookup)
        for idx in df.index
    ]
    return display

def _pichia_round1_workbook_bytes(df: pd.DataFrame) -> bytes:
    """Polished .xlsx twin of the design: colored rows by run_type, Chinese
    headers, an auto-generated notes column, a legend, and a frozen header --
    the same look already approved from an earlier one-off script, now built
    dynamically from whatever design was actually generated (any mix of
    baseline/OFAT/LHS) instead of a fixed example."""
    run_type = df["run_type"] if "run_type" in df.columns else pd.Series("", index=df.index)
    changed_variable = df["changed_variable"] if "changed_variable" in df.columns else pd.Series(None, index=df.index)
    baseline_lookup = _pichia_baseline_lookup(df)

    font_name = "Calibri"
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(name=font_name, size=11, bold=True, color="FFFFFF")
    baseline_fill = PatternFill("solid", fgColor="E2EFDA")
    combo_fill = PatternFill("solid", fgColor="DDEBF7")
    fillin_fill = PatternFill("solid", fgColor="FFF2CC")
    thin = Side(style="thin", color="B7B7B7")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    center = Alignment(horizontal="center", vertical="center")
    wrap_center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    wrap_left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    # only "type" and "note" hold variable-length sentences -- everything else
    # is a short id/number, so it stays single-line and the table stays compact
    wrap_keys = {"run_type", "note"}

    columns: list[tuple[str, str, float]] = [("run_id", "编号", 9), ("run_type", "类型", 18)]
    for variable in PICHIA_VARIABLES:
        if variable in df.columns:
            columns.append((variable, PICHIA_VARIABLE_LABELS.get(variable, variable), 13))
    columns.append((PICHIA_OD_COL, "收获时OD600(待填)", 12))
    columns.append((PICHIA_TARGET_COL, "hLF产量(待填)", 12))
    columns.append(("note", "备注/目的", 34))

    wb = Workbook()
    ws = wb.active
    ws.title = "Round1设计"
    ws.sheet_view.showGridLines = False

    for col_idx, (_key, label, width) in enumerate(columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=label)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = border
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    # deliberately no explicit row_dimensions[1].height: leaving customHeight
    # unset lets Excel auto-fit each row to its own wrapped content on open,
    # which is what actually works when note length varies this much between
    # row types -- any single fixed height is either too tall for short
    # baseline notes or clipped for the longer combo/ofat notes.
    ws.freeze_panes = "A2"

    row_fill_by_type = {"baseline": baseline_fill, "combo": combo_fill}

    for offset, idx in enumerate(df.index):
        row_idx = offset + 2
        rt = run_type.loc[idx]
        row = df.loc[idx]
        note = _pichia_row_note(rt, changed_variable.loc[idx], row, baseline_lookup)
        values: dict[str, Any] = {
            "run_id": row.get("run_id"),
            "run_type": _pichia_type_label(rt, changed_variable.loc[idx]),
            PICHIA_OD_COL: row.get(PICHIA_OD_COL),
            PICHIA_TARGET_COL: row.get(PICHIA_TARGET_COL),
            "note": note,
        }
        for variable in PICHIA_VARIABLES:
            if variable in df.columns:
                values[variable] = row.get(variable)

        row_fill = row_fill_by_type.get(rt)
        for col_idx, (key, _label, _width) in enumerate(columns, start=1):
            value = values.get(key)
            if isinstance(value, float) and pd.isna(value):
                value = None
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.font = Font(name=font_name, size=10)
            cell.border = border
            if key == "note":
                cell.alignment = wrap_left
            elif key in wrap_keys:
                cell.alignment = wrap_center
            else:
                cell.alignment = center
            if key in (PICHIA_OD_COL, PICHIA_TARGET_COL):
                cell.fill = fillin_fill
            elif row_fill is not None:
                cell.fill = row_fill

    # legend lives on its own sheet, not appended below the data table --
    # otherwise pd.read_excel() on reimport reads it as extra data rows
    # (confirmed: it silently inflated an 18-row design to 23 rows).
    legend_ws = wb.create_sheet("图例说明")
    legend_ws.sheet_view.showGridLines = False
    legend_ws.column_dimensions["A"].width = 4
    legend_ws.column_dimensions["B"].width = 40
    legend_items = [
        (baseline_fill, "基线重复（估计批次噪声）"),
        (combo_fill, "联合探索点（LHS，多变量同时变化）"),
        (fillin_fill, "需要填写的结果列"),
    ]
    for row_idx, (fill, text) in enumerate(legend_items, start=1):
        marker = legend_ws.cell(row=row_idx, column=1, value=" ")
        marker.fill = fill
        marker.border = border
        label_cell = legend_ws.cell(row=row_idx, column=2, value=text)
        label_cell.font = Font(name=font_name, size=10)

    buffer = BytesIO()
    wb.save(buffer)
    return buffer.getvalue()

def _pichia_remap_uploaded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recognizes the polished Excel export's Chinese headers, in addition to
    the plain English round1_template_columns() schema, so a design that was
    downloaded, filled in by hand in Excel, and re-uploaded round-trips
    correctly. Also reconstructs run_type/changed_variable from the combined
    "类型" column (e.g. "单变量-发酵温度 (℃)") -- round2's significance
    analysis and the result charts both key off those two columns, so losing
    them on reimport would silently break Round 2, not just cosmetics."""
    label_to_variable = {label: variable for variable, label in PICHIA_VARIABLE_LABELS.items()}
    rename_map = {
        "编号": "run_id",
        "收获时OD600(待填)": PICHIA_OD_COL,
        "hLF产量(待填)": PICHIA_TARGET_COL,
    }
    rename_map.update(label_to_variable)
    renamed = df.rename(columns=rename_map)

    if "类型" in renamed.columns and "run_type" not in renamed.columns:
        run_types: list[str | None] = []
        changed_vars: list[str | None] = []
        for raw in renamed["类型"]:
            text = str(raw).strip()
            if text == PICHIA_RUN_TYPE_LABELS.get("baseline"):
                run_types.append("baseline")
                changed_vars.append(None)
            elif text == PICHIA_RUN_TYPE_LABELS.get("combo"):
                run_types.append("combo")
                changed_vars.append(None)
            elif text.startswith("单变量-"):
                run_types.append("ofat")
                changed_vars.append(label_to_variable.get(text[len("单变量-"):]))
            else:
                run_types.append(None)
                changed_vars.append(None)
        renamed["run_type"] = run_types
        renamed["changed_variable"] = changed_vars
        renamed = renamed.drop(columns=["类型"])

    renamed = renamed.drop(columns=["备注/目的"], errors="ignore")

    # defensive: drop any row missing a variable value -- every real design
    # row has all 6 populated, so this filters out a legend/blank row picked
    # up from an older export that still had the legend below the table (or
    # any stray manual annotation row) without needing to special-case it.
    variable_columns = [variable for variable in PICHIA_VARIABLES if variable in renamed.columns]
    if variable_columns:
        renamed = renamed[renamed[variable_columns].notna().all(axis=1)].reset_index(drop=True)

    return renamed

def _pichia_round1_tab() -> None:
    _ensure_pichia_data_area()
    st.markdown("### Round 1：实验设计构建器")
    st.caption(
        "温度只能是 20/25/30℃，补料间隔只能是 12/24h——这两个变量的档位一旦被单变量法测全，"
        "就已经穷尽了离散选项，Round 2 不会再对它们做响应面细化。"
    )

    with st.expander("方案配置", expanded="round1_results_df" not in st.session_state):
        _pichia_round1_builder()

    if "round1_results_df" not in st.session_state:
        st.info("配置好方案后点击「生成 Round 1 设计」，或直接用「直接使用已验证方案」一键生成。")
        return

    st.markdown("#### 上传已回填结果（可选）")
    uploaded = st.file_uploader(
        "上传已回填 yield_g_per_l / od600 的 CSV，或者上面下载的 Excel 填完后原样传回来",
        type=["csv", "xlsx"],
        key="round1_results_upload",
    )
    if uploaded is not None:
        try:
            payload = uploaded.getvalue()
            if uploaded.name.lower().endswith(".xlsx"):
                uploaded_df = pd.read_excel(BytesIO(payload))
            else:
                uploaded_df = pd.read_csv(BytesIO(payload))
            uploaded_df = _pichia_remap_uploaded_columns(uploaded_df)
        except Exception as exc:
            st.error(f"读取失败：{exc}")
        else:
            missing = [column for column in ["run_id", *PICHIA_VARIABLES, PICHIA_TARGET_COL, PICHIA_OD_COL] if column not in uploaded_df.columns]
            if missing:
                st.error(f"上传文件缺少字段：{', '.join(missing)}")
            else:
                _ensure_pichia_data_area()
                archive_name = Path(uploaded.name).name or "pichia_round1_uploaded.csv"
                (PICHIA_UPLOAD_DIR / archive_name).write_bytes(payload)
                st.session_state["round1_results_df"] = uploaded_df
                st.success("已加载并归档上传的 Round 1 结果。")

    working_df = st.session_state["round1_results_df"]

    st.markdown("#### 设计总览")
    excel_bytes = _pichia_round1_workbook_bytes(working_df)
    st.download_button(
        "下载设计表格（Excel，配色+备注，和示例一致）",
        data=excel_bytes,
        file_name="pichia_round1_design.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_round1_excel",
    )
    st.caption("配色、图例和备注列在 Excel 里最完整；下面是网页内的快速预览（不带颜色）。")
    st.dataframe(_pichia_design_display_frame(working_df), width="stretch", hide_index=True)

    st.markdown("#### 回填实测结果")
    st.caption("按编号对应到上面总览表里的样本，这里只需要填产量和 OD600 这两列。")
    entry_view = working_df[["run_id", PICHIA_TARGET_COL, PICHIA_OD_COL]].reset_index(drop=True)
    edited_entries = st.data_editor(
        entry_view,
        width="stretch",
        num_rows="fixed",
        hide_index=True,
        key="round1_data_editor",
        disabled=["run_id"],
        column_config={
            "run_id": "编号",
            PICHIA_TARGET_COL: "hLF产量",
            PICHIA_OD_COL: "收获时OD600",
        },
    )
    edited = working_df.copy()
    edited[PICHIA_TARGET_COL] = edited_entries[PICHIA_TARGET_COL].to_numpy()
    edited[PICHIA_OD_COL] = edited_entries[PICHIA_OD_COL].to_numpy()
    st.session_state["round1_results_df"] = edited

    numeric = _pichia_numeric_results(edited)
    filled = int(numeric[[PICHIA_TARGET_COL, PICHIA_OD_COL]].notna().all(axis=1).sum())
    st.caption(f"已回填 {filled}/{len(edited)} 行完整结果（yield_g_per_l 和 od600 都有值才算完成）。")

    action_cols = st.columns(2)
    with action_cols[0]:
        csv_bytes = edited.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载当前表格",
            data=csv_bytes,
            file_name="pichia_round1_results.csv",
            mime="text/csv",
            key="download_round1_results",
        )
    with action_cols[1]:
        if st.button("保存到 data/pichia/final/（供 Round 2 和下次打开使用）", key="save_round1_to_final"):
            _ensure_pichia_data_area()
            edited.to_csv(PICHIA_DEFAULT_DATASET_PATH, index=False, encoding="utf-8-sig")
            st.success(f"已保存到 {PICHIA_DEFAULT_DATASET_PATH}")

    if int(numeric[PICHIA_TARGET_COL].notna().sum()) >= 3:
        st.markdown("#### 结果可视化")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.plotly_chart(
                _pichia_yield_scatter_chart(numeric, PICHIA_TARGET_COL, "产量分布（按样本类型）"),
                width="stretch",
            )
        with chart_cols[1]:
            st.plotly_chart(
                _pichia_yield_scatter_chart(numeric, PICHIA_OD_COL, "OD600 分布（按样本类型）"),
                width="stretch",
            )
        st.plotly_chart(_pichia_correlation_heatmap(numeric), width="stretch")
    else:
        st.caption("至少回填 3 行产量数据后，这里会显示分布图和相关性热力图。")

def _pichia_effect_magnitude_chart(effects: dict[str, FactorEffect], threshold: float) -> go.Figure:
    """Horizontal bar of each variable's effect_magnitude, descending, with a
    confidence-interval error bar and a reference line at the significance
    threshold -- this is what actually decides K (how many variables become
    "active" in resolve_round2_variables). Emphasis coloring (accent vs muted)
    rather than a full categorical palette, since the only distinction that
    matters here is "crossed the line" vs "didn't"."""

    ordered = sorted(effects.values(), key=lambda effect: effect.effect_magnitude, reverse=True)
    labels = [PICHIA_VARIABLE_LABELS.get(effect.variable, effect.variable) for effect in ordered]
    magnitudes = [effect.effect_magnitude for effect in ordered]
    colors = [PICHIA_ACCENT_COLOR if effect.significant else PICHIA_MUTED_COLOR for effect in ordered]
    error_plus = [max(effect.ci_high - effect.effect_magnitude, 0.0) for effect in ordered]
    error_minus = [max(effect.effect_magnitude - effect.ci_low, 0.0) for effect in ordered]

    fig = go.Figure(
        go.Bar(
            x=magnitudes,
            y=labels,
            orientation="h",
            marker_color=colors,
            error_x=dict(type="data", symmetric=False, array=error_plus, arrayminus=error_minus, color="#c3c2b7"),
            hovertemplate="%{y}: %{x:.3g}<extra></extra>",
        )
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#c3c2b7",
        annotation_text=f"显著性阈值 {threshold:.3g}",
        annotation_position="top",
    )
    fig.update_layout(
        title="各变量效应量（误差棒=置信区间，仅 3 次基线重复，df=2，区间偏宽属实情）",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="效应量 |偏离基线|",
        margin=dict(t=50, b=40, l=140, r=30),
        height=340,
    )
    return fig

def _pichia_round2_tab() -> None:
    st.markdown("### Round 2：显著性分析 → 响应面（CCD）+ 约束贝叶斯优化")

    results_df = st.session_state.get("round1_results_df")
    if results_df is None or results_df.empty:
        st.info("请先在「Round 1：实验设计」页签生成设计并回填实测结果。")
        return

    run_df = _pichia_numeric_results(results_df)
    filled = int(run_df[[PICHIA_TARGET_COL, PICHIA_OD_COL]].notna().all(axis=1).sum())
    st.caption(f"当前已回填 {filled}/{len(run_df)} 行完整结果。")
    if filled < len(run_df):
        st.warning("还有行没有回填完整结果；下面的分析只会用到已经有产量和 OD600 的行，结论可能随着数据补全而变化。")

    with st.expander("分析参数", expanded=False):
        param_cols = st.columns(3)
        with param_cols[0]:
            max_active_variables = st.number_input(
                "最多活跃变量数 (K 上限)",
                min_value=1,
                max_value=6,
                value=3,
                help="超过这个数量的显著变量会按效应量排序，排不进的先固定在 round 1 最优水平。",
                key="round2_max_active_variables",
            )
        with param_cols[1]:
            ccd_step_fraction = st.slider(
                "CCD 步长比例",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="响应面设计的轴向步长 = 该比例 x 单变量法(OFAT)的水平间距。",
                key="round2_ccd_step_fraction",
            )
        with param_cols[2]:
            od_threshold_fraction = st.slider(
                "OD600 约束比例",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="OD600 约束阈值 = 该比例 x 基线 OD600 均值，工程默认值，非生物学判断。",
                key="round2_od_threshold_fraction",
            )

    try:
        plan = plan_round2(
            run_df,
            PICHIA_ROUND1_BASELINE,
            od_threshold_fraction=float(od_threshold_fraction),
            ccd_step_fraction=float(ccd_step_fraction),
            max_active_variables=int(max_active_variables),
        )
    except ValueError as exc:
        st.error(f"Round 2 分析暂时无法运行：{exc}")
        return

    st.markdown("#### 噪声与显著性阈值")
    noise = plan.noise
    cols = st.columns(3)
    cols[0].metric("基线重复数", noise["n_replicates"])
    cols[1].metric("基线均值", _num(noise["baseline_mean"]))
    cols[2].metric("显著性阈值", _num(noise["threshold"]))

    st.markdown("#### 固定变量（不显著，或离散档位已测完）")
    fixed_rows = [
        {"变量": PICHIA_VARIABLE_LABELS.get(name, name), "固定取值": _num(value)}
        for name, value in plan.fixed_values.items()
    ]
    st.dataframe(pd.DataFrame(fixed_rows), width="stretch", hide_index=True)

    st.markdown("#### 活跃变量（进入响应面设计）")
    st.metric("活跃变量数 (K)", len(plan.active_variables))
    st.plotly_chart(
        _pichia_effect_magnitude_chart(plan.effects, noise["threshold"]),
        width="stretch",
    )
    if not plan.active_variables:
        st.info("当前数据下没有变量的效应大到需要响应面细化；可以直接看下面的贝叶斯优化建议，或考虑扩大探索范围重新走一轮。")
    for variable, note in plan.boundary_notes.items():
        st.warning(f"{PICHIA_VARIABLE_LABELS.get(variable, variable)}：{note}")
    for variable, note in plan.overflow_notes.items():
        st.info(f"{PICHIA_VARIABLE_LABELS.get(variable, variable)}：{note}")

    if plan.combo_interactions:
        st.markdown("#### 联合探索点的可加性检查")
        st.caption("检查每个联合探索样本的实测值和「按单变量效应线性叠加」的预测值之间的差距；差距超过阈值提示可能存在变量间交互。")
        interaction_rows = [
            {
                "样本": item["row_id"],
                "改变的变量": "、".join(PICHIA_VARIABLE_LABELS.get(name, name) for name in item["changed_variables"]),
                "实测": _num(item["observed"]),
                "可加性预测": _num(item["additive_prediction"]),
                "残差": _num(item["residual"]),
                "可能存在交互": "是" if item["possible_interaction"] else "否",
            }
            for item in plan.combo_interactions
        ]
        st.dataframe(pd.DataFrame(interaction_rows), width="stretch", hide_index=True)

    if plan.active_variables:
        st.markdown("#### Round 2 响应面设计（Face-centered CCD）")
        ccd_rows = []
        for row in plan.design_rows:
            display = _pichia_variable_display(row)
            display["复用 Round1 样本"] = row.get("reused_from_round1") or ""
            ccd_rows.append(display)
        st.dataframe(pd.DataFrame(ccd_rows), width="stretch", hide_index=True)
        reused_count = sum(1 for row in plan.design_rows if row.get("reused_from_round1"))
        st.caption(f"共 {len(plan.design_rows)} 个设计点，其中 {reused_count} 个和 Round 1 已有数据重合，可以直接复用、不用重新做。")

    st.markdown("#### 约束贝叶斯优化建议批次")
    st.caption(
        f"OD600 约束阈值：{_num(plan.od_threshold['threshold'])}"
        f"（= 基线 OD600 均值 x {plan.od_threshold['fraction']}，工程默认值，请研发组结合实际需求确认）。"
    )
    n_batch = st.slider("建议批次大小", min_value=3, max_value=15, value=9, key="round2_bo_batch_size")
    if st.button("生成 Round 2 贝叶斯优化建议", type="primary", key="run_round2_bo"):
        try:
            bo_result = recommend_round2_bo_batch(
                run_df,
                fixed_values=plan.fixed_values,
                active_variables=plan.active_variables or list(PICHIA_CONTINUOUS_BOUNDS),
                od_threshold=plan.od_threshold["threshold"],
                n_batch=int(n_batch),
            )
        except ImportError:
            st.warning("当前环境未安装 torch/botorch/gpytorch，无法运行贝叶斯优化建议；以上响应面(CCD)部分不受影响。")
        except ValueError as exc:
            st.warning(f"未生成建议：{exc}")
        else:
            st.session_state["round2_bo_result"] = bo_result
            st.success(
                f"已生成 {len(bo_result['recommendations'])} 个建议"
                f"（候选池 {bo_result['n_candidates']} 个，满足约束 {bo_result['n_feasible']} 个）。"
            )

    bo_result = st.session_state.get("round2_bo_result")
    if bo_result:
        bo_rows = []
        for rec in bo_result["recommendations"]:
            row = _pichia_variable_display(rec)
            row["预测产量"] = _num(rec.get("predicted_yield"))
            row["预测产量标准差"] = _num(rec.get("predicted_yield_std"))
            row["预测 OD600"] = _num(rec.get("predicted_od600"))
            bo_rows.append(row)
        st.dataframe(pd.DataFrame(bo_rows), width="stretch", hide_index=True)

        if st.button("保存本次 Round 2 分析到历史记录", key="save_round2_snapshot"):
            created_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            record_id = f"r2_{hashlib.sha256(created_at.encode('utf-8')).hexdigest()[:10]}"
            _pichia_ui_records().append(
                {
                    "record_id": record_id,
                    "record_name": f"Round2 {created_at}",
                    "created_at": created_at,
                    "fixed_values": plan.fixed_values,
                    "active_variables": plan.active_variables,
                    "od_threshold": plan.od_threshold,
                    "bo_recommendations": bo_result["recommendations"],
                }
            )
            st.session_state["pichia_ui_record_notice"] = "已保存本次 Round 2 分析"
            _remember_ui_cache()
            st.rerun()

def _pichia_history_tab() -> None:
    st.markdown("### 历史记录（当前会话）")
    st.caption(
        "这里保存的是本次会话里生成的 Round 2 分析快照；刷新页面会恢复，重启应用后清空，不写入文件。"
        "Round 1 的实测结果请用「Round 1」页签里的下载/保存按钮处理，不依赖这里的缓存。"
    )
    notice = st.session_state.pop("pichia_ui_record_notice", None)
    if notice:
        st.success(str(notice))

    clear_cols = st.columns([1, 1, 3])
    with clear_cols[0]:
        clear_confirm = st.checkbox("确认清空界面缓存", key="confirm_clear_pichia_ui_cache")
    with clear_cols[1]:
        if st.button("清空界面缓存", disabled=not clear_confirm, key="clear_pichia_ui_cache"):
            _clear_ui_cache()
            st.rerun()

    records = _pichia_ui_records()
    if not records:
        st.info("暂无历史记录。在「Round 2」页签生成贝叶斯优化建议后可以保存快照。")
        return

    summary_rows = [
        {
            "记录名": record.get("record_name"),
            "时间": record.get("created_at"),
            "活跃变量": "、".join(PICHIA_VARIABLE_LABELS.get(name, name) for name in record.get("active_variables", [])),
            "建议数": len(record.get("bo_recommendations", [])),
        }
        for record in records
    ]
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    labels = [f"{record.get('created_at', '')} | {record.get('record_name', '')}" for record in records]
    selected_label = st.selectbox("查看记录", labels, key="pichia_record_selector")
    selected_index = labels.index(selected_label)
    record = records[selected_index]

    with st.expander("选中记录详情", expanded=False):
        fixed_rows = [
            {"变量": PICHIA_VARIABLE_LABELS.get(name, name), "固定取值": _num(value)}
            for name, value in (record.get("fixed_values") or {}).items()
        ]
        st.dataframe(pd.DataFrame(fixed_rows), width="stretch", hide_index=True)
        if record.get("bo_recommendations"):
            rec_rows = []
            for rec in record["bo_recommendations"]:
                row = _pichia_variable_display(rec)
                row["预测产量"] = _num(rec.get("predicted_yield"))
                rec_rows.append(row)
            st.dataframe(pd.DataFrame(rec_rows), width="stretch", hide_index=True)

        selected_id = str(record.get("record_id"))
        delete_confirm = st.checkbox("确认删除选中记录", key=f"confirm_delete_{selected_id}")
        if st.button("删除选中记录", disabled=not delete_confirm, key=f"delete_record_{selected_id}"):
            del records[selected_index]
            st.session_state["pichia_ui_record_notice"] = "已删除选中记录"
            _remember_ui_cache()
            st.rerun()

def _pichia_hlf_page() -> None:
    _ensure_pichia_data_area()
    st.caption(
        "当前模式：毕赤酵母 hLF 摇瓶实验设计。Round 1 是可配置的基线+单变量+联合探索设计构建器，"
        "Round 2 基于 Round 1 实测结果做显著性分析、响应面(CCD)设计和约束贝叶斯优化。"
    )

    round1_tab, round2_tab, history_tab = st.tabs(["Round 1：实验设计", "Round 2：响应面 + 贝叶斯优化", "历史记录"])
    with round1_tab:
        _pichia_round1_tab()
    with round2_tab:
        _pichia_round2_tab()
    with history_tab:
        _pichia_history_tab()
