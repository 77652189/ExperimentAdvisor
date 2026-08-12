"""Round 1 tab: the design builder, its two result charts, and the tab body.

Split out of App/pages_pichia.py (see docs/adr/0017).
"""
from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_advisor.recommendation.round1_design import (
    count_round1_rows,
    generate_round1_design,
)
from App.pichia_common import (
    PICHIA_CONTINUOUS_BOUNDS,
    PICHIA_DEFAULT_DATASET_PATH,
    PICHIA_DIVERGING_COLORSCALE,
    PICHIA_FIXED_LEVELS,
    PICHIA_OD_COL,
    PICHIA_OFAT_LEVELS,
    PICHIA_ROUND1_BASELINE,
    PICHIA_RUN_TYPE_COLORS,
    PICHIA_RUN_TYPE_LABELS,
    PICHIA_TARGET_COL,
    PICHIA_UPLOAD_DIR,
    PICHIA_VARIABLE_LABELS,
    PICHIA_VARIABLES,
    _ensure_pichia_data_area,
    _pichia_design_display_frame,
    _pichia_numeric_results,
    _pichia_restore_persisted_dataset,
)
from App.pichia_results_io import (
    _pichia_remap_uploaded_columns,
    _pichia_round1_workbook_bytes,
)

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
            st.success(f"已生成 {counts['total']} 行 Round 1 设计。")
    with reset_col:
        if st.button("直接使用已验证方案（18 样本，不改上面的表单）", width="stretch", key="round1_use_preset"):
            st.session_state["round1_results_df"] = generate_round1_design()
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

def _pichia_round1_tab() -> None:
    _ensure_pichia_data_area()
    if _pichia_restore_persisted_dataset("round1_results_df", PICHIA_DEFAULT_DATASET_PATH):
        st.toast(f"已从 {PICHIA_DEFAULT_DATASET_PATH.name} 恢复上次保存的 Round 1 结果。", icon="✅")
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
            uploaded_df, replicate_spread, consistency_warnings = _pichia_remap_uploaded_columns(uploaded_df)
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
                for warning_text in consistency_warnings:
                    st.warning(warning_text)
                if not replicate_spread.empty:
                    st.markdown("##### 同一编号的重复测量")
                    st.caption(
                        "检测到部分编号有 2 次及以上重复测量，后续显著性/响应面分析用的是均值；"
                        "原始的每次重复数值和差值没有被丢弃，一直跟着这一行数据（存到 final 时也在），"
                        "下表按 hLF产量的重复间差值从大到小排列，供你判断重复间是否足够一致。"
                    )
                    sort_column = next(
                        (column for column in replicate_spread.columns if column.endswith("重复间差值")), None
                    )
                    if sort_column:
                        replicate_spread = replicate_spread.sort_values(sort_column, ascending=False)
                    st.dataframe(replicate_spread, width="stretch", hide_index=True)

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
