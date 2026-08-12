"""Round 2 tab: the sections that assemble the views into a page, and the tab
body that routes between them.

This is the only Round 2 module that owns page state -- it decides which
st.session_state keys exist ("round2_full_design_df",
"round2_combined_bo_result", the widget keys) and in what order they are
written and read. The views it calls are stateless by design (see
docs/adr/0017), so the read/write ordering lives here and nowhere else.
"""
from __future__ import annotations

import itertools
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_advisor.recommendation.round2_design import (
    FactorEffect,
    Round2Plan,
    analyze_interval_interaction,
    assemble_round2_design,
    classify_response_surface_case,
    evaluate_response_surface,
    fit_ccd_response_surface,
    plan_round2,
    predict_with_confidence_interval,
    recommend_round2_bo_batch,
    summarize_response_surface,
)
from App.pichia_common import (
    PICHIA_ACCENT_COLOR,
    PICHIA_CONTINUOUS_BOUNDS,
    PICHIA_MUTED_COLOR,
    PICHIA_OD_COL,
    PICHIA_ROUND1_BASELINE,
    PICHIA_ROUND2_DATASET_PATH,
    PICHIA_RUN_TYPE_LABELS,
    PICHIA_SEQUENTIAL_COLORSCALE_DEFAULT,
    PICHIA_SEQUENTIAL_COLORSCALE_OPTIONS,
    PICHIA_TARGET_COL,
    PICHIA_UPLOAD_DIR,
    PICHIA_VARIABLE_LABELS,
    PICHIA_VARIABLES,
    _ensure_pichia_data_area,
    _num,
    _pichia_numeric_results,
    _pichia_restore_persisted_dataset,
    _pichia_variable_display,
)
from App.pichia_results_io import (
    _pichia_pooled_technical_noise,
    _pichia_remap_uploaded_columns,
    _pichia_round2_workbook_bytes,
)
from App.pichia_round2_surface_views import (
    _pichia_coefficient_column_config,
    _pichia_coefficient_table,
    _pichia_render_response_surface_deep_dive,
    _pichia_render_response_surface_verdicts,
    _pichia_response_surface_3d_chart,
    _pichia_response_surface_contour_chart,
    _pichia_response_surface_curve_chart,
    _pichia_response_surface_residual_chart,
)
from App.pichia_round2_bo_views import _pichia_render_bo_recommendation_section

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

def _pichia_round2_significance_section(plan: Round2Plan, run_df: pd.DataFrame) -> None:
    """Which round 1 variables earned a place in the Round 2 design, and on what
    evidence: the baseline noise the significance threshold is derived from, the
    technical-replicate noise that deliberately stays out of that threshold (see
    ADR-0011), the effect sizes themselves, and the additivity check on whatever
    joint-exploration points round 1 happened to include."""
    st.markdown("#### 噪声与显著性阈值")
    noise = plan.noise
    cols = st.columns(3)
    cols[0].metric("基线重复数", noise["n_replicates"])
    cols[1].metric("基线均值", _num(noise["baseline_mean"]))
    cols[2].metric("显著性阈值", _num(noise["threshold"]))

    technical_noise = _pichia_pooled_technical_noise(run_df)
    if technical_noise:
        st.caption(
            f"另一种噪声：同一样本测 2 次（技术重复）的差异。汇总全部 {technical_noise['n_runs']} 个有重复的编号后，"
            f"技术重复噪声(SD)≈{_num(technical_noise['pooled_sd'])}（自由度 {technical_noise['dof']}，"
            f"比上面基线批次噪声的自由度 {noise['n_replicates'] - 1} 高很多，估得更稳）。"
            "有意不并入上面的显著性阈值：每个 round 1 样本统一测 2 次技术重复，"
            "批次噪声本身已经是「run 间会差多少」的正确尺度；这里只作为独立的数据质量参考——"
            "谁的技术重复差异远超其他编号（下面会标出来），值得单独复核，但不改变谁进入响应面设计。"
        )
        if technical_noise["outliers"]:
            outlier_text = "、".join(f"{run_id}(SD≈{_num(sd)})" for run_id, sd in technical_noise["outliers"])
            st.warning(f"这些编号的技术重复差异明显超过其余编号的普遍水平（>2x 汇总 SD）：{outlier_text}，建议重点核实。")

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
        st.info(
            "当前数据下没有变量的效应大到需要响应面细化。两条路：一是在「② 设计生成与回填」子页照常生成设计表"
            "（没有活跃变量时它仍会出补料间隔交互点和 LHS 空间填充点），跑完回填后到「④ 合并数据贝叶斯优化」"
            "子页拿建议；二是考虑扩大探索范围重新走一轮 Round 1。"
        )
    for variable, note in plan.untested_notes.items():
        st.warning(f"⚠️ 「{PICHIA_VARIABLE_LABELS.get(variable, variable)}」本轮未测，不是「测过发现不显著」：{note}")
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

def _pichia_simulate_round2_results(design: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Fills currently-blank yield/od600 cells with deterministic, plausible
    (not random-garbage) synthetic values -- same formula and seed used to
    build the one-off demo Excel generated earlier this project (a clean
    quadratic in glucose/ph/volume centered near round 1's own best region,
    a modest fixed penalty for the untested 36h interval, small Gaussian
    noise). Never touches a cell that already has a real value: this exists
    so someone can exercise the CCD-fit/interaction/BO-visualization pipeline
    before real Round 2 results exist, not to backfill Round 2 for them."""
    result = design.copy()
    rng = np.random.default_rng(seed)
    center = {"glucose_pct": 1.0, "ph": 6.2, "volume_ml": 50.0}
    still_blank = result[PICHIA_TARGET_COL].isna()
    if not still_blank.any():
        return result

    glucose = result.loc[still_blank, "glucose_pct"]
    ph = result.loc[still_blank, "ph"]
    volume = result.loc[still_blank, "volume_ml"]
    interval = result.loc[still_blank, "interval_h"]
    yield_value = (
        0.013
        - 0.04 * (glucose - center["glucose_pct"]) ** 2
        - 0.015 * (ph - center["ph"]) ** 2
        + 0.00015 * (volume - center["volume_ml"])
        - 0.0000005 * (volume - center["volume_ml"]) ** 2
    )
    yield_value = yield_value.where(interval != 36.0, yield_value * 0.85)
    yield_value = (yield_value + rng.normal(0.0, 0.0003, int(still_blank.sum()))).clip(lower=0.00005)
    od_base = pd.Series(np.where(interval.to_numpy() == 24.0, 32.0, 26.0), index=interval.index)
    od_value = (od_base + rng.normal(0.0, 1.0, int(still_blank.sum()))).clip(lower=5.0)

    result.loc[still_blank, PICHIA_TARGET_COL] = yield_value
    result.loc[still_blank, PICHIA_OD_COL] = od_value
    return result

def _pichia_round2_design_and_backfill_section(plan: Round2Plan, round1_df: pd.DataFrame) -> None:
    """Extends plan's CCD table with two blocks generate_ccd can never
    produce on its own -- a feed-interval level round 1 never tried, crossed
    with one continuous variable to check for interaction, plus Latin
    Hypercube space-filling points for whichever BO round consumes the
    combined dataset next (see round2_design.generate_round2_extension_design
    for why both are structurally separate from the CCD track) -- then gives
    the combined sheet the same download / upload-backfill / save-to-final
    loop Round 1's design has, so results collected under it are reproducible
    and don't dead-end in a chat transcript."""
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

    _ensure_pichia_data_area()
    if _pichia_restore_persisted_dataset("round2_full_design_df", PICHIA_ROUND2_DATASET_PATH):
        st.toast(f"已从 {PICHIA_ROUND2_DATASET_PATH.name} 恢复上次保存的 Round 2 结果。", icon="✅")
    st.markdown("#### 完整 Round 2 设计表（CCD + 补料间隔交互 + LHS）")
    st.caption(
        "在上面的响应面(CCD)之外，再加两块：一是测试一个 round 1 没试过的补料间隔水平，"
        "看它的效应是否跟某个连续变量有交互；二是拉丁超立方(LHS)随机点，专门填补 CCD 网格之间的空隙，"
        "供后续贝叶斯优化的 GP 模型使用。同样的输入（round 1 数据 + 下面这些参数）永远生成同一张表，方便留档核对。"
    )

    interaction_choices = plan.active_variables or list(PICHIA_CONTINUOUS_BOUNDS)
    param_cols = st.columns(4)
    with param_cols[0]:
        extra_interval = st.number_input(
            "新补料间隔 (h)",
            min_value=0.0,
            value=36.0,
            step=1.0,
            help="round 1 从未测过的一个补料间隔水平，例如比现有 24h 更长的档位；具体数值需先跟研发组确认设备/排期可行。",
            key="round2_extra_interval",
        )
    with param_cols[1]:
        interaction_variable = st.selectbox(
            "跟新间隔做交互的变量",
            options=interaction_choices,
            format_func=lambda name: PICHIA_VARIABLE_LABELS.get(name, name),
            help="通常选一个 round 1 已经显著的变量，交互测试才有决策价值。",
            key="round2_interaction_variable",
        )
    with param_cols[2]:
        n_noise_reference = st.number_input(
            "新间隔噪声参考重复数",
            min_value=0,
            max_value=6,
            value=2,
            help="新补料间隔区域没有任何历史噪声数据，这里的重复测量专门给它一个噪声估计，不是可选的锦上添花。",
            key="round2_n_noise_reference",
        )
    with param_cols[3]:
        n_lhs = st.number_input(
            "LHS 点数",
            min_value=0,
            max_value=30,
            value=10,
            help="纯随机空间填充点，不进入响应面拟合，只用于让后续贝叶斯优化的 GP 模型看到更多样的条件组合。",
            key="round2_n_lhs",
        )

    if st.button("生成完整 Round 2 设计", type="primary", key="generate_round2_full_design"):
        full_design = assemble_round2_design(
            plan,
            PICHIA_ROUND1_BASELINE,
            round1_df,
            extra_interval_levels=[float(extra_interval)],
            interaction_variable=interaction_variable,
            n_noise_reference=int(n_noise_reference),
            n_lhs=int(n_lhs),
            seed=42,
        )
        st.session_state["round2_full_design_df"] = full_design
        st.success(f"已生成 {len(full_design)} 行设计（种子固定为 42，同样的参数下次重跑结果完全一致）。")

    full_design = st.session_state.get("round2_full_design_df")
    if full_design is None or full_design.empty:
        return

    st.caption("各模块行数：" + "、".join(f"{PICHIA_RUN_TYPE_LABELS.get(rt, rt)} {count}" for rt, count in full_design["run_type"].value_counts().items()))

    excel_bytes = _pichia_round2_workbook_bytes(full_design)
    st.download_button(
        "下载 Round 2 设计表格（Excel）",
        data=excel_bytes,
        file_name="pichia_round2_design.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_round2_excel",
    )

    st.markdown("##### 上传已回填结果（可选）")
    uploaded = st.file_uploader(
        "上传已回填 yield_g_per_l / od600 的 CSV，或者上面下载的 Excel 填完后原样传回来",
        type=["csv", "xlsx"],
        key="round2_results_upload",
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
                archive_name = Path(uploaded.name).name or "pichia_round2_uploaded.csv"
                (PICHIA_UPLOAD_DIR / archive_name).write_bytes(payload)
                st.session_state["round2_full_design_df"] = uploaded_df
                full_design = uploaded_df
                st.success("已加载并归档上传的 Round 2 结果。")
                for warning_text in consistency_warnings:
                    st.warning(warning_text)
                if not replicate_spread.empty:
                    sort_column = next((column for column in replicate_spread.columns if column.endswith("重复间差值")), None)
                    if sort_column:
                        replicate_spread = replicate_spread.sort_values(sort_column, ascending=False)
                    st.dataframe(replicate_spread, width="stretch", hide_index=True)

    with st.expander("🧪 开发/测试用：填入模拟数据", expanded=False):
        st.caption(
            "只填当前还是空的产量/OD600格子，编造的示意数值，不是真实实验结果——已经回填的真实数据不会被覆盖。"
            "用于在真实 Round 2 结果回来之前，先试一遍下面响应面拟合/交互检验/贝叶斯优化这几块的界面是否正常。"
        )
        confirm_simulate = st.checkbox("我知道这是假数据，只是用来测试界面", key="confirm_simulate_round2")
        # the explicit "and confirm_simulate" doesn't just duplicate disabled=
        # for show -- it's what actually gates the fill; disabled= only stops
        # a mouse click in a real browser, not a widget state set some other
        # way, and the one thing that must never happen is filling fake data
        # in without the checkbox truly being on.
        if st.button("填入模拟数据", key="simulate_round2_results", disabled=not confirm_simulate) and confirm_simulate:
            already_filled = int(full_design[PICHIA_TARGET_COL].notna().sum())
            full_design = _pichia_simulate_round2_results(full_design)
            st.session_state["round2_full_design_df"] = full_design
            newly_filled = int(full_design[PICHIA_TARGET_COL].notna().sum()) - already_filled
            st.success(f"已给 {newly_filled} 行空白结果填入模拟数据；原有 {already_filled} 行真实回填数据未改动。")

    st.markdown("##### 回填实测结果")
    entry_view = full_design[["run_id", PICHIA_TARGET_COL, PICHIA_OD_COL]].reset_index(drop=True)
    edited_entries = st.data_editor(
        entry_view,
        width="stretch",
        num_rows="fixed",
        hide_index=True,
        key="round2_data_editor",
        disabled=["run_id"],
        column_config={"run_id": "编号", PICHIA_TARGET_COL: "hLF产量", PICHIA_OD_COL: "收获时OD600"},
    )
    edited = full_design.copy()
    edited[PICHIA_TARGET_COL] = edited_entries[PICHIA_TARGET_COL].to_numpy()
    edited[PICHIA_OD_COL] = edited_entries[PICHIA_OD_COL].to_numpy()
    st.session_state["round2_full_design_df"] = edited

    numeric_edited = _pichia_numeric_results(edited)
    filled = int(numeric_edited[[PICHIA_TARGET_COL, PICHIA_OD_COL]].notna().all(axis=1).sum())
    st.caption(f"已回填 {filled}/{len(edited)} 行完整结果。")

    if st.button("保存到 data/pichia/final/（供下一轮和历史查阅使用）", key="save_round2_to_final"):
        _ensure_pichia_data_area()
        edited.to_csv(PICHIA_ROUND2_DATASET_PATH, index=False, encoding="utf-8-sig")
        st.success(f"已保存到 {PICHIA_ROUND2_DATASET_PATH}")

PICHIA_ROUND2_NO_RESULTS_HINT = (
    "还没有可分析的 Round 2 结果。先在「② 设计生成与回填」子页生成设计表，"
    "再把产量和 OD600 填回去——只有两个都填了的行才会进入分析。"
)

def _pichia_round2_filled_design() -> pd.DataFrame | None:
    """The Round 2 rows that are actually analyzable: both yield and OD600
    present. None when the sheet doesn't exist yet or nothing is backfilled.

    Lives here rather than as two early returns inside the response-surface and
    BO sections because both need the same view of the same sheet, and after the
    split into sub-tabs both also need to *say* something when it's empty --
    a silently blank sub-tab reads as a broken page, not as "no data yet"."""
    full_design = st.session_state.get("round2_full_design_df")
    if full_design is None or full_design.empty:
        return None
    numeric_full = _pichia_numeric_results(full_design)
    filled_mask = numeric_full[[PICHIA_TARGET_COL, PICHIA_OD_COL]].notna().all(axis=1)
    if not filled_mask.any():
        return None
    return numeric_full.loc[filled_mask]

def _pichia_round2_surface_section(plan: Round2Plan, round1_df: pd.DataFrame) -> None:
    """What this round's own designed points say: the fitted response surface
    over the CCD block (plus the deep-dive reads of it) and the feed-interval
    interaction verdict. Generating the design sheet only produces conditions to
    test, not conclusions -- this is where the backfilled numbers turn into one.

    Deliberately separate from the combined-dataset BO section: that answers a
    different question (where to go next given everything measured so far), and
    keeping them in one sub-tab was most of why the Round 2 page had become an
    unnavigable scroll."""
    filled_design = _pichia_round2_filled_design()
    if filled_design is None:
        st.info(PICHIA_ROUND2_NO_RESULTS_HINT)
        return
    full_design = st.session_state["round2_full_design_df"]

    if plan.active_variables:
        st.markdown("#### 响应面 (CCD) 拟合")
        ccd_rows = filled_design[filled_design["run_type"] == "ccd"]
        try:
            fit = fit_ccd_response_surface(ccd_rows, plan.active_variables)
        except ValueError as exc:
            st.info(f"CCD 拟合暂时无法运行：{exc}")
        else:
            od_fit: Any | None = None
            try:
                od_fit = fit_ccd_response_surface(ccd_rows, plan.active_variables, target_col=PICHIA_OD_COL)
            except ValueError:
                pass  # OD600 fit is a supplementary feasibility overlay -- yield analysis still stands without it
            od_threshold = plan.od_threshold["threshold"]

            fit_cols = st.columns(3)
            fit_cols[0].metric(
                "R²", _num(fit.r_squared), help="拟合优度：模型解释了多少比例的产量变化，1.0=完美拟合，越低说明二次模型解释力越弱。"
            )
            fit_cols[1].metric(
                "样本数 / 参数数",
                f"{fit.n_points} / {fit.n_params}",
                help="用于拟合的 CCD 样本数，和模型需要估计的参数个数（截距+一次项+二次项+交互项）；样本数远大于参数数时，估计才稳。",
            )
            if fit.lack_of_fit is None:
                fit_cols[2].metric("失拟检验", "样本不足，无法判断")
            else:
                verdict = "存在失拟" if fit.lack_of_fit["significant_lack_of_fit"] else "无显著失拟"
                fit_cols[2].metric(
                    "失拟检验",
                    verdict,
                    help=f"p={_num(fit.lack_of_fit['p_value'])}。检验「二次模型这个形式对不对」，和某一项系数是否显著是两件独立的事。",
                )

            optimum_point = {**plan.fixed_values, **fit.optimum}
            optimum_display = {PICHIA_VARIABLE_LABELS.get(name, name): _num(value) for name, value in optimum_point.items()}
            optimum_display["预测产量"] = _num(fit.predicted_optimum)
            optimum_ci = predict_with_confidence_interval(fit, pd.DataFrame([fit.optimum]))
            optimum_display["预测产量 95% CI"] = f"[{_num(optimum_ci.loc[0, 'ci_low'])}, {_num(optimum_ci.loc[0, 'ci_high'])}]"
            if od_fit is not None:
                predicted_od_at_optimum = float(evaluate_response_surface(od_fit, pd.DataFrame([fit.optimum]))[0])
                optimum_display["预测OD600"] = _num(predicted_od_at_optimum)
                optimum_display["OD600可行"] = "是" if predicted_od_at_optimum >= od_threshold else "否"
            optimum_help: dict[str, str] = {
                PICHIA_VARIABLE_LABELS.get(name, name): f"响应面拟合在本轮实际测试范围内搜索得到的「{PICHIA_VARIABLE_LABELS.get(name, name)}」最优取值。"
                for name in optimum_point
            }
            optimum_help["预测产量"] = "把最优点代入拟合模型算出的预测产量——模型估计值，不是实测值。"
            optimum_help["预测产量 95% CI"] = (
                "这个预测值本身的 95% 置信区间：区间越宽，说明这个「最优点」的产量估计越不确定，"
                "换一批数据重新做同样的实验，结果可能有明显差异；区间很窄才说明这个数字站得住。"
            )
            if od_fit is not None:
                optimum_help["预测OD600"] = "把最优点代入 OD600 的响应面拟合算出的预测生长量，同样是模型估计值。"
                optimum_help["OD600可行"] = "预测OD600 是否达到可行阈值；「否」表示这个产量最优点在生长可行性上不成立，不能直接采用。"
            st.dataframe(
                pd.DataFrame([optimum_display]),
                width="stretch",
                hide_index=True,
                column_config={col: st.column_config.Column(help=text) for col, text in optimum_help.items()},
            )
            st.caption("预测最优点的搜索范围限制在 CCD 实际测试过的区间内，不外推到 round 1 原始边界之外。")

            _pichia_render_response_surface_verdicts(
                summarize_response_surface(fit, ccd_rows, od_fit=od_fit, od_threshold=od_threshold if od_fit is not None else None)
            )

            with st.expander("系数明细（每一项是否显著）", expanded=False):
                st.caption(
                    "失拟检验看的是「模型整体形式对不对」；这里的显著性看的是「这一项本身是不是噪声」，两者是独立的两件事。"
                    "显著性：*** p<0.001，** p<0.01，* p<0.05，· p<0.1，空白=不显著——星号只是 p 值的快速参考，具体数值看 p 值列。"
                )
                st.dataframe(
                    _pichia_coefficient_table(fit),
                    width="stretch",
                    hide_index=True,
                    column_config=_pichia_coefficient_column_config(),
                )

            st.plotly_chart(_pichia_response_surface_residual_chart(fit, ccd_rows), width="stretch")

            if len(plan.active_variables) == 1:
                st.plotly_chart(
                    _pichia_response_surface_curve_chart(fit, ccd_rows, plan.active_variables[0]), width="stretch"
                )
            else:
                st.caption(
                    "每组图固定其余活跃变量在预测最优点，○是 CCD 实测点，等高线图上的×同理，红色菱形/星标是模型给出的联合最优点；"
                    + ("白色虚线内是预测 OD600 达到可行阈值的区域。" if od_fit is not None else "OD600 拟合暂时无法运行，没有可行边界线。")
                )
                colorscale_name = st.selectbox(
                    "曲面配色",
                    list(PICHIA_SEQUENTIAL_COLORSCALE_OPTIONS),
                    index=list(PICHIA_SEQUENTIAL_COLORSCALE_OPTIONS).index(PICHIA_SEQUENTIAL_COLORSCALE_DEFAULT),
                    help="推荐选项在保持深色主题蓝调的前提下拉宽了明度跨度，比原配色更容易分清相邻等高线档位；"
                    "越靠后的选项辨识度越高，但和界面主题的贴合度也越低。",
                    key="pichia_response_surface_colorscale",
                )
                colorscale = PICHIA_SEQUENTIAL_COLORSCALE_OPTIONS[colorscale_name]
                pairs = list(itertools.combinations(plan.active_variables, 2))
                for x_variable, y_variable in pairs:
                    view_cols = st.columns(2)
                    with view_cols[0]:
                        st.plotly_chart(
                            _pichia_response_surface_3d_chart(fit, ccd_rows, x_variable, y_variable, colorscale),
                            width="stretch",
                        )
                    with view_cols[1]:
                        st.plotly_chart(
                            _pichia_response_surface_contour_chart(
                                fit, ccd_rows, x_variable, y_variable, colorscale, od_fit=od_fit, od_threshold=od_threshold
                            ),
                            width="stretch",
                        )

            st.markdown("#### 深入分析")
            case = classify_response_surface_case(
                fit, ccd_rows, od_fit=od_fit, od_threshold=od_threshold if od_fit is not None else None
            )
            _pichia_render_response_surface_deep_dive(fit, ccd_rows, case, od_fit, od_threshold if od_fit is not None else None)
    else:
        st.caption("本轮没有活跃变量进入响应面，跳过 CCD 拟合。")

    interaction_rows = full_design[full_design["run_type"] == "interval_interaction"]
    if not interaction_rows.empty:
        st.markdown("#### 补料间隔交互检验")
        interaction_variable = interaction_rows["changed_variable"].iloc[0]
        extra_interval_level = float(interaction_rows["interval_h"].iloc[0])
        interaction_levels = sorted(interaction_rows[interaction_variable].dropna().unique().tolist())
        result = analyze_interval_interaction(
            full_design, round1_df, PICHIA_ROUND1_BASELINE, interaction_variable, extra_interval_level, interaction_levels
        )
        label = PICHIA_VARIABLE_LABELS.get(interaction_variable, interaction_variable)
        if result["noise_sd"] is None:
            st.info(f"新区域噪声参考样本不足（{result['noise_n']} 个），暂时无法判断显著性，仅展示效应量。")
        rows = [
            {
                "对比": f"{label}={result['low_level']:g}",
                "新间隔-旧间隔差值": _num(result["effect_at_low"]),
                "显著": {True: "是", False: "否", None: "样本不足"}[result["interval_effect_significant_at_low"]],
            },
            {
                "对比": f"{label}={result['high_level']:g}",
                "新间隔-旧间隔差值": _num(result["effect_at_high"]),
                "显著": {True: "是", False: "否", None: "样本不足"}[result["interval_effect_significant_at_high"]],
            },
            {
                "对比": "交互效应（两者差值）",
                "新间隔-旧间隔差值": _num(result["interaction_effect"]),
                "显著": {True: "是", False: "否", None: "样本不足"}[result["interaction_significant"]],
            },
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if result["interaction_significant"] is True:
            st.caption(f"「补料间隔 {extra_interval_level:g}h」的效应随「{label}」水平变化——不是一个固定的主效应，用它时要连着「{label}」一起看。")
        elif result["interaction_significant"] is False:
            st.caption(f"「补料间隔 {extra_interval_level:g}h」的效应在两个「{label}」水平下差不多，可以当一个不依赖「{label}」的主效应来看。")

def _pichia_round2_combined_bo_section(plan: Round2Plan, round1_df: pd.DataFrame) -> None:
    """Bayesian optimisation over Round 1 + whatever Round 2 is backfilled so
    far -- the ADR-0009 track that keeps running alongside the response surface
    instead of competing with it, and the only remaining BO entry after
    ADR-0016 removed the Round1-only preview."""
    filled_design = _pichia_round2_filled_design()
    if filled_design is None:
        st.info(PICHIA_ROUND2_NO_RESULTS_HINT)
        return

    st.caption("用 Round 1 + 本轮已回填的 Round 2 数据一起重新训练 GP——数据量和覆盖面都比只用 Round 1 时更完整。")
    combined_columns = ["run_id", *PICHIA_VARIABLES, PICHIA_TARGET_COL, PICHIA_OD_COL]
    combined_df = pd.concat(
        [round1_df[combined_columns], filled_design[combined_columns]],
        ignore_index=True,
    )
    n_batch_combined = st.slider("建议批次大小", min_value=3, max_value=15, value=9, key="round2_combined_bo_batch_size")
    if st.button("用合并数据生成贝叶斯优化建议", type="primary", key="run_combined_bo"):
        try:
            bo_result = recommend_round2_bo_batch(
                combined_df,
                fixed_values=plan.fixed_values,
                active_variables=plan.active_variables or list(PICHIA_CONTINUOUS_BOUNDS),
                od_threshold=plan.od_threshold["threshold"],
                n_batch=int(n_batch_combined),
            )
        except ImportError:
            st.warning("当前环境未安装 torch/botorch/gpytorch，无法运行贝叶斯优化建议。")
        except ValueError as exc:
            st.warning(f"未生成建议：{exc}")
        except Exception as exc:
            st.warning(f"贝叶斯优化建议生成失败（样本量很小时，GP 拟合可能数值不稳定）：{exc}")
        else:
            st.session_state["round2_combined_bo_result"] = bo_result
            st.success(
                f"已用 {len(combined_df)} 行合并数据生成 {len(bo_result['recommendations'])} 个建议"
                f"（候选池 {bo_result['n_candidates']} 个，满足约束 {bo_result['n_feasible']} 个）。"
            )

    combined_bo_result = st.session_state.get("round2_combined_bo_result")
    if combined_bo_result:
        _pichia_render_bo_recommendation_section(
            combined_bo_result, plan.active_variables, combined_df, "round2_combined_bo_cv_result"
        )

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

    # The four sub-tab bodies all run on every rerun in the order written here
    # -- st.tabs only hides them visually, it does not defer them. Design and
    # backfill therefore has to stay above the response-surface and BO tabs:
    # its st.data_editor is what writes "round2_full_design_df", which those two
    # then read, and flipping the order would make a freshly typed-in result
    # need a second interaction before the analysis picked it up.
    significance_tab, design_tab, surface_tab, bo_tab = st.tabs(
        ["① 显著性分析", "② 设计生成与回填", "③ 响应面结果", "④ 合并数据贝叶斯优化"]
    )
    with significance_tab:
        _pichia_round2_significance_section(plan, run_df)
    with design_tab:
        _pichia_round2_design_and_backfill_section(plan, run_df)
    with surface_tab:
        _pichia_round2_surface_section(plan, run_df)
    with bo_tab:
        _pichia_round2_combined_bo_section(plan, run_df)
