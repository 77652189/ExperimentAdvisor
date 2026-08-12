"""Round 2 response-surface views: the tables, term labels, tooltip text,
narrative-verdict rendering and every chart the CCD fit is read through.

Pure view layer (see docs/adr/0017): every function here takes an already
fitted model plus data and returns a figure / frame, or writes st.* output.
Nothing here reads or writes st.session_state and no widget here carries a
key -- that is the boundary that keeps this module free of page state, and it
is the rule to check a new function against before adding it here.
"""
from __future__ import annotations

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
    canonical_analysis,
    evaluate_response_surface,
    optimize_joint_desirability,
    response_surface_grid,
    sensitivity_analysis,
)
from App.pichia_common import (
    PICHIA_ACCENT_COLOR,
    PICHIA_MUTED_COLOR,
    PICHIA_TARGET_COL,
    PICHIA_VARIABLE_LABELS,
    _num,
)

def _pichia_response_surface_term_label(term: str) -> str:
    """Chinese-friendly label for a fitted response-surface term name
    ("ph" / "ph^2" / "ph*glucose_pct") -- reuses PICHIA_VARIABLE_LABELS so it
    stays in sync with every other Chinese label in this file rather than
    hand-maintaining a second translation table."""
    if term == "intercept":
        return "截距"
    if "*" in term:
        first, second = term.split("*")
        return f"{PICHIA_VARIABLE_LABELS.get(first, first)} × {PICHIA_VARIABLE_LABELS.get(second, second)}"
    if term.endswith("^2"):
        base = term[:-2]
        return f"{PICHIA_VARIABLE_LABELS.get(base, base)}²"
    return PICHIA_VARIABLE_LABELS.get(term, term)

def _pichia_significance_stars(p_value: float | None) -> str:
    """Standard significance-star convention (R/statsmodels-style) -- a
    single p-value column is precise but a binary 是/否 throws away exactly
    how close to the cutoff a term is; stars keep both the exact number and
    a quick-scan read."""
    if p_value is None or pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "·"
    return ""

# help text for st.dataframe(column_config=...) -- centralized here (rather
# than inline at each call site) since several tables below share the same
# jargon (p值/显著性/满意度 etc.) and a hover tooltip is the only place this
# explanation lives; keeping it in one dict makes it easy to keep the wording
# consistent across tables instead of drifting per call site.
PICHIA_COEFFICIENT_HELP: dict[str, str] = {
    "项": "拟合模型里的一项：截距、某个变量的一次项、二次项（²，弯曲/封顶效应），或两个变量的交互项（×）。",
    "系数": "这一项对预测产量的贡献方向和大小；正负号表示增加该变量取值是让预测产量升高还是降低，不能跨项直接比较大小（单位不同）。",
    "标准误": "这个系数估计值本身的不确定度——数值越大，说明换一批数据重新拟合，这个系数可能差很多，估计不稳。",
    "p值": "这一项是否只是噪声的统计检验：p 值越小，这一项真实存在（不是噪声）的把握越大；一般 p<0.05 认为显著。",
    "显著性": "p 值的快速参考：*** p<0.001，** p<0.01，* p<0.05，· p<0.1，空白=不显著（p≥0.1）。",
}


def _pichia_coefficient_column_config() -> dict[str, Any]:
    return {column: st.column_config.Column(help=text) for column, text in PICHIA_COEFFICIENT_HELP.items()}


def _pichia_coefficient_table(fit: Any) -> pd.DataFrame:
    rows = []
    for term in fit.term_names:
        stats_dict = fit.coefficient_significance.get(term)
        p_value = stats_dict["p_value"] if stats_dict else None
        rows.append(
            {
                "项": _pichia_response_surface_term_label(term),
                "系数": _num(fit.coefficients[term]),
                "标准误": _num(stats_dict["se"]) if stats_dict else None,
                "p值": _num(p_value) if stats_dict else None,
                "显著性": _pichia_significance_stars(p_value) if stats_dict else "样本不足",
            }
        )
    return pd.DataFrame(rows)

PICHIA_VERDICT_RENDERERS = {"success": st.success, "info": st.info, "warning": st.warning}

def _pichia_render_verdicts(verdicts: list[Any], translate: dict[str, str] | None = None) -> None:
    """Shared severity->st.* dispatch for any verdict list shaped like
    ResponseSurfaceVerdict/BoRecommendationVerdict (a .severity/.message
    pair). `translate`, when given, swaps bare「name」placeholders for
    Chinese labels -- round2_design.py's response-surface verdicts need
    this (it stays UI-agnostic, no label dict of its own); its BO verdicts
    don't, since summarize_bo_recommendation already writes fully Chinese
    messages with no placeholders to translate."""
    for verdict in verdicts:
        message = verdict.message
        if translate:
            for name, label in translate.items():
                message = message.replace(f"「{name}」", f"「{label}」")
        PICHIA_VERDICT_RENDERERS.get(verdict.severity, st.info)(message)

def _pichia_render_response_surface_verdicts(verdicts: list[Any]) -> None:
    _pichia_render_verdicts(verdicts, translate=PICHIA_VARIABLE_LABELS)

PICHIA_SENSITIVITY_HELP: dict[str, str] = {
    "变量": "该活跃变量。",
    "峰值点": "其余活跃变量固定在联合最优点时，单独扫这一个变量得到的预测产量最高点——K≥2 时一般很接近联合最优点，但不必完全相同。",
    "峰值预测产量": "峰值点对应的预测产量。",
    "容许范围(±5%)": "峰值点附近、预测产量仍不低于峰值 95% 的取值区间——区间越窄，说明这个变量需要控制得越精确才能拿到接近最优的产量；区间越宽，说明这个变量比较「皮实」，微调影响不大。",
    "范围宽度": "容许范围的绝对宽度（上限-下限），单位与该变量本身一致。",
    "占测试范围比例": "容许范围宽度 ÷ 本轮该变量实际测试范围的宽度，例如 0.3 表示容许范围只占本轮测试跨度的 30%。",
    "触及测试范围边界": "容许范围是否顶到了本轮实际测试范围的边缘——如果是，真实的容许范围可能比这里算出的更宽，只是被测试范围本身卡住了，不代表这个变量真的这么敏感。",
}

def _pichia_sensitivity_table(sensitivity: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for variable, result in sensitivity.items():
        touches = []
        if result.touches_lower_bound:
            touches.append("下限")
        if result.touches_upper_bound:
            touches.append("上限")
        rows.append(
            {
                "变量": PICHIA_VARIABLE_LABELS.get(variable, variable),
                "峰值点": _num(result.peak_x),
                "峰值预测产量": _num(result.peak_value),
                "容许范围(±5%)": f"[{_num(result.plateau_low)}, {_num(result.plateau_high)}]",
                "范围宽度": _num(result.plateau_width),
                "占测试范围比例": _num(result.plateau_width_fraction),
                "触及测试范围边界": "、".join(touches) if touches else "否",
            }
        )
    return pd.DataFrame(rows)

def _pichia_sensitivity_chart(result: Any, variable_label: str) -> go.Figure:
    """The sweep sensitivity_analysis's own plateau bounds were read off of,
    made visible: the shaded band is the "容许范围" from the table, the
    dashed line is the 95%-of-peak cutoff that band was measured against."""
    cutoff = result.peak_value - abs(result.peak_value) * 0.05
    fig = go.Figure()
    fig.add_vrect(x0=result.plateau_low, x1=result.plateau_high, fillcolor=PICHIA_ACCENT_COLOR, opacity=0.15, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=[result.tested_low, result.tested_high],
            y=[cutoff, cutoff],
            mode="lines",
            line=dict(color=PICHIA_MUTED_COLOR, dash="dash"),
            name="峰值的95%",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(x=result.x_values, y=result.predicted, mode="lines", line=dict(color=PICHIA_ACCENT_COLOR, width=2), name="预测产量")
    )
    fig.add_trace(
        go.Scatter(
            x=[result.peak_x], y=[result.peak_value], mode="markers", marker=dict(color="#e66767", size=10, symbol="star"), name="峰值点"
        )
    )
    fig.update_layout(
        title=f"{variable_label}：灵敏度扫描（阴影=容许范围）",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=variable_label,
        yaxis_title="预测产量",
        showlegend=False,
        margin=dict(t=50, b=40, l=60, r=30),
        height=300,
    )
    return fig

_PICHIA_CANONICAL_LABELS: dict[str, str] = {
    "maximum": "真极大值（有明确的单一峰值）",
    "minimum": "真极小值（有明确的单一谷值——如果目标是最大化产量，这个方向不是好消息）",
    "saddle": "鞍点（沿一个方向升高、另一个方向降低，不是真正的「峰」）",
    "ridge": "岭线/脊线（至少有一个方向几乎没有曲率，可能有一整条线同样好，不存在单一最优点）",
    "flat": "近似平坦（拟合面在活跃变量范围内几乎没有曲率，模型给不出「哪里最优」的有效信息）",
}

def _pichia_canonical_classification_label(classification: str) -> str:
    return _PICHIA_CANONICAL_LABELS.get(classification, classification)

PICHIA_CANONICAL_HELP: dict[str, str] = {
    "分类": "经典响应面 canonical analysis 判别：不看网格搜索给出的「测试范围内最好的点」，只看拟合公式本身的曲率，判断这个曲面真实的形状是不是真的有峰。",
    "无约束理论最优点": "不考虑测试范围限制、只看拟合公式本身梯度为零的点——鞍点/岭线情形下这个点的意义有限，请配合下面两列一起看。",
    "该点在测试范围内": "理论最优点是否落在本轮实际测试范围内；「否」时它更多是数学参考点，不能直接当作可以验证的条件。",
    "结果可靠": "「否」表示分类是鞍点以外的退化情形（岭线/近似平坦），此时「理论最优点」不唯一或没有实际意义，不建议照搬这个点去验证。",
    "该点预测产量": "把理论最优点代入拟合模型算出的预测产量，同样只在「结果可靠」为「是」时才有直接参考价值。",
}

def _pichia_canonical_table(canonical: Any) -> pd.DataFrame:
    point_text = "、".join(
        f"{PICHIA_VARIABLE_LABELS.get(name, name)}={_num(value)}" for name, value in canonical.stationary_point.items()
    )
    row = {
        "分类": _pichia_canonical_classification_label(canonical.classification),
        "无约束理论最优点": point_text,
        "该点在测试范围内": "是" if canonical.stationary_point_in_tested_range else "否",
        "结果可靠": "是" if canonical.stationary_point_reliable else "否",
        "该点预测产量": _num(canonical.predicted_at_stationary_point),
    }
    return pd.DataFrame([row])

def _pichia_canonical_eigenvalue_chart(canonical: Any, ridge_tolerance: float = 0.05) -> go.Figure:
    """The raw evidence the max/min/saddle/ridge classification in the table
    above is actually computed from -- otherwise the classification is just
    an assertion to take on faith. Bars inside the shaded band are the
    "no real curvature in this direction" eigenvalues (see canonical_analysis's
    own ridge_tolerance); a ridge/saddle classification is one bar in that
    band (or of the opposite sign from the rest) away from being a clean
    maximum/minimum."""
    values = canonical.eigenvalues
    scale = max((abs(v) for v in values), default=0.0)
    threshold = ridge_tolerance * scale
    if len(canonical.active_variables) == 1:
        labels = [PICHIA_VARIABLE_LABELS.get(canonical.active_variables[0], canonical.active_variables[0])]
    else:
        labels = [f"曲率方向 {index + 1}" for index in range(len(values))]
    colors = [PICHIA_MUTED_COLOR if abs(value) < threshold else ("#e66767" if value < 0 else PICHIA_ACCENT_COLOR) for value in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, hovertemplate="%{x}<br>特征值: %{y:.4g}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=PICHIA_MUTED_COLOR, width=1))
    if scale > 0:
        fig.add_hrect(y0=-threshold, y1=threshold, fillcolor=PICHIA_MUTED_COLOR, opacity=0.15, line_width=0)
    fig.update_layout(
        title="曲率特征值（灰色带内视为「没有真实曲率」）",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="特征值",
        margin=dict(t=50, b=40, l=60, r=30),
        height=280,
    )
    return fig

PICHIA_DESIRABILITY_HELP: dict[str, str] = {
    "方案": "两个候选点的对比：只看产量能拿到多少，vs. 同时兼顾产量和 OD600 可行性能拿到多少。",
    "预测产量": "该点代入产量响应面拟合算出的预测产量。",
    "预测OD600": "该点代入 OD600 响应面拟合算出的预测生长量——纯产量最优点这一行也算了这个值，用来说明选它到底在 OD600 上差多远，不是完全没有这个信息。",
    "满意度(0~1)": "Derringer-Suich 满意度：产量和 OD600 可行性各自打 0~1 分后取几何平均——只要有一项是 0（例如 OD600 完全不可行），总分就是 0，不会被另一项的高分「平均」掉。纯产量最优点这一行留空，因为选它时完全没考虑 OD600，谈不上一个联合满意度分数。",
}

def _pichia_desirability_table(result: Any) -> pd.DataFrame:
    def _row(
        label: str, point: dict[str, float], yield_value: float, od_value: float | None, desirability: float | None
    ) -> dict[str, Any]:
        row: dict[str, Any] = {"方案": label}
        for name, value in point.items():
            row[PICHIA_VARIABLE_LABELS.get(name, name)] = _num(value)
        row["预测产量"] = _num(yield_value)
        # None (not a "-" placeholder string) so the column stays a clean
        # float dtype -- mixing floats and a literal placeholder string in
        # one column makes it an "object" dtype, which pyarrow's Streamlit
        # serialization only handles via a slow, warning-logging fallback
        # path; a plain missing value renders as an empty cell either way.
        row["预测OD600"] = _num(od_value) if od_value is not None else None
        row["满意度(0~1)"] = _num(desirability) if desirability is not None else None
        return row

    rows = [
        _row(
            "纯产量最优点（不考虑OD600）",
            result.pure_yield_optimum,
            result.pure_yield_optimum_value,
            result.pure_yield_optimum_od600,
            None,
        ),
        _row("联合满意度最优点", result.point, result.predicted_yield, result.predicted_od600, result.composite_desirability),
    ]
    return pd.DataFrame(rows)

def _pichia_desirability_column_config(result: Any) -> dict[str, Any]:
    config = dict(PICHIA_DESIRABILITY_HELP)
    for name in result.point:
        label = PICHIA_VARIABLE_LABELS.get(name, name)
        config.setdefault(label, f"「{label}」在该方案下的取值。")
    return {column: st.column_config.Column(help=text) for column, text in config.items()}

def _pichia_desirability_tradeoff_chart(result: Any, od_threshold: float) -> go.Figure:
    """Yield-vs-OD600 scatter comparing the two candidate points -- they sit
    at different process conditions, but what the trade-off is actually
    about is where each one lands on these two axes, not which conditions
    produced them (already shown in the table)."""
    fig = go.Figure()
    fig.add_vline(line=dict(color=PICHIA_MUTED_COLOR, dash="dash"), x=od_threshold)
    fig.add_annotation(x=od_threshold, y=1.02, yref="paper", text="OD600 可行阈值", showarrow=False, font=dict(color=PICHIA_MUTED_COLOR, size=11))
    fig.add_trace(
        go.Scatter(
            x=[result.pure_yield_optimum_od600],
            y=[result.pure_yield_optimum_value],
            mode="markers",
            marker=dict(color="#e6e4da", size=13, symbol="circle"),
            name="纯产量最优点",
            hovertemplate="纯产量最优点<br>预测产量: %{y:.4g}<br>预测OD600: %{x:.3g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[result.predicted_od600],
            y=[result.predicted_yield],
            mode="markers",
            marker=dict(color="#e66767", size=13, symbol="diamond"),
            name="联合满意度最优点",
            hovertemplate="联合满意度最优点<br>预测产量: %{y:.4g}<br>预测OD600: %{x:.3g}<extra></extra>",
        )
    )
    fig.update_layout(
        title="产量 vs OD600 权衡",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="预测 OD600",
        yaxis_title="预测产量",
        margin=dict(t=50, b=40, l=60, r=30),
        height=340,
    )
    return fig

def _pichia_render_response_surface_deep_dive(
    fit: Any,
    ccd_rows: pd.DataFrame,
    case: str,
    od_fit: Any | None,
    od_threshold: float | None,
) -> None:
    """The three deeper analyses beyond the optimum's own CI (already shown
    inline on the optimum table): sensitivity/plateau width, canonical
    (saddle/ridge/peak) classification, and joint yield+OD600 desirability.
    Framed differently depending on `case` (classify_response_surface_case's
    output, same priority order as summarize_response_surface's own verdict)
    since the same numbers mean different things depending on whether the
    fit/data are already trustworthy or not -- e.g. a narrow sensitivity
    plateau is reassuring in the "normal" case but beside the point when the
    model has significant lack-of-fit in the first place."""
    if case == "lack_of_fit":
        st.warning(
            "⚠️ 存在显著失拟：下面的灵敏度/鞍点判别/联合优化都基于当前这个（形式可能不对的）二次模型，"
            "结论只能参考，不能替代重新检查模型形式。"
        )

    with st.expander("灵敏度分析（最优点附近多「皮实」）", expanded=False):
        st.caption("固定其余活跃变量在联合最优点，单独扫一个变量：预测产量掉到峰值 95% 以下之前，这个变量还有多大的浮动空间。")
        sensitivity = sensitivity_analysis(fit, ccd_rows)
        st.dataframe(
            _pichia_sensitivity_table(sensitivity),
            width="stretch",
            hide_index=True,
            column_config={column: st.column_config.Column(help=text) for column, text in PICHIA_SENSITIVITY_HELP.items()},
        )
        if case == "boundary" and any(result.touches_lower_bound or result.touches_upper_bound for result in sensitivity.values()):
            st.caption("留意「触及测试范围边界」为「是」的行——这些变量的真实容许范围可能比表里算出的更宽，只是被本轮测试范围卡住了。")
        sensitivity_cols = st.columns(2)
        for index, (variable, result) in enumerate(sensitivity.items()):
            with sensitivity_cols[index % 2]:
                st.plotly_chart(
                    _pichia_sensitivity_chart(result, PICHIA_VARIABLE_LABELS.get(variable, variable)), width="stretch"
                )

    with st.expander("鞍点/岭线判别（曲面到底是不是真的有峰）", expanded=(case == "boundary")):
        st.caption("经典响应面 canonical analysis：抛开网格搜索给出的「测试范围内最好的点」，只看拟合公式本身的曲率，判断曲面真实的形状。")
        canonical = canonical_analysis(fit, ccd_rows)
        st.dataframe(
            _pichia_canonical_table(canonical),
            width="stretch",
            hide_index=True,
            column_config={column: st.column_config.Column(help=text) for column, text in PICHIA_CANONICAL_HELP.items()},
        )
        st.plotly_chart(_pichia_canonical_eigenvalue_chart(canonical), width="stretch")
        if case == "boundary":
            if canonical.classification in ("ridge", "flat"):
                st.info(
                    "这和上面「最优点卡边界」的现象是一致的：曲面在活跃变量的某个方向上几乎没有曲率（呈岭线/近似平坦），"
                    "网格搜索自然找不到真正的封顶，只能停在测试范围的边缘——不是这个点真的最优，是没有曲率能定出唯一最优点。"
                )
            elif not canonical.stationary_point_in_tested_range:
                point_text = "、".join(
                    f"{PICHIA_VARIABLE_LABELS.get(name, name)}={_num(value)}" for name, value in canonical.stationary_point.items()
                )
                st.info(f"无约束理论最优点在 {point_text}，已经超出本轮测试范围——和「最优点卡边界」的现象一致，建议下一轮把测试范围扩大到能覆盖这个点。")
        elif case == "normal" and canonical.classification == "maximum" and canonical.stationary_point_in_tested_range:
            st.caption("曲面呈真极大值、理论最优点也落在测试范围内——和「一切正常」的结论一致，这个联合最优点值得安排验证批次。")

    if od_fit is not None and od_threshold is not None:
        with st.expander("产量 + OD600 联合优化（满意度）", expanded=(case == "od_infeasible")):
            st.caption("不把 OD600 当成硬性的「是/否」过滤，而是把产量和 OD600 可行性都打分（0~1），找同时兼顾两者的折中点，并和「只看产量」的最优点对比。")
            desirability = optimize_joint_desirability(fit, od_fit, ccd_rows, od_threshold)
            st.dataframe(
                _pichia_desirability_table(desirability),
                width="stretch",
                hide_index=True,
                column_config=_pichia_desirability_column_config(desirability),
            )
            st.plotly_chart(_pichia_desirability_tradeoff_chart(desirability, od_threshold), width="stretch")
            if case == "od_infeasible":
                yield_cost = (
                    1.0 - desirability.predicted_yield / desirability.pure_yield_optimum_value
                    if desirability.pure_yield_optimum_value
                    else 0.0
                )
                feasibility_note = "可行" if desirability.od_desirability >= 1.0 - 1e-6 else "更接近可行（仍未完全达标）"
                st.success(
                    f"这就是「折中点」：把最优条件从纯产量最优点换成联合满意度最优点，预测产量下降约 {yield_cost:.1%}，"
                    f"换来预测 OD600 从不可行变为{feasibility_note}——建议下一轮把这个点也纳入验证批次，而不是只验证纯产量最优点。"
                )
            elif desirability.od_desirability >= 1.0 - 1e-6:
                st.caption("纯产量最优点本身已经满足 OD600 可行性，联合满意度最优点和它基本重合，不存在实质性权衡。")

    if case == "normal":
        st.caption("以上分析（含上方最优点的置信区间）互相印证：模型形式没问题、最优点不在边界、（若已检验）OD600 可行——建议按前面「下一步建议」安排验证批次。")

def _pichia_predicted_vs_actual_chart(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    title: str,
    x_title: str,
    y_title: str,
    point_name: str,
    hovertemplate: str,
    height: int = 340,
    text: pd.Series | None = None,
    error_y: np.ndarray | None = None,
) -> go.Figure:
    """Shared predicted-vs-actual scatter + y=x reference line -- the same
    diagnostic shape (a fitted model's own predictions on points it's
    evaluated against, however those predictions were obtained) whether
    the caller is the CCD fit's plain residual chart or the GP leave-one-out
    CV's held-out-prediction chart; only the labels/data/error bars differ."""
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=PICHIA_MUTED_COLOR, dash="dash"), name="y=x", hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode="markers",
            marker=dict(color=PICHIA_ACCENT_COLOR, size=8),
            text=text,
            error_y=dict(type="data", array=error_y, visible=True, color=PICHIA_MUTED_COLOR) if error_y is not None else None,
            hovertemplate=hovertemplate,
            name=point_name,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=x_title,
        yaxis_title=y_title,
        margin=dict(t=50, b=40, l=60, r=30),
        height=height,
    )
    return fig

def _pichia_response_surface_residual_chart(fit: Any, df: pd.DataFrame) -> go.Figure:
    predicted = evaluate_response_surface(fit, df)
    actual = df[PICHIA_TARGET_COL].to_numpy(dtype=float)
    return _pichia_predicted_vs_actual_chart(
        actual,
        predicted,
        title="预测值 vs 实测值（残差诊断，越贴近虚线越好）",
        x_title="实测产量",
        y_title="预测产量",
        point_name="CCD 样本",
        hovertemplate="%{text}<br>实测: %{x:.4g}<br>预测: %{y:.4g}<extra></extra>",
        text=df["run_id"] if "run_id" in df.columns else None,
    )

def _pichia_response_surface_curve_chart(fit: Any, df: pd.DataFrame, variable: str) -> go.Figure:
    x_values = np.linspace(float(df[variable].min()), float(df[variable].max()), 61)
    predicted = evaluate_response_surface(fit, pd.DataFrame({variable: x_values}))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=predicted, mode="lines", line=dict(color=PICHIA_ACCENT_COLOR, width=2), name="拟合曲线"))
    fig.add_trace(
        go.Scatter(
            x=df[variable], y=df[PICHIA_TARGET_COL], mode="markers", marker=dict(color="#e6e4da", size=8), name="实测点"
        )
    )
    fig.update_layout(
        title=f"{PICHIA_VARIABLE_LABELS.get(variable, variable)} 响应曲线",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=PICHIA_VARIABLE_LABELS.get(variable, variable),
        yaxis_title="产量",
        margin=dict(t=50, b=40, l=60, r=30),
        height=340,
    )
    return fig

def _pichia_response_surface_3d_chart(
    fit: Any, df: pd.DataFrame, x_variable: str, y_variable: str, colorscale: list[list[Any]]
) -> go.Figure:
    """3D twin of _pichia_response_surface_contour_chart, same underlying
    response_surface_grid data -- a surface gives the curvature/peak shape
    away at a glance (is it a sharp peak, a broad plateau, a ridge?), which a
    2D contour states just as correctly but less immediately; shown side by
    side with the contour rather than replacing it, since reading exact
    values back off a rotated 3D surface is much harder than off a contour's
    color scale and axis lines."""
    grid = response_surface_grid(fit, df, x_variable, y_variable)
    x_label = PICHIA_VARIABLE_LABELS.get(x_variable, x_variable)
    y_label = PICHIA_VARIABLE_LABELS.get(y_variable, y_variable)
    fig = go.Figure(
        go.Surface(
            x=grid["x_values"],
            y=grid["y_values"],
            z=grid["z"],
            colorscale=colorscale,
            showscale=False,
            opacity=0.92,
            hovertemplate=f"{x_label}: %{{x:.3g}}<br>{y_label}: %{{y:.3g}}<br>预测产量: %{{z:.4g}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=df[x_variable],
            y=df[y_variable],
            z=df[PICHIA_TARGET_COL],
            mode="markers",
            marker=dict(color="#e6e4da", size=4),
            name="实测点",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[fit.optimum[x_variable]],
            y=[fit.optimum[y_variable]],
            z=[fit.predicted_optimum],
            mode="markers",
            marker=dict(color="#e66767", size=6, symbol="diamond"),
            name="联合最优点",
        )
    )
    fig.update_layout(
        title=f"{x_label} × {y_label}（3D）",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title="预测产量",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=50, b=10, l=10, r=10),
        height=420,
    )
    return fig

def _pichia_response_surface_contour_chart(
    fit: Any,
    df: pd.DataFrame,
    x_variable: str,
    y_variable: str,
    colorscale: list[list[Any]],
    od_fit: Any | None = None,
    od_threshold: float | None = None,
) -> go.Figure:
    grid = response_surface_grid(fit, df, x_variable, y_variable)
    x_label = PICHIA_VARIABLE_LABELS.get(x_variable, x_variable)
    y_label = PICHIA_VARIABLE_LABELS.get(y_variable, y_variable)
    fig = go.Figure(
        go.Contour(
            x=grid["x_values"],
            y=grid["y_values"],
            z=grid["z"],
            colorscale=colorscale,
            colorbar=dict(title="预测产量"),
            hovertemplate=f"{x_label}: %{{x:.3g}}<br>{y_label}: %{{y:.3g}}<br>预测产量: %{{z:.4g}}<extra></extra>",
        )
    )
    if od_fit is not None and od_threshold is not None:
        # binarize OD600's own fitted surface at the feasibility threshold and
        # contour *that* at level 0.5 -- draws exactly the feasible/infeasible
        # boundary regardless of OD600's actual value range, rather than
        # fighting Plotly's contour-level picking to land a line precisely on
        # od_threshold in the raw OD600 units.
        od_grid = response_surface_grid(od_fit, df, x_variable, y_variable, fixed_at=grid["fixed_at"])
        feasible_mask = (od_grid["z"] >= od_threshold).astype(float)
        fig.add_trace(
            go.Contour(
                x=od_grid["x_values"],
                y=od_grid["y_values"],
                z=feasible_mask,
                contours=dict(start=0.5, end=0.5, size=0.5, coloring="lines"),
                line=dict(color="#ffffff", width=2.5, dash="dot"),
                showscale=False,
                name=f"OD600≥{od_threshold:.3g} 可行边界",
                hovertemplate=f"OD600 可行边界（阈值 {od_threshold:.3g}）<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=df[x_variable],
            y=df[y_variable],
            mode="markers",
            marker=dict(color="#e6e4da", size=7, symbol="x"),
            name="实测点",
            hoverinfo="skip",
        )
    )
    title = f"{x_label} × {y_label}"
    other_note = "、".join(f"{PICHIA_VARIABLE_LABELS.get(name, name)}={value:g}" for name, value in grid["fixed_at"].items())
    if other_note:
        title += f"（其余变量固定：{other_note}）"
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=x_label,
        yaxis_title=y_label,
        margin=dict(t=60, b=40, l=60, r=30),
        height=380,
    )
    return fig
