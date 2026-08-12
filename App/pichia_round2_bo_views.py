"""Round 2 Bayesian-optimisation views: the recommendation batch's table,
partial-dependence charts, leave-one-out cross-validation block and narrative
verdicts.

Reuses _pichia_render_verdicts / _pichia_predicted_vs_actual_chart from
App.pichia_round2_surface_views rather than keeping second copies -- both were
extracted for exactly this sharing. Unlike the surface views this module does
touch session state, but only through the cv_session_key its caller passes in;
it never hard-codes a key of its own.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_advisor.recommendation.round2_design import (
    gp_leave_one_out_cv,
    gp_partial_dependence,
    summarize_bo_recommendation,
)
from App.pichia_common import (
    PICHIA_ACCENT_COLOR,
    PICHIA_CONTINUOUS_BOUNDS,
    PICHIA_OD_COL,
    PICHIA_TARGET_COL,
    PICHIA_VARIABLE_LABELS,
    _num,
    _pichia_variable_display,
)
from App.pichia_round2_surface_views import (
    _pichia_predicted_vs_actual_chart,
    _pichia_render_verdicts,
)

def _pichia_gp_pdp_chart(model: Any, feature_cols: list[str], variable: str, anchor: dict[str, Any]) -> go.Figure:
    """Partial-dependence read of an already-fitted BO GP (see
    round2_design.gp_partial_dependence -- never refits): sweeps `variable`
    across its full original bounds holding every other feature at anchor
    (the recommendation this chart is attached to), so the shaded band shows
    where the GP is actually confident (near training points) versus
    guessing (gaps between them)."""
    lower, upper = PICHIA_CONTINUOUS_BOUNDS[variable]
    pdp = gp_partial_dependence(model, feature_cols, variable, anchor, lower=lower, upper=upper, resolution=41)
    x_values = list(pdp["x"])
    mean = pdp["mean"]
    band = 1.96 * pdp["std"]  # ~95% band under the GP's own Gaussian posterior
    upper_band = list(mean + band)
    lower_band = list(mean - band)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values + x_values[::-1],
            y=upper_band + lower_band[::-1],
            fill="toself",
            fillcolor="rgba(57,135,229,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% 置信区间",
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=x_values, y=mean, mode="lines", line=dict(color=PICHIA_ACCENT_COLOR, width=2), name="GP 预测均值"))
    if variable in anchor:
        fig.add_trace(
            go.Scatter(
                x=[anchor[variable]],
                y=[anchor.get("predicted_yield")],
                mode="markers",
                marker=dict(color="#e66767", size=11, symbol="star"),
                name="当前推荐点",
            )
        )
    fig.update_layout(
        title=f"{PICHIA_VARIABLE_LABELS.get(variable, variable)} 偏依赖（其余变量固定在推荐点）",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=PICHIA_VARIABLE_LABELS.get(variable, variable),
        yaxis_title="预测产量",
        margin=dict(t=50, b=40, l=60, r=30),
        height=340,
    )
    return fig

def _pichia_render_bo_gp_pdp(bo_result: dict[str, Any], active_variables: list[str]) -> None:
    model = bo_result.get("yield_model")
    feature_cols = bo_result.get("feature_cols")
    if model is None or not feature_cols or not active_variables:
        return
    anchor = bo_result["recommendations"][0]
    with st.expander("GP 偏依赖图（模型在没测过的区域有多不确定）", expanded=False):
        st.caption("阴影是 GP 自己算出的 95% 置信区间，不是另外估的——区间越宽，说明这一段离训练数据越远、模型越没底。")
        pdp_cols = st.columns(2)
        for index, variable in enumerate(active_variables):
            with pdp_cols[index % 2]:
                st.plotly_chart(_pichia_gp_pdp_chart(model, feature_cols, variable, anchor), width="stretch")

def _pichia_render_bo_verdicts(bo_result: dict[str, Any], cv_result: dict[str, Any] | None) -> None:
    """Narrative read of a BO recommendation batch (summarize_bo_recommendation's
    output), analogous to _pichia_render_response_surface_verdicts for the CCD
    fit -- unlike that one, the messages here are already fully Chinese (no
    bare variable-name placeholders to translate), since round2_design.py
    already writes them that way for this function (see its own docstring)."""
    yield_cv = cv_result["yield"] if cv_result else None
    od_cv = cv_result["od"] if cv_result else None
    verdicts = summarize_bo_recommendation(bo_result, yield_cv=yield_cv, od_cv=od_cv)
    _pichia_render_verdicts(verdicts)

def _pichia_bo_cv_residual_chart(cv: dict[str, Any], label: str) -> go.Figure:
    return _pichia_predicted_vs_actual_chart(
        cv["actual"],
        cv["predicted"],
        title=f"{label}：留一法预测 vs 实测（误差线是 GP 自己的预测标准差）",
        x_title="实测",
        y_title="留一法预测",
        point_name="留一法预测",
        hovertemplate="实测: %{x:.4g}<br>留一法预测: %{y:.4g}<extra></extra>",
        height=320,
        error_y=cv["predicted_std"],
    )

PICHIA_BO_CV_HELP: dict[str, str] = {
    "Q²": "留一法交叉验证的预测能力：每次把一个样本从训练集里拿掉、用剩下的重新拟合再预测这个点，和实测值比较算出的类 R² 指标。"
    "Q²≥0.5 一般认为有实用的预测能力；Q²<0 说明比直接猜训练集均值还差，模型目前不可信。",
    "留一法 RMSE": "留一法预测误差的均方根，单位和该指标本身一致——越小说明模型对没见过的新条件预测得越准。",
}

PICHIA_BO_CV_TARGETS: list[tuple[str, str]] = [("产量", "yield"), ("OD600", "od")]

def _pichia_render_bo_cv_result(cv_result: dict[str, Any]) -> None:
    cols = st.columns(2)
    for col, (label, key) in zip(cols, PICHIA_BO_CV_TARGETS):
        cv = cv_result[key]
        with col:
            st.metric("Q²", _num(cv["q_squared"]), help=PICHIA_BO_CV_HELP["Q²"])
            st.metric("留一法 RMSE", _num(cv["rmse"]), help=PICHIA_BO_CV_HELP["留一法 RMSE"])
            st.plotly_chart(_pichia_bo_cv_residual_chart(cv, label), width="stretch")

def _pichia_bo_cv_training_rows(train_df: pd.DataFrame) -> pd.DataFrame:
    """Rows with both target columns present -- the same row selection
    recommend_round2_bo_batch itself uses to train yield_model/od_model.
    Cross-validation must filter on both, not just whichever single target
    it's validating this call: whenever backfill is partial (yield filled
    but OD600 not, or vice versa), filtering on only one column would
    validate against a different, larger row set than the one that
    actually produced the deployed model."""
    return train_df.dropna(subset=[PICHIA_TARGET_COL, PICHIA_OD_COL])

def _pichia_render_bo_recommendation_section(
    bo_result: dict[str, Any],
    active_variables: list[str],
    train_df: pd.DataFrame,
    cv_session_key: str,
) -> None:
    """Recommendation table + narrative read + GP partial-dependence + an
    on-demand leave-one-out cross-validation -- shared by both BO entry
    points (Round-1-only and the combined Round1+Round2 dataset), which
    otherwise built an identical table from an identical result-dict shape.
    Consolidating this avoids the two call sites silently drifting apart on
    formatting once there's new shared content (the verdicts/CV) to add."""
    rows = []
    for rec in bo_result["recommendations"]:
        row = _pichia_variable_display(rec)
        row["预测产量"] = _num(rec.get("predicted_yield"))
        row["预测产量标准差"] = _num(rec.get("predicted_yield_std"))
        row["预测 OD600"] = _num(rec.get("predicted_od600"))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("模型校验（留一法交叉验证，判断这个 GP 能信到什么程度）", expanded=False):
        st.caption(
            "对训练数据逐个样本做「留一法」：把这个样本从训练集里拿掉，用剩下的重新拟合 GP，预测被拿掉的这个点，"
            "再和它的实测值比较——这样得到的误差才是真正的「预测」误差，不是模型看过这个点之后回头报的拟合误差。"
            "样本量不大，但每次点击都会重新拟合多次模型，需要几秒到几十秒，不会自动运行。"
        )
        if st.button("运行交叉验证", key=f"{cv_session_key}_run_button"):
            cv_train_df = _pichia_bo_cv_training_rows(train_df)
            try:
                yield_cv = gp_leave_one_out_cv(cv_train_df, bo_result["feature_cols"], PICHIA_TARGET_COL)
                od_cv = gp_leave_one_out_cv(cv_train_df, bo_result["feature_cols"], PICHIA_OD_COL)
            except ImportError:
                st.warning("当前环境未安装 torch/botorch/gpytorch，无法运行交叉验证。")
            except ValueError as exc:
                st.warning(f"交叉验证暂时无法运行：{exc}")
            except Exception as exc:
                st.warning(f"交叉验证运行失败（样本量很小时，某一折的模型拟合可能数值不稳定）：{exc}")
            else:
                st.session_state[cv_session_key] = {"yield": yield_cv, "od": od_cv}

        # read session_state here, after the button block above may have
        # just written to it -- lets a fresh result show immediately on the
        # same click without needing st.rerun() (which would otherwise
        # re-execute this whole page, including refitting the CCD model for
        # the combined-data call site, just to re-read a variable that was
        # bound too early).
        cv_result = st.session_state.get(cv_session_key)
        if cv_result:
            _pichia_render_bo_cv_result(cv_result)

    _pichia_render_bo_verdicts(bo_result, st.session_state.get(cv_session_key))
    _pichia_render_bo_gp_pdp(bo_result, active_variables)
