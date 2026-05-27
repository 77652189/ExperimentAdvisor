from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_advisor.ingestion.run_level import TARGET_COL, build_run_level_dataset, training_view
from experiment_advisor.recommendation.service import compare_recommenders
from experiment_advisor.report import generate_recommendation_report

METHOD_LABELS = {
    "standard_bo_ei": "标准 GP-BO（EI）",
    "xgp_bo_ei": "XGBoost + GP 残差 BO（EI）",
}

METHOD_EXPLANATIONS = {
    "standard_bo_ei": "当前主推荐方法。Gaussian Process 直接拟合产量，并用 EI 选择预期改进较大的候选。",
    "xgp_bo_ei": "候选参考方法。XGBoost 预测产量均值，GP 只拟合 XGBoost 残差，用于补充查看非线性模型下的建议。",
}

FEATURE_LABELS = {
    "temperature_growth_phase_c": "生长期温度",
    "temperature_shift_time_h": "温度切换时间",
    "temperature_production_phase_c": "生产期温度",
    "temperature_c_mean": "平均温度",
    "ph_mean": "平均 pH",
    "feed1_total_ml": "Feed 1 总量",
    "feed1_start_time_h": "Feed 1 开始时间",
    "feed1_end_time_h": "Feed 1 结束时间",
    "feed1_ml_final": "Feed 1 最终量",
    "feed2_total_ml": "Feed 2 总量",
    "feed2_start_time_h": "Feed 2 开始时间",
    "feed2_end_time_h": "Feed 2 结束时间",
    "feed2_ml_final": "Feed 2 最终量",
    "lactose_total_ml": "乳糖总量",
    "lactose_first_add_time_h": "乳糖首次添加时间",
    "lactose_last_add_time_h": "乳糖末次添加时间",
    "lactose_ml_final": "乳糖最终补加量",
    "base_ml_final": "碱液最终补加量",
    "od600_max": "OD600 最大值",
    "fermentation_duration_h": "发酵时长",
}

UNCERTAINTY_LABELS = {
    "gp_posterior_std": "GP 后验标准差",
    "xgp_gp_residual_std": "GP 残差后验标准差",
}

RISK_LABELS = {"low": "低", "medium": "中", "high": "高"}
FLAG_LABELS = {
    "far_from_history": "远离历史实验",
    "near_search_boundary": "接近参数边界",
    "high_residual_uncertainty": "残差不确定性较高",
}


def _load_default_dataset() -> pd.DataFrame:
    return build_run_level_dataset(
        source_dir=PROJECT_ROOT / "data" / "csv_from_excel",
        output_path=PROJECT_ROOT / "data" / "final" / "run_level_modeling_dataset.csv",
    )


def _training_data(df: pd.DataFrame) -> pd.DataFrame:
    if "exclude_from_training" in df.columns:
        return training_view(df, TARGET_COL)
    return df.dropna(subset=[TARGET_COL])


def _name(value: str) -> str:
    return FEATURE_LABELS.get(value, value)


def _num(value: Any) -> Any:
    return round(float(value), 4) if isinstance(value, int | float) else value


def _flags(flags: list[str] | None) -> str:
    if not flags:
        return "无明显风险标记"
    return "；".join(FLAG_LABELS.get(flag, flag) for flag in flags)


def _candidate_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        row = {
            "排序": item.get("rank"),
            "预测产量": _num(item.get("predicted_yield")),
            "XGBoost 均值": _num(item.get("xgb_prediction")),
            "GP 残差修正": _num(item.get("gp_residual_mean")),
            "不确定性": _num(item.get("model_uncertainty")),
            "不确定性定义": UNCERTAINTY_LABELS.get(item.get("uncertainty_type"), item.get("uncertainty_type")),
            "历史距离": _num(item.get("history_distance")),
            "边界风险": _num(item.get("boundary_risk")),
            "风险等级": RISK_LABELS.get(item.get("risk_level"), item.get("risk_level")),
            "风险标记": _flags(item.get("quality_flags")),
            "推荐得分": _num(item.get("acquisition_score")),
        }
        for key, value in item.get("params", {}).items():
            row[_name(key)] = _num(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _overview(df: pd.DataFrame) -> None:
    train_df = _training_data(df)
    cols = st.columns(4)
    cols[0].metric("总 run 数", len(df))
    cols[1].metric("可训练 run 数", len(train_df))
    cols[2].metric("排除 run 数", len(df) - len(train_df))
    cols[3].metric("目标字段", TARGET_COL)

    if not train_df.empty:
        y = pd.to_numeric(train_df[TARGET_COL], errors="coerce").dropna()
        stats = st.columns(4)
        stats[0].metric("历史最低产量", f"{y.min():.3g}")
        stats[1].metric("历史中位产量", f"{y.median():.3g}")
        stats[2].metric("历史最高产量", f"{y.max():.3g}")
        stats[3].metric("历史平均产量", f"{y.mean():.3g}")

    with st.expander("训练数据筛选说明", expanded=False):
        st.write("缺少产量、污染、异常或失败备注的 run 会保留在数据表中用于审计，但不会参与模型训练。")
        if "exclusion_reason" in df.columns:
            counts = df["exclusion_reason"].fillna("").replace("", "可训练").value_counts().reset_index()
            counts.columns = ["原因", "数量"]
            st.dataframe(counts, width="stretch", hide_index=True)

    with st.expander("数据预览", expanded=False):
        st.dataframe(df.head(30), width="stretch")


def _nearest_history(df: pd.DataFrame, params: dict[str, float], top_n: int = 3) -> pd.DataFrame:
    features = [name for name in params if name in df.columns]
    history = _training_data(df).dropna(subset=features)
    if history.empty or not features:
        return pd.DataFrame()
    means = history[features].mean()
    stds = history[features].std(ddof=0).replace(0, 1.0)
    candidate = pd.Series(params)[features]
    distances = (((history[features] - means) / stds - ((candidate - means) / stds)) ** 2).sum(axis=1) ** 0.5
    nearest = history.assign(相似距离=distances).sort_values("相似距离").head(top_n)
    columns = [col for col in ["fermenter_run_id", "experiment_date", TARGET_COL, "相似距离", *features] if col in nearest.columns]
    return nearest[columns].rename(columns={col: _name(col) for col in columns})


def _method_block(method: str, items: list[dict[str, Any]]) -> None:
    st.subheader(METHOD_LABELS.get(method, method))
    st.caption(METHOD_EXPLANATIONS.get(method, ""))
    if items:
        st.dataframe(_candidate_table(items), width="stretch", hide_index=True)
    else:
        st.info("暂无结果。")


def _standard_bo_summary(top: dict[str, Any]) -> None:
    rows = [
        {"项目": "预测产量", "数值": _num(top.get("predicted_yield")), "说明": "标准 GP 模型对候选点的产量均值预测。"},
        {"项目": "GP 后验标准差", "数值": _num(top.get("model_uncertainty")), "说明": "标准 GP 对该候选点预测不确定性的估计。"},
        {"项目": "EI 推荐得分", "数值": _num(top.get("acquisition_score")), "说明": "Expected Improvement，越高表示相对当前历史最好值越值得尝试。"},
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _xgp_summary(top: dict[str, Any]) -> None:
    rows = [
        {"项目": "XGBoost 均值预测", "数值": _num(top.get("xgb_prediction")), "说明": "学习非线性产量结构。"},
        {"项目": "GP 残差修正", "数值": _num(top.get("gp_residual_mean")), "说明": "对 XGBoost 未解释残差做局部修正。"},
        {"项目": "最终预测产量", "数值": _num(top.get("predicted_yield")), "说明": "XGBoost 均值 + GP 残差修正。"},
        {"项目": "GP 残差后验标准差", "数值": _num(top.get("model_uncertainty")), "说明": "残差模型的不确定性。"},
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _gp_health(items: list[dict[str, Any]]) -> None:
    health = (items[0].get("gp_health") if items else None) or {}
    if not health:
        st.info("当前结果没有残差 GP 健康诊断。")
        return

    cols = st.columns(4)
    cols[0].metric("残差均值", _num(health.get("residual_mean")))
    cols[1].metric("残差标准差", _num(health.get("residual_std")))
    cols[2].metric("最大绝对残差", _num(health.get("residual_max_abs")))
    cols[3].metric("不确定性种类数", health.get("candidate_uncertainty_unique_rounded"))

    gp_features = health.get("gp_feature_cols") or []
    st.write("GP 实际使用的特征")
    st.dataframe(pd.DataFrame({"显示名": [_name(col) for col in gp_features], "字段名": gp_features}), width="stretch", hide_index=True)

    if health.get("candidate_uncertainty_degenerate"):
        st.error("候选点不确定性几乎完全相同，残差 GP 可能退化。")
    else:
        st.success("候选点不确定性可以区分不同候选，残差 GP 未表现为常数退化。")

    st.dataframe(
        pd.DataFrame(
            [
                {"指标": "候选不确定性最小值", "数值": _num(health.get("candidate_uncertainty_min"))},
                {"指标": "候选不确定性最大值", "数值": _num(health.get("candidate_uncertainty_max"))},
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    warnings = health.get("warnings") or []
    if warnings:
        st.warning("GP 训练出现警告，建议谨慎解释残差不确定性。")
        for message in warnings:
            st.caption(message)

    with st.expander("GP kernel 详情", expanded=False):
        st.code(str(health.get("kernel", "")), language="text")


def _metric_explanations() -> None:
    st.markdown("### 主推荐：标准 GP-BO（EI）")
    standard_rows = [
        {"指标": "预测产量", "如何产生": "GP 直接拟合历史产量后，对候选点输出 posterior mean。", "如何解读": "模型预测值，不是实测值。"},
        {"指标": "GP 后验标准差", "如何产生": "GP 对候选点预测分布的 posterior std。", "如何解读": "越高表示模型在该区域越不确定。"},
        {"指标": "EI 推荐得分", "如何产生": "Expected Improvement，综合预测均值、后验标准差和当前历史最好值。", "如何解读": "越高表示越有可能带来改进。"},
    ]
    st.dataframe(pd.DataFrame(standard_rows), width="stretch", hide_index=True)

    st.markdown("### 候选参考：XGBoost + GP 残差 BO（EI）")
    xgp_rows = [
        {"指标": "XGBoost 均值", "如何产生": "XGBoost 使用全部搜索特征预测产量。", "如何解读": "反映非线性模型学到的主要产量结构。"},
        {"指标": "GP 残差修正", "如何产生": "GP 只拟合 XGBoost 的训练残差。", "如何解读": "正值表示局部上调预测，负值表示局部下调预测。"},
        {"指标": "GP 残差后验标准差", "如何产生": "残差 GP 的 posterior std。", "如何解读": "只表示残差模型不确定性，不是湿实验置信区间。"},
        {"指标": "历史距离", "如何产生": "候选点到最近历史实验的标准化距离。", "如何解读": "越高越像外推，工艺采纳风险越高。"},
        {"指标": "边界风险", "如何产生": "候选点靠近搜索空间上下限的程度。", "如何解读": "越高越靠边界，需要检查是否可执行。"},
    ]
    st.dataframe(pd.DataFrame(xgp_rows), width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="发酵工艺优化推荐系统", layout="wide")
    st.title("发酵工艺优化推荐系统")
    st.caption("主方法：标准 GP-BO（EI）。候选参考：XGBoost + GP 残差 BO（EI）。")

    with st.sidebar:
        st.header("数据入口")
        source = st.radio("选择数据来源", ["使用 data/csv_from_excel", "上传 run-level CSV"])
        top_k = st.slider("推荐数量", min_value=1, max_value=10, value=5)
        run_button = st.button("运行推荐", type="primary", width="stretch")
        st.divider()
        st.markdown("### 默认决策")
        st.write("默认采用标准 GP-BO（EI）作为主推荐；XGBoost + GP 残差 BO（EI）只作为候选参考。")

    uploaded_file = st.file_uploader("上传已整理好的 run-level CSV", type=["csv"]) if source == "上传 run-level CSV" else None

    try:
        df = _load_default_dataset() if source == "使用 data/csv_from_excel" else pd.read_csv(uploaded_file) if uploaded_file else None
    except Exception as exc:
        st.error(f"数据加载失败：{exc}")
        return

    if df is None:
        st.info("请上传 run-level CSV，或选择默认数据目录。")
        return
    if TARGET_COL not in df.columns:
        st.error(f"数据缺少目标字段 `{TARGET_COL}`。")
        return

    _overview(df)
    if not run_button:
        st.info("点击左侧“运行推荐”后，系统会训练模型、生成候选点，并输出诊断信息。")
        return

    with st.spinner("正在训练标准 GP-BO、XGP 候选模型并生成推荐..."):
        comparison = compare_recommenders(df, top_k=top_k)
        report_path = PROJECT_ROOT / "summary" / "recommendation_report.md"
        report_md = generate_recommendation_report(comparison, output_path=report_path)

    selected_method = comparison.get("selected_method", "standard_bo_ei")
    selected = comparison.get("selected_recommendations", [])
    xgp_items = comparison.get("recommendations", {}).get("xgp_bo_ei", [])
    decision = comparison.get("decision", {})

    st.success("推荐已生成")
    cols = st.columns(3)
    cols[0].metric("主推荐方法", METHOD_LABELS.get(selected_method, selected_method))
    cols[1].metric("需要人工审议", "是" if decision.get("needs_human_review") else "否")
    cols[2].metric("训练 run 数", comparison.get("n_training_rows"))
    st.info(decision.get("reason", "默认采用标准 GP-BO（EI）。"))

    tabs = st.tabs(["主推荐", "XGP 候选", "残差 GP 健康", "指标说明", "Markdown 报告"])
    with tabs[0]:
        _method_block(selected_method, selected)
        if selected:
            _standard_bo_summary(selected[0])
            nearest = _nearest_history(df, selected[0].get("params", {}))
            if not nearest.empty:
                st.write("最相似的历史实验")
                st.dataframe(nearest, width="stretch", hide_index=True)
    with tabs[1]:
        _method_block("xgp_bo_ei", xgp_items)
        if xgp_items:
            _xgp_summary(xgp_items[0])
    with tabs[2]:
        _gp_health(xgp_items)
    with tabs[3]:
        _metric_explanations()
    with tabs[4]:
        st.markdown(report_md)
        st.caption(f"报告已写入：{report_path}")


if __name__ == "__main__":
    main()
