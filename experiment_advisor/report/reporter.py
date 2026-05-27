from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int | float):
        return f"{float(value):.3g}"
    return str(value)


def _params_text(item: dict[str, Any]) -> str:
    return ", ".join(f"{key}={_fmt(value)}" for key, value in item.get("params", {}).items())


def _recommendation_table(items: list[dict]) -> list[str]:
    lines = [
        "| Rank | 预测产量 | XGBoost 均值 | GP 残差修正 | 不确定性 | 历史距离 | 边界风险 | 风险等级 | 推荐得分 | 参数 |",
        "|------|----------|--------------|--------------|----------|----------|----------|----------|----------|------|",
    ]
    for item in items:
        lines.append(
            "| {rank} | {pred} | {xgb} | {resid} | {unc} | {dist} | {boundary} | {risk} | {score} | {params} |".format(
                rank=item.get("rank", "-"),
                pred=_fmt(item.get("predicted_yield")),
                xgb=_fmt(item.get("xgb_prediction")),
                resid=_fmt(item.get("gp_residual_mean")),
                unc=_fmt(item.get("model_uncertainty")),
                dist=_fmt(item.get("history_distance")),
                boundary=_fmt(item.get("boundary_risk")),
                risk=item.get("risk_level", "-"),
                score=_fmt(item.get("acquisition_score")),
                params=_params_text(item),
            )
        )
    return lines


def _xgp_health_lines(items: list[dict]) -> list[str]:
    health = (items[0].get("gp_health") if items else None) or {}
    if not health:
        return ["暂无残差 GP 健康诊断。"]
    gp_features = ", ".join(health.get("gp_feature_cols", []))
    lines = [
        f"- GP 使用特征：{gp_features}",
        f"- 训练残差均值：{_fmt(health.get('residual_mean'))}",
        f"- 训练残差标准差：{_fmt(health.get('residual_std'))}",
        f"- 最大绝对残差：{_fmt(health.get('residual_max_abs'))}",
        f"- 候选不确定性范围：{_fmt(health.get('candidate_uncertainty_min'))} 到 {_fmt(health.get('candidate_uncertainty_max'))}",
        f"- 不确定性是否退化为常数：{health.get('candidate_uncertainty_degenerate')}",
    ]
    warnings = health.get("warnings") or []
    if warnings:
        lines.append("- GP 训练警告：")
        lines.extend(f"  - {message}" for message in warnings)
    return lines


def generate_recommendation_report(
    comparison_result: dict[str, Any] | list[dict],
    offline_analysis: dict | None = None,
    output_path: str | Path | None = None,
) -> str:
    """生成 XGP-first 推荐报告。"""

    if isinstance(comparison_result, list):
        comparison = {
            "n_training_rows": None,
            "model_metrics": {},
            "recommendations": {"xgp_bo_ei": comparison_result},
            "selected_method": "xgp_bo_ei",
            "selected_recommendations": comparison_result,
        }
    else:
        comparison = comparison_result

    decision = comparison.get("decision", {})
    selected_method = comparison.get("selected_method", "xgp_bo_ei")
    selected = comparison.get("selected_recommendations") or comparison.get("recommendations", {}).get(selected_method, [])
    xgp_items = comparison.get("recommendations", {}).get("xgp_bo_ei", selected)

    lines = [
        "# 发酵工艺优化推荐报告",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 数据与默认方法",
        f"- 训练 run 数：{_fmt(comparison.get('n_training_rows'))}",
        f"- 目标字段：{comparison.get('target_col', 'yield_g_per_l')}",
        f"- 默认主推荐：{selected_method}",
        f"- 是否需要人工审议：{decision.get('needs_human_review', False)}",
        f"- 决策说明：{decision.get('reason', '默认采用 xgp_bo_ei。')}",
        "",
        "## 主推荐",
        *_recommendation_table(selected),
        "",
        "## XGP 机制说明",
        "- XGBoost 负责学习参数到产量的非线性均值。",
        "- GP 只拟合 XGBoost 的训练残差，并提供残差后验标准差。",
        "- 最终预测产量 = XGBoost 均值预测 + GP 残差修正。",
        "- 历史距离、边界风险和残差不确定性用于判断候选是否需要人工审议。",
        "",
        "## 残差 GP 健康检查",
        *_xgp_health_lines(xgp_items),
        "",
        "## 方法对照",
    ]

    for method, items in comparison.get("recommendations", {}).items():
        lines.extend(["", f"### {method}", *_recommendation_table(items)])

    metrics = comparison.get("model_metrics", {})
    if metrics:
        lines.extend(["", "## 模型验证指标", "| 模型 | LOO MAE | LOO R2 |", "|------|---------|--------|"])
        for name, values in metrics.items():
            lines.append(f"| {name} | {_fmt(values.get('mae_loocv'))} | {_fmt(values.get('r2_loocv'))} |")

    lines.extend(
        [
            "",
            "## 风险说明",
            "- XGP 的不确定性是残差 GP 后验标准差，不是严格湿实验置信区间。",
            "- 历史距离高表示候选点更像外推，需要工艺可行性复核。",
            "- 边界风险高表示候选点靠近搜索空间边界，不建议直接视作稳妥方案。",
            "- 标准 GP-BO 只作为对照，不是默认决策。",
        ]
    )

    if offline_analysis and offline_analysis.get("shap_importance"):
        lines.extend(["", "## 关键变量影响"])
        for rank, (feature, value) in enumerate(offline_analysis["shap_importance"].items(), start=1):
            lines.append(f"{rank}. {feature}: SHAP {_fmt(value)}")

    markdown = "\n".join(lines) + "\n"
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
    return markdown
