from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


STANDARD_BO_KEY = "standard_bo_qnei"


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
        "| Rank | 预测产量 | GP 后验标准差 | 推荐展示分 | 参数 |",
        "|------|----------|----------------|------------|------|",
    ]
    for item in items:
        lines.append(
            "| {rank} | {pred} | {unc} | {score} | {params} |".format(
                rank=item.get("rank", "-"),
                pred=_fmt(item.get("predicted_yield")),
                unc=_fmt(item.get("model_uncertainty")),
                score=_fmt(item.get("acquisition_score")),
                params=_params_text(item),
            )
        )
    return lines


def generate_recommendation_report(
    comparison_result: dict[str, Any] | list[dict],
    offline_analysis: dict | None = None,
    output_path: str | Path | None = None,
) -> str:
    """Generate a Markdown report for the qNEI standard GP-BO recommendation."""

    if isinstance(comparison_result, list):
        comparison = {
            "n_training_rows": None,
            "model_metrics": {},
            "recommendations": {STANDARD_BO_KEY: comparison_result},
            "selected_method": STANDARD_BO_KEY,
            "selected_recommendations": comparison_result,
        }
    else:
        comparison = comparison_result

    decision = comparison.get("decision", {})
    selected_method = comparison.get("selected_method", STANDARD_BO_KEY)
    selected = comparison.get("selected_recommendations") or comparison.get("recommendations", {}).get(selected_method, [])

    lines = [
        "# 发酵工艺优化推荐报告",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 数据与默认方法",
        f"- 训练 run 数：{_fmt(comparison.get('n_training_rows'))}",
        f"- 目标字段：{comparison.get('target_col', 'yield_g_per_l')}",
        f"- 主推荐方法：{selected_method}",
        "- 优化器：BoTorch qNEI + MLE SingleTaskGP",
        f"- 是否需要人工审议：{decision.get('needs_human_review', False)}",
        f"- 决策说明：{decision.get('reason', '默认采用 standard_bo_qnei。')}",
        "",
        f"## 主推荐：{STANDARD_BO_KEY}",
        *_recommendation_table(selected),
    ]

    metrics = comparison.get("model_metrics", {})
    if metrics:
        lines.extend(["", "## 模型验证指标", "| 模型 | LOO MAE | LOO R2 |", "|------|---------|--------|"])
        for name, values in metrics.items():
            lines.append(f"| {name} | {_fmt(values.get('mae_loocv'))} | {_fmt(values.get('r2_loocv'))} |")

    lines.extend(
        [
            "",
            "## 风险说明",
            "- standard_bo_qnei 是当前主推荐方法；其不确定性来自 GP 直接拟合产量后的后验标准差。",
            "- qNEI 联合优化整批候选点，能够缓解逐点 EI top-k 造成的 batch 聚集。",
            "- qNEI 显式处理观测噪声，相比直接使用历史最大值的 EI 更不容易被噪声高点牵引。",
            "- GP 后验标准差不是湿实验置信区间，最终采纳仍需结合工艺可执行性复核。",
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
