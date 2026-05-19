from __future__ import annotations

from typing import Any

FALLBACK_TEMPLATE = (
    "第 {trial_index} 批建议参数：\n"
    "{param_lines}\n"
    "预测结果：\n{outcome_lines}\n"
    "（置信度：{confidence}）\n"
    "各目标历史最优：\n{best_lines}"
)


def render_fallback_report(trial: dict[str, Any]) -> str:
    param_lines = "\n".join(f"- {key}: {value}" for key, value in trial.get("parameters", {}).items())
    predicted = trial.get("predicted_outcomes", {})
    outcome_lines = "\n".join(
        f"- {key}: {payload.get('range')} ({payload.get('direction')})" for key, payload in predicted.items()
    )
    best = trial.get("best_outcomes_so_far", {})
    best_lines = "\n".join(f"- {key}: {payload.get('value')}" for key, payload in best.items())
    return FALLBACK_TEMPLATE.format(
        trial_index=trial.get("trial_index"),
        param_lines=param_lines or "- 暂无",
        outcome_lines=outcome_lines or "- 暂无",
        confidence=trial.get("confidence", "low"),
        best_lines=best_lines or "- 暂无",
    )
