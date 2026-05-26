from __future__ import annotations

from pathlib import Path

import logging

LOGGER = logging.getLogger(__name__)


def save(ax_client, path: str | Path) -> None:
    """将 AxClient 序列化为 JSON 文件。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ax_client.save_to_json_file(str(output))


def load(path: str | Path):
    """从 JSON 文件恢复 AxClient，并打印观测数量和当前最优点摘要。"""

    try:
        from ax.service.ax_client import AxClient
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError("Ax is required. Install dependencies with: pip install -r requirements.txt") from exc

    ax_client = AxClient.load_from_json_file(str(path))
    trials = getattr(ax_client.experiment, "trials", {})
    completed = len([trial for trial in trials.values() if getattr(trial.status, "is_completed", False)])
    try:
        best = ax_client.get_best_parameters()
    except Exception:
        best = None
    LOGGER.info("已恢复 Ax 状态：%s 条已完成观测，当前最优点=%s", completed, best)
    return ax_client
