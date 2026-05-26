from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _require_ax():
    try:
        from ax.service.ax_client import AxClient
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError("Ax is required. Install dependencies with: pip install -r requirements.txt") from exc
    return AxClient


def _parameter_names(search_space: Any) -> list[str]:
    parameters = getattr(search_space, "parameters", {})
    if isinstance(parameters, dict):
        return list(parameters)
    return [parameter.name for parameter in parameters]


class BOEngine:
    """历史数据冷启动的 Bayesian Optimization 引擎。"""

    def __init__(
        self,
        search_space: Any,
        objective_name: str = "yield_g_per_l",
        noise_std: float | None = None,
        seed: int = 42,
    ):
        AxClient = _require_ax()
        self.search_space = search_space
        self.objective_name = objective_name
        self.noise_std = noise_std
        self.seed = seed
        self.ax_client = AxClient(random_seed=seed)
        self._observations: list[tuple[dict[str, float], float]] = []
        self._create_experiment()

    def _create_experiment(self) -> None:
        try:
            self.ax_client.create_experiment(
                name="hmo_fermentation_bo",
                search_space=self.search_space,
                objective_name=self.objective_name,
                minimize=False,
            )
        except TypeError:
            from ax.service.utils.instantiation import ObjectiveProperties

            self.ax_client.create_experiment(
                name="hmo_fermentation_bo",
                search_space=self.search_space,
                objectives={self.objective_name: ObjectiveProperties(minimize=False)},
            )

    def _raw_data(self, observed_yield: float) -> dict[str, float | tuple[float, float]]:
        objective_name = getattr(self, "objective_name", "yield_g_per_l")
        noise_std = getattr(self, "noise_std", None)
        if noise_std is None:
            return {objective_name: float(observed_yield)}
        return {objective_name: (float(observed_yield), float(noise_std))}

    def _current_search_space(self):
        if hasattr(self, "search_space"):
            return self.search_space
        experiment = getattr(self.ax_client, "experiment", None)
        if experiment is None or getattr(experiment, "search_space", None) is None:
            raise ValueError("BOEngine has no search_space; construct normally or restore a full AxClient state")
        return experiment.search_space

    def _observation_count(self) -> int:
        if hasattr(self, "_observations"):
            return len(self._observations)
        experiment = getattr(self.ax_client, "experiment", None)
        trials = getattr(experiment, "trials", {}) if experiment is not None else {}
        return len([trial for trial in trials.values() if getattr(trial.status, "is_completed", False)])

    def cold_start(self, df: pd.DataFrame) -> None:
        """
        将历史数据批量注入 Ax，作为 BO 的先验观测。

        输入 DataFrame 必须包含搜索空间参数列和 objective_name 列。
        """

        param_names = _parameter_names(self._current_search_space())
        missing = [column for column in [*param_names, self.objective_name] if column not in df.columns]
        if missing:
            raise ValueError(f"cold_start missing required columns: {', '.join(missing)}")

        clean = df.dropna(subset=[*param_names, self.objective_name])
        if clean.empty:
            raise ValueError("cold_start requires at least one complete historical batch")

        for _, row in clean.iterrows():
            params = {name: float(row[name]) for name in param_names}
            _, trial_index = self.ax_client.attach_trial(params)
            self.ax_client.complete_trial(trial_index=trial_index, raw_data=self._raw_data(float(row[self.objective_name])))
            self._observations.append((params, float(row[self.objective_name])))

        LOGGER.info("已加载 %s 批历史数据，当前模型 LOO-CV R²=待 diagnostics.run_loocv 评估", len(clean))

    def _recommendation_mode(self) -> str:
        dimension = max(len(_parameter_names(self._current_search_space())), 1)
        if self._observation_count() < 2 * dimension:
            return "explore"
        values = [value for _, value in getattr(self, "_observations", [])]
        if len(values) >= 2 and pd.Series(values).std(ddof=1) > 0:
            return "exploit"
        return "explore"

    def _prediction_summary(self) -> tuple[float | None, float | None]:
        values = [value for _, value in getattr(self, "_observations", [])]
        if not values:
            return None, None
        mean = float(sum(values) / len(values))
        std = float(pd.Series(values).std(ddof=1)) if len(values) > 1 else 0.0
        return mean, std

    def recommend(self, n: int = 1) -> list[dict[str, Any]]:
        """
        返回 n 组推荐参数。

        n=1 面向 EI 使用场景；n>1 连续请求 Ax 候选点，作为 batch qEI 的接口等价层。
        """

        if n <= 0:
            raise ValueError("n must be positive")

        recommendations: list[dict[str, Any]] = []
        mode = self._recommendation_mode()
        predicted_mean, predicted_std = self._prediction_summary()
        for _ in range(n):
            params, trial_index = self.ax_client.get_next_trial()
            recommendations.append(
                {
                    "trial_index": trial_index,
                    "params": {name: float(value) for name, value in params.items()},
                    "predicted_mean": predicted_mean,
                    "predicted_std": predicted_std,
                    "mode": mode,
                    "acquisition": "EI" if n == 1 else "qEI",
                }
            )
        return recommendations

    def update(self, params: dict[str, float], observed_yield: float) -> None:
        """录入一批新实验结果并更新 Ax 后验。"""

        _, trial_index = self.ax_client.attach_trial({name: float(value) for name, value in params.items()})
        self.ax_client.complete_trial(trial_index=trial_index, raw_data=self._raw_data(float(observed_yield)))
        if not hasattr(self, "_observations"):
            self._observations = []
        self._observations.append(({name: float(value) for name, value in params.items()}, float(observed_yield)))

    def get_best(self) -> dict[str, Any]:
        """返回当前最优参数组合及已观测或模型估计产量。"""

        try:
            params, values = self.ax_client.get_best_parameters()
            objective = values[0].get(self.objective_name) if isinstance(values, tuple) and values else values
            return {"params": params, "predicted_yield": objective}
        except Exception:
            if not getattr(self, "_observations", []):
                raise ValueError("No observations available")
            params, value = max(self._observations, key=lambda item: item[1])
            return {"params": params, "predicted_yield": value}
