from __future__ import annotations

import random


# 此模块为备用 DOE 工具，当前主流程使用历史数据冷启动。
# 若未来有机会重新设计实验（如菌株改造后），可重新启用。
def latin_hypercube(n_trials: int, n_vars: int, seed: int = 42) -> list[list[float]]:
    """生成单位超立方内的 LHS 样本。"""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if n_vars <= 0:
        raise ValueError("n_vars must be positive")
    try:
        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=n_vars, seed=seed)
        return sampler.random(n=n_trials).tolist()
    except Exception:
        rng = random.Random(seed)
        columns = []
        for _ in range(n_vars):
            values = [(index + rng.random()) / n_trials for index in range(n_trials)]
            rng.shuffle(values)
            columns.append(values)
        return [[columns[col][row] for col in range(n_vars)] for row in range(n_trials)]
