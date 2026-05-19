# ExperimentAdvisor

HMO 发酵实验建议模块。它接收知识约束和研究员配置，先生成 DOE 探索性实验矩阵，再根据实验结果切换到贝叶斯建议阶段。

## 范围

本项目只负责 `ExperimentAdvisor` 目录内的内容。阿里云同步时作为 `hmoAgent/ExperimentAdvisor` 子项目维护，不访问、不修改 `hmoAgent` 下其它目录。

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

核心 Python API：

```python
from experiment_advisor import initialize, get_next_trial, complete_trial

initialize(optimization_mode="maximize_yield")
trial = get_next_trial()
complete_trial(trial["trial_index"], {"yield": 88.5})
```

运行测试：

```bash
python -m pytest -q
```

## 架构文档

见 `docs/experiment-advisor-architecture.md`。
