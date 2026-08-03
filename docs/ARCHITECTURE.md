# ExperimentAdvisor 架构

## 分层与数据流

```text
Streamlit UI (App/)
    -> Pichia workflow orchestration
        -> recommendation/round1_design.py and round2_design.py
            -> utility/statistical and GP libraries
    -> user-controlled import/export and session state

Legacy UI -> ingestion/ + optimizer/ + recommendation/service.py
```

## 边界

- `App/` 只负责用户交互、文件交接和会话态；实验设计、效应计算和候选生成必须由 `experiment_advisor/` 的领域模块完成。
- `recommendation/round1_design.py` 是无历史数据的 Round 1 设计权威；`round2_design.py` 只消费回填的 Round 1 结果来形成 Round 2 分析和候选。
- `ingestion/`、`optimizer/`、`recommendation/service.py` 服务于保留的历史 HMO/2FL 路径，不能成为活跃 hLF 推荐的数据来源。
- 原始、处理后及用户上传的发酵数据都留在 `data/`；该目录默认不纳入版本控制。可跟踪的仅限毕赤酵母空目录标记和模板。
- Streamlit `session_state` 只保存当前浏览器会话的工作态与界面历史；关键实验结果必须经下载、上传或明确保存流程交接，不能假定会话重启后仍存在。

## 不变量

- 温度只能使用 20/25/30℃，补料时间间隔只能使用 12/24h；Round 1 的联合探索不得绕过这些离散设备约束。
- 产量与 OD600 分别建模/评估，OD600 作为 Round 2 候选的可行性筛选条件，而不是产量的替代指标。
- 历史 HMO/2FL 内容可以用于理解旧实现，但不能以其数据或输出支持当前 hLF 实验决策。
