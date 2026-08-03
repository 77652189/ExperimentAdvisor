# ExperimentAdvisor 执行计划

本文件是项目进度、优先级与授权边界的唯一权威。

## 当前状态

活跃路线已切换为毕赤酵母 hLF 摇瓶项目；Round 1/2 的设计、回填、分析与界面流程已实现，并有相应单元测试。当前尚无可用的 hLF 历史实验数据，因此下一步价值来自真实 Round 1 执行与回填，而不是继续训练旧数据模型。

## 已完成能力摘要

- Round 1 支持基线重复、OFAT、LHS 的灵活组合，以及受设备限制的温度和补料间隔。
- Round 2 支持基线噪声估计、效应筛选、CCD 和带 OD600 可行性筛选的候选批次。
- Streamlit 默认入口已面向 hLF；旧大肠杆菌/HMO 页面被保留为历史参考。
- `App/app.py` 按 UI 关注点拆分为 `ui_shared.py`（跨会话缓存 + 共享格式化）、`pages_pichia.py`、`pages_legacy_ecoli.py`；`ingestion/excel_schema_converter.py` 拆分为解析核心与 `migration_audit.py`（新旧数据迁移审计工具）。细节见 git 历史（`archive/REFACTOR_PLAN.md` 留有当时的函数级拆分记录）。

## 待启动 / 待数据工作

| 工作 | 门控 | 现状 |
|---|---|---|
| 执行并回填 Round 1 | 获得经实验团队确认的摇瓶条件与实测产量、OD600 | 等待真实实验数据 |
| 评审 Round 2 分析 | Round 1 数据完整、基线重复足以估计噪声 | 未启动 |
| 决定是否执行候选点 | 研发和工艺团队审阅候选、风险与资源 | 不由软件自动授权 |
| 决定 `optimizer/standard_bo.py`（历史 E.coli 路径）与 `recommendation/round2_design.py` 的 `recommend_round2_bo_batch`（Pichia Round 2）两套贝叶斯优化实现是否合并 | 需要真实 Round 2 数据暴露出现有实现的具体不足，或产品侧认为有必要 | 待决策，不阻塞其他工作 |

## 明确不做

- 不恢复或清洗 HMO/2FL 发酵罐历史数据来推动 hLF 推荐：其有效性已被否定。
- 不在没有 hLF 实测数据时声明模型已学得 hLF 最优条件。
- 不将原始发酵数据、用户上传结果或生成的敏感报告提交到仓库。

## 重新评估点

完成一轮真实 Round 1 回填后，重新评估样本量、基线噪声、有效变量数量和是否值得进入 Round 2；在此之前不扩大为通用发酵预测平台。
