# Archive：HMO 发酵罐阶段产出（已停用）

此目录归档 2026-07 之前 HMO（2FL，大肠杆菌发酵罐）阶段的分析产出：

- `supporting_reports/`：EDA 报告、字段中英对照表、建模前数据质量报告、run 级建模表设计决策、液相目标匹配推断记录，以及一份从未真正落地实现的旧架构设计稿（`experiment-advisor-architecture.md`，描述的 config/doe/bayes/agent 分层与当前 `experiment_advisor/` 实际代码结构不符）。
- `recommendation_report.md`：旧 `standard_bo_qnei` 推荐生成的示例报告。

**停用原因：** 项目已从 HMO 发酵罐转向毕赤酵母（Pichia）摇瓶生产 hLF，且之前 HMO 的实验数据已确认完全无效。这批文档不再代表当前项目的数据、方法或结论。

**保留原因：** 部分方法论和文档结构仍可复用（例如字段字典的中英对照格式、EDA 报告的模块划分、数据质量审计的呈现方式），保留供后续 Pichia 摇瓶数据积累后参考格式，不作为当前结论来源。

**注意：** `supporting_reports/` 和 `recommendation_report.md` 本身仍在 `.gitignore` 中（可能包含真实实验数值），不会被提交到 GitHub；本 `README.md` 不含敏感数据，正常纳入版本控制。
