# ADR 索引

| 编号 | 状态 | 摘要 |
|---|---|---|
| [0001](0001-pichia-hlf-shake-flask-is-active-route.md) | accepted | 以从零开始的毕赤酵母 hLF 摇瓶闭环取代无效 HMO/2FL 数据驱动路线。 |
| [0002](0002-fermentation-data-stays-out-of-version-control.md) | accepted | 原始、处理后和用户上传的发酵数据默认不进入版本控制。 |
| [0003](0003-legacy-ecoli-fermenter-path-retained-as-reference.md) | accepted | 大肠杆菌发酵罐路径（代码+UI）保留为历史参考，不作为 hLF 决策依据。 |
| [0004](0004-standard-recommender-converges-on-gp-qnei.md) | accepted | 标准推荐路径收敛为 GP-qNEI，否掉 XGBoost/UCB/集成/随机基线等候选方法。 |
| [0005](0005-standard-gp-visualization-reuses-fitted-model.md) | accepted | 标准 GP-BO 的可视化复用推荐阶段拟合的同一 GP 对象，不重新训练。 |
| [0006](0006-soft-filter-failures-grow-pool-not-backfill.md) | accepted | 软过滤未通过的推荐通过扩大候选池补足，不用落选候选递补。 |
| [0007](0007-round1-variable-set-and-ccd-boundary-rules.md) | accepted | Round 1 移除 `glucose_start_time_h`、水平不取全局 min/max；CCD 在硬边界整体平移。 |
| [0008](0008-ofat-levels-are-user-extensible.md) | accepted | Round 1 的 OFAT 水平允许用户运行时新增自定义值，不锁定为协议默认值。 |
