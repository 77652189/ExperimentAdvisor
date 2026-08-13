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
| [0009](0009-round2-primary-method-is-ccd-not-pure-bo.md) | accepted | Round 2 主方法是响应面(CCD)，贝叶斯优化在合并数据集上持续跑，不是二选一。 |
| [0010](0010-round2-replication-is-structural-not-per-point.md) | accepted | Round 2 的重复靠设计结构（CCD 中心点），角点/轴点/LHS 点不再默认每点测 2 瓶。 |
| [0011](0011-technical-noise-is-a-separate-diagnostic.md) | accepted | 技术重复噪声只作独立诊断展示，不并入现有显著性阈值。 |
| [0012](0012-new-interval-level-stays-out-of-fixed-levels.md) | accepted | 新补料间隔水平(36h)通过专门交互子设计探索，不写入共享的 `FIXED_LEVELS`。 |
| [0013](0013-ccd-response-surface-fit-design.md) | accepted | CCD 响应面拟合：全二次 OLS 模型、中心点估纯误差、最优点只搜已测范围。 |
| [0014](0014-response-surface-deep-dive-four-analyses.md) | accepted | 响应面深入分析四件套：置信区间、灵敏度、鞍点/岭线判别、联合满意度，四种边界情况共用一个判据函数。 |
| [0015](0015-bo-leave-one-out-cv-and-narrative-verdicts.md) | accepted | 贝叶斯优化补上留一法交叉验证(Q²)和叙述性解读，两个 BO 入口合并共用同一段渲染代码。 |
| [0016](0016-remove-round1-only-bo-preview-and-history-tab.md) | accepted | 删除脱节于 CCD 方法论的 Round1-only 贝叶斯优化预览批次，连带删除唯一依赖它的历史记录页签。 |
| [0017](0017-pichia-ui-module-boundaries-and-round2-subtabs.md) | accepted | `pages_pichia.py` 拆成 6 个模块（三条边界有测试守着），Round 2 页签切四个子页；接受 `st.tabs` 不懒加载。 |
| [0018](0018-round3-is-a-single-composite-confirmation-round.md) | accepted | Round 3 是一次性 32 瓶复合确认轮而非循环迭代；K 永久为 3；确认判据报两个不合并。方案见 `docs/ROUND3_PLAN.md`。 |
| [0019](0019-fill-volume-cap-and-temperature-stay-fixed.md) | accepted | 装液量上限 75 mL 不放宽（250 mL 瓶已 30% 装液率）；温度因摇床是共享参数推迟到后期；溶氧探索移到发酵罐但须用配气而非降转速归因。 |
| [0020](0020-ingest-time-normalisation-and-not-detected-as-zero.md) | accepted | 回填结果在读取阶段一次性规范化（单位换算、技术重复取均值且保留原始值）；「未检测到」按 0 而非缺失值，代价是低估真值。 |
| [0021](0021-upr-collected-but-not-analysed.md) | accepted | UPR 采集入库但不进任何分析；搁置前提（方向未知）已因真实 Round 1 数据部分不成立，Round 2 后应重评为诊断维度。 |
