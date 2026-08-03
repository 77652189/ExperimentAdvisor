# ADR-0004：标准推荐路径收敛为 GP-qNEI，否掉 XGBoost/UCB/集成/随机基线等候选

## 元数据

- 决策发生时间（推断自会话记录）：2026-05-26
- 本记录补记时间：2026-07-31

## 背景

早期架构（2026-05-19 起的"experiment-advisor 架构设计文档 v2/v3"）规划过基于 Ax 平台的 DOE + 贝叶斯优化两阶段流程。2026-05-26 前后一度提议删除 DOE、改用 XGBoost + 贝叶斯优化，理由是"不做激进黑箱优化，做保守型小样本贝叶斯推荐系统"。

从 XGBoost 等数据驱动 ML 方法转向 DOE + 贝叶斯学习，根本原因是数据质量和数量：一旦从"挖历史大肠杆菌发酵罐数据"转为"从零开始设计小样本实验"，样本量天然很小，XGBoost 这类依赖较大样本量才能学到可靠模式的方法不再适用；DOE 也没有被真的删掉——Round 1 至今仍是基线重复 + OFAT + LHS 的screening design 风格。

实现迭代过程中，先后比较过至少 7 个候选方法/方法名：`standard_bo_ei`、`standard_bo_ucb`、`xgp_bo_ei`、`xgp_bo_ucb`、`conservative_ensemble`、`random_safe`、`single_xgboost`。

## 决策

主推荐方法只保留 `standard_bo_qnei`（BoTorch `SingleTaskGP` + qNEI 批量采集函数）一种；上述 7 种候选方法均不作为主推荐或次选方法暴露给用户。

## 后果

`tests/test_recommender_comparison.py::test_compare_recommenders_returns_standard_bo_qnei_primary` 对这 7 个方法名逐一做了"不存在于结果中"的断言，是这条决策的常设回归守卫——该测试红了就意味着这条决策被破坏。以后如果要重新引入 EI/UCB/集成/随机基线等方法作为可选项，需要新开一条 ADR 说明当年否掉的理由为什么不再成立，而不是直接改测试放行。

## 取代关系

无。
