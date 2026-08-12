# ADR-0012：新补料间隔水平(36h)通过专门的交互子设计探索，不写入 FIXED_LEVELS

## 元数据

- 决策发生时间：2026-08-11（本次会话）
- 本记录补记时间：2026-08-11

## 背景

Round 1 的补料间隔只测过 12h/24h（`FIXED_LEVELS["interval_h"]`），24h 明显更好。这轮想追加了解一个更长的新间隔（36h，已与研发组排期确认可行）。`recommend_round2_bo_batch`（贝叶斯优化）的候选采样（`_sample_bo_candidates`）会遍历 `FIXED_LEVELS` 里每一个补料间隔水平去构造候选点网格，不管当前训练数据里有没有那个水平的真实观测。

## 决策

不把 36h 加进模块级常量 `FIXED_LEVELS["interval_h"]`；36h 只作为 `generate_round2_extension_design`/`assemble_round2_design` 的显式参数（`extra_interval_levels`）存在，专门生成一个小型交互子设计（36h × 一个连续变量的两个水平 + 噪声参考点），不去扩大 CCD 或 BO 的候选空间。

理由：`FIXED_LEVELS` 是全局共享常量，一旦加入 36h，`recommend_round2_bo_batch` 会立刻开始把 36h 当作候选水平之一去外推预测——但这时训练数据里还没有任何 36h 的真实观测，GP 会在一个完全没见过的维度取值上给出预测，这不是"基于数据的预测"，是纯粹的外推盲猜，容易给出虚假的高置信度建议。等这轮 36h 的真实数据回填之后，`FIXED_LEVELS` 要不要正式扩展，需要另外评估——不在这条 ADR 的范围内。

## 后果

- `generate_round2_extension_design` 的交互子设计（2 个交互点 + 2 个噪声参考点）不受 CCD/BO 现有代码路径影响，是独立生成的一块。
- 在真实 36h 数据回填并验证之前，`recommend_round2_bo_batch` 的候选网格永远只包含 12h/24h，即使这轮已经拿到了一些 36h 的结果。这是有意的限制，不是遗漏。

## 验证

`tests/test_round2_design.py::test_interval_h_fixed_levels_excludes_untested_36h` 直接断言 `FIXED_LEVELS["interval_h"] == [12.0, 24.0]`，防止以后有人顺手把 36h（或其他新水平）加进这个共享常量而没有意识到会影响 BO 候选空间。

## 取代关系

无。
