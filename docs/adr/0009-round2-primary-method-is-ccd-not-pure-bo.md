# ADR-0009：Round 2 主方法是响应面(CCD)，贝叶斯优化在合并数据集上持续跑，不是二选一

## 元数据

- 决策发生时间：2026-08-11（本次会话）
- 本记录补记时间：2026-08-11

## 背景

Y103 真实 Round 1 数据回填后，`plan_round2` 算出 K=3（`glucose_pct`/`ph`/`volume_ml` 活跃）。此时面临选择：

- CCD 在 K=3 需要 18 个设计点（含 1 个和 Round 1 重合、可直接复用的点），代表性 slide 材料里同一档位 BBD（Box-Behnken）理论上只需 15 个——但 BBD 在这个代码库里从未实现过，只有 CCD（`generate_ccd`）是真实可跑的。
- 项目目标是产量最大化，不是发表级别的统计推断；贝叶斯优化（`recommend_round2_bo_batch`）明确是为"预算紧张、尽快找到最优解"设计的方法。这两点单独看，似乎更支持整轮预算都投给 BO，不做 CCD。

## 决策

Round 2 以 CCD 响应面为主线设计（`assemble_round2_design`/`fit_ccd_response_surface`），贝叶斯优化不是被 CCD 取代，而是在 Round 1 + 当轮 Round 2 回填数据的合并数据集上持续可用（`_pichia_round2_results_analysis_section` 里"合并数据后的贝叶斯优化建议"）。

理由：GP（BoTorch `SingleTaskGP`）不挑训练数据的来源——不管一个点是 OFAT、CCD 角点/轴点/中心点，还是 LHS 随机点，只要有条件+产量+OD600 就能喂进去训练。所以 CCD 产出的 18 个点不是"和 BO 争预算"，是同时喂给两条线：CCD 自己拟合响应面方程（有 R²、失拟检验这类正式统计量），同一批数据也让后续任何一轮的 BO 训练得更准。反过来（拿 BO 选出的点做 CCD 式的正式统计推断）不成立：BO 选点偏向"看起来产量高"的区域，分布不均匀，回归系数估计会更吵——这个不对等在当前目标（产量最大化，不是发表级别推断）下不影响决策。

## 后果

- `assemble_round2_design`把 CCD 的 18 个点和补料间隔交互、LHS 两块放进同一张设计表，共用一套 run_id 编号。
- Round 2 页签有两个独立的贝叶斯优化入口：一个只用 Round 1 数据（预览/尽快出结果用），一个用合并数据集重新训练（`run_combined_bo`按钮）——两者不会互相覆盖对方的 session_state。
- 这不是通用规则："CCD 优先"只在当前场景成立（K=3、样本预算 20-32 瓶、BBD 未实现）；如果以后 K 值或预算发生实质变化，需要重新评估，不能直接援引这条 ADR。

## 验证

CCD/合并数据 BO 两条线各自的能力（数据生成、拟合、推荐）由 `tests/test_round2_design.py` 里 `fit_ccd_response_surface`/`assemble_round2_design`/`recommend_round2_bo_batch` 相关测试覆盖；两个入口在界面上互不干扰这一点，用 AppTest 脚本手动跑过一次（生成设计→回填→CCD 拟合展示→点击合并数据 BO 按钮→确认两个 session_state key 各自独立），未写成正式 pytest 守卫。

## 取代关系

无。
