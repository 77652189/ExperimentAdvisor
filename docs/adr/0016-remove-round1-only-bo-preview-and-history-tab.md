# ADR-0016：删除 Round1-only 贝叶斯优化预览批次和历史记录页签

## 元数据

- 决策发生时间：2026-08-12（本次会话）
- 本记录补记时间：2026-08-12

## 背景

Round 2 页签里一直有两条独立的贝叶斯优化入口：

1. "约束贝叶斯优化建议批次"——只用 Round 1 数据训练两个 GP，在 CCD 响应面设计/回填分析之前就能点。这是本项目早期（Round 2 方法论还没定下来时）的探索性入口，回答"如果只看 Round 1 数据，BO 会建议往哪走"。
2. "合并数据后的贝叶斯优化建议"（在"Round 2 结果分析"区域）——用 Round 1 + 已回填 Round 2 数据训练，是 ADR-0009 里"响应面(CCD)是主方法，贝叶斯优化在合并数据集上持续跑"这个决策的具体落地。

ADR-0009 定下 Round 2 主方法是 CCD 之后，第 1 条入口的产出就从没被真正用来决定 Round 2 该测哪些点——Round 2 的 32 个条件（CCD 18 + 补料间隔交互 2 + 噪声参考 2 + LHS 10）是直接照 CCD 方法论生成的，不是这个 BO 入口的建议。这条入口继续留着，容易让后来者误以为它是 Round 2 设计依据之一，实际上只是一段没有被采纳过的历史遗留探索。

同时，"历史记录"页签的唯一数据来源就是入口 1 下面的"保存本次 Round 2 分析到历史记录"按钮——两者是绑在一起的一套功能，不是各自独立的。

## 决策

1. **删除入口 1（"约束贝叶斯优化建议批次"，Round-1-only）及其"保存到历史记录"按钮**：这条入口和 Round 2 实际采用的方法论（CCD+LHS）已经脱节，继续留着是维护负担、也是对后来者的误导，不是"以防以后用得上"的可选功能。
2. **连带删除"历史记录"页签（`_pichia_history_tab`）**：删除入口 1 之后，历史记录页签再没有任何路径能写入新记录（`_pichia_ui_records().append(...)` 只在入口 1 的保存按钮里调用过），留着只会变成一个永远空白、点进去只有"暂无历史记录"提示的死页签，删比留着更诚实。
3. **明确保留入口 2（合并数据后的贝叶斯优化建议）**：这是 ADR-0009 决策的实际执行部分，本次删除不涉及它——`_pichia_render_bo_recommendation_section`/`_pichia_render_bo_verdicts`/`gp_leave_one_out_cv` 等这次会话新加的贝叶斯优化诊断能力，全部继续为入口 2 服务。

## 后果

- `App/pages_pichia.py` 净删除约 123 行：`_pichia_history_tab` 函数、`_pichia_ui_records` 辅助函数整体、`hashlib` 导入（唯一用途就是给历史记录生成 `record_id`）、`App.ui_shared` 里 `_clear_ui_cache`/`_remember_ui_cache` 两个导入（`_num` 保留）、`_pichia_round1_builder` 里两处对已不存在的 `round2_bo_result` 的 `session_state.pop` 清理调用。
- Pichia hLF 页面从三个页签（Round 1 / Round 2 / 历史记录）变成两个（Round 1 / Round 2）。
- 已经存在于某个用户会话 `st.session_state` 里的 `round2_bo_result`/`pichia_ui_design_records` 等旧键不会主动迁移或报错——它们只是不再被任何代码读写，等这次进程重启或会话结束后自然消失；不需要写迁移逻辑，因为 `st.session_state` 本来就不持久化到磁盘（这几个键从没被 `_pichia_restore_persisted_dataset` 之类的机制落盘过）。

## 验证

- 全仓库检索确认没有测试或文档引用 `_pichia_history_tab`/`_pichia_ui_records`/`round2_bo_result`/`run_round2_bo`/`save_round2_snapshot` 等符号——删除前后 `python -m pytest -q`（120 项）结果不变。
- 用真实 Y103 数据完整跑一遍 AppTest：确认页面只剩两个页签、被删的两个按钮 key 已不存在、"合并数据后的贝叶斯优化建议"入口依然能正常生成建议并渲染叙述解读，全程无异常。

## 取代关系

无。
