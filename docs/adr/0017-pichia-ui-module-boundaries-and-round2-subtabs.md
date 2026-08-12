# ADR-0017：Pichia 页面的模块边界与 Round 2 子页签

## 元数据

- 决策发生时间：2026-08-12（本次会话）
- 本记录补记时间：2026-08-12

## 背景

`App/pages_pichia.py` 长到 2482 行，Round 2 相关代码占一半以上。两个具体后果：

1. **改一处要在一个文件里翻两千行**。Round 1 的上传解析、Round 2 的响应面图表、贝叶斯优化的交叉验证全在同一个文件里，"要改 X 该打开哪个文件"这个问题没有短答案。
2. **Round 2 页签是一条无法导航的竖直滚动流**。从噪声估计一路往下：显著性筛选 → CCD 设计表 → 下载/上传/回填 → 响应面拟合 → 19 张图 → 四项深入分析 → 补料间隔交互 → 合并数据贝叶斯优化。想回到某一节只能滚。

`archive/REFACTOR_PLAN.md` 记录过上一次同类拆分（`App/app.py` → `ui_shared.py` + 两个页面模块），那次没有留 ADR，只留了函数级清单。这次留 ADR，是因为这次真正需要固定下来的不是"哪个函数搬到哪儿"（那是 git 历史能回答的），而是**以后新增功能时该往哪个文件放、哪些边界不能越**——那些规则不写下来就会在两三轮迭代后失效。

## 决策

### 1. 拆成 6 个模块 + 1 个路由，扁平放在 `App/` 下

| 模块 | 职责 |
|---|---|
| `pichia_common.py` | 路径常量、中文标签、图表配色、两轮共用的行级展示辅助 |
| `pichia_results_io.py` | 设计表导出 Excel + 回传结果解析（两轮共用） |
| `pichia_round1.py` | Round 1 构建器、图表、页签 |
| `pichia_round2_surface_views.py` | 响应面的表格/图表/叙述解读 |
| `pichia_round2_bo_views.py` | 贝叶斯优化批次、PDP、留一法交叉验证、解读 |
| `pichia_round2_sections.py` | Round 2 的各个小节 + 页签本体 |
| `pages_pichia.py` | 只剩顶层 Round 1 / Round 2 页签路由 |

不建 `App/pichia/` 子包，保持和现有 `App/`（`pages_legacy_ecoli.py` / `ui_shared.py`）一致的扁平布局。

文档里原先写的是 4 个模块，实际按行数切下来"Round2 图表与解读"单独就有 800 行，所以再分两刀：把两轮共用的 Excel 导出+上传解析从共享层里独立出来，把 BO 视图从响应面视图里独立出来。最大文件约 713 行。

### 2. 三条硬边界（有测试守着，不只是文档）

- **`pichia_common.py` 不 import 任何其他 `App.pichia_*` 模块**。它是依赖图的最底层，所有人都可以依赖它；只要依赖关系不反向，就不会有循环导入。
- **`pichia_round2_surface_views.py` 不读写 `st.session_state`、不拥有任何 widget key**。它是纯视图层：接收已拟合的模型和数据，返回图/表或直接渲染。这是判断一个新函数该不该放进这个文件的判据。
- **页面状态只归 `pichia_round2_sections.py`**。哪些 session_state 键存在、谁先写谁后读，只有这一个文件知道。`pichia_round2_bo_views.py` 是个受控例外：它确实读写 session_state，但只通过调用方传进来的 `cv_session_key`，自己不硬编码任何键名。

三条都在 `tests/test_adr_invariants.py` 里有对应断言，并且都验证过"规则被破坏时测试确实会失败"（不是永真断言）。

### 3. 常量别名集中在 common，算法函数不经 common 中转

`PICHIA_TARGET_COL` / `PICHIA_VARIABLES` / `PICHIA_ROUND1_BASELINE` 这类给 `experiment_advisor` 常量起的 `PICHIA_` 前缀别名，全部在 `pichia_common.py` 转出，其他模块从那里取——UI 的词汇表映射到算法层名字的地方只有一处。但 `fit_ccd_response_surface`、`sensitivity_analysis` 这类**函数**由用它的模块直接从 `experiment_advisor.recommendation.*` 导入，不在 common 里转一手：常量是词汇表，函数不是，把函数也塞进 common 会让它慢慢长成第二个巨型文件。

### 4. Round 2 页签内部加一层 `st.tabs`，切四个子页

① 显著性分析 / ② 设计生成与回填 / ③ 响应面结果 / ④ 合并数据贝叶斯优化。沿用现有"外层 Round1/Round2 tabs"的模式，不引入新导航框架。

「分析参数」expander 和 `plan_round2()` 留在四个子页**上方**——四个子页都依赖 `plan`，塞进任何一个子页都会让另外三个拿不到。

CCD 设计预览表从①移到②的顶部：它预览的正是②要生成的那张表。

### 5. 明确接受：`st.tabs` 不省计算

`st.tabs` 只是把内容隐藏起来，**不做懒加载**——四个子页的代码每次 rerun 都会按源码顺序全部执行。停在①的时候③的 CCD 拟合照样在跑。

这和分页前的行为完全一致（分页前也是从上到下全跑），不是性能回退，但也**不要指望分页带来任何加速**。真要懒执行得换成 `st.radio`/`st.segmented_control` 路由，本次有意不做：那会改变"一次 rerun 渲染整页"的模型，而当前的计算量（一次 CCD OLS 拟合 + 若干 plotly 图）并没有慢到值得为此换掉 Streamlit 最标准的分页控件。

由此产生一条**执行顺序约束**：②的 `st.data_editor` 会把用户填的结果写回 `round2_full_design_df`，③④要读它，所以②的 `with` 块必须写在③④前面。否则用户刚填进去的数字要多点一次界面才会被下面的分析看到。这条约束靠 `tests/test_adr_invariants.py::test_adr_0017_round2_backfill_subtab_precedes_the_subtabs_reading_it` 守着。

### 6. 空态要出声

分页前，没有回填结果时整块"Round 2 结果分析"直接 `return`，界面上什么都不显示——在一条长滚动流里这不明显。分页后，一个空白子页会被读成"页面坏了"。所以③④在没有可分析数据时各给一句 `st.info`，指向②。这是本次改动里唯一有意的行为变化。

## 后果

- `pages_pichia.py` 从 2482 行降到 33 行；最大的单个模块 713 行。
- **不留 re-export 兼容层**。`pages_pichia.py` 只导出 `_pichia_hlf_page`（`App/app.py` 按这个名字导入）。继续从旧路径转出那些名字等于把耦合原样搬过来，拆分就白做了；两个测试模块改成从各自的归属模块导入。
- `tests/test_adr_invariants.py` 里 ADR-0008 那条断言改读 `App/pichia_round1.py`（Round 1 构建器的新家）。
- 顺带清掉 `App/ui_shared.py` 的 `PICHIA_UI_CACHE_KEYS` 里两个死键（`round2_bo_batch_size` / `pichia_ui_design_records`）——都属于 ADR-0016 删掉的功能，已经没有任何代码写它们。
- 以后加 Round 2 功能时的落点判断：新图表/新表格 → `*_views.py`；新的"把图表拼成一节"的编排 → `pichia_round2_sections.py`，并且要想清楚它属于哪个子页；新的两轮共用工具 → `pichia_common.py`（不带状态）或 `pichia_results_io.py`（涉及文件读写）。

## 验证

- **拆分是纯搬家，不是重写**：拆完的 7 个文件和拆分前文件的行做多重集比对，原文件 2422 行正文一行没丢；再用 AppTest 把整页元素（每张图的标题/坐标轴/trace、每张表的列名和形状、每个 metric/caption/markdown 字符串、每个 button/slider/selectbox 的 key）导出成 JSON，拆分前后**完全一致**。快照脚本用真实 Y103 数据驱动四个交互阶段（初始 → 生成设计 → 填模拟数据 → 生成 BO 建议），且验证过脚本本身连跑两次结果相同（否则比对没有意义）。
- **分页只改了该改的**：同一套快照比对，差异只有四个子页签标题、被移除的分组标题+提升一级的三个标题、以及两条新的空态提示；图表、表格、指标、widget key 全部未变。
- `python -m pytest -q`：拆分后 120 项全过（和拆分前一致），加上 3 条新的 ADR-0017 守卫共 123 项。
- 三条守卫测试都做了反向验证：人为破坏对应规则后测试确实失败（其中执行顺序那条是靠断言而不是靠异常抓到的）。

## 取代关系

无。补充 ADR-0016（它删掉的两个功能留下的死缓存键在本次一并清掉）。
