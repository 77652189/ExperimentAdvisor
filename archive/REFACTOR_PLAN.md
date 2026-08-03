# ExperimentAdvisor 重构计划

草拟日期：2026-07-28
状态：**Phase 1、Phase 2、Phase 3 已完成并通过验证（2026-07-29）；Phase 4 待做（决策项，暂不安排）**

**2026-07-31 归档说明**：本文件不再是进度的权威来源，归档到此处只为保留函数级拆分的历史记录。当前状态与 Phase 4 决策项见 [`docs/EXECUTION_PLAN.md`](../docs/EXECUTION_PLAN.md)。

## 0. 先澄清一件事：哪些"旧问题"其实已经解决了

在写这份计划之前核实了一遍现状，发现两件之前担心的事其实已经在 `81513f0`（pivot to Pichia）和 `cd548c0`（flexible Round1 builder）里做完了，**本计划不再重复处理**：

- 导航顺序：`App/app.py` 侧边栏 `mode = st.radio(...)` 里"毕赤酵母 hLF"已经排在第一位，"大肠杆菌 BO"选项文案已经是"（历史，数据已作废）"。
- Pichia 页面对接 round1/round2 方法论：`_pichia_round1_tab()`/`_pichia_round2_tab()` 已经在调用 `round1_design.generate_round1_design()` 和 `round2_design.py` 的函数；仓库里搜不到任何"菌株借鉴框架"或"单轮上限4个建议"的残留代码。

真正还没解决的，是上一轮耦合度分析里发现的**结构性问题**：`app.py`过大、excel_schema_converter.py职责混杂、两套独立BO实现、`__init__.py`导出不一致。这份计划只针对这些。

## 1. 目标 / 非目标

**目标**：降低`app.py`体积和扇出，让新旧两条路径（Pichia / E.coli历史路径）在文件层面就分开，不再共享一个2381行的文件；顺手解决`excel_schema_converter.py`的职责混杂和`__init__.py`导出不一致。

**非目标（明确不做）**：
- 不改任何算法/数值逻辑（BO、CCD、GP拟合的计算结果必须前后一致）
- 不改UI可见行为、不改`session_state` key字符串、不引入Streamlit原生`pages/`多页路由（那是更大的UX改动，如果以后需要可以单独讨论）
- 不动`data/scripts/`下的一次性脚本和`archive/`
- 是否合并两套BO实现，作为决策项列出，不在本计划内直接执行

## 2. 现状数据回顾

| 文件 | 行数 | 问题 |
|---|---|---|
| `App/app.py` | 2381 | 62函数/0 class，扇出到9个内部模块，新旧两条路径混在一起 |
| `experiment_advisor/ingestion/excel_schema_converter.py` | 1317 | Excel解析 + 历史数据迁移审计工具混在一个文件 |

`app.py`一个文件占了`experiment_advisor`+`App`全部代码量的约60%。

## 3. Phase 1（优先级最高）：拆分 `App/app.py` — ✅ 已完成

### 3.1 目标结构（实际落地时把 ui_cache.py/ui_common.py 合并成了一个文件，见下方说明）

```
App/
  app.py                    # 只保留 page_config + 侧边栏 mode 单选 + 分发，实际 38 行
  ui_shared.py                # 缓存 + 通用显示函数，实际 80 行
  pages_legacy_ecoli.py        # 大肠杆菌历史路径整页，实际 1286 行（新增 _ecoli_legacy_page() 包装函数）
  pages_pichia.py              # 毕赤酵母页面（round1/round2/history 三个tab），实际 972 行
```

**和原计划的偏差**：原计划里`ui_cache.py`/`ui_common.py`是分开的两个文件。实际逐个检查每个函数在Pichia代码里有没有被调用后发现，`_display_name`/`_name`/`_display_dataframe`/`_flags`/`_is_empty_value`/`_drop_empty_columns`/`_deduplicate_columns`/`_training_data`/`_load_default_dataset`/`_dataset_fingerprint`/`_load_field_labels`这些"看起来通用"的函数，实际上只有大肠杆菌路径在用，Pichia路径根本没调用过（Pichia的图表全是plotly，连`_configure_plot_fonts`这个matplotlib字体配置函数都用不上）。真正两边都用的只有缓存机制（`_ui_cache_store`等）和`_num`一个格式化函数，所以合并成了一个`ui_shared.py`，避免为了一个2行函数单开一个文件。

**验证结果**（2026-07-29）：
- 原63个函数逐个核对，拆分后一个不少、没有重复（脚本化按行号切分，不是手动重打）
- 4个文件`py_compile`全部通过，4个模块`import`全部无报错
- `pytest`全量48个测试全部通过（含`tests/test_app_helpers.py`的6处`from App import app`改成`from App import pages_legacy_ecoli as app`）
- `streamlit run App/app.py`能正常启动，无报错
- 用`streamlit.testing.v1.AppTest`无头驱动了毕赤酵母模式（三个tab全渲染）和大肠杆菌历史模式，两边`exception`都是空
- 没能拿到浏览器截图做像素级视觉确认（这次会话里Browser pane打不开，环境限制，和代码无关）

### 3.2 函数归属（按当前 app.py 内容分类，行数为函数体估算，不含import/常量）

**`ui_cache.py`**（约90行）：`_ui_cache_store`、`_is_pichia_ui_cache_key`、`_restore_ui_cache_to_session`、`_remember_ui_cache`、`_clear_ui_cache`、`_read_recommendation_cache`、`_write_recommendation_cache`、`_dataset_fingerprint`

**`ui_common.py`**（约85行）：`_configure_plot_fonts`、`_load_field_labels`、`_load_default_dataset`、`_name`、`_display_name`、`_display_dataframe`、`_num`、`_flags`、`_is_empty_value`、`_drop_empty_columns`、`_deduplicate_columns`、`_metric_value`、`_metric_explanations`、`_training_data`

**`pages_legacy_ecoli.py`**（约1050行，含从`main()`里搬出来的~250行内联逻辑）：`_compare_recommenders`、`_recommendation_pool_size`、`_ensure_strategy_quality`、`_select_without_soft_filters`、`_history_sigma_ranges`、`_history_range_violation`、`_apply_soft_filters`、`_candidate_table`、`_overview`、`_nearest_history`、`_method_block`、`_standard_bo_summary`、`_standard_gp_slice_frame`、`_standard_gp_plot`、`_pdp_curve`、`_pdp_direction`、`_pdp_summary`、`_gp_pdp`、`_loocv_scatter`、`_nearest_history_validation`、`_strategy_quality_block`，外加新写一个`_ecoli_legacy_page()`把`main()`里`mode != "毕赤酵母 hLF"`分支的sidebar控件、运行按钮、数据加载、缓存读写整体包起来

**`pages_pichia.py`**（约890行）：`_pichia_hlf_page`、`_pichia_round1_tab`、`_pichia_round2_tab`、`_pichia_history_tab`、`_ensure_pichia_data_area`、`_pichia_numeric_results`、`_pichia_ui_records`、`_pichia_variable_display`、`_pichia_round1_builder`、`_pichia_yield_scatter_chart`、`_pichia_correlation_heatmap`、`_pichia_baseline_lookup`、`_pichia_format_value`、`_pichia_type_label`、`_pichia_row_note`、`_pichia_design_display_frame`、`_pichia_round1_workbook_bytes`、`_pichia_remap_uploaded_columns`、`_pichia_effect_magnitude_chart`

### 3.3 执行顺序（每步都可独立验证，避免一次性大改）

1. 先搬`ui_cache.py`/`ui_common.py`（纯函数，没有Streamlit控件顺序依赖，风险最低）
2. 再搬`pages_pichia.py`（`_pichia_hlf_page()`已经是干净的分发入口，整体剪切即可）
3. 再处理`pages_legacy_ecoli.py`（需要把`main()`里那~250行内联代码包装成`_ecoli_legacy_page()`，比单纯搬函数多一步）
4. 最后把`main()`缩到只剩`page_config` + 侧边栏 `mode` 单选 + 两行分发
5. 同步修`tests/test_app_helpers.py`里对`App.app`的import路径

### 3.4 硬约束

- 所有`st.session_state`的**key字符串**（`"recommendation_mode"`、`"bo_method"`、`"bo_seed"`、`"enable_soft_filter"`等）必须原样保留，搬移时不能顺手改名——改了就会让用户已有的浏览器session缓存失效
- 只做搬移+补import，不顺带精简函数内部逻辑（`_pichia_round1_builder`、`main()`原内联块等内部复杂度留到后续单独一轮再处理）

### 3.5 验证方式

- `pytest tests/test_app_helpers.py tests/test_recommender_comparison.py`（这两个文件目前直接`from App.app import ...`，改完import路径后必须仍然全绿）
- `streamlit run App/app.py`跑一遍手动冒烟：Pichia模式的Round1/Round2/History三个tab，以及大肠杆菌模式的运行推荐按钮，都要各点一次确认无异常

## 4. Phase 2：拆分 `excel_schema_converter.py` — ✅ 已完成

拆成两个文件：
- `excel_schema_converter.py`保留：`convert_excel_directory`、`_parse_liquid_long`、`_header_map`、`_read_time_series`等"Excel→结构化行"的解析核心（1-687行，逐个函数核对openpyxl/Worksheet依赖只出现在这一段）
- 新增`migration_audit.py`：`audit_old_nonblank_value_coverage`、`compare_csv_directories`、`write_detailed_diff_files`、`_detailed_diff`等"新旧数据迁移审计"工具（688-1317行，只处理CSV目录，不依赖openpyxl）

**注意**：这几个audit函数除了在`ingestion/__init__.py`里被re-export之外，`App/`、`tests/`、`scripts/`都没有实际调用——大概率是HMO迁移期间用过一次就没再用的工具。这次选择了**保留（搬到新文件）而不是删除**：搬移是可逆的（还在git历史里，随时能找回），删除则需要更确定这些工具真的不会再用到，这个判断留给你，我不替你做。如果确认要删，之后单独说一声就行。

**验证结果**（2026-07-29）：40个原有函数/类逐个核对，拆分后一个不少不重复；两个文件`py_compile`+`import`都通过；`ingestion/__init__.py`改成分别从两个模块导入后，`pytest`全量48个测试仍然全部通过。

## 5. Phase 3：修 `__init__.py` 导出不一致 — ✅ 已完成

`experiment_advisor/__init__.py`和`recommendation/__init__.py`目前只re-export旧路径的`service.compare_recommenders`/`recommend_next`，`round1_design`/`round2_design`完全没有被导出，`app.py`是绕开包接口直接import子模块路径的。

已给`recommendation/__init__.py`补上`generate_round1_design`、`plan_round2`、`recommend_round2_bo_batch`三个入口函数的导出（跟`compare_recommenders`/`recommend_next`同一层级——都是"做事"的函数，不是`BASELINE`/`ALL_VARIABLES`这类schema常量）。`App/pages_pichia.py`目前仍然是直接import子模块路径，没有改成走包接口——这次只补导出，不强制改调用方的import风格。

## 6. Phase 4（决策项，不在本计划内执行）：两套BO实现要不要合并

`experiment_advisor/optimizer/standard_bo.py`（BoTorch/gpytorch的qNEI GP，服务大肠杆菌历史路径）和`round2_design.py`里的`recommend_round2_bo_batch`（纯numpy/pandas自实现）是两条完全独立、互不复用的贝叶斯优化代码路径。是否值得合并取决于两者的约束处理逻辑是否真的可以共享——这需要对着数值细节单独评估，不适合放在这轮结构性重构里顺手做，列在这里留作后续决策。

## 7. 建议的推进节奏

Phase 1 > Phase 3 > Phase 2 的顺序做（1收益最大，3最快最安全，2需要先确认audit函数去留）。每个phase结束跑一次上面提到的验证方式，确认无误再进下一个。Phase 4 单独找时间讨论，不卡前三个phase的进度。
