# ExperimentAdvisor 执行计划

本文件是项目进度、优先级与授权边界的唯一权威。

## 当前状态

活跃路线是毕赤酵母 hLF 摇瓶项目。2026-08 已拿到第一批真实 Round 1 数据（菌株 Y103，16 个条件），完成回填与显著性分析，K=3（`glucose_pct`/`ph`/`volume_ml` 活跃）。基于这批真实数据生成了真实的 Round 2 设计（CCD 18 点 + 补料间隔交互 2 点 + 噪声参考 2 点 + LHS 10 点，共 32 个条件），已具备下载/回填/分析的完整闭环，但**尚未有真实 Round 2 实测结果**——负责这轮设计的同事已离职，真实数据回填后由后续接手的人使用下面"已完成能力"里的分析流程，而不需要重新设计。

## 已完成能力摘要

- Round 1 支持基线重复、OFAT、LHS 的灵活组合，以及受设备限制的温度和补料间隔；上传回填时能识别单位标注（如"mg/L"自动换算）、"未检测到"等文本结果、同一编号的多次技术重复（自动取均值，原始值保留在 `_n`/`_spread`/`_reps` companion 列，不丢弃），设计变量在重复行之间不一致会报警告而不是静默按第一行处理。
- Round 2 支持基线噪声估计、效应筛选（含"从未测试"和"测过但不显著"的区分）、技术重复噪声独立诊断（`_pichia_pooled_technical_noise`，不并入显著性阈值，见 ADR-0011）、CCD 响应面设计与带 OD600 可行性筛选的贝叶斯优化候选批次。
- Round 2 新增一整套"完整设计"能力（见 ADR-0009/0010/0012）：CCD + 补料间隔交互子设计 + LHS 空间填充点的组合生成（`assemble_round2_design`，种子固定、可复现）、配套的下载/上传回填/存档界面，以及回填后的结果分析——CCD 响应面拟合（`fit_ccd_response_surface`：全二次模型、失拟检验、系数显著性、预测最优点，见 ADR-0013）、补料间隔交互显著性检验（`analyze_interval_interaction`）、Round 1+Round 2 合并数据集上的贝叶斯优化重新推荐。
- **未完成**：响应面拟合结果目前只有数值/表格展示，缺少等高线图、系数表的界面渲染、预测值-实测值残差图——底层函数（`evaluate_response_surface`/`response_surface_grid`）已实现并测试，UI 尚未接上，是当前唯一的已知半成品。
- App 重启（不是浏览器刷新）会清空 `st.session_state`；`_pichia_restore_persisted_dataset` 在两个页签打开时自动从 `data/pichia/final/` 对应的 CSV 读回，避免"明明保存过、重启后又要重新上传"的体验。
- Streamlit 默认入口已面向 hLF；旧大肠杆菌/HMO 页面被保留为历史参考。
- `App/app.py` 按 UI 关注点拆分为 `ui_shared.py`（跨会话缓存 + 共享格式化）、`pages_pichia.py`、`pages_legacy_ecoli.py`；`ingestion/excel_schema_converter.py` 拆分为解析核心与 `migration_audit.py`（新旧数据迁移审计工具）。细节见 git 历史（`archive/REFACTOR_PLAN.md` 留有当时的函数级拆分记录）。

## 待启动 / 待数据工作

| 工作 | 门控 | 现状 |
|---|---|---|
| 执行并回填真实 Round 2 结果 | 32 个条件（CCD+补料间隔交互+LHS）已生成，等待摇瓶实验实际执行、产量/OD600 回填 | 等待真实实验数据；分析流程已就位但从未跑过真实数据，数字是否合理需要接手人自己核对 |
| 补全 Round 2 结果分析的可视化 | 无门控，随时可继续 | 已暂停：等高线图/系数表界面/残差图未做，底层计算函数已完成并测试（`evaluate_response_surface`/`response_surface_grid`） |
| 评估是否要正式扩展补料间隔到 36h（写入 `FIXED_LEVELS`） | 真实 36h 数据回填、且交互检验结果支持 | 见 ADR-0012，当前有意不做 |
| 决定是否执行候选点 | 研发和工艺团队审阅候选、风险与资源 | 不由软件自动授权 |
| 决定 `optimizer/standard_bo.py`（历史 E.coli 路径）与 `recommendation/round2_design.py` 的 `recommend_round2_bo_batch`（Pichia Round 2）两套贝叶斯优化实现是否合并 | 需要真实 Round 2 数据暴露出现有实现的具体不足，或产品侧认为有必要 | 待决策，不阻塞其他工作 |

## 明确不做

- 不恢复或清洗 HMO/2FL 发酵罐历史数据来推动 hLF 推荐：其有效性已被否定。
- 不在没有 hLF 实测数据时声明模型已学得 hLF 最优条件。
- 不将原始发酵数据、用户上传结果或生成的敏感报告提交到仓库。
- 不把探索性的新补料间隔水平写入共享的 `FIXED_LEVELS` 常量，除非已有真实数据支撑（见 ADR-0012）。

## 重新评估点

完成一轮真实 Round 2 回填后，重新评估 CCD 拟合的失拟检验结果、补料间隔交互结论是否成立、以及是否需要 Round 3；在此之前不扩大为通用发酵预测平台。
