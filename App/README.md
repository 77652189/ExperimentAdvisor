# App

此目录是 Streamlit 交付入口，提供中文可解释界面。侧栏切换两种模式：**毕赤酵母 hLF**（当前活跃，默认）和**大肠杆菌 BO**（历史，数据已作废）。

运行方式：

```bash
streamlit run App/app.py
```

## 毕赤酵母 hLF 模式（当前，默认）

三个页签，对应 `experiment_advisor/recommendation/round1_design.py` 和 `round2_design.py`：

### Round 1：实验设计

- **方案构建器**：基线取值（4 个连续变量数字输入 + 温度/补料间隔下拉选固定档位）、基线重复次数、OFAT 参与变量与测试水平、LHS 联合探索点数与参与变量，三个模块可独立开关、任意组合。顶部有实时预计行数提示，也可一键套用已和研发组确认过的方案（18 样本）。
- **OFAT 测试水平**：连续变量的推荐水平可以直接删掉，也可以在专门的输入框里填新数值、点「+ 添加」增加自定义水平；温度和补料时间间隔受设备限制，只能在 20/25/30℃、12/24h 里选，不能新增。
- **输出**：设计生成后可下载配色 Excel（基线/联合探索行底色、"备注/目的"说明列、独立图例 sheet、Excel 自动适配行高），也可以在网页表格里直接回填产量/OD600。填好的 Excel（下载的原文件）或 CSV 都可以原样传回来继续用——上传时会自动把 Excel 的中文表头和"类型"列（如"单变量-发酵温度 (℃)"）还原成内部字段名和 run_type/changed_variable。
- **可视化**：回填 ≥3 行产量后显示产量/OD600 分布图和变量相关性热力图（Plotly）。

### Round 2：响应面 + 贝叶斯优化

- 可调节显著性分析参数（最大活跃变量数、CCD 步长比例、OD600 阈值比例）
- 展示活跃变量数 (K) 指标和效应量条形图（按 `effect_magnitude` 排序，含置信区间误差棒和显著性阈值线；置信区间用基线重复的纯误差方差 + t 分布估计，重复数少时区间会偏宽，这是真实的信息量，不是缺陷）
- 据此生成响应面（CCD）设计表，以及两个独立 GP（产量、OD600）做可行性过滤的约束贝叶斯优化候选点

### 历史记录

本次会话内的 Round 2 快照缓存（`st.session_state`，重启应用后清空，不写入文件）；Round 1 的实测结果走文件保存/下载，不依赖这里的缓存。

## 大肠杆菌 BO 模式（历史，仅供参考）

界面支持两种数据入口：

- 使用 `data/final/run_level_modeling_dataset.csv`
- 上传已经整理好的 run-level CSV

本页产出统一落在 `archive/summary/`（模块常量 `LEGACY_SUMMARY_DIR`），与该路径的归档定位一致。字段中文显示读取 `archive/summary/supporting_reports/field_dictionary.csv`；该文件不入版本控制，缺失时"字段中英对照"显示为"尚未生成字段字典"，重新运行 `python data/scripts/generate_field_dictionary.py` 即可生成。该页签展示的是 `standard_bo_qnei`（BoTorch `SingleTaskGP` + qNEI 联合优化整批候选点）推荐、代理模型验证、推荐策略质量、GP 偏依赖图和指标说明；由于 HMO 实验数据已确认无效，此页签的输出不代表当前项目结论，保留仅为界面参考。
