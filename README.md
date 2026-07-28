# ExperimentAdvisor

ExperimentAdvisor 是一个实验推荐 Python 项目：基于历史/阶段性实验数据生成下一批实验建议，并给出预测值、不确定性和风险解释，供研发人员审阅后再和工艺团队确认。

## 当前项目：毕赤酵母摇瓶生产 hLF（活跃）

当前目标是毕赤酵母（*Pichia pastoris*）摇瓶发酵生产 hLF（人乳铁蛋白），**从零开始**——尚无可用历史数据。此前 HMO（2FL，大肠杆菌发酵罐）阶段的实验数据已确认**完全无效**，相关分析产出已归档到 [`archive/summary/archive_hmo/`](archive/summary/archive_hmo/README.md)，代码中的对应路径（见下文）保留但视为历史，不再是当前推荐目标。

摇瓶实验节奏：

- 每轮 15~20 个样本（摇瓶数量远高于发酵罐，可支持更宽的探索批量）
- 默认 3 轮；结果较稳定可减至 2 轮，结果不稳定可追加 1~2 轮（即实际 2~5 轮弹性调整）
- 每轮结束后回填产量结果，再生成下一轮建议

### 两轮方法论

- **Round 1（实验设计）**：灵活方案构建器——基线重复、单变量法(OFAT)、联合探索(LHS) 三个模块可独立开关、任意组合（能拼出纯 LHS、纯 OFAT 或混合方案），不限定固定的"三段式"结构。种子OD600、葡萄糖浓度、pH、装液量四个连续变量的 OFAT 测试水平可自由增删；温度（20/25/30℃）和补料时间间隔（12/24h）受设备限制，锁定为固定档位，不能自定义。
- **Round 2（响应面 + 贝叶斯优化）**：基于 Round 1 实测结果，用基线重复估计的纯误差方差做显著性检验（含置信区间，t 分布，df = 基线重复数 − 1），筛出"活跃变量"（数量记作 K）；对活跃的连续变量做响应面（CCD，Central Composite Design）细化；用两个独立 GP（产量、OD600）做可行性过滤的约束贝叶斯优化，推荐下一批候选点。

## 两条推荐路径

| | Pichia 摇瓶（当前） | HMO/2FL 发酵罐（历史，数据已作废） |
|---|---|---|
| 代码位置 | `experiment_advisor/recommendation/round1_design.py` + `round2_design.py` | `experiment_advisor/optimizer/` + `recommendation/service.py` |
| 方法 | 见上「两轮方法论」 | `standard_bo_qnei`：BoTorch `SingleTaskGP` + MLE 超参 + qNEI 联合优化整批候选点，显式处理观测噪声 |
| 数据前提 | 零历史数据起步；Round 1 是纯设计生成，不需要历史数据，Round 2 依赖 Round 1 实测回填 | 依赖历史 run-level 数据集（45+ 可训练 run） |
| 输出 | Round 1：设计表格（网页预览 + 可下载配色 Excel）；Round 2：效应量图、K 值、CCD 设计表、约束 BO 候选点 | 推荐排行榜、代理模型验证、GP 偏依赖图、Markdown 报告 |
| 状态 | 活跃开发目标 | 保留代码供参考，不再用于生成实际推荐 |

## 项目结构

```text
experiment_advisor/
  ingestion/          # 数据读取、校验、run-level 聚合（HMO/2FL 发酵罐数据管线，历史用途）
  optimizer/          # 标准 GP-BO（qNEI）、搜索空间与约束（HMO/2FL 路径，历史用途）
  recommendation/
    round1_design.py  # Pichia 摇瓶 Round 1：基线+OFAT+LHS 灵活设计构建器（当前活跃）
    round2_design.py  # Pichia 摇瓶 Round 2：显著性分析、CCD 响应面、约束贝叶斯优化（当前活跃）
    service.py        # HMO/2FL：GP-BO 推荐服务（历史用途）
    quality.py        # 推荐质量评估
  report/             # Markdown 推荐报告生成（HMO/2FL 路径）
  analysis/           # 离线分析与诊断工具
  utils/              # LHS 采样等工具函数
App/
  app.py              # Streamlit 界面：毕赤酵母 hLF（当前，默认页）+ 大肠杆菌 BO（历史，次要入口）
data/
  pichia/             # Pichia 摇瓶数据区：final/ 保存确认结果，uploads/ 归档上传文件，templates/ 提供模板
  （其余目录为 HMO/2FL 历史数据）
archive/              # 归档目录，非当前工作范围（见 archive/README.md）
  summary/            # 一页总结 + HMO 历史报告；参考另一个课程项目带来的模板，非本项目交付要求
  Slides/             # 展示材料说明；同上，非本项目交付要求
tests/                # 自动化测试
```

根目录 `data/` 存放真实发酵实验数据，必须保持本地私有，不上传 GitHub。

> `summary/recommendation_report.md` 的生成路径（HMO/2FL 页签的"运行推荐"按钮，以及下面 Python 示例里的 `output_path`）目前仍写死指向根目录 `summary/`，不是 `archive/summary/`——用一次那个按钮会在根目录下重新生成一个只含这一份文件的 `summary/` 文件夹。这是本轮纯文档整理没有同步修改的已知代码路径（该文件本身在 `.gitignore` 里，不会被提交）。

## 安装

```bash
pip install -r requirements.txt
```

## 运行推荐

### Streamlit UI（推荐入口）

```bash
streamlit run App/app.py
```

侧栏可切换两个模式，默认是**毕赤酵母 hLF**：

- **毕赤酵母 hLF**（默认）：三个页签——
  - *Round 1：实验设计*——方案构建器（基线取值、基线重复次数、OFAT 参与变量与测试水平、LHS 联合探索点数与参与变量），带实时预计行数提示；可一键套用已验证方案，也可从零配置任意组合。生成后可下载配色 Excel（基线/联合探索行底色、"备注/目的"说明列、单独图例 sheet）或在网页表格直接回填产量/OD600，也可以把填好的 Excel 或 CSV 传回来继续使用。回填 ≥3 行产量后显示产量/OD600 分布图和变量相关性热力图。
  - *Round 2：响应面 + 贝叶斯优化*——可调节显著性分析参数，展示活跃变量数 (K) 和效应量条形图（含置信区间与显著性阈值线），据此生成 CCD 设计表和约束贝叶斯优化候选点。
  - *历史记录*——本次会话内的 Round 2 快照缓存（重启应用后清空，不写入文件）。
- **大肠杆菌 BO（历史，数据已作废）**：读取 `data/final/run_level_modeling_dataset.csv`，展示 `standard_bo_qnei` 推荐、代理模型验证、GP 偏依赖图等；数据已作废，仅保留界面供参考。

### Python 调用（HMO/2FL 历史路径示例）

```python
from experiment_advisor.ingestion import build_run_level_dataset
from experiment_advisor.recommendation import compare_recommenders
from experiment_advisor.report import generate_recommendation_report

df = build_run_level_dataset(
    source_dir="data/csv_from_excel",
    output_path="data/final/run_level_modeling_dataset.csv",
)

comparison = compare_recommenders(df, top_k=5)
print("主推荐方法：", comparison["selected_method"])
print(comparison["selected_recommendations"][0])

generate_recommendation_report(
    comparison,
    output_path="summary/recommendation_report.md",
)
```

## 测试

```bash
python -m pytest -q
```

当前测试覆盖 run-level 数据构建、搜索空间、报告生成、qNEI 标准 BO（历史路径），以及 Pichia 摇瓶 Round 1 设计构建器和 Round 2 显著性分析/CCD/约束 BO（`tests/test_round1_design.py`、`tests/test_round2_design.py`）。
