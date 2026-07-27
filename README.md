# ExperimentAdvisor

ExperimentAdvisor 是一个实验推荐 Python 项目：基于历史/阶段性实验数据生成下一批实验建议，并给出预测值、不确定性和风险解释，供研发人员审阅后再和工艺团队确认。

## 当前项目：毕赤酵母摇瓶生产 hLF（活跃）

当前目标是毕赤酵母（*Pichia pastoris*）摇瓶发酵生产 hLF（人乳铁蛋白），**从零开始**——尚无可用历史数据。此前 HMO（2FL，大肠杆菌发酵罐）阶段的实验数据已确认**完全无效**，相关分析产出已归档到 [`summary/archive_hmo/`](summary/archive_hmo/README.md)，代码中的对应路径（见下文）保留但视为历史，不再是当前推荐目标。

摇瓶实验节奏：

- 每轮 15~20 个样本（摇瓶数量远高于发酵罐，可支持更宽的探索批量）
- 默认 3 轮；结果较稳定可减至 2 轮，结果不稳定可追加 1~2 轮（即实际 2~5 轮弹性调整）
- 每轮结束后回填产量结果，再生成下一轮建议

### 已知差距（下一轮代码工作，本轮暂不处理）

`experiment_advisor/recommendation/pichia.py` 目前仍按**发酵罐口径**实现：单轮最多 4 个建议、参数包含风扇转速、生长期/生产期 pH 等罐体控制变量。这与摇瓶口径（单轮 15~20 样本）明显不匹配，批量规模和参数体系都需要针对摇瓶重新设计，留待后续代码迭代处理。

## 两条推荐路径

| | Pichia 摇瓶（当前） | HMO/2FL 发酵罐（历史，数据已作废） |
|---|---|---|
| 代码位置 | `experiment_advisor/recommendation/pichia.py` | `experiment_advisor/optimizer/` + `recommendation/service.py` |
| 方法 | 基准点（同菌种最优/最近成功、亲本菌种借鉴、手动输入）+ 小样本探索（联合 LHS 扰动 / 序贯 2 因子 DOE / 单变量验证） | `standard_bo_qnei`：BoTorch `SingleTaskGP` + MLE 超参 + qNEI 联合优化整批候选点，显式处理观测噪声 |
| 数据前提 | 零历史数据起步，依赖基准点和菌种借鉴 | 依赖历史 run-level 数据集（45+ 可训练 run） |
| 状态 | 活跃开发目标；批量口径待适配摇瓶（见上） | 保留代码供参考，不再用于生成实际推荐 |

## 项目结构

```text
experiment_advisor/
  ingestion/          # 数据读取、校验、run-level 聚合（HMO/2FL 发酵罐数据管线，历史用途）
  optimizer/          # 标准 GP-BO（qNEI）、搜索空间与约束（HMO/2FL 路径，历史用途）
  recommendation/
    pichia.py         # Pichia 摇瓶：基准点 + 小样本探索/序贯 DOE 推荐（当前活跃）
    service.py        # HMO/2FL：GP-BO 推荐服务（历史用途）
    quality.py        # 推荐质量评估
  report/             # Markdown 推荐报告生成
  analysis/           # 离线分析与诊断工具
  space/              # 参数空间定义
  utils/              # LHS 采样等工具函数
App/
  app.py              # Streamlit 界面，含 Pichia（当前）和 HMO/2FL（历史）两个页签
data/
  pichia/             # Pichia 摇瓶数据区，当前只有空模板，等待第一轮实验回填
  （其余目录为 HMO/2FL 历史数据）
summary/              # 一页总结；archive_hmo/ 归档 HMO 历史报告
Slides/               # 展示材料说明
tests/                # 自动化测试
```

根目录 `data/` 存放真实发酵实验数据，必须保持本地私有，不上传 GitHub。

## 安装

```bash
pip install -r requirements.txt
```

## 运行推荐

### Streamlit UI（推荐入口）

```bash
streamlit run App/app.py
```

UI 提供两个页签：

- **Pichia 摇瓶**：读取 `data/pichia/final/` 下的 run-level CSV（或手动录入/上传），选择基准点来源和探索方式，生成下一批建议；可下载数据模板 `data/pichia/templates/pichia_run_level_template.csv`。
- **HMO/2FL（历史）**：读取 `data/final/run_level_modeling_dataset.csv`，展示 `standard_bo_qnei` 推荐、代理模型验证、GP 偏依赖图等；数据已作废，仅保留界面供参考。

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

当前测试覆盖 run-level 数据构建、搜索空间、报告生成、qNEI 标准 BO（历史路径）和 Pichia 摇瓶推荐逻辑。
