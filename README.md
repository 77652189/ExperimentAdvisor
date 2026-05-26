# ExperimentAdvisor

ExperimentAdvisor 是面向 HMO 发酵工艺优化的数据科学项目。项目按课程 PDF 的三段式要求组织：

1. **EXTRACT**：从 Excel/CSV 整理发酵实验数据，完成清洗、校验和特征工程。
2. **LEARN**：训练 surrogate 模型，并对标准 BO 与保守 ensemble 推荐器做对照。
3. **PREDICT**：输出下一批实验推荐、预测值、风险说明和可复现实验报告。

本项目主题为 **HMO fermentation yield recommendation**，问题类型以产量 `yield_g_per_l` 的回归建模和候选实验推荐为主。

## 课程交付目录

```text
data/              # PDF 中的 /Data：原始 Excel、CSV、final 数据、清洗脚本
Notebooks/         # EDA、建模、预测 demo notebook
App/               # Streamlit app，占 bonus 交付位
Model/             # 保存 best model、feature columns、model info
Slides/            # 5-10 页展示材料或视频说明
summary/           # 一页总结和 supporting reports
README.md          # 运行说明
```

代码实现仍保留在 `experiment_advisor/`，测试在 `tests/`。根目录 `data/` 是真实数据目录，保持原位不搬动。

## 数据目录

```text
data/
  excel/           # 原始 Excel 文件
  csv/             # 旧 CSV 或对照 CSV
  csv_from_excel/  # 从 Excel 整理得到的可信 CSV
  final/           # 校验、建模、特征工程后的正式入模数据
  scripts/         # 数据转换和清洗脚本
```

旧 DOE 运行态 JSON 已从主流程移除。

## 快速开始

```bash
pip install -r requirements.txt
```

将 Excel 转换为 schema CSV 并生成对比报告：

```bash
python data/scripts/convert_excel_to_schema_csv.py
```

运行测试：

```bash
python -m pytest -q
```

## Python 示例

```python
from experiment_advisor.ingestion import build_final_dataset
from experiment_advisor.space import build_search_space
from experiment_advisor.analysis import estimate_noise, run_offline_analysis
from experiment_advisor.optimizer import BOEngine
from experiment_advisor.optimizer.state import save
from experiment_advisor.report import generate_recommendation_report

df = build_final_dataset("data/excel/history.xlsx")
analysis = run_offline_analysis(df, output_dir="summary/supporting_reports")

engine = BOEngine(build_search_space(), noise_std=estimate_noise(df))
engine.cold_start(df)
recommendations = engine.recommend(n=1)

save(engine.ax_client, "Model/experiment_state.json")
report_md = generate_recommendation_report(
    recommendations,
    offline_analysis=analysis,
    output_path="summary/recommendation.md",
)
print(report_md)
```
