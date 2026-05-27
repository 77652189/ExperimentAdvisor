# ExperimentAdvisor

ExperimentAdvisor 是一个面向 HMO 发酵工艺优化的 Python 项目。当前定位不是自动替代实验员决策，而是基于历史实验数据生成下一批实验建议，并同时给出预测值、不确定性和风险解释，供研发人员审阅后再和工艺团队确认。

## 当前推荐策略

主推荐方法是 `standard_bo_ei`：

- 使用 Gaussian Process 直接拟合历史 `yield_g_per_l`
- 在同一搜索空间内采样合法候选点
- 使用 EI（Expected Improvement）对候选排序
- 输出下一批建议参数、预测产量和 GP 后验不确定性

候选参考方法是 `xgp_bo_ei`：

- XGBoost 学习参数到产量的非线性均值预测
- GP 只拟合 XGBoost 的训练残差
- 输出 XGP 候选建议，帮助判断标准 BO 推荐是否和非线性模型明显分歧

## 项目结构

```text
experiment_advisor/
  ingestion/          # 数据读取、校验、run-level 聚合和训练视图构建
  optimizer/          # 标准 GP-BO、XGP-BO、搜索空间与约束
  recommendation/     # 高层推荐服务：主推荐 + 候选参考
  report/             # Markdown 推荐报告
  analysis/           # 离线分析与诊断工具
  model/              # 旧模型训练/注册工具，当前不作为主推荐路径
App/
  app.py              # Streamlit bonus 入口，中文可解释界面
summary/              # 一页总结与 supporting reports
Slides/               # 展示材料说明
tests/                # 自动化测试
```

根目录 `data/` 存放真实发酵实验数据，必须保持本地私有，不上传 GitHub。

## 数据目录

```text
data/
  excel/           # 原始 Excel 文件
  csv/             # 旧 CSV 或对照 CSV
  csv_from_excel/  # 从 Excel 整理得到的可用 CSV
  final/           # run-level 建模数据
  scripts/         # 数据转换和清洗脚本
```

`data/` 已在 `.gitignore` 中忽略。

## 安装

```bash
pip install -r requirements.txt
```

## 运行推荐

### Python 调用

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

### Streamlit UI

```bash
streamlit run App/app.py
```

UI 默认读取 `data/final/run_level_modeling_dataset.csv`，使用 `standard_bo_ei` 作为主推荐，并在单独页签中展示 `xgp_bo_ei` 候选参考、残差 GP 健康诊断和指标说明。

## 测试

```bash
python -m pytest -q
```

当前测试覆盖 run-level 数据构建、搜索空间、报告生成、标准 BO 主推荐和 XGP 候选推荐。
