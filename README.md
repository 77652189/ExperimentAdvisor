# ExperimentAdvisor

ExperimentAdvisor 是一个面向 HMO 发酵工艺优化的 Python 项目。它基于历史实验数据生成下一批实验建议，并给出预测值、不确定性和风险解释，供研发人员审阅后再和工艺团队确认。

## 当前推荐策略

主推荐方法是 `standard_bo_qnei`：

- 使用 BoTorch `SingleTaskGP` 直接拟合历史 `yield_g_per_l`
- 使用 MLE 优化 GP 超参数
- 使用 `qNoisyExpectedImprovement` 联合优化整批候选点
- 显式处理观测噪声，缓解历史最优值被噪声向上偏移的问题
- 输出下一批建议参数、预测产量和 GP 后验不确定性

XGBoost + GP 残差 BO 已废弃，相关优化器、服务调用、UI 页签和测试已删除。

## 项目结构

```text
experiment_advisor/
  ingestion/          # 数据读取、校验、run-level 聚合和训练视图构建
  optimizer/          # 标准 GP-BO、搜索空间与约束
  recommendation/     # 高层推荐服务
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

UI 默认读取 `data/final/run_level_modeling_dataset.csv`，使用 `standard_bo_qnei` 作为主推荐，并展示代理模型验证、推荐策略质量、GP 偏依赖图、指标说明和 Markdown 报告。

## 测试

```bash
python -m pytest -q
```

当前测试覆盖 run-level 数据构建、搜索空间、报告生成和 qNEI 标准 BO 主推荐。
