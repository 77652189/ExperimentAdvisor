# App

此目录是 Streamlit bonus 交付入口，提供中文可解释界面。

运行方式：

```bash
streamlit run App/app.py
```

界面支持两种数据入口：

- 使用 `data/final/run_level_modeling_dataset.csv`
- 上传已经整理好的 run-level CSV

字段中文显示来自 `summary/supporting_reports/field_dictionary.csv`。如果数据字段有新增或改名，先运行：

```bash
python data/scripts/generate_field_dictionary.py
```

当前主推荐方法是 `standard_bo_qnei`：BoTorch `SingleTaskGP` 直接拟合产量，并用 qNEI 联合优化整批候选点。界面展示主推荐、代理模型验证、推荐策略质量、GP 偏依赖图、指标说明和 Markdown 推荐报告。
