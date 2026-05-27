# App

此目录是 Streamlit bonus 交付入口，提供中文可解释界面。

运行方式：

```bash
streamlit run App/app.py
```

界面支持两种数据入口：

- 使用 `data/final/run_level_modeling_dataset.csv`
- 上传已经整理好的 run-level CSV

当前主推荐方法是 `standard_bo_ei`：标准 GP-BO 直接拟合产量，并用 EI 选择预期改进较大的候选。

`xgp_bo_ei` 作为候选参考，用 XGBoost 预测产量均值，再用 GP 拟合 XGBoost 残差。界面会单独展示残差 GP 健康诊断、指标说明和 Markdown 推荐报告。
