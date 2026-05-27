# App

此目录是 Streamlit bonus 交付入口。

运行方式：

```bash
streamlit run App/app.py
```

界面支持两种数据入口：

- 使用 `data/csv_from_excel` 构建 run-level dataset
- 上传已经整理好的 run-level CSV

当前主推荐方法是 `standard_bo_ei`：标准 GP-BO 直接拟合产量，并用 EI 选择预期改进较大的候选。
`xgp_bo_ei` 作为候选参考，用 XGBoost 预测产量均值，再用 GP 拟合 XGBoost 残差。
