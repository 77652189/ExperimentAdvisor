# App

此目录是课程 PDF 的 Streamlit bonus 交付入口。

运行方式：

```bash
streamlit run App/app.py
```

界面支持两种数据入口：

- 使用 `data/csv_from_excel` 构建 run-level dataset
- 上传已经整理好的 run-level CSV

当前主推荐器是 `xgp_bo_ei`：XGBoost 负责产量均值预测，GP 拟合 XGBoost 残差并提供后验不确定性。标准 GP-BO 只作为对照方法展示。
