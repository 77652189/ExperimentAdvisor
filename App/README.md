# App

此目录是 Streamlit 交付入口，提供中文可解释界面，含两个页签：**Pichia 摇瓶**（当前活跃）和 **HMO/2FL**（历史，数据已作废）。

运行方式：

```bash
streamlit run App/app.py
```

## Pichia 摇瓶页签（当前）

- 数据入口：`data/pichia/final/pichia_run_level_dataset.csv`，或上传/手动录入 run-level 数据
- 可下载空模板：`data/pichia/templates/pichia_run_level_template.csv`
- 选择基准点来源（同菌种历史最优/最近成功、亲本菌种借鉴、手动输入）和探索方式（联合探索 LHS、序贯 2 因子 DOE、单变量验证），生成下一批建议
- 序贯 DOE 模式支持回填实测产量后自动计算主效应/交互效应，给出下一轮基准点和范围建议
- 参数体系目前仍是发酵罐口径（见根目录 `README.md` 的已知差距说明），摇瓶批量适配待后续版本

## HMO/2FL 页签（历史，仅供参考）

界面支持两种数据入口：

- 使用 `data/final/run_level_modeling_dataset.csv`
- 上传已经整理好的 run-level CSV

字段中文显示读取代码里写死的 `summary/supporting_reports/field_dictionary.csv`。该文件的 HMO 版本已归档到 `summary/archive_hmo/supporting_reports/field_dictionary.csv`（本轮文档整理只搬移文件，未同步改代码路径），因此当前这个页签的“字段中英对照”会显示为“尚未生成字段字典”，除非重新运行 `python data/scripts/generate_field_dictionary.py`。同样，“运行推荐”会在 `summary/recommendation_report.md`（旧路径）重新生成报告。该页签展示的是 `standard_bo_qnei`（BoTorch `SingleTaskGP` + qNEI 联合优化整批候选点）推荐、代理模型验证、推荐策略质量、GP 偏依赖图和指标说明；由于 HMO 实验数据已确认无效，此页签的输出不代表当前项目结论，保留仅为界面参考。
