# ExperimentAdvisor 交接

```yaml
slice_status: in_progress
current_slice: prepare_and_execute_pichia_hlf_round1
next_action: obtain_reviewed_round1_measurements
```

## 当前目标

使用已实现的 Round 1 设计完成毕赤酵母 hLF 摇瓶首轮实验，并将真实产量与 OD600 回填到应用中，为 Round 2 分析提供输入。

## 下一步

1. 从默认 Pichia hLF 页面导出或生成 Round 1 实验表。
2. 由实验团队确认条件后执行摇瓶实验，记录每个样本的产量和 OD600。
3. 上传/回填结果，检查基线重复与数据完整性，再评审 Round 2 输出。

## 必读材料

1. [需求](REQUIREMENTS.md)：目标、验收与不可作出的声明。
2. [架构](ARCHITECTURE.md)：活跃/历史路径和数据边界。
3. `tests/test_round1_design.py`、`tests/test_round2_design.py`：设计和分析的可执行约束。

## 验证方式

运行 `python -m pytest -q`；如改动 Streamlit 页面，再用 `streamlit run App/app.py` 检查默认 Pichia 页面、Round 1 导出/回填与 Round 2 前置提示。

## 硬约束

- 当前项目是从零开始的毕赤酵母 hLF 摇瓶实验；没有可用的 hLF 历史数据。
- HMO/2FL 发酵罐数据已确认无效，相关代码和归档只能作为历史参考，不能用于当前实际推荐。
- 原始、处理后和用户上传的发酵数据不得提交到版本库。
- 软件建议必须由研发人员审阅，并与工艺团队确认后才可执行；软件不自动授权实验。
