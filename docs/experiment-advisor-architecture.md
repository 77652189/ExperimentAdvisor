# ExperimentAdvisor 架构设计文档 v3

## 0. 决策记录（已确认，不再讨论）

| 问题 | 决策 |
|---|---|
| DOE批次 | 可配置，默认8批 |
| 变量数量 | 不设上限，LHS天然支持任意维度 |
| 优化模式 | 四种模式：产量优先、成本优先、周期优先、自定义权重；默认产量优先 |
| 多实验并行 | 不支持，单实验会话 |
| Ax状态恢复 | 每次从 `trial_results.json` 重建，不持久化Ax内部状态 |
| DashScope失败降级 | 返回格式化模板文本，不报错 |
| 命名约定 | 仓库目录：`ExperimentAdvisor`，Python包名：`experiment_advisor` |

---

## 0.1 v3补充决策（实现时以本节为准）

### 优化模式

系统不直接暴露复杂 Pareto 多目标优化，MVP 使用四种清晰模式：

| 模式 | mode | 目标 | 说明 |
|---|---|---|---|
| 产量优先 | `maximize_yield` | 最大化 `yield` | 默认模式，最适合第一版 |
| 成本优先 | `minimize_cost` | 最小化 `cost` | 需要录入成本 |
| 周期优先 | `minimize_duration` | 最小化 `duration` | 需要录入周期 |
| 自定义权重 | `weighted_custom` | 最大化 `advisor_score` | UI中配置 yield/cost/duration 权重 |

自定义权重模式不把 Ax 配置为 MultiObjective，而是把多个实测指标转换为单个派生分数 `advisor_score`，再按单目标最大化处理。这样第一版更稳定，也更容易解释。

### 自定义权重评分

`advisor_score` 按历史已完成实验的 min/max 做归一化：

```text
yield_score    = normalize_max(yield)
cost_score     = normalize_min(cost)
duration_score = normalize_min(duration)
advisor_score  = w_yield * yield_score + w_cost * cost_score + w_duration * duration_score
```

规则：
- 权重来自 UI，范围 0~1，保存前自动归一化，总和必须大于 0。
- `yield` 越高越好，`cost` 和 `duration` 越低越好。
- 某个权重大于 0 时，`complete_trial()` 必须录入对应 outcome，缺失则抛出 `ValueError`。
- 若某个指标历史 min=max，该指标本轮归一化分数记为 0.5，避免除零。
- 所有原始 outcomes 仍完整保存，`advisor_score` 作为派生字段写入 trial 记录，便于审计。

### primary_objective

`experiment_state.json` 增加：

```json
{
  "optimization_mode": "maximize_yield",
  "primary_objective": "yield",
  "objective_weights": {"yield": 1.0, "cost": 0.0, "duration": 0.0}
}
```

映射规则：
- `maximize_yield` -> `primary_objective="yield"`
- `minimize_cost` -> `primary_objective="cost"`
- `minimize_duration` -> `primary_objective="duration"`
- `weighted_custom` -> `primary_objective="advisor_score"`

`fixed_vars` 固定值取 primary objective 最优 trial 中对应变量的实际值。若 `primary_objective="advisor_score"`，则取 `advisor_score` 最高的 trial。

### Ax实现策略

Ax 第一版统一按单目标运行：
- `maximize_yield`：Ax objective 为 `yield`，maximize=True。
- `minimize_cost`：Ax objective 为 `cost`，maximize=False。
- `minimize_duration`：Ax objective 为 `duration`，maximize=False。
- `weighted_custom`：Ax objective 为 `advisor_score`，maximize=True。

warm_start_points 只 attach 满足当前模式必要 outcome 字段的点；字段不完整的 warm start 跳过并记录 warning，不假设 Ax 会自动忽略缺失目标。

### 变量数量提示

LHS 实现层面不限制变量数量，但默认 8 批在高维空间中信息量不足。MVP 建议 2~6 个变量；当变量数 > 6 且 DOE 批次仍为 8 时，UI 给出提示：建议增加 DOE 批次或减少变量。

---
## 1. 模块定位

`experiment_advisor` 是「HMO发酵助手」项目的子模块，由一名工程师独立负责。

**核心职责：**
- 接收文献模块（papersort）输出的结构化知识JSON
- 支持实验人员通过Streamlit UI录入并管理参数配置
- 将三路来源（文献 / 研究员配置 / 系统默认）合并为最终参数空间
- DOE阶段（前8批）：自动生成LHS探索性采样矩阵，分析主效应，变量数量不设上限
- 贝叶斯阶段（第9批起）：按优化模式输出下一批最优参数建议
- 通过LLM将建议翻译成中文自然语言报告

**不负责：**
- 文献解析与知识提取（knowledge层留空占位，接口预留）
- 其他模块的数据分析、异常识别

---

## 2. 命名约定

- 仓库/目录名：`ExperimentAdvisor`
- Python包名：`experiment_advisor`（下划线，用于import）
- 文档/注释统一用：`ExperimentAdvisor`

---

## 3. 外部依赖与接口

### 3.1 输入：knowledge_rules.json

由papersort模块单独运行后输出，放置于 `data/knowledge_rules.json`。
本模块只读取，不写入。**此文件不存在时模块正常运行，降级为文献来源为空。**

**固定Schema（papersort必须严格遵守此格式）：**

```json
{
  "schema_version": "1.0",
  "variables": [
    {
      "name": "lactose_flow",
      "unit": "g/L/h",
      "bounds": [2.0, 5.5],
      "focus_range": [3.0, 4.5],
      "source": "literature"
    },
    {
      "name": "temperature",
      "unit": "°C",
      "bounds": [30.0, 34.0],
      "focus_range": [31.0, 33.0],
      "source": "literature"
    }
  ],
  "hard_constraints": [
    {
      "var": "lactose_flow",
      "op": ">",
      "value": 6.0,
      "reason": "乙酸积累风险",
      "confidence": "high"
    },
    {
      "conditions": [
        {"var": "temperature", "op": ">", "value": 34.0},
        {"var": "lactose_flow", "op": ">", "value": 4.0}
      ],
      "logic": "and",
      "reason": "高温高流速失败组合",
      "confidence": "medium"
    }
  ],
  "warm_start_points": [
    {"lactose_flow": 3.5, "temperature": 32.0, "yield": 92.0, "cost": 1.2, "duration": 48.0},
    {"lactose_flow": 4.0, "temperature": 31.0, "yield": 85.0, "cost": 1.1, "duration": 52.0}
  ]
}
```

**约束格式说明（不使用eval，全部结构化）：**

单变量约束：`{"var": "x", "op": ">/</>=/<=", "value": 数字}`

多变量组合约束：`{"conditions": [...], "logic": "and/or"}`

constraint_handler只处理这两种结构，不接受字符串表达式。

### 3.2 对外接口

```python
# Python函数接口，后续可套FastAPI壳，函数签名不变

def initialize(
    researcher_config: dict | None = None,
    optimization_mode: str = "maximize_yield",
    objective_weights: dict | None = None
) -> pd.DataFrame:
    """
    初始化实验会话：
    - optimization_mode：四选一：maximize_yield / minimize_cost / minimize_duration / weighted_custom
    - objective_weights：仅 weighted_custom 使用，示例 {"yield": 0.6, "cost": 0.2, "duration": 0.2}
    - 合并三路参数空间
    - 生成DOE矩阵，写入doe_design.json
    - 初始化experiment_state.json（phase="doe", completed_count=0, optimization_mode等）
    - 返回DOE实验矩阵（DataFrame）
    researcher_config：UI"直接使用"时传入，None则从active_config读取
    """

def complete_trial(
    trial_index: int,
    outcomes: dict,
    notes: str = ""
) -> None:
    """
    录入实验结果：
    - outcomes：字典，key为目标名，value为实测值
                示例：{"yield": 88.5, "cost": 1.3, "duration": 50.0}
                需录入当前optimization_mode要求的字段，多余key原样保存但不参与评分
    - 追加写入trial_results.json
    - 更新experiment_state.json
    """

def get_next_trial() -> dict:
    """
    获取下一批建议：
    - DOE阶段：返回doe_design中的下一行，completed_count达到上限时触发effect_analyzer并切换贝叶斯
    - 贝叶斯阶段：从trial_results重建Ax → 输出建议
    返回格式见7.4节optimizer
    """
```

---

## 4. 目录结构

```
ExperimentAdvisor/
│
├── data/
│   ├── knowledge_rules.json         # papersort输出（只读，可缺失）
│   ├── parameter_defaults.json      # 系统兜底默认值（手动维护）
│   ├── parameter_configs/           # 研究员保存的命名配置
│   │   └── {config_name}.json
│   ├── doe_design.json              # DOE实验矩阵（initialize时生成）
│   ├── trial_results.json           # 所有已完成实验结果（append-only）
│   ├── pending_trials.json          # 已推荐但未录入结果的trial（见第6节）
│   └── experiment_state.json        # 实验会话状态（见第6节）
│
├── knowledge/                       # 【留空占位】
│   └── __init__.py
│
├── experiment_advisor/              # Python包根目录
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config_manager.py        # 配置增删改查
│   │   └── space_merger.py          # 三路合并
│   │
│   ├── doe/
│   │   ├── __init__.py
│   │   ├── space_builder.py         # 调用space_merger，构建参数空间
│   │   ├── design_generator.py      # LHS生成实验矩阵 + 约束过滤
│   │   └── effect_analyzer.py       # 主效应分析
│   │
│   ├── bayes/
│   │   ├── __init__.py
│   │   ├── initializer.py           # 从trial_results重建AxClient
│   │   ├── constraint_handler.py    # 结构化约束过滤（无eval）
│   │   └── optimizer.py             # Ax Client封装
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py                 # LangGraph状态定义
│   │   ├── graph.py                 # 节点编排
│   │   └── nodes/
│   │       ├── doe_node.py
│   │       ├── effect_node.py
│   │       ├── bayes_node.py
│   │       └── report_node.py
│   │
│   └── api/
│       └── endpoints.py             # 三个对外函数
│
├── ui/
│   └── parameter_config_ui.py       # Streamlit界面
│
├── tests/
│   ├── test_config.py
│   ├── test_doe.py
│   ├── test_bayes.py
│   ├── test_constraints.py
│   └── fixtures/
│       ├── mock_knowledge_rules.json
│       └── mock_trial_results.json
│
├── main.py                          # 端到端手动调试入口
└── requirements.txt
```

---

## 5. 技术栈

| 用途 | 库 |
|---|---|
| DOE实验设计 | `scipy.stats.qmc.LatinHypercube`（LHS采样）|
| 贝叶斯优化 | `ax-platform`（底层BoTorch + GPyTorch）|
| Agent编排 | `langgraph` |
| LLM调用 | `dashscope`（Qwen，失败时降级为模板）|
| 参数配置UI | `streamlit` |
| 数据处理 | `pandas`, `numpy` |
| 持久化 | 标准库 `json` |
| 接口预留 | `fastapi`（预留，当前不启动）|

```
# requirements.txt
ax-platform>=0.3.0
langgraph>=0.1.0
dashscope>=1.14.0
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0          # LHS采样
fastapi>=0.100.0       # 预留
uvicorn>=0.23.0        # 预留
```

---

## 6. 状态持久化设计

### 6.1 experiment_state.json（核心）

进程重启后所有状态从此文件恢复，Ax每次从trial_results.json重建，不持久化内部状态。

```json
{
  "phase": "doe",
  "doe_batch_limit": 8,
  "completed_count": 3,
  "next_doe_index": 3,
  "optimization_mode": "maximize_yield",
  "primary_objective": "yield",
  "objective_weights": {"yield": 1.0, "cost": 0.0, "duration": 0.0},
  "effect_report": null,
  "best_outcomes": {
    "yield": {"value": 82.5, "trial_index": 2},
    "cost":  {"value": 1.1,  "trial_index": 1}
  },
  "initialized_at": "2026-05-19T10:00:00",
  "last_updated": "2026-05-19T15:30:00"
}
```

字段说明：
- `phase`：`"doe"` 或 `"bayes"`，切换后不可逆
- `next_doe_index`：DOE阶段下次应返回doe_design的第几行，**只在complete_trial成功后推进，不在get_next_trial时推进**
- `completed_count`：已录入结果的批次数
- `optimization_mode`：本次实验的优化模式
- `primary_objective`：用于Ax建模和fixed_vars选择的主目标
- `objective_weights`：自定义权重模式下的归一化权重
- `effect_report`：DOE结束后写入，贝叶斯初始化时读取
- `best_outcomes`：每个目标的历史最优值及对应批次，供report_node对比描述

### 6.2 pending_trials.json（防重放机制）

`get_next_trial()` 返回建议后立即写入此文件。`complete_trial()` 先从此文件校验参数，再转入trial_results.json。

```json
{
  "pending": [
    {
      "trial_index": 3,
      "phase": "doe",
      "parameters": {"lactose_flow": 3.0, "temperature": 31.0},
      "suggested_at": "2026-05-19T14:00:00"
    }
  ]
}
```

**关键规则：**
- 任何时刻pending列表里最多只有1条记录（单会话）
- `get_next_trial()` 被重复调用时：若pending非空，直接返回pending中的记录，不重新生成
- `complete_trial()` 成功后：从pending中移除该trial_index，再推进next_doe_index（DOE阶段）
- 进程重启后：pending中的记录仍然有效，用户可以直接录入结果

### 6.3 Ax状态恢复机制

每次调用 `get_next_trial()` 进入贝叶斯阶段时：

```
读取 trial_results.json（所有已完成记录，DOE和bayes均包含）
        ↓
读取 knowledge_rules.json 的 warm_start_points（若存在）
        ↓
重建 AxClient：
  warm_start_points → attach，noise=0.2（文献置信度低）
  DOE结果          → attach，noise=0.0（真实实验）
  bayes结果        → attach，noise=0.0（真实实验）
        ↓
获取下一个候选点
        ↓
写入 pending_trials.json，返回结果（不保存AxClient实例）
```

**为什么DOE结果也要进Ax：**
DOE的8批真实实验是质量最高的训练数据，全部丢弃会让贝叶斯优化从接近零样本出发，浪费了最有价值的探索信息。

代价：每次重建约1~2秒（数据量小时可接受）。
优点：无需处理Ax序列化，进程重启后天然恢复。

### 6.4 trial_results.json

append-only，不修改历史记录。`parameters` 用嵌套对象，支持变量数变化时向前兼容。

```json
{
  "trials": [
    {
      "trial_index": 0,
      "phase": "doe",
      "parameters": {"lactose_flow": 3.0, "temperature": 31.0},
      "outcomes": {"yield": 78.5, "cost": 1.2},
      "notes": "",
      "recorded_at": "2026-05-19T14:00:00"
    }
  ]
}
```

读写规范：读取全量 → 追加 → 整体写回，`indent=2`，文件不存在时自动初始化空结构。

---

## 7. 各层详细设计

### 7.1 config层

参数空间的唯一入口，所有其他层通过它获取变量配置，不直接读取JSON文件。

#### config_manager.py

```python
class ConfigManager:
    def list_configs(self) -> list[dict]:
        """返回所有已保存配置，含name、created_at、is_default"""

    def load_config(self, config_name: str) -> dict:
        """加载指定配置，文件不存在抛出ValueError"""

    def save_config(self, config_name: str, variables: list) -> None:
        """保存到 data/parameter_configs/{config_name}.json
        config_name不能含路径分隔符，写入前校验"""

    def set_default(self, config_name: str) -> None:
        """将目标config的is_default置True，其余全部置False"""

    def delete_config(self, config_name: str) -> None:
        """删除配置文件，若为default则同时清除标记"""

    def get_active_config(self) -> dict | None:
        """返回is_default=True的配置，无则返回None"""
```

#### space_merger.py

三路合并，优先级：researcher_config > literature > defaults

```python
def merge_space(
    defaults: dict,
    knowledge_rules: dict | None,
    researcher_config: dict | None
) -> tuple[dict, dict]:
    """
    合并规则：
    1. 以defaults["variables"]为基础建立变量字典
    2. knowledge_rules存在时，用其variables覆盖同名变量的所有字段
    3. researcher_config存在时，用其variables再次覆盖（整变量替换）
    4. knowledge_rules为None时静默跳过

    返回：(space, merge_log)
    space格式：
    {
        "lactose_flow": {"bounds": [2.0, 5.5], "focus": [3.0, 4.5], "unit": "g/L/h"},
        "temperature":  {"bounds": [30.0, 34.0], "focus": [31.0, 33.0], "unit": "°C"},
    }
    merge_log格式：{"lactose_flow": "researcher", "temperature": "literature"}
    """
```

---

### 7.2 UI层（Streamlit）

启动命令：`streamlit run ui/parameter_config_ui.py`

**页面布局：**

```
┌─────────────────────────────────────────┐
│  📋 实验参数配置                          │
│                                         │
│  已保存配置: [下拉▼]  [加载] [删除]       │
│                       [设为默认]          │
├─────────────────────────────────────────┤
│  优化目标（至少选1个）                    │
│  ☑ 产量 yield (g/L)                     │
│  ☐ 成本 cost (元/g)                     │
│  ☐ 周期 duration (h)                    │
├─────────────────────────────────────────┤
│  当前参数设置                             │
│  变量: lactose_flow  单位: g/L/h         │
│    范围: [2.0] ~ [5.5]                  │
│    重点区间: [3.0] ~ [4.5]              │
│  变量: temperature   单位: °C           │
│    范围: [30.0] ~ [34.0]                │
│    重点区间: [31.0] ~ [33.0]            │
│  [＋ 添加变量]  [－ 删除最后一个]         │
├─────────────────────────────────────────┤
│  参数来源预览                             │
│  ✅ lactose_flow — researcher            │
│  📖 temperature  — literature           │
├─────────────────────────────────────────┤
│  配置名称: [___________]                 │
│  [保存配置]  [设为默认并保存]  [直接使用] │
└─────────────────────────────────────────┘
```

**交互逻辑：**

- 页面加载：读active_config填充表单，无则读系统defaults
- 优化模式：单选框，默认产量优先；选择自定义权重时展示 yield/cost/duration 权重输入并校验权重总和 > 0
- 加载：用选中配置覆盖表单，不自动保存
- 删除：delete_config，刷新下拉列表
- 保存配置：config_name不能为空，不能含 `/\` 字符，同时保存 optimization_mode 和 objective_weights
- 设为默认并保存：save + set_default原子执行
- 直接使用：不保存，将表单值、optimization_mode 和 objective_weights 传入initialize()
- 参数来源预览：调用space_merger获取merge_log，实时展示

---

### 7.3 doe层

#### space_builder.py

```python
def build_space(researcher_config: dict | None = None) -> tuple[dict, dict]:
    """
    1. 加载 data/parameter_defaults.json
    2. 尝试加载 data/knowledge_rules.json，失败则传None
    3. researcher_config由外部传入或从get_active_config()读取
    4. 调用space_merger.merge_space()
    返回：(space, merge_log)
    """
```

#### design_generator.py

**DOE设计策略（使用LHS而非pbdesign/ff2n）：**

pbdesign的批次数由变量数决定（非8），ff2n是全因子设计变量多时爆炸。
**采用LHS（拉丁超立方采样）**：批次数完全由n_trials控制，变量数无上限，均匀覆盖性好。

```python
def generate_design(space: dict, n_trials: int = 8) -> pd.DataFrame:
    """
    使用 scipy.stats.qmc.LatinHypercube 生成 n_trials 行设计矩阵
    采样范围：focus_range（若无则用bounds）

    生成后对每个点执行constraint_handler.is_valid()：
    - 不合法的点用focus_range中心点替换，记录warning
    - 替换比例超过20%时抛出ValueError，提示约束过于严格

    写入 data/doe_design.json
    返回 pd.DataFrame
    """
```

#### effect_analyzer.py

```python
def analyze_effects(space: dict) -> dict:
    """
    从trial_results.json读取phase="doe"的所有记录
    对每个变量计算主效应：高值组产量均值 - 低值组产量均值（以中位数分界）

    返回：
    {
        "significant_vars": ["lactose_flow"],
        "fixed_vars": ["temperature"],
        "effect_sizes": {"lactose_flow": 12.3, "temperature": 2.1},
        "ready_for_bayes": True
    }
    同时将此结果写入experiment_state.json["effect_report"]
    """
```

---

### 7.4 bayes层

#### constraint_handler.py

**结构化约束，不使用eval：**

```python
import operator
SUPPORTED_OPS = {">": operator.gt, "<": operator.lt,
                 ">=": operator.ge, "<=": operator.le, "==": operator.eq}

def _check_single(constraint: dict, params: dict) -> bool:
    """{"var": "x", "op": ">", "value": 1.0}"""
    var, op, val = constraint["var"], constraint["op"], constraint["value"]
    if var not in params:
        return False  # 变量不存在则跳过此约束
    return SUPPORTED_OPS[op](params[var], val)

def _check_compound(constraint: dict, params: dict) -> bool:
    """{"conditions": [...], "logic": "and/or"}"""
    results = [_check_single(c, params) for c in constraint["conditions"]]
    return all(results) if constraint["logic"] == "and" else any(results)

def is_valid(params: dict, hard_constraints: list) -> bool:
    """命中任一约束返回False（禁止区域）"""
    for c in hard_constraints:
        hit = _check_compound(c, params) if "conditions" in c else _check_single(c, params)
        if hit:
            return False
    return True
```

#### initializer.py

```python
def build_ax_client(space: dict, constraints: list, optimization_mode: str, objective_weights: dict | None) -> AxClient:
    """
    每次调用时从trial_results.json完整重建AxClient，不依赖持久化状态。

    步骤：
    1. 创建新AxClient，experiment名固定为 "hmo_fermentation"
    2. 用space定义parameters（仅significant_vars，fixed_vars不参与优化）
    3. 根据optimization_mode配置单目标Ax：
         maximize_yield    -> objective=yield, maximize=True
         minimize_cost     -> objective=cost, maximize=False
         minimize_duration -> objective=duration, maximize=False
         weighted_custom   -> objective=advisor_score, maximize=True
    4. 读knowledge_rules的warm_start_points，attach为历史trial（noise=0.2）
       缺少当前模式必要outcome字段的warm_start跳过并记录warning
    5. 读trial_results.json中全部记录（DOE和bayes均包含），attach为真实trial（noise=0.0）
       DOE结果是质量最高的训练数据，全量纳入可显著提升第9批建议的准确性
    6. 返回已fit的AxClient
    """
```

#### optimizer.py

```python
class ExperimentOptimizer:
    def __init__(self, space: dict, constraints: list):
        self.space = space
        self.constraints = constraints

    def get_next_trial(self) -> dict:
        """
        每次调用都重建AxClient（调用initializer.build_ax_client）
        获取候选点后执行constraint_handler.is_valid()：
          不合法则重新采样，最多5次
          超过5次取focus_range中心点，记录warning

        返回格式（多目标时predicted_outcomes包含所有选定目标）：
        {
            "trial_index": 9,
            "phase": "bayes",
            "parameters": {"lactose_flow": 3.8, "temperature": 32.0},
            "predicted_outcomes": {
                "yield":    {"range": [85.0, 93.0], "direction": "maximize"},
                "cost":     {"range": [1.0, 1.3],  "direction": "minimize"},
                "duration": {"range": [46.0, 52.0],"direction": "minimize"}
            },
            "confidence": "medium",      # <10批=low，10~20=medium，>20=high
            "best_outcomes_so_far": {
                "yield": {"value": 82.5, "trial_index": 6},
                "cost":  {"value": 1.1,  "trial_index": 4}
            }
        }
        """

    def complete_trial(self, trial_index: int, outcomes: dict, notes: str = "") -> None:
        """
        追加写入trial_results.json（outcomes字典格式）
        更新experiment_state.json（completed_count, best_outcomes）
        不保持AxClient实例
        """
```

---

### 7.5 agent层

#### state.py

```python
from typing import TypedDict, Optional
import pandas as pd

class AdvisorState(TypedDict):
    researcher_config: Optional[dict]       # UI传入的参数配置
    optimization_mode: str                  # maximize_yield / minimize_cost / minimize_duration / weighted_custom
    primary_objective: str                  # yield / cost / duration / advisor_score
    objective_weights: dict                 # 自定义权重，非weighted模式也保存归一化默认值
    space: Optional[dict]                   # space_builder输出
    merge_log: Optional[dict]               # 参数来源记录
    doe_design: Optional[pd.DataFrame]      # DOE矩阵
    effect_report: Optional[dict]           # 主效应分析结果
    current_trial: Optional[dict]           # 当前轮次建议（optimizer输出）
    report: Optional[str]                   # LLM报告或降级模板
    phase: str                              # "doe" 或 "bayes"
    doe_batch_limit: int                    # 默认8
    completed_count: int                    # 已完成批次数
    best_outcomes: Optional[dict]           # 每个目标的历史最优
    error: Optional[str]
```

**注意：optimizer实例不放入state，避免LangGraph序列化问题。**
每个节点内部临时创建ExperimentOptimizer，用完即弃。

#### graph.py

```
节点：
  doe_node      调用design_generator，返回实验矩阵
  effect_node   调用effect_analyzer，更新state和experiment_state.json
  bayes_node    临时创建ExperimentOptimizer，调用get_next_trial()
  report_node   调用DashScope，失败时返回FALLBACK_TEMPLATE

边逻辑：
  START → doe_node → END
  （等待外部complete_trial触发，下次以effect_node为入口重入）

  effect_node → 条件路由：
    phase=="doe" and completed_count < doe_batch_limit → doe_node
    phase=="doe" and completed_count >= doe_batch_limit → bayes_node（切换）
    phase=="bayes" → bayes_node

  bayes_node → report_node → END
```

#### nodes/report_node.py

```python
FALLBACK_TEMPLATE = (
    "第 {trial_index} 批建议参数：\n"
    "{param_lines}\n"
    "预测结果：\n{outcome_lines}\n"
    "（置信度：{confidence}）\n"
    "各目标历史最优：\n{best_lines}"
)

def report_node(state: AdvisorState) -> AdvisorState:
    """
    尝试调用DashScope qwen-turbo：
      system: 你是HMO发酵实验AI助手，帮助研究员理解贝叶斯优化建议
      user: 建议参数{params}，预测结果{predicted_outcomes}（含所有选定目标的预测区间），
            置信度{conf}，各目标历史最优{best_outcomes}，
            请用不超过200字说明建议理由，并指出各目标的权衡关系和主要风险

    失败时（任何异常）：
      记录warning，不写入state["error"]
      用FALLBACK_TEMPLATE格式化返回，保证流程不中断
    """
```

---

## 8. 阶段切换逻辑

```
initialize() → phase="doe", next_doe_index=0, pending_trials=[]

get_next_trial()：

  # 第一步：检查pending，有则直接返回，不重新生成
  if pending_trials非空:
      return pending_trials[0]

  # 第二步：判断是否需要切换（先判断再取数，避免数组越界）
  if phase=="doe" and completed_count >= doe_batch_limit:
      调用effect_analyzer()，写入experiment_state.json["effect_report"]
      phase = "bayes"（写入experiment_state.json，不可逆）

  # 第三步：根据当前phase生成建议
  if phase=="doe":
      result = doe_design[next_doe_index]（next_doe_index此时不推进）
      写入pending_trials.json
      return result

  if phase=="bayes":
      重建AxClient（warm_start + 全部trial_results）
      result = get_next_trial()
      写入pending_trials.json
      return result

complete_trial(trial_index, outcomes)：
  校验trial_index存在于pending_trials.json中，不存在则抛出ValueError
  追加写入trial_results.json
  更新experiment_state.json（completed_count+1，best_outcomes）
  从pending_trials.json中移除该trial_index
  if phase=="doe":
      next_doe_index += 1  ← 只在此处推进，不在get_next_trial推进
```

**fixed_vars的取值策略（MVP）：**

effect_analyzer标记为fixed_vars的变量不参与贝叶斯优化，但最终建议仍需给出完整参数。
取值规则：固定为**primary_objective历史最优trial中该变量的实际值**。

```python
# initializer.py中的fixed_vars处理
best_trial = trial_results[state["best_outcomes"][state["primary_objective"]]["trial_index"]]
fixed_values = {var: best_trial["parameters"][var] for var in effect_report["fixed_vars"]}
# optimizer.get_next_trial()返回时，在parameters中补入fixed_values
```

选择"主目标历史最优trial对应值"的理由：简单、可解释（"我们在主目标表现最好的那批实验里，该变量是X，所以沿用"），优于focus_range中心值（没有实验依据）。

---

## 9. 错误处理规范

| 场景 | 处理方式 |
|---|---|
| knowledge_rules.json缺失 | 静默跳过，以defaults + researcher_config继续 |
| researcher_config缺失 | 以defaults + literature继续 |
| DOE生成点命中约束 | 用focus_range中心点替换，记录warning |
| 约束太严导致>20%替换 | 抛出ValueError，提示研究员放宽约束范围 |
| Ax采样5次仍不合法 | 返回focus_range中心点，记录warning |
| DashScope调用失败 | 降级为FALLBACK_TEMPLATE，不中断流程 |
| 节点内部异常 | 写入state["error"]，graph跳过后续节点直达END |
| trial_results.json不存在 | 自动初始化空结构 |
| experiment_state.json不存在 | 自动初始化，phase="doe" |

---

## 10. 开发顺序

```
Step 1:  data/ 初始化脚本（创建所有JSON空文件和parameter_configs/目录）
Step 2:  experiment_advisor/config/config_manager.py + test_config.py
Step 3:  experiment_advisor/config/space_merger.py + test_config.py补充
Step 4:  ui/parameter_config_ui.py（可独立开发，不阻塞后续）
Step 5:  experiment_advisor/doe/space_builder.py
Step 6:  experiment_advisor/bayes/constraint_handler.py + test_constraints.py
         （必须在Step 7之前完成，DOE生成后需要过滤约束）
Step 7:  experiment_advisor/doe/design_generator.py（LHS + 约束过滤）+ test_doe.py
Step 8:  experiment_advisor/doe/effect_analyzer.py + test_doe.py补充
Step 9:  experiment_advisor/bayes/initializer.py（重建AxClient）
Step 10: experiment_advisor/bayes/optimizer.py + test_bayes.py
Step 11: experiment_advisor/agent/state.py
Step 12: experiment_advisor/agent/nodes/（四个节点文件）
Step 13: experiment_advisor/agent/graph.py
Step 14: experiment_advisor/api/endpoints.py
Step 15: main.py（端到端：initialize → 模拟8批DOE → 切换贝叶斯 → get_next_trial）
```


