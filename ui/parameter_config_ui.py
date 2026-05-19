from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_advisor.api.endpoints import complete_trial, get_next_trial, initialize
from experiment_advisor.bayes.scoring import normalize_weights, required_outcomes
from experiment_advisor.config.config_manager import ConfigManager
from experiment_advisor.config.space_merger import merge_space
from experiment_advisor.data_access import load_design, load_pending, load_state, load_trials
from experiment_advisor.paths import PARAMETER_DEFAULTS_PATH
from experiment_advisor.storage import read_json

MODE_LABELS = {
    "产量优先": "maximize_yield",
    "成本优先": "minimize_cost",
    "周期优先": "minimize_duration",
    "自定义权重": "weighted_custom",
}
MODE_NAMES = {value: key for key, value in MODE_LABELS.items()}
OBJECTIVE_LABELS = {
    "yield": "产量",
    "cost": "成本",
    "duration": "周期",
    "advisor_score": "综合评分",
}


def _objective_label(key: str) -> str:
    return OBJECTIVE_LABELS.get(key, key)


def _default_variables() -> list[dict[str, Any]]:
    defaults = read_json(PARAMETER_DEFAULTS_PATH, {"variables": []})
    return defaults.get("variables", [])


def _ensure_form_state() -> None:
    if "variables" not in st.session_state:
        active = ConfigManager().get_active_config()
        st.session_state.variables = active.get("variables", _default_variables()) if active else _default_variables()
        st.session_state.optimization_mode = active.get("optimization_mode", "maximize_yield") if active else "maximize_yield"
        st.session_state.objective_weights = active.get(
            "objective_weights", {"yield": 1.0, "cost": 0.0, "duration": 0.0}
        ) if active else {"yield": 1.0, "cost": 0.0, "duration": 0.0}
        st.session_state.doe_batch_limit = 8
        st.session_state.bayes_trial_limit = 0


def _mode_index(mode: str) -> int:
    labels = list(MODE_LABELS)
    return labels.index(MODE_NAMES.get(mode, "产量优先"))


def _variable_editor() -> list[dict[str, Any]]:
    variables: list[dict[str, Any]] = []
    for index, variable in enumerate(st.session_state.variables):
        with st.container(border=True):
            cols = st.columns([1.4, 1, 1, 1, 1, 1])
            name = cols[0].text_input("变量名", value=variable.get("name", ""), key=f"name_{index}")
            unit = cols[1].text_input("单位", value=variable.get("unit", ""), key=f"unit_{index}")
            bounds = variable.get("bounds", [0.0, 1.0])
            focus = variable.get("focus_range") or variable.get("focus") or bounds
            lower = cols[2].number_input("下界", value=float(bounds[0]), key=f"lower_{index}")
            upper = cols[3].number_input("上界", value=float(bounds[1]), key=f"upper_{index}")
            focus_lower = cols[4].number_input("重点下界", value=float(focus[0]), key=f"focus_lower_{index}")
            focus_upper = cols[5].number_input("重点上界", value=float(focus[1]), key=f"focus_upper_{index}")
            if name:
                variables.append(
                    {
                        "name": name.strip(),
                        "unit": unit.strip(),
                        "bounds": [lower, upper],
                        "focus_range": [focus_lower, focus_upper],
                    }
                )
    add_col, remove_col = st.columns([1, 5])
    if add_col.button("添加变量", width="stretch"):
        st.session_state.variables.append({"name": "", "unit": "", "bounds": [0.0, 1.0], "focus_range": [0.0, 1.0]})
        st.rerun()
    if remove_col.button("删除最后一个变量", disabled=not st.session_state.variables):
        st.session_state.variables = st.session_state.variables[:-1]
        st.rerun()
    return variables


def _optimization_controls() -> tuple[str, dict[str, float] | None]:
    labels = list(MODE_LABELS)
    label = st.radio("优化模式", labels, index=_mode_index(st.session_state.optimization_mode), horizontal=True)
    mode = MODE_LABELS[label]
    weights = None
    if mode == "weighted_custom":
        current = st.session_state.objective_weights
        col1, col2, col3 = st.columns(3)
        weights = {
            "yield": col1.number_input("产量权重", min_value=0.0, value=float(current.get("yield", 0.6))),
            "cost": col2.number_input("成本权重", min_value=0.0, value=float(current.get("cost", 0.2))),
            "duration": col3.number_input("周期权重", min_value=0.0, value=float(current.get("duration", 0.2))),
        }
        normalized = normalize_weights(weights, mode)
        st.caption(
            "归一化权重："
            f"产量 {normalized['yield']:.2f}，成本 {normalized['cost']:.2f}，周期 {normalized['duration']:.2f}"
        )
    st.session_state.optimization_mode = mode
    st.session_state.objective_weights = weights or normalize_weights(None, mode)
    return mode, weights


def _config_panel(variables: list[dict[str, Any]], mode: str, weights: dict[str, float] | None) -> None:
    manager = ConfigManager()
    configs = manager.list_configs()
    names = [item["name"] for item in configs]
    selected = st.selectbox("已保存配置", names, index=0 if names else None, placeholder="暂无配置")
    col1, col2, col3 = st.columns(3)
    if col1.button("加载", disabled=not selected, width="stretch"):
        payload = manager.load_config(selected)
        st.session_state.variables = payload.get("variables", _default_variables())
        st.session_state.optimization_mode = payload.get("optimization_mode", "maximize_yield")
        st.session_state.objective_weights = payload.get(
            "objective_weights", {"yield": 1.0, "cost": 0.0, "duration": 0.0}
        )
        st.rerun()
    if col2.button("设为默认", disabled=not selected, width="stretch"):
        manager.set_default(selected)
        st.success(f"已设为默认：{selected}")
    if col3.button("删除", disabled=not selected, width="stretch"):
        manager.delete_config(selected)
        st.success(f"已删除：{selected}")
        st.rerun()

    config_name = st.text_input("配置名称", placeholder="例如：高流速验证方案")
    save_col, default_col = st.columns(2)
    if save_col.button("保存配置", width="stretch"):
        manager.save_config(config_name, variables, mode, normalize_weights(weights, mode))
        st.success("配置已保存")
    if default_col.button("设为默认并保存", width="stretch"):
        manager.save_config(config_name, variables, mode, normalize_weights(weights, mode), is_default=True)
        manager.set_default(config_name)
        st.success("默认配置已保存")


def _source_preview(variables: list[dict[str, Any]]) -> None:
    defaults = read_json(PARAMETER_DEFAULTS_PATH, {"variables": []})
    researcher_config = {"variables": variables}
    try:
        _, merge_log = merge_space(defaults, None, researcher_config)
    except ValueError as exc:
        st.warning(str(exc))
        return
    st.write("参数来源预览")
    source_names = {"researcher": "研究员配置", "literature": "文献知识", "defaults": "系统默认"}
    rows = [{"变量": name, "来源": source_names.get(source, source)} for name, source in merge_log.items()]
    st.dataframe(rows, width="stretch", hide_index=True)


def _parameter_rows(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"变量": name, "建议值": value} for name, value in parameters.items()]


def _outcome_rows(predicted: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    direction_names = {"maximize": "越高越好", "minimize": "越低越好"}
    for name, payload in predicted.items():
        value_range = payload.get("range", [])
        if value_range and value_range[0] is not None:
            display_range = f"{value_range[0]} ~ {value_range[1]}"
        else:
            display_range = "暂无预测"
        rows.append(
            {
                "指标": _objective_label(name),
                "预测区间": display_range,
                "方向": direction_names.get(payload.get("direction"), payload.get("direction", "")),
            }
        )
    return rows


def _best_rows(best_outcomes: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"指标": _objective_label(name), "历史最优": payload.get("value"), "批次": payload.get("trial_index")}
        for name, payload in best_outcomes.items()
    ]


def _recommendation_card(trial: dict[str, Any]) -> None:
    phase_name = "DOE探索" if trial.get("phase") == "doe" else "Bayes优化"
    st.info(f"第 {trial['trial_index']} 批建议 · {phase_name}")
    st.write("建议参数")
    st.dataframe(_parameter_rows(trial.get("parameters", {})), width="stretch", hide_index=True)

    predicted = trial.get("predicted_outcomes")
    if predicted:
        st.write("预测结果")
        st.dataframe(_outcome_rows(predicted), width="stretch", hide_index=True)
    if trial.get("confidence"):
        st.caption(f"当前置信度：{trial['confidence']}")
    if trial.get("best_outcomes_so_far"):
        with st.expander("历史最优", expanded=False):
            st.dataframe(_best_rows(trial["best_outcomes_so_far"]), width="stretch", hide_index=True)
    with st.expander("调试信息", expanded=False):
        st.json(trial)


def _experiment_controls(variables: list[dict[str, Any]], mode: str, weights: dict[str, float] | None) -> None:
    state = load_state()
    trials = load_trials()
    doe_done = len([trial for trial in trials if trial.get("phase") == "doe"])
    bayes_done = len([trial for trial in trials if trial.get("phase") == "bayes"])
    st.subheader("实验流程")
    cols = st.columns(4)
    cols[0].metric("阶段", state.get("phase", "doe"))
    cols[1].metric("DOE", f"{doe_done}/{state.get('doe_batch_limit', 8)}")
    bayes_limit = state.get("bayes_trial_limit")
    bayes_label = f"{bayes_done}/{bayes_limit}" if bayes_limit is not None else f"{bayes_done}/不限"
    cols[2].metric("Bayes", bayes_label)
    cols[3].metric("主目标", _objective_label(state.get("primary_objective", "yield")))

    limit_col1, limit_col2 = st.columns(2)
    doe_batch_limit = int(
        limit_col1.number_input(
            "DOE批次数",
            min_value=0,
            max_value=100,
            value=int(st.session_state.get("doe_batch_limit", state.get("doe_batch_limit", 8))),
            step=1,
        )
    )
    bayes_trial_limit_raw = int(
        limit_col2.number_input(
            "Bayes批次数上限（0表示不限）",
            min_value=0,
            max_value=500,
            value=int(st.session_state.get("bayes_trial_limit", state.get("bayes_trial_limit") or 0)),
            step=1,
        )
    )
    st.session_state.doe_batch_limit = doe_batch_limit
    st.session_state.bayes_trial_limit = bayes_trial_limit_raw
    bayes_trial_limit = None if bayes_trial_limit_raw == 0 else bayes_trial_limit_raw

    init_col, next_col = st.columns(2)
    if init_col.button("初始化实验", type="primary", width="stretch"):
        design = initialize(
            researcher_config={"variables": variables},
            optimization_mode=mode,
            objective_weights=weights,
            doe_batch_limit=doe_batch_limit,
            bayes_trial_limit=bayes_trial_limit,
        )
        st.session_state.latest_design = design
        st.success("实验流程已初始化")
        st.rerun()
    if next_col.button("获取下一批建议", width="stretch"):
        try:
            st.session_state.latest_trial = get_next_trial()
            st.rerun()
        except ValueError as exc:
            st.warning(str(exc))

    pending = load_pending()
    if pending:
        trial = pending[0]
        _recommendation_card(trial)
        with st.form("complete_trial_form"):
            st.write("录入实验结果")
            current_state = load_state()
            needed = required_outcomes(
                current_state.get("optimization_mode", "maximize_yield"),
                current_state.get("objective_weights", {"yield": 1.0}),
            )
            outcomes = {}
            for key in needed:
                outcomes[key] = st.number_input(_objective_label(key), value=0.0, key=f"outcome_{key}")
            notes = st.text_area("备注", "")
            submitted = st.form_submit_button("提交结果")
            if submitted:
                complete_trial(trial["trial_index"], outcomes, notes)
                st.success("实验结果已录入")
                st.rerun()
    else:
        st.caption("当前没有待执行 trial。")


def _data_views() -> None:
    tab_design, tab_trials, tab_state = st.tabs(["DOE矩阵", "实验结果", "状态"])
    with tab_design:
        design = load_design().get("design", [])
        rows = [{"batch_index": item["batch_index"], **item["parameters"]} for item in design]
        st.dataframe(rows, width="stretch")
    with tab_trials:
        trials = load_trials()
        rows = [
            {
                "trial_index": item["trial_index"],
                "phase": item["phase"],
                **item.get("parameters", {}),
                **{_objective_label(key): value for key, value in item.get("outcomes", {}).items()},
                "notes": item.get("notes", ""),
            }
            for item in trials
        ]
        st.dataframe(rows, width="stretch")
    with tab_state:
        state = load_state()
        trials = load_trials()
        doe_done = len([trial for trial in trials if trial.get("phase") == "doe"])
        bayes_done = len([trial for trial in trials if trial.get("phase") == "bayes"])
        cols = st.columns(5)
        cols[0].metric("阶段", state.get("phase", "doe"))
        cols[1].metric("DOE", f"{doe_done}/{state.get('doe_batch_limit', 8)}")
        bayes_limit = state.get("bayes_trial_limit")
        cols[2].metric("Bayes", f"{bayes_done}/{bayes_limit}" if bayes_limit is not None else f"{bayes_done}/不限")
        cols[3].metric("下一DOE", state.get("next_doe_index", 0))
        cols[4].metric("主目标", _objective_label(state.get("primary_objective", "yield")))
        if state.get("best_outcomes"):
            st.write("历史最优")
            st.dataframe(_best_rows(state["best_outcomes"]), width="stretch", hide_index=True)
        with st.expander("调试信息", expanded=False):
            st.json(state)


def _configuration_page() -> None:
    st.header("参数配置")
    mode, weights = _optimization_controls()
    variables = _variable_editor()
    _source_preview(variables)
    _config_panel(variables, mode, weights)


def _workflow_page() -> None:
    st.header("实验流程")
    mode = st.session_state.optimization_mode
    weights = st.session_state.objective_weights if mode == "weighted_custom" else None
    variables = [
        variable
        for variable in st.session_state.variables
        if variable.get("name")
    ]
    if not variables:
        st.warning("请先在“参数配置”菜单中至少配置一个变量。")
        return
    with st.expander("当前使用的参数配置", expanded=False):
        st.write(f"优化模式：{MODE_NAMES.get(mode, mode)}")
        if mode == "weighted_custom":
            st.write(
                "权重："
                f"产量 {st.session_state.objective_weights.get('yield', 0):.2f}，"
                f"成本 {st.session_state.objective_weights.get('cost', 0):.2f}，"
                f"周期 {st.session_state.objective_weights.get('duration', 0):.2f}"
            )
        st.dataframe(
            [
                {
                    "变量": item.get("name"),
                    "单位": item.get("unit", ""),
                    "范围": f"{item.get('bounds', ['', ''])[0]} ~ {item.get('bounds', ['', ''])[1]}",
                    "重点区间": f"{item.get('focus_range', ['', ''])[0]} ~ {item.get('focus_range', ['', ''])[1]}",
                }
                for item in variables
            ],
            width="stretch",
            hide_index=True,
        )
    _experiment_controls(variables, mode, weights)
    _data_views()


def main() -> None:
    st.set_page_config(page_title="ExperimentAdvisor", layout="wide")
    _ensure_form_state()
    st.sidebar.title("ExperimentAdvisor")
    page = st.sidebar.radio("菜单", ["参数配置", "实验流程"], label_visibility="collapsed")
    st.title(page)
    if page == "参数配置":
        _configuration_page()
    else:
        _workflow_page()


if __name__ == "__main__":
    main()
