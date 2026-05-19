from __future__ import annotations

import streamlit as st

from experiment_advisor.api.endpoints import get_next_trial, initialize
from experiment_advisor.bayes.scoring import normalize_weights


def _mode_from_label(label: str) -> str:
    return {
        "产量优先": "maximize_yield",
        "成本优先": "minimize_cost",
        "周期优先": "minimize_duration",
        "自定义权重": "weighted_custom",
    }[label]


st.set_page_config(page_title="ExperimentAdvisor", layout="wide")
st.title("ExperimentAdvisor")

mode_label = st.radio("优化模式", ["产量优先", "成本优先", "周期优先", "自定义权重"], horizontal=True)
mode = _mode_from_label(mode_label)

weights = None
if mode == "weighted_custom":
    col1, col2, col3 = st.columns(3)
    weights = {
        "yield": col1.number_input("yield 权重", min_value=0.0, value=0.6),
        "cost": col2.number_input("cost 权重", min_value=0.0, value=0.2),
        "duration": col3.number_input("duration 权重", min_value=0.0, value=0.2),
    }
    st.caption(f"归一化权重：{normalize_weights(weights, mode)}")

if st.button("初始化 DOE"):
    st.session_state["design"] = initialize(optimization_mode=mode, objective_weights=weights)

if "design" in st.session_state:
    st.dataframe(st.session_state["design"])

if st.button("获取下一批建议"):
    st.json(get_next_trial())
