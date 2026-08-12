"""Pichia hLF shake-flask page: the top-level Round 1 / Round 2 tab router.

Everything the two tabs are built from lives in the App.pichia_* modules (see
docs/adr/0017 for what belongs where); this file stays deliberately thin so
that "which file do I open to change X" has a short answer.
"""
from __future__ import annotations

import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from App.pichia_common import _ensure_pichia_data_area
from App.pichia_round1 import _pichia_round1_tab
from App.pichia_round2_sections import _pichia_round2_tab

def _pichia_hlf_page() -> None:
    _ensure_pichia_data_area()
    st.caption(
        "当前模式：毕赤酵母 hLF 摇瓶实验设计。Round 1 是可配置的基线+单变量+联合探索设计构建器，"
        "Round 2 基于 Round 1 实测结果做显著性分析、响应面(CCD)设计和约束贝叶斯优化。"
    )

    round1_tab, round2_tab = st.tabs(["Round 1：实验设计", "Round 2：响应面 + 贝叶斯优化"])
    with round1_tab:
        _pichia_round1_tab()
    with round2_tab:
        _pichia_round2_tab()
