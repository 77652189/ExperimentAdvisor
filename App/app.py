from __future__ import annotations

import streamlit as st

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from App.pages_legacy_ecoli import _ecoli_legacy_page
from App.pages_pichia import _pichia_hlf_page
from App.ui_shared import _remember_ui_cache, _restore_ui_cache_to_session


def main() -> None:
    st.set_page_config(
        page_title="发酵工艺优化推荐系统",
        page_icon=r"C:\Users\63097\Documents\LauncherIcons\experimentadvisor_8505.png",
        layout="wide",
    )
    _restore_ui_cache_to_session()
    st.title("发酵工艺优化推荐系统")
    with st.sidebar:
        mode = st.radio("推荐模式", ["毕赤酵母 hLF", "大肠杆菌 BO（历史，数据已作废）"], key="recommendation_mode")
    _remember_ui_cache()

    if mode == "毕赤酵母 hLF":
        _pichia_hlf_page()
        _remember_ui_cache()
        return

    _ecoli_legacy_page()


if __name__ == "__main__":
    main()
