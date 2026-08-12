"""Shared Streamlit UI helpers used by both the Pichia and legacy E.coli pages.

Split out of App/app.py: cross-session UI cache (survives Streamlit session
resets via st.cache_resource) plus the one generic display helper (_num) that
both page modules need.
"""
from __future__ import annotations

import copy
from typing import Any

import streamlit as st

UI_CACHE_VERSION = "ui-cache-v1"

PICHIA_UI_CACHE_KEYS = {
    "recommendation_mode",
    "round2_max_active_variables",
    "round2_ccd_step_fraction",
    "round2_od_threshold_fraction",
}
# "round2_bo_batch_size" and "pichia_ui_design_records" used to be listed here;
# both belonged to the Round1-only BO batch and the history tab that ADR-0016
# deleted, so nothing writes them any more and caching them cached nothing.

PICHIA_UI_CACHE_PREFIXES: tuple[str, ...] = ()
# Note: round1_builder_* widget keys deliberately are NOT in the cross-session
# cache -- they pass explicit value=/default=/index= defaults on every render,
# which Streamlit forbids combining with a pre-populated session_state entry
# (raises "widget created with a default value but also had its value set via
# the Session State API"). They're normal per-session widget state instead;
# the thing that actually needs to survive is the generated round1_results_df,
# which is a plain session_state entry, not a cached widget key.

@st.cache_resource
def _ui_cache_store() -> dict[str, Any]:
    return {"version": UI_CACHE_VERSION, "session_state": {}}

def _is_pichia_ui_cache_key(key: str) -> bool:
    if key in PICHIA_UI_CACHE_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in PICHIA_UI_CACHE_PREFIXES)

def _restore_ui_cache_to_session() -> None:
    store = _ui_cache_store()
    if store.get("version") != UI_CACHE_VERSION:
        store.clear()
        store.update({"version": UI_CACHE_VERSION, "session_state": {}})
        return
    cached = store.get("session_state") or {}
    for key, value in cached.items():
        if key in st.session_state:
            continue
        try:
            st.session_state[key] = copy.deepcopy(value)
        except Exception:
            st.session_state[key] = value

def _remember_ui_cache() -> None:
    store = _ui_cache_store()
    cached: dict[str, Any] = {}
    for key, value in st.session_state.items():
        if not _is_pichia_ui_cache_key(str(key)):
            continue
        try:
            cached[str(key)] = copy.deepcopy(value)
        except Exception:
            cached[str(key)] = value
    store["version"] = UI_CACHE_VERSION
    store["session_state"] = cached

def _clear_ui_cache() -> None:
    store = _ui_cache_store()
    store.clear()
    store.update({"version": UI_CACHE_VERSION, "session_state": {}})
    for key in list(st.session_state.keys()):
        if _is_pichia_ui_cache_key(str(key)):
            del st.session_state[key]

def _num(value: Any) -> Any:
    return round(float(value), 4) if isinstance(value, int | float) else value
