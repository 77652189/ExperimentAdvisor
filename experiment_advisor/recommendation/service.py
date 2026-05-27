from __future__ import annotations

from typing import Any

import pandas as pd

from experiment_advisor.ingestion.run_level import TARGET_COL, training_view
from experiment_advisor.optimizer.search_space import SearchSpace, build_search_space_from_history, generate_candidates
from experiment_advisor.optimizer.standard_bo import recommend_standard_bo


def _decision_summary(recommendations: dict[str, list[dict]], review_threshold: float) -> dict[str, Any]:
    xgp = (recommendations.get("xgp_bo_ei") or [{}])[0]
    standard_bo = (recommendations.get("standard_bo_ei") or [{}])[0]
    xgp_yield = xgp.get("predicted_yield")
    bo_yield = standard_bo.get("predicted_yield")
    selected_method = "xgp_bo_ei" if recommendations.get("xgp_bo_ei") else "standard_bo_ei"
    needs_review = False
    reason = "默认采用 xgp_bo_ei：XGBoost 负责产量均值预测，GP 只拟合残差并给出后验不确定性。"

    if xgp_yield is not None and bo_yield is not None and xgp_yield:
        relative_gap = abs(float(xgp_yield) - float(bo_yield)) / abs(float(xgp_yield))
        if relative_gap > review_threshold:
            needs_review = True
            reason = f"xgp_bo_ei 与 standard_bo_ei 预测产量差超过 {review_threshold:.0%}，建议人工审议两种贝叶斯方案的差异。"
    elif selected_method == "standard_bo_ei":
        reason = "xgp_bo_ei 不可用，回退到 standard_bo_ei。"

    return {
        "selected_method": selected_method,
        "needs_human_review": needs_review,
        "review_threshold": review_threshold,
        "reason": reason,
    }


def compare_recommenders(
    df: pd.DataFrame,
    search_space: SearchSpace | None = None,
    target_col: str = TARGET_COL,
    top_k: int = 5,
    review_threshold: float = 0.10,
) -> dict[str, Any]:
    """Train models and compare XGP-BO with standard GP-BO baselines."""

    history = training_view(df, target_col) if "exclude_from_training" in df.columns else df.dropna(subset=[target_col])
    space = search_space or build_search_space_from_history(history)
    candidates = generate_candidates(space, n=5000, seed=42)
    recommendations: dict[str, list[dict]] = {}
    result: dict[str, Any] = {
        "target_col": target_col,
        "n_training_rows": int(len(history)),
        "search_space": space.bounds,
        "model_info": {
            "primary_method": "xgp_bo_ei",
            "comparison_methods": ["xgp_bo_ucb", "standard_bo_ei", "standard_bo_ucb"],
            "feature_columns": list(space.bounds),
        },
        "model_metrics": {},
        "reference_model_metrics": {},
        "recommendations": recommendations,
    }
    try:
        recommendations["standard_bo_ei"] = recommend_standard_bo(
            history,
            space,
            acquisition="ei",
            top_k=top_k,
            target_col=target_col,
            feature_cols=list(space.bounds),
            candidates=candidates,
        )
        recommendations["standard_bo_ucb"] = recommend_standard_bo(
            history,
            space,
            acquisition="ucb",
            top_k=top_k,
            target_col=target_col,
            feature_cols=list(space.bounds),
            candidates=candidates,
        )
    except ImportError as exc:
        result["standard_bo_error"] = str(exc)
    try:
        from experiment_advisor.optimizer.xgp_bo import recommend_xgp_bo

        recommendations["xgp_bo_ei"] = recommend_xgp_bo(
            history,
            space,
            acquisition="ei",
            top_k=top_k,
            target_col=target_col,
            feature_cols=list(space.bounds),
            candidates=candidates,
        )
        recommendations["xgp_bo_ucb"] = recommend_xgp_bo(
            history,
            space,
            acquisition="ucb",
            top_k=top_k,
            target_col=target_col,
            feature_cols=list(space.bounds),
            candidates=candidates,
        )
    except ImportError as exc:
        result["xgp_bo_error"] = str(exc)
    result["decision"] = _decision_summary(recommendations, review_threshold)
    result["selected_method"] = result["decision"]["selected_method"]
    result["selected_recommendations"] = recommendations.get(result["selected_method"], [])
    return result


def recommend_next(df: pd.DataFrame, top_k: int = 5, use_standard_bo: bool = False) -> dict[str, Any]:
    """Return the next recommendation, defaulting to xgp_bo_ei when available."""

    comparison = compare_recommenders(df, top_k=top_k)
    key = "standard_bo_ei" if use_standard_bo else comparison["selected_method"]
    comparison["selected_method"] = key
    comparison["selected_recommendations"] = comparison["recommendations"].get(key, [])
    return comparison
