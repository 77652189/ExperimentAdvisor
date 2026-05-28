from __future__ import annotations

from typing import Any

import pandas as pd

from experiment_advisor.ingestion.run_level import TARGET_COL, training_view
from experiment_advisor.optimizer.search_space import SearchSpace, build_search_space_from_history
from experiment_advisor.recommendation.quality import evaluate_recommendation_quality
from experiment_advisor.optimizer.standard_bo import recommend_standard_bo, recommend_standard_bo_ei


STANDARD_BO_KEY = "standard_bo_qnei"
STANDARD_EI_KEY = "standard_bo_ei"

_METHOD_REASONS = {
    STANDARD_EI_KEY: (
        "采用 standard_bo_ei：顺序贪心单点 EI，每步用 set_X_pending 推开已选候选点，"
        "结果可解释性强。"
    ),
    STANDARD_BO_KEY: (
        "采用 standard_bo_qnei：BoTorch qNEI 联合优化 batch，并显式处理观测噪声。"
    ),
}


def _decision_summary(
    recommendations: dict[str, list[dict]],
    review_threshold: float,
    method_key: str,
) -> dict[str, Any]:
    """Build the decision metadata for the active recommendation method."""

    selected_method = method_key if recommendations.get(method_key) else ""
    reason = _METHOD_REASONS.get(method_key, f"采用 {method_key}。")
    if not selected_method:
        reason = f"{method_key} 不可用，未生成推荐。"

    return {
        "selected_method": selected_method,
        "needs_human_review": False,
        "review_threshold": review_threshold,
        "reason": reason,
    }


def compare_recommenders(
    df: pd.DataFrame,
    search_space: SearchSpace | None = None,
    target_col: str = TARGET_COL,
    top_k: int = 5,
    review_threshold: float = 0.10,
    seed: int = 0,
    method: str = "qnei",
) -> dict[str, Any]:
    """Run the selected GP-BO method and return the recommendation bundle.

    method: "ei"   — sequential-greedy single-point EI with set_X_pending
            "qnei" — joint batch qLogNEI (original behaviour, kept for tests)
    """

    history = (
        training_view(df, target_col)
        if "exclude_from_training" in df.columns
        else df.dropna(subset=[target_col])
    )
    space = search_space or build_search_space_from_history(history)
    recommendations: dict[str, list[dict]] = {}

    method_key = STANDARD_EI_KEY if method == "ei" else STANDARD_BO_KEY
    bo_fn = recommend_standard_bo_ei if method == "ei" else recommend_standard_bo

    result: dict[str, Any] = {
        "target_col": target_col,
        "n_training_rows": int(len(history)),
        "search_space": space.bounds,
        "model_info": {
            "primary_method": method_key,
            "candidate_methods": [],
            "feature_columns": list(space.bounds),
        },
        "model_metrics": {},
        "reference_model_metrics": {},
        "recommendations": recommendations,
    }

    try:
        bo_result = bo_fn(
            history,
            space,
            top_k=top_k,
            target_col=target_col,
            feature_cols=list(space.bounds),
            seed=seed,
        )
        recommendations[method_key] = bo_result["recommendations"]
        result["model_info"]["fitted_standard_bo_gp"] = bo_result["fitted_gp"]
        result["model_info"]["standard_bo_feature_cols"] = bo_result["feature_cols"]
        result["strategy_quality"] = evaluate_recommendation_quality(
            recommendations[method_key],
            history,
            space,
            feature_cols=bo_result["feature_cols"],
            target_col=target_col,
        )
    except ImportError as exc:
        result["standard_bo_error"] = str(exc)

    result["decision"] = _decision_summary(recommendations, review_threshold, method_key)
    result["selected_method"] = result["decision"]["selected_method"]
    result["selected_recommendations"] = recommendations.get(result["selected_method"], [])
    return result


def recommend_next(df: pd.DataFrame, top_k: int = 5, method: str = "qnei") -> dict[str, Any]:
    """Return the next batch recommendation using the specified method."""

    comparison = compare_recommenders(df, top_k=top_k, method=method)
    comparison["selected_method"] = comparison["decision"]["selected_method"]
    comparison["selected_recommendations"] = comparison["recommendations"].get(
        comparison["selected_method"], []
    )
    return comparison
