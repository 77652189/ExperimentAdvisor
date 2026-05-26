from __future__ import annotations

from typing import Any

import pandas as pd

from experiment_advisor.ingestion.run_level import TARGET_COL, training_view
from experiment_advisor.model.trainer import ModelBundle, train_surrogate_ensemble
from experiment_advisor.optimizer.conservative import ConservativeWeights, recommend_conservative
from experiment_advisor.optimizer.search_space import SearchSpace, build_search_space_from_history, generate_candidates
from experiment_advisor.optimizer.standard_bo import recommend_standard_bo


def _single_xgboost_baseline(
    model_bundle: ModelBundle,
    history_df: pd.DataFrame,
    search_space: SearchSpace,
    candidates: pd.DataFrame,
) -> list[dict]:
    if "xgboost" not in model_bundle.models:
        return []
    xgb_bundle = ModelBundle(
        models={"xgboost": model_bundle.models["xgboost"]},
        reference_models={},
        feature_columns=model_bundle.feature_columns,
        target_col=model_bundle.target_col,
        metrics={"xgboost": model_bundle.metrics.get("xgboost", {})},
        reference_metrics={},
        model_info=model_bundle.model_info,
    )
    result = recommend_conservative(
        xgb_bundle,
        history_df,
        search_space,
        top_k=5,
        candidates=candidates,
        weights=ConservativeWeights(exploration_weight=0.0, distance_penalty=0.0, boundary_penalty=0.0),
    )
    for item in result:
        item["method"] = "single_xgboost"
    return result


def _random_safe_baseline(search_space: SearchSpace, candidates: pd.DataFrame) -> list[dict]:
    rows = candidates.head(5)
    return [
        {
            "method": "random_safe",
            "rank": int(rank),
            "params": {name: float(row[name]) for name in search_space.bounds},
            "predicted_yield": None,
            "model_uncertainty": None,
            "acquisition_score": None,
        }
        for rank, (_, row) in enumerate(rows.iterrows(), start=1)
    ]


def _decision_summary(recommendations: dict[str, list[dict]], review_threshold: float) -> dict[str, Any]:
    conservative = (recommendations.get("conservative_ensemble") or [{}])[0]
    xgp = (recommendations.get("xgp_bo_ei") or [{}])[0]
    bo = (recommendations.get("standard_bo_ei") or [{}])[0]
    conservative_yield = conservative.get("predicted_yield")
    xgp_yield = xgp.get("predicted_yield")
    bo_yield = bo.get("predicted_yield")
    needs_review = False
    selected_method = "xgp_bo_ei" if recommendations.get("xgp_bo_ei") else "conservative_ensemble"
    reason = "默认采用 xgp_bo_ei：XGBoost 负责产量均值预测，GP 只拟合残差并给出后验不确定性。"

    if xgp_yield is not None and conservative_yield is not None and conservative_yield:
        relative_gap = abs(float(xgp_yield) - float(conservative_yield)) / abs(float(conservative_yield))
        if relative_gap > review_threshold:
            needs_review = True
            reason = f"xgp_bo_ei 与 conservative_ensemble 预测产量差超过 {review_threshold:.0%}，建议人工审议两种方案的风险差异。"
    elif conservative_yield is not None and bo_yield is not None and conservative_yield:
        reason = "xgp_bo_ei 不可用，回退到 conservative_ensemble，并参考 standard_bo_ei 做人工审议判断。"
        relative_gap = (float(bo_yield) - float(conservative_yield)) / abs(float(conservative_yield))
        if relative_gap > review_threshold:
            needs_review = True
            reason = f"standard_bo_ei 的预测产量高于 conservative_ensemble 超过 {review_threshold:.0%}，建议人工审议两种方案的风险差异。"

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
    conservative_weights: ConservativeWeights | dict[str, float] | None = None,
    review_threshold: float = 0.10,
) -> dict[str, Any]:
    """Train models and compare conservative, standard BO, XGP-BO, XGBoost, and random-safe recommenders."""

    history = training_view(df, target_col) if "exclude_from_training" in df.columns else df.dropna(subset=[target_col])
    space = search_space or build_search_space_from_history(history)
    candidates = generate_candidates(space, n=5000, seed=42)
    model_bundle = train_surrogate_ensemble(history, target_col=target_col, feature_cols=list(space.bounds))
    conservative = recommend_conservative(
        model_bundle,
        history,
        space,
        top_k=top_k,
        candidates=candidates,
        weights=conservative_weights,
    )
    recommendations: dict[str, list[dict]] = {
        "conservative_ensemble": conservative,
        "random_safe": _random_safe_baseline(space, candidates),
    }
    result: dict[str, Any] = {
        "target_col": target_col,
        "n_training_rows": int(len(history)),
        "search_space": space.bounds,
        "model_info": model_bundle.model_info,
        "model_metrics": model_bundle.metrics,
        "reference_model_metrics": model_bundle.reference_metrics,
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
    recommendations["single_xgboost"] = _single_xgboost_baseline(model_bundle, history, space, candidates)
    result["decision"] = _decision_summary(recommendations, review_threshold)
    result["selected_method"] = result["decision"]["selected_method"]
    result["selected_recommendations"] = recommendations.get(result["selected_method"], [])
    return result


def recommend_next(df: pd.DataFrame, top_k: int = 5, conservative: bool = True) -> dict[str, Any]:
    """Return the next recommendation, defaulting to xgp_bo_ei when available."""

    comparison = compare_recommenders(df, top_k=top_k)
    key = comparison["selected_method"] if conservative else "standard_bo_ei"
    comparison["selected_method"] = key
    comparison["selected_recommendations"] = comparison["recommendations"].get(key, [])
    return comparison
