from __future__ import annotations

import pandas as pd
import pytest


def _synthetic_df(n: int = 16) -> pd.DataFrame:
    """Synthetic run-level data using production-like MODEL_FEATURES names."""

    rows = []
    for i in range(n):
        shift_time = 20.0 + (i % 4) * 2.0
        prod_temp = 29.0 + (i % 3) * 0.5
        lactose = 500 + (i % 5) * 40
        feed1 = 700 + i * 8
        feed2 = 80 + (i % 4) * 15
        lac_start = 20.0 + (i % 3) * 2.0
        y = (
            140.0
            - 2.0 * (shift_time - 20.0)
            + 1.5 * (prod_temp - 29.0)
            - 0.05 * (lactose - 500.0)
            + 0.02 * (feed1 - 700.0)
            - 0.10 * (feed2 - 80.0)
            - 0.30 * (lac_start - 20.0)
        )
        rows.append(
            {
                "fermenter_run_id": f"R{i:02d}",
                "temperature_shift_time_h": shift_time,
                "temperature_production_phase_c": prod_temp,
                "lactose_total_ml": lactose,
                "feed1_total_ml": feed1,
                "feed2_total_ml": feed2,
                "lactose_first_add_time_h": lac_start,
                "yield_g_per_l": y,
                "exclude_from_training": False,
            }
        )
    return pd.DataFrame(rows)


def test_compare_recommenders_returns_standard_bo_qnei_primary():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")
    from experiment_advisor.optimizer.search_space import build_search_space_from_history
    from experiment_advisor.recommendation.service import compare_recommenders

    df = _synthetic_df()
    result = compare_recommenders(df, build_search_space_from_history(df), top_k=3)

    assert "standard_bo_qnei" in result["recommendations"]
    assert "standard_bo_ei" not in result["recommendations"]
    assert "standard_bo_ucb" not in result["recommendations"]
    assert "xgp_bo_ei" not in result["recommendations"]
    assert "xgp_bo_ucb" not in result["recommendations"]
    assert "conservative_ensemble" not in result["recommendations"]
    assert "random_safe" not in result["recommendations"]
    assert "single_xgboost" not in result["recommendations"]
    assert len(result["recommendations"]["standard_bo_qnei"]) == 3
    assert result["model_info"]["primary_method"] == "standard_bo_qnei"
    assert result["model_info"]["candidate_methods"] == []
    assert result["decision"]["selected_method"] == "standard_bo_qnei"
    assert result["strategy_quality"]["batch_diversity"]["n_recommendations"] == 3
    assert "mean_nearest_history_distance" in result["strategy_quality"]["history_support"]


def test_standard_bo_qnei_recommendations_are_batch_diverse_and_visualizable():
    pytest.importorskip("torch")
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")
    from experiment_advisor.optimizer.search_space import build_search_space_from_history
    from experiment_advisor.optimizer.standard_bo import recommend_standard_bo

    df = _synthetic_df()
    search_space = build_search_space_from_history(df)
    result = recommend_standard_bo(df, search_space, top_k=5)

    assert len(result["recommendations"]) == 5
    assert all(item["method"] == "standard_bo_qnei" for item in result["recommendations"])

    param_sets = [
        frozenset(item["params"].items())
        for item in result["recommendations"]
    ]
    assert len(set(param_sets)) > 1
    assert hasattr(result["fitted_gp"], "predict")


def test_recommendation_quality_metrics_describe_batch_support_and_risk():
    from experiment_advisor.optimizer.search_space import build_search_space_from_history
    from experiment_advisor.recommendation.quality import evaluate_recommendation_quality

    df = _synthetic_df()
    search_space = build_search_space_from_history(df)
    features = list(search_space.bounds)
    recommendations = [
        {
            "rank": 1,
            "params": {feature: search_space.bounds[feature][0] for feature in features},
            "predicted_yield": 120.0,
            "model_uncertainty": 2.0,
        },
        {
            "rank": 2,
            "params": {feature: search_space.bounds[feature][1] for feature in features},
            "predicted_yield": 121.0,
            "model_uncertainty": 3.0,
        },
    ]

    quality = evaluate_recommendation_quality(recommendations, df, search_space, features)

    assert quality["batch_diversity"]["n_recommendations"] == 2
    assert quality["batch_diversity"]["min_pairwise_distance"] > 0
    assert quality["history_support"]["max_nearest_history_distance"] is not None
    assert quality["boundary_risk"]["max_boundary_risk"] == 1.0
    assert len(quality["per_recommendation"]) == 2


def test_recommendation_report_mentions_qnei_only():
    from experiment_advisor.report import generate_recommendation_report

    report = generate_recommendation_report(
        {
            "target_col": "yield_g_per_l",
            "n_training_rows": 12,
            "model_metrics": {"ridge": {"mae_loocv": 1.2, "r2_loocv": 0.3}},
            "recommendations": {
                "standard_bo_qnei": [
                    {
                        "rank": 1,
                        "params": {"temperature_c_mean": 31.5},
                        "predicted_yield": 118,
                    }
                ],
            },
            "selected_method": "standard_bo_qnei",
            "selected_recommendations": [
                {"rank": 1, "params": {"temperature_c_mean": 31.5}, "predicted_yield": 118}
            ],
        }
    )

    assert "conservative_ensemble" not in report
    assert "single_xgboost" not in report
    assert "random_safe" not in report
    assert "standard_bo_ucb" not in report
    assert "standard_bo_ei" not in report
    assert "xgp_bo_ei" not in report
    assert "standard_bo_qnei" in report
    assert "qNEI" in report
