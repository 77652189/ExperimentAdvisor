from __future__ import annotations

import pandas as pd
import pytest


def _synthetic_df(n: int = 16) -> pd.DataFrame:
    rows = []
    for i in range(n):
        temp = 31.0 + (i % 4) * 0.4
        ph = 6.7 + (i % 3) * 0.08
        feed = 100 + i * 4
        lactose = 200 + (i % 5) * 20
        duration = 70 + i
        y = 90 + 4 * (temp - 31.5) - 8 * abs(ph - 6.82) + 0.12 * feed + 0.04 * lactose
        rows.append(
            {
                "fermenter_run_id": f"R{i:02d}",
                "temperature_c_mean": temp,
                "ph_mean": ph,
                "feed1_ml_final": feed,
                "feed2_ml_final": 50 + i,
                "lactose_ml_final": lactose,
                "base_ml_final": 30 + i,
                "od600_max": 60 + i * 0.5,
                "fermentation_duration_h": duration,
                "yield_g_per_l": y,
                "exclude_from_training": False,
            }
        )
    return pd.DataFrame(rows)


def test_compare_recommenders_returns_xgp_and_standard_bo_only():
    pytest.importorskip("sklearn")
    from experiment_advisor.optimizer.search_space import build_search_space_from_history
    from experiment_advisor.recommendation.service import compare_recommenders

    df = _synthetic_df()
    result = compare_recommenders(df, build_search_space_from_history(df), top_k=3)

    assert "xgp_bo_ei" in result["recommendations"]
    assert "xgp_bo_ucb" in result["recommendations"]
    assert "standard_bo_ei" in result["recommendations"]
    assert "standard_bo_ucb" in result["recommendations"]
    assert "conservative_ensemble" not in result["recommendations"]
    assert "random_safe" not in result["recommendations"]
    assert "single_xgboost" not in result["recommendations"]
    assert len(result["recommendations"]["xgp_bo_ei"]) == 3
    assert result["model_info"]["primary_method"] == "xgp_bo_ei"
    assert result["model_info"]["comparison_methods"] == ["xgp_bo_ucb", "standard_bo_ei", "standard_bo_ucb"]
    assert result["decision"]["selected_method"] == "xgp_bo_ei"


def test_xgp_bo_in_compare_recommenders():
    pytest.importorskip("sklearn")
    pytest.importorskip("xgboost")
    from experiment_advisor.recommendation.service import compare_recommenders

    df = _synthetic_df()
    result = compare_recommenders(df, top_k=3)

    assert "xgp_bo_ei" in result["recommendations"]
    assert "xgp_bo_ucb" in result["recommendations"]
    assert len(result["recommendations"]["xgp_bo_ei"]) == 3

    first = result["recommendations"]["xgp_bo_ei"][0]
    assert "xgb_prediction" in first
    assert "gp_residual_mean" in first
    assert first["uncertainty_type"] == "xgp_gp_residual_std"
    assert abs(first["predicted_yield"] - (first["xgb_prediction"] + first["gp_residual_mean"])) < 1e-6
    assert "gp_feature_cols" in first
    assert len(first["gp_feature_cols"]) <= 4
    assert "history_distance" in first
    assert "boundary_risk" in first
    assert "risk_level" in first
    assert "quality_flags" in first
    assert "gp_health" in first
    assert first["gp_health"]["gp_feature_cols"] == first["gp_feature_cols"]
    assert "candidate_uncertainty_degenerate" in first["gp_health"]

    uncertainties = [
        item["model_uncertainty"]
        for item in result["recommendations"]["xgp_bo_ei"]
    ]
    assert len(set(round(u, 6) for u in uncertainties)) > 1, (
        "GP model_uncertainty is identical for all candidates - "
        "GP may have degenerated to constant prediction"
    )

    assert result["selected_method"] == "xgp_bo_ei"


def test_recommendation_report_mentions_both_methods():
    from experiment_advisor.report import generate_recommendation_report

    report = generate_recommendation_report(
        {
            "target_col": "yield_g_per_l",
            "n_training_rows": 12,
            "model_metrics": {"ridge": {"mae_loocv": 1.2, "r2_loocv": 0.3}},
            "recommendations": {
                "xgp_bo_ei": [{"rank": 1, "params": {"temperature_c_mean": 32.0}, "predicted_yield": 120}],
                "standard_bo_ei": [{"rank": 1, "params": {"temperature_c_mean": 31.5}, "predicted_yield": 118}],
            },
            "selected_method": "xgp_bo_ei",
            "selected_recommendations": [
                {"rank": 1, "params": {"temperature_c_mean": 32.0}, "predicted_yield": 120}
            ],
        }
    )

    assert "conservative_ensemble" not in report
    assert "single_xgboost" not in report
    assert "random_safe" not in report
    assert "standard_bo_ei" in report
    assert "xgp_bo_ei" in report
    assert "XGBoost" in report
