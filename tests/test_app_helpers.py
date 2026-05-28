from __future__ import annotations

import pandas as pd


def test_nearest_history_deduplicates_display_columns(monkeypatch):
    from App import app

    monkeypatch.setattr(
        app,
        "FIELD_LABELS",
        {
            "fermenter_run_id": "重复列",
            "temperature_shift_time_h": "重复列",
            "temperature_production_phase_c": "重复列",
        },
    )
    df = pd.DataFrame(
        [
            {
                "fermenter_run_id": "R01",
                "yield_g_per_l": 100.0,
                "temperature_shift_time_h": 20.0,
                "temperature_production_phase_c": 29.0,
                "exclude_from_training": False,
            },
            {
                "fermenter_run_id": "R02",
                "yield_g_per_l": 110.0,
                "temperature_shift_time_h": 22.0,
                "temperature_production_phase_c": 30.0,
                "exclude_from_training": False,
            },
        ]
    )

    nearest = app._nearest_history(
        df,
        {
            "temperature_shift_time_h": 21.0,
            "temperature_production_phase_c": 29.5,
        },
    )

    assert not nearest.columns.duplicated().any()
    assert list(nearest.columns).count("重复列") == 1
    assert "重复列_2" in nearest.columns


def test_soft_filter_uses_larger_pool_instead_of_supplementing_failures():
    from App import app

    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "yield_g_per_l": [100.0, 101.0, 102.0, 103.0, 104.0],
            "exclude_from_training": [False] * 5,
        }
    )
    comparison = {
        "selected_recommendations": [
            {"rank": 1, "params": {"x": 2.0}, "predicted_yield": 110.0},
            {"rank": 2, "params": {"x": 10.0}, "predicted_yield": 120.0},
            {"rank": 3, "params": {"x": 3.0}, "predicted_yield": 105.0},
        ],
        "strategy_quality": {
            "per_recommendation": [
                {"rank": 1, "nearest_history_distance": 0.0, "boundary_risk": 0.0},
                {"rank": 2, "nearest_history_distance": 0.0, "boundary_risk": 0.0},
                {"rank": 3, "nearest_history_distance": 0.0, "boundary_risk": 0.0},
            ]
        },
    }

    result = app._apply_soft_filters(
        comparison,
        df,
        ["x"],
        max_nearest_history_distance=2.0,
        max_boundary_risk=0.8,
        history_sigma=2.0,
        target_count=2,
    )

    selected = result["selected_recommendations"]
    assert len(selected) == 2
    assert [item["rank"] for item in selected] == [1, 3]
    assert all(item["soft_filter_status"] == "通过" for item in selected)
    assert result["soft_filter"]["failed_history_range_ranks"] == [2]
    assert result["soft_filter"]["failure_counts"]["history_range"] == 1
    assert result["soft_filter"]["target_count"] == 2


def test_select_without_soft_filters_restores_original_pool():
    from App import app

    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "yield_g_per_l": [100.0, 101.0, 102.0, 103.0, 104.0],
            "exclude_from_training": [False] * 5,
        }
    )
    comparison = {
        "selected_method": "standard_bo_qnei",
        "search_space": {"x": (0.0, 4.0)},
        "unfiltered_selected_recommendations": [
            {"rank": 1, "params": {"x": 2.0}, "predicted_yield": 110.0, "soft_filter_status": "通过"},
            {"rank": 2, "params": {"x": 10.0}, "predicted_yield": 120.0, "soft_filter_status": "通过"},
            {"rank": 3, "params": {"x": 3.0}, "predicted_yield": 105.0, "soft_filter_status": "通过"},
        ],
        "selected_recommendations": [
            {"rank": 1, "params": {"x": 2.0}, "predicted_yield": 110.0, "soft_filter_status": "通过"},
        ],
    }

    result = app._select_without_soft_filters(comparison, df, ["x"], target_count=2)

    selected = result["selected_recommendations"]
    assert [item["rank"] for item in selected] == [1, 2]
    assert all("soft_filter_status" not in item for item in selected)
    assert result["soft_filter"] == {
        "enabled": False,
        "n_before": 3,
        "n_after": 2,
        "target_count": 2,
    }


def test_soft_filter_reports_failure_reason_breakdown():
    from App import app

    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "yield_g_per_l": [100.0, 101.0, 102.0, 103.0, 104.0],
            "exclude_from_training": [False] * 5,
        }
    )
    comparison = {
        "selected_recommendations": [
            {"rank": 1, "params": {"x": 2.0}, "predicted_yield": 110.0},
            {"rank": 2, "params": {"x": 10.0}, "predicted_yield": 120.0},
            {"rank": 3, "params": {"x": 3.0}, "predicted_yield": 105.0},
        ],
        "strategy_quality": {
            "per_recommendation": [
                {"rank": 1, "nearest_history_distance": 0.0, "boundary_risk": 0.0},
                {"rank": 2, "nearest_history_distance": 3.0, "boundary_risk": 0.9},
                {"rank": 3, "nearest_history_distance": 2.5, "boundary_risk": 0.0},
            ]
        },
    }

    result = app._apply_soft_filters(
        comparison,
        df,
        ["x"],
        max_nearest_history_distance=2.0,
        max_boundary_risk=0.8,
        history_sigma=2.0,
        target_count=2,
    )

    soft_filter = result["soft_filter"]
    assert soft_filter["failed_nearest_history_ranks"] == [2, 3]
    assert soft_filter["failed_boundary_risk_ranks"] == [2]
    assert soft_filter["failed_history_range_ranks"] == [2]
    assert soft_filter["failure_counts"] == {
        "nearest_history_distance": 2,
        "boundary_risk": 1,
        "history_range": 1,
    }
