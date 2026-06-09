from __future__ import annotations

import pandas as pd


def _pichia_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "A01",
                "experiment_date": "2026-01-01",
                "strain_id": "A",
                "parent_strain_id": "",
                "growth_phase_ph": 5.4,
                "production_phase_ph": 5.2,
                "growth_phase_temp_c": 28.0,
                "production_phase_temp_c": 24.0,
                "glucose_start_time_h": 30.0,
                "glucose_concentration_g_l": 8.0,
                "fan_speed_rpm": 450.0,
                "yield_g_per_l": 0.6,
            },
            {
                "run_id": "B01",
                "experiment_date": "2026-02-01",
                "strain_id": "B",
                "parent_strain_id": "A",
                "growth_phase_ph": 5.5,
                "production_phase_ph": 5.3,
                "growth_phase_temp_c": 27.0,
                "production_phase_temp_c": 23.0,
                "glucose_start_time_h": 32.0,
                "glucose_concentration_g_l": 9.0,
                "fan_speed_rpm": 500.0,
                "yield_g_per_l": 0.8,
            },
            {
                "run_id": "B02",
                "experiment_date": "2026-02-08",
                "strain_id": "B",
                "parent_strain_id": "A",
                "growth_phase_ph": 5.6,
                "production_phase_ph": 5.4,
                "growth_phase_temp_c": 26.0,
                "production_phase_temp_c": 22.0,
                "glucose_start_time_h": 36.0,
                "glucose_concentration_g_l": 10.0,
                "fan_speed_rpm": 550.0,
                "yield_g_per_l": 1.1,
            },
            {
                "run_id": "C01",
                "experiment_date": "2026-03-01",
                "strain_id": "C",
                "parent_strain_id": "B",
                "growth_phase_ph": 5.4,
                "production_phase_ph": 5.1,
                "growth_phase_temp_c": 25.0,
                "production_phase_temp_c": 21.0,
                "glucose_start_time_h": 34.0,
                "glucose_concentration_g_l": 7.0,
                "fan_speed_rpm": 600.0,
                "yield_g_per_l": 0.7,
            },
        ]
    )


def test_changed_variable_count_matches_coupling_rule():
    from experiment_advisor.recommendation.pichia import changed_variable_count, uniform_single_variable_levels

    assert changed_variable_count(0.0, 4) == 1
    assert changed_variable_count(0.33, 4) == 2
    assert changed_variable_count(0.66, 4) == 3
    assert changed_variable_count(1.0, 4) == 4
    assert uniform_single_variable_levels("growth_phase_temp_c", 3, lower=25, upper=30) == [25.0, 27.5, 30.0]
    assert uniform_single_variable_levels("growth_phase_temp_c", 4, lower=24, upper=30) == [24.0, 26.0, 28.0, 30.0]


def test_recommend_pichia_design_uses_same_strain_baseline_and_hard_bounds():
    from experiment_advisor.recommendation.pichia import (
        PICHIA_PARAMETER_SPECS,
        changed_variable_count,
        recommend_pichia_design,
    )

    variables = list(PICHIA_PARAMETER_SPECS)
    result = recommend_pichia_design(
        _pichia_history(),
        strain_id="B",
        parent_strain_id="A",
        baseline_source="同菌种历史最优",
        variables=variables,
        coupling=0.33,
        n_recommendations=4,
        seed=7,
    )

    assert result["baseline_meta"]["run_id"] == "B02"
    assert len(result["recommendations"]) == 4
    assert result["changed_variable_count"] == changed_variable_count(0.33, len(variables))

    for item in result["recommendations"]:
        assert item["strain_support"] == "同菌种支持"
        assert item["cross_strain_risk"] == "低"
        assert len(item["changed_variables"]) == result["changed_variable_count"]
        assert "glycerol" not in item["params"]
        assert "od" not in item["params"]
        for name, spec in PICHIA_PARAMETER_SPECS.items():
            assert spec.lower <= item["params"][name] <= spec.upper


def test_recommend_pichia_design_marks_parent_borrowing_risk():
    from experiment_advisor.recommendation.pichia import recommend_pichia_design

    result = recommend_pichia_design(
        _pichia_history(),
        strain_id="C",
        parent_strain_id="B",
        baseline_source="亲本菌种历史最优",
        variables=["growth_phase_ph", "production_phase_ph"],
        coupling=1.0,
        n_recommendations=2,
        seed=2,
    )

    assert result["baseline_meta"]["support_type"] == "parent_strain"
    assert result["recommendations"][0]["strain_support"] == "亲本菌种借鉴"
    assert result["recommendations"][0]["cross_strain_risk"] == "中"
    assert result["recommendations"][0]["recommendation_type"] == "全变量探索"


def test_single_variable_design_evenly_covers_range_and_keeps_others_fixed():
    from experiment_advisor.recommendation.pichia import PICHIA_PARAMETER_SPECS, recommend_pichia_design

    result = recommend_pichia_design(
        _pichia_history(),
        strain_id="B",
        parent_strain_id="A",
        baseline_source="同菌种历史最优",
        variables=list(PICHIA_PARAMETER_SPECS),
        single_variable="growth_phase_temp_c",
        single_variable_bounds=(25.0, 30.0),
        n_recommendations=3,
        seed=999,
    )

    levels = [item["params"]["growth_phase_temp_c"] for item in result["recommendations"]]

    assert levels == [25.0, 27.5, 30.0]
    assert result["single_variable_levels"] == levels
    assert result["changed_variable_count"] == 1
    for item in result["recommendations"]:
        assert item["changed_variables"] == ["growth_phase_temp_c"]
        assert item["recommendation_type"] == "单变量验证"
        for name in PICHIA_PARAMETER_SPECS:
            if name != "growth_phase_temp_c":
                assert item["params"][name] == result["baseline"][name]


def test_two_factor_sequential_doe_estimates_main_and_interaction_terms():
    from experiment_advisor.recommendation.pichia import PICHIA_PARAMETER_SPECS, recommend_pichia_design

    result = recommend_pichia_design(
        _pichia_history(),
        strain_id="B",
        parent_strain_id="A",
        baseline_source="同菌种历史最优",
        doe_variables=["growth_phase_ph", "production_phase_ph"],
        doe_bounds={
            "growth_phase_ph": (5.2, 5.8),
            "production_phase_ph": (5.1, 5.7),
        },
        n_recommendations=4,
        seed=999,
    )

    pairs = [
        (item["params"]["growth_phase_ph"], item["params"]["production_phase_ph"])
        for item in result["recommendations"]
    ]

    assert pairs == [(5.2, 5.1), (5.8, 5.1), (5.2, 5.7), (5.8, 5.7)]
    assert result["mode"] == "pichia_sequential_doe"
    assert result["changed_variable_count"] == 2
    assert result["information_gain"]["can_estimate_main_effects"] is True
    assert result["information_gain"]["can_estimate_interaction"] is True
    assert "交互效应" in "；".join(result["information_gain"]["estimable_terms"])

    for item in result["recommendations"]:
        assert item["changed_variables"] == ["growth_phase_ph", "production_phase_ph"]
        for name in PICHIA_PARAMETER_SPECS:
            if name not in {"growth_phase_ph", "production_phase_ph"}:
                assert item["params"][name] == result["baseline"][name]


def test_analyze_pichia_doe_feedback_calculates_effects_and_suggestion():
    from experiment_advisor.recommendation.pichia import analyze_pichia_doe_feedback, recommend_pichia_design

    result = recommend_pichia_design(
        _pichia_history(),
        strain_id="B",
        parent_strain_id="A",
        baseline_source="同菌种历史最优",
        doe_variables=["growth_phase_ph", "production_phase_ph"],
        doe_bounds={
            "growth_phase_ph": (5.2, 5.8),
            "production_phase_ph": (5.1, 5.7),
        },
        n_recommendations=4,
    )

    feedback = analyze_pichia_doe_feedback(
        result,
        {
            1: 1.0,
            2: 2.0,
            3: 1.5,
            4: 4.0,
        },
        practical_threshold=0.2,
    )

    effects = {item["variable"]: item for item in feedback["main_effects"]}

    assert feedback["status"] == "complete"
    assert feedback["best_rank"] == 4
    assert feedback["best_yield"] == 4.0
    assert effects["growth_phase_ph"]["effect"] == 1.75
    assert effects["growth_phase_ph"]["direction"] == "高水平更好"
    assert effects["production_phase_ph"]["effect"] == 1.25
    assert effects["production_phase_ph"]["direction"] == "高水平更好"
    assert feedback["interaction_effect"]["effect"] == 0.75
    assert feedback["interaction_effect"]["direction"] == "同向组合更好"
    assert "交互效应明显" in feedback["suggestion"]


def test_pichia_template_columns_are_isolated_from_ecoli_features():
    from experiment_advisor.recommendation.pichia import pichia_template_columns

    columns = pichia_template_columns()

    assert "growth_phase_ph" in columns
    assert "production_phase_ph" in columns
    assert "glucose_start_time_h" in columns
    assert "glucose_concentration_g_l" in columns
    assert "fan_speed_rpm" in columns
    assert "temperature_shift_time_h" not in columns
    assert "lactose_first_add_time_h" not in columns
    assert "od" not in columns
    assert "glycerol" not in columns
