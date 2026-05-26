from __future__ import annotations

import pytest

from experiment_advisor.report import generate_recommendation_report


def test_reporter_generates_markdown_with_recommendation(tmp_path):
    output_path = tmp_path / "recommendation.md"
    markdown = generate_recommendation_report(
        [
            {
                "params": {"temperature": 32.5, "ph": 7.0},
                "predicted_mean": 135.2,
                "predicted_std": 5.1,
                "mode": "exploit",
            }
        ],
        offline_analysis={"shap_importance": {"induction_time": 0.42}},
        output_path=output_path,
    )

    assert "# 发酵工艺优化推荐报告" in markdown
    assert "temperature=32.5" in markdown
    assert output_path.read_text(encoding="utf-8") == markdown


def test_search_space_defaults_and_constraint():
    pytest.importorskip("ax")
    from experiment_advisor.space import build_search_space

    space = build_search_space()

    assert set(space.parameters) >= {"temperature", "ph", "feed_amount", "feed_time", "induction_time", "inducer_dose"}
    assert space.parameter_constraints
