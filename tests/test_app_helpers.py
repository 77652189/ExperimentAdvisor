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
