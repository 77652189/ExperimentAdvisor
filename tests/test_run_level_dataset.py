from __future__ import annotations

from pathlib import Path

from experiment_advisor.ingestion.run_level import build_run_level_dataset, training_view


def test_build_run_level_dataset_from_real_csv_from_excel(tmp_path):
    source = Path("data/csv_from_excel")
    output = tmp_path / "run_level_modeling_dataset.csv"

    df = build_run_level_dataset(source, output)
    train = training_view(df)

    assert output.exists()
    assert len(df) >= 50
    assert "fermenter_run_id" in df.columns
    assert "yield_g_per_l" in df.columns
    assert "target_yield_g_per_l" in df.columns
    assert "target_source" in df.columns
    assert "feed1_start_time_h" in df.columns
    assert "feed1_before_24h_ml" in df.columns
    assert "lactose_first_add_time_h" in df.columns
    assert "temperature_c_mean" in df.columns
    assert "temperature_growth_phase_c" in df.columns
    assert "temperature_shift_time_h" in df.columns
    assert "temperature_production_phase_c" in df.columns
    assert "feed1_total_ml" in df.columns
    assert "od600_outlier_corrected_count" in df.columns
    assert df["od600_max"].max() < 1000
    assert train["yield_g_per_l"].notna().all()
    assert len(train) >= 20
    assert (df["target_source"] == "liquid_long_data.extracellular_yield_g_per_l").sum() > 0
    assert "liquid_label_sequence_fallback" not in set(df["target_match_method"].dropna())
