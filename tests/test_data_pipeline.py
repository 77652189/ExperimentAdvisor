from __future__ import annotations

import pandas as pd
import pytest

from experiment_advisor.ingestion import build_final_dataset, engineer_features, load_fermentation_data, validate


def _sample_rows(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "批次": [f"B{i:02d}" for i in range(n)],
            "发酵温度": [32.0 + (i % 3) * 0.2 for i in range(n)],
            "pH": [6.8 + (i % 2) * 0.1 for i in range(n)],
            "补料量": [120.0 + i for i in range(n)],
            "补料时间点": [8.0 + (i % 3) for i in range(n)],
            "诱导时间点": [14.0 + (i % 4) for i in range(n)],
            "诱导剂用量": [0.3 + (i % 2) * 0.05 for i in range(n)],
            "产量": [100.0 + i * 1.5 for i in range(n)],
            "未知列": ["ignored"] * n,
        }
    )


def test_loader_maps_chinese_aliases_and_normalizes_dtypes(tmp_path):
    path = tmp_path / "history.csv"
    _sample_rows().to_csv(path, index=False, encoding="utf-8")

    df = load_fermentation_data(path)

    assert list(df.columns) == [
        "batch_id",
        "temperature",
        "ph",
        "feed_amount",
        "feed_time",
        "induction_time",
        "inducer_dose",
        "yield_g_per_l",
    ]
    assert str(df["batch_id"].dtype) in {"object", "str", "string"}
    assert str(df["yield_g_per_l"].dtype) == "float64"
    assert df.loc[0, "batch_id"] == "B00"


def test_loader_detects_header_below_first_row(tmp_path):
    path = tmp_path / "history.csv"
    sample = _sample_rows()
    with path.open("w", encoding="utf-8") as file:
        file.write("metadata,metadata,metadata,metadata,metadata,metadata,metadata,metadata,metadata\n")
        file.write(",".join(sample.columns) + "\n")
        for row in sample.astype(str).itertuples(index=False):
            file.write(",".join(row) + "\n")

    df = load_fermentation_data(path)

    assert df.loc[0, "batch_id"] == "B00"
    assert df.loc[0, "yield_g_per_l"] == 100.0


def test_validator_reports_missing_and_outliers():
    df = load_fermentation_data_from_frame(_sample_rows())
    df.loc[0, "yield_g_per_l"] = None
    df.loc[1, "feed_amount"] = 9999

    report = validate(df)

    assert report["passed"] is False
    assert report["missing_rate"]["yield_g_per_l"] > 0
    assert "B01" in report["outliers"]["feed_amount"]


def test_engineer_features_adds_expected_columns_and_nan_for_zero_denominator():
    df = load_fermentation_data_from_frame(_sample_rows())
    df.loc[0, "induction_time"] = 0.0

    engineered = engineer_features(df)

    assert "feat_feed_rate_proxy" in engineered.columns
    assert pd.isna(engineered.loc[0, "feat_feed_rate_proxy"])
    assert engineered.loc[1, "feat_feed_to_induction_interval"] == engineered.loc[1, "induction_time"] - engineered.loc[1, "feed_time"]


def test_build_final_dataset_writes_featured_csv(tmp_path):
    input_path = tmp_path / "history.csv"
    output_path = tmp_path / "final" / "fermentation_modeling_dataset.csv"
    _sample_rows().to_csv(input_path, index=False, encoding="utf-8")

    final_df = build_final_dataset(input_path, output_path)

    assert output_path.exists()
    assert "feat_carbon_load_proxy" in final_df.columns


def test_excel_loader_smoke(tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "history.xlsx"
    _sample_rows().to_excel(path, index=False)

    df = load_fermentation_data(path)

    assert len(df) == 12
    assert df.loc[0, "temperature"] == 32.0


def load_fermentation_data_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    from experiment_advisor.ingestion.loader import _rename_columns

    df = _rename_columns(frame)
    for column in ["batch_id", "temperature", "ph", "feed_amount", "feed_time", "induction_time", "inducer_dose", "yield_g_per_l"]:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[["batch_id", "temperature", "ph", "feed_amount", "feed_time", "induction_time", "inducer_dose", "yield_g_per_l"]].copy()
    df["batch_id"] = df["batch_id"].astype(str)
    for column in df.columns:
        if column != "batch_id":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    return df
