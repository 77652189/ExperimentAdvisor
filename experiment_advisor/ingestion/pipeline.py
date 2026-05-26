from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment_advisor.ingestion.features import engineer_features
from experiment_advisor.ingestion.loader import load_fermentation_data
from experiment_advisor.ingestion.run_level import build_run_level_dataset
from experiment_advisor.ingestion.validator import validate


def build_final_dataset(
    input_path: str | Path,
    output_path: str | Path = "data/final/fermentation_modeling_dataset.csv",
) -> pd.DataFrame:
    """
    从 Excel/CSV 历史数据构建 BO 正式入模数据，并写入 data/final。

    若数据质量检查未通过，抛出 ValueError，避免把不可用数据写入最终入模目录。
    """

    raw = load_fermentation_data(input_path)
    report = validate(raw)
    if not report["passed"]:
        raise ValueError(f"Data quality validation failed: {report}")
    final_df = engineer_features(raw)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output, index=False, encoding="utf-8")
    return final_df


__all__ = ["build_final_dataset", "build_run_level_dataset"]
