from __future__ import annotations

from typing import Any

import pandas as pd

from experiment_advisor.ingestion.loader import NUMERIC_COLUMNS, STANDARD_COLUMNS


def _outlier_batch_ids(df: pd.DataFrame, column: str) -> list[str]:
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return []
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = pd.to_numeric(df[column], errors="coerce").lt(lower) | pd.to_numeric(df[column], errors="coerce").gt(upper)
    return df.loc[mask, "batch_id"].astype(str).tolist() if "batch_id" in df.columns else df.index[mask].astype(str).tolist()


def validate(df: pd.DataFrame) -> dict[str, Any]:
    """
    检查历史发酵数据质量，返回验证报告，不修改原始 DataFrame。

    passed=False 的条件：产量列缺失率 > 5%，或完整可用批次数 < 10。
    """

    missing_rate = {column: float(df[column].isna().mean()) if column in df.columns else 1.0 for column in STANDARD_COLUMNS}
    warnings = [
        f"{column} missing rate exceeds 20%: {rate:.1%}"
        for column, rate in missing_rate.items()
        if rate > 0.2
    ]

    outliers = {
        column: _outlier_batch_ids(df, column)
        for column in NUMERIC_COLUMNS
        if column in df.columns
    }

    homogeneity_warnings: list[str] = []
    date_column = next((column for column in ("batch_date", "date", "experiment_date") if column in df.columns), None)
    if date_column:
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if not dates.empty and (dates.max() - dates.min()).days > 365:
            homogeneity_warnings.append("Batch date span exceeds 12 months; strain/process homogeneity should be reviewed.")

    n_usable_batches = int(df.dropna(subset=[column for column in STANDARD_COLUMNS if column in df.columns]).shape[0])
    passed = missing_rate.get("yield_g_per_l", 1.0) <= 0.05 and n_usable_batches >= 10
    return {
        "passed": bool(passed),
        "missing_rate": missing_rate,
        "outliers": outliers,
        "homogeneity_warnings": homogeneity_warnings + warnings,
        "n_usable_batches": n_usable_batches,
    }
