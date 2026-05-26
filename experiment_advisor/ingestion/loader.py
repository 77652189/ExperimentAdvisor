from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

STANDARD_COLUMNS = [
    "batch_id",
    "temperature",
    "ph",
    "feed_amount",
    "feed_time",
    "induction_time",
    "inducer_dose",
    "yield_g_per_l",
]
NUMERIC_COLUMNS = [column for column in STANDARD_COLUMNS if column != "batch_id"]

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "batch_id": ("batch_id", "batch", "batch_no", "batch_number", "run_id", "批次", "批号", "批次号", "实验批次"),
    "temperature": ("temperature", "temp", "fermentation_temperature", "温度", "发酵温度"),
    "ph": ("ph", "p_h", "pH", "PH", "酸碱度"),
    "feed_amount": ("feed_amount", "feed", "feeding_amount", "补料量", "补料总量", "碳源补料量"),
    "feed_time": ("feed_time", "feeding_time", "feed_timing", "补料时间", "补料时间点"),
    "induction_time": ("induction_time", "induction_timing", "诱导时间", "诱导时间点"),
    "inducer_dose": ("inducer_dose", "inducer_amount", "inducer", "诱导剂用量", "诱导剂浓度"),
    "yield_g_per_l": ("yield_g_per_l", "yield", "titer", "产量", "目标产物产量", "g_l", "g/L"),
}

_ALIAS_TO_STANDARD = {
    re.sub(r"[\s_\-()/\\]+", "", alias).casefold(): standard
    for standard, aliases in COLUMN_ALIASES.items()
    for alias in aliases
}


def _read_table(path: Path, header: int | None = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, header=header)
    if suffix in {".xlsx", ".xls"}:
        kwargs = {"header": header}
        if suffix == ".xlsx":
            kwargs["engine"] = "openpyxl"
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported data file type: {path.suffix}")


def _normalize_name(name: object) -> str:
    return re.sub(r"[\s_\-()/\\]+", "", str(name).strip()).casefold()


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[object, str] = {}
    unknown: list[str] = []
    for column in df.columns:
        standard = _ALIAS_TO_STANDARD.get(_normalize_name(column))
        if standard:
            rename_map[column] = standard
        else:
            unknown.append(str(column))
    if unknown:
        LOGGER.warning("Ignoring unrecognized columns: %s", ", ".join(unknown))
    renamed = df.rename(columns=rename_map)
    return renamed[[column for column in renamed.columns if column in STANDARD_COLUMNS]]


def _read_with_detected_header(path: Path) -> pd.DataFrame:
    first_pass = _rename_columns(_read_table(path, header=0))
    if len(first_pass.columns) >= 3:
        return first_pass

    raw = _read_table(path, header=None)
    best_index = 0
    best_score = -1
    for index in range(min(10, len(raw))):
        score = sum(1 for value in raw.iloc[index].tolist() if _normalize_name(value) in _ALIAS_TO_STANDARD)
        if score > best_score:
            best_index = index
            best_score = score
    if best_score <= 0:
        return first_pass
    detected = raw.iloc[best_index + 1 :].copy()
    detected.columns = raw.iloc[best_index].tolist()
    return _rename_columns(detected)


def load_fermentation_data(path: str | Path) -> pd.DataFrame:
    """
    读取历史发酵 Excel/CSV 数据，并统一为标准 DataFrame。

    支持 .xlsx/.xls/.csv；自动识别常见中英文表头别名。返回列固定为
    batch_id, temperature, ph, feed_amount, feed_time, induction_time,
    inducer_dose, yield_g_per_l。无法识别的源列会被忽略并写入 warning。
    """

    table_path = Path(path)
    if not table_path.exists():
        raise ValueError(f"Data file does not exist: {table_path}")

    df = _read_with_detected_header(table_path)
    for column in STANDARD_COLUMNS:
        if column not in df.columns:
            LOGGER.warning("Missing expected column %s; filling with NaN", column)
            df[column] = pd.NA
    df = df[STANDARD_COLUMNS].copy().reset_index(drop=True)
    df["batch_id"] = df["batch_id"].astype("string").fillna("").astype(str)
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")

    LOGGER.info("Loaded fermentation data shape=%s dtypes=%s", df.shape, df.dtypes.astype(str).to_dict())
    return df
