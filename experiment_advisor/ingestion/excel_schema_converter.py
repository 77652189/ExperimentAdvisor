from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet


PRODUCT_COLUMNS = ["code", "full_name", "created_at"]
STRAIN_COLUMNS = ["name", "created_at"]
EXPERIMENT_COLUMNS = [
    "id",
    "product_code",
    "experiment_date",
    "file_name",
    "recorder",
    "source_file_md5",
    "created_at",
]
FERMENTER_RUN_COLUMNS = [
    "id",
    "experiment_id",
    "fermenter_label",
    "sheet_name",
    "strain_name",
    "batch_number",
    "inoculum_ratio",
    "seed_culture_time",
    "seed_od_value",
    "inoculation_time",
    "fermentation_end_time",
    "condition_notes",
    "created_at",
]
TIME_SERIES_COLUMNS = [
    "id",
    "fermenter_run_id",
    "fermentation_time_h",
    "temperature_c",
    "ph",
    "feed1_ml",
    "feed2_ml",
    "base_ml",
    "lactose_ml",
    "volume_ml",
    "od600",
    "yield_g_per_l",
    "lactose_g_per_l",
    "remarks",
    "created_at",
]
HPLC_COLUMNS = [
    "id",
    "fermenter_run_id",
    "sample_time_h",
    "extracellular_yield_g_per_l",
    "inactivated_yield_g_per_l",
    "extracellular_lactose_g_per_l",
    "extracellular_lactose_peak_area",
    "inactivated_lactose_g_per_l",
    "inactivated_lactose_peak_area",
    "extracellular_acetate_g_per_l",
    "inactivated_acetate_g_per_l",
    "created_at",
]
EXCEL_CELL_COLUMNS = [
    "file_name",
    "sheet_name",
    "row",
    "column",
    "cell",
    "value",
    "formula",
    "is_formula",
]
SUPPLEMENTAL_CELL_COLUMNS = [
    "file_name",
    "sheet_name",
    "fermenter_run_id",
    "row",
    "column",
    "cell",
    "value",
    "formula",
]
LIQUID_LONG_COLUMNS = [
    "id",
    "experiment_id",
    "file_name",
    "sheet_name",
    "section",
    "sample_label",
    "sample_time_h",
    "value",
    "formula",
    "source_cell",
    "created_at",
]

_KNOWN_METADATA_LABELS = (
    "文件名称",
    "发酵罐编号",
    "发酵批次",
    "页码",
    "发酵菌株名称",
    "种子液接种比例",
    "种子液培养时间",
    "种子液最终 OD 值",
    "接种上罐时间",
    "发酵结束时间",
    "发酵条件优化操作",
)


@dataclass(frozen=True)
class ConversionResult:
    output_dir: Path
    tables: dict[str, pd.DataFrame]
    skipped_sheets: list[dict[str, str]]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _cell_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_header(value: Any) -> str:
    return re.sub(r"[\s_（）()/-]+", "", _text(value)).casefold()


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_date_from_filename(path: Path) -> str:
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", path.name)
    if not match:
        raise ValueError(f"Cannot parse experiment date from filename: {path.name}")
    return "-".join(match.groups())


def _parse_product_code(path: Path) -> str:
    match = re.search(r"(\d+)[-\s_]*(FL|SL)", path.name, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)}-{match.group(2).upper()}"
    for code in ("LNnT", "LNFP", "LNT", "SA"):
        if code.casefold() in path.name.casefold():
            return code
    return "UNKNOWN"


def _experiment_id(path: Path, product_code: str) -> str:
    date = datetime.strptime(_parse_date_from_filename(path), "%Y-%m-%d")
    compact_product = product_code.replace("-", "")
    return f"EXP-{date:%y%m%d}-{compact_product}-01"


def _find_labeled_value(ws: Worksheet, label_keyword: str, max_row: int = 8) -> str:
    for row in range(1, min(ws.max_row, max_row) + 1):
        for col in range(1, ws.max_column + 1):
            value = _text(ws.cell(row, col).value)
            if label_keyword in value:
                after_colon = re.split(r"[:：]", value, maxsplit=1)
                if len(after_colon) == 2 and after_colon[1].strip():
                    return after_colon[1].strip()
                max_offset = 3 if col <= 2 else 4
                for offset in range(1, max_offset + 1):
                    candidate = _text(ws.cell(row, col + offset).value)
                    if candidate and not any(label in candidate for label in _KNOWN_METADATA_LABELS):
                        return candidate
    return ""


def _find_labeled_values(ws: Worksheet, label_keyword: str, max_row: int = 8) -> list[str]:
    values: list[str] = []
    for row in range(1, min(ws.max_row, max_row) + 1):
        for col in range(1, ws.max_column + 1):
            value = _text(ws.cell(row, col).value)
            if label_keyword not in value:
                continue
            max_offset = 3 if col <= 2 else 4
            for offset in range(1, max_offset + 1):
                candidate = _text(ws.cell(row, col + offset).value)
                if candidate and not any(label in candidate for label in _KNOWN_METADATA_LABELS):
                    values.append(candidate)
                    break
    return values


def _batch_number(ws: Worksheet) -> str:
    values = _find_labeled_values(ws, "发酵批次")
    if not values:
        return ""
    values = sorted(set(values), key=lambda value: (bool(re.search(r"-\d{1,3}$", value)), len(value)), reverse=True)
    return values[0]


def _condition_notes(ws: Worksheet) -> str:
    for row in range(1, min(ws.max_row, 8) + 1):
        for col in range(1, ws.max_column + 1):
            value = _text(ws.cell(row, col).value)
            if "发酵条件优化操作" in value:
                parts = re.split(r"[:：]", value, maxsplit=1)
                return parts[1].strip() if len(parts) == 2 else value
    return ""


def _strain_name(raw: str) -> str:
    match = re.search(r"[（(]([^）)]+)[）)]", raw)
    value = match.group(1) if match else raw
    value = value.replace(" ", "").strip()
    if value.endswith("S") and value[:-1]:
        return value[:-1]
    return value or "UNKNOWN"


def _id_suffix(value: str) -> str:
    suffix = re.sub(r"\s+", "_", value.strip())
    suffix = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "", suffix)
    return suffix or "RUN"


def _header_map(ws: Worksheet, header_row: int = 9) -> dict[str, int]:
    result: dict[str, int] = {}
    for col in range(1, ws.max_column + 1):
        raw = _text(ws.cell(header_row, col).value)
        header = _normalize_header(raw)
        if "发酵时长" in raw:
            result["fermentation_time_h"] = col
        elif "温度" in raw:
            result["temperature_c"] = col
        elif header == "ph":
            result["ph"] = col
        elif "补料1" in raw:
            result["feed1_ml"] = col
        elif "补料2" in raw:
            result["feed2_ml"] = col
        elif "补碱" in raw:
            result["base_ml"] = col
        elif "乳糖" in raw and ("ml" in raw.casefold() or "ｍｌ" in raw.casefold()):
            result["lactose_ml"] = col
        elif ("实时体积" in raw or raw == "体积") and "volume_ml" not in result:
            result["volume_ml"] = col
        elif "OD 600" in raw or header == "od600":
            result["od600"] = col
        elif "产量" in raw and "总产量" not in raw and "yield_g_per_l" not in result:
            result["yield_g_per_l"] = col
        elif "乳糖" in raw and "g/L" in raw and "lactose_g_per_l" not in result:
            result["lactose_g_per_l"] = col
        elif "备注" in raw and "remarks" not in result:
            result["remarks"] = col
    return result


def _is_fermentation_sheet(ws: Worksheet) -> bool:
    headers = _header_map(ws)
    required = {"fermentation_time_h", "temperature_c", "ph"}
    return required.issubset(headers)


def _read_time_series(ws: Worksheet, run_id: str, created_at: str) -> list[dict[str, Any]]:
    headers = _header_map(ws)
    rows: list[dict[str, Any]] = []
    blank_streak = 0
    row_number = 1
    for row in range(10, ws.max_row + 1):
        time_col = headers.get("fermentation_time_h")
        time_value = _number(ws.cell(row, time_col).value) if time_col else None
        if time_value is None:
            if rows:
                blank_streak += 1
                if blank_streak >= 3:
                    break
            continue
        blank_streak = 0
        item: dict[str, Any] = {
            "id": f"{run_id}-{row_number:04d}",
            "fermenter_run_id": run_id,
            "fermentation_time_h": time_value,
            "created_at": created_at,
        }
        for name in TIME_SERIES_COLUMNS:
            if name in {"id", "fermenter_run_id", "fermentation_time_h", "created_at"}:
                continue
            col = headers.get(name)
            value = ws.cell(row, col).value if col else None
            item[name] = _number(value) if name not in {"remarks"} else _text(value)
        rows.append(item)
        row_number += 1
    return rows


def _main_table_row_set(ws: Worksheet) -> set[int]:
    headers = _header_map(ws)
    time_col = headers.get("fermentation_time_h")
    if not time_col:
        return set()
    rows: set[int] = set()
    blank_streak = 0
    for row in range(10, ws.max_row + 1):
        time_value = _number(ws.cell(row, time_col).value)
        if time_value is None:
            if rows:
                blank_streak += 1
                if blank_streak >= 3:
                    break
            continue
        blank_streak = 0
        rows.add(row)
    return rows


def _dump_workbook_cells(workbook_path: Path, values_workbook: Any, formulas_workbook: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for values_ws in values_workbook.worksheets:
        formulas_ws = formulas_workbook[values_ws.title]
        for row in range(1, values_ws.max_row + 1):
            for col in range(1, values_ws.max_column + 1):
                value = values_ws.cell(row, col).value
                formula_or_raw = formulas_ws.cell(row, col).value
                if value is None and formula_or_raw is None:
                    continue
                formula = formula_or_raw if isinstance(formula_or_raw, str) and formula_or_raw.startswith("=") else ""
                rows.append(
                    {
                        "file_name": workbook_path.name,
                        "sheet_name": values_ws.title,
                        "row": row,
                        "column": col,
                        "cell": f"{get_column_letter(col)}{row}",
                        "value": _cell_value(value),
                        "formula": formula,
                        "is_formula": bool(formula),
                    }
                )
    return rows


def _supplemental_cells(workbook_path: Path, values_ws: Worksheet, formulas_ws: Worksheet, run_id: str) -> list[dict[str, Any]]:
    main_rows = _main_table_row_set(values_ws)
    rows: list[dict[str, Any]] = []
    for row in range(10, values_ws.max_row + 1):
        if row in main_rows:
            continue
        for col in range(1, values_ws.max_column + 1):
            value = values_ws.cell(row, col).value
            formula_or_raw = formulas_ws.cell(row, col).value
            if value is None and formula_or_raw is None:
                continue
            formula = formula_or_raw if isinstance(formula_or_raw, str) and formula_or_raw.startswith("=") else ""
            rows.append(
                {
                    "file_name": workbook_path.name,
                    "sheet_name": values_ws.title,
                    "fermenter_run_id": run_id,
                    "row": row,
                    "column": col,
                    "cell": f"{get_column_letter(col)}{row}",
                    "value": _cell_value(value),
                    "formula": formula,
                }
            )
    return rows


def _looks_like_liquid_sheet(ws: Worksheet) -> bool:
    title = ws.title.casefold()
    if "hplc" in title:
        return True
    if "\u6db2" in ws.title:
        return True
    for row in range(1, min(ws.max_row, 90) + 1):
        for col in range(1, min(ws.max_column, 8) + 1):
            value = _text(ws.cell(row, col).value)
            if "\u80de\u5916\u4ea7\u91cf" in value or "\u706d\u6d3b\u4ea7\u91cf" in value or "\u4e59\u9178" in value:
                return True
    return False


def _section_name(value: str, current: str) -> str:
    if not value:
        return current
    if "OD600" in value or value.upper() == "OD600":
        return "od600"
    if "\u80de\u5916\u4ea7\u91cf" in value:
        return "extracellular_yield_g_per_l"
    if "\u706d\u6d3b\u4ea7\u91cf" in value:
        return "inactivated_yield_g_per_l"
    if "\u4e73\u7cd6" in value:
        return "lactose_g_per_l"
    if "\u4e59\u9178" in value:
        return "acetate_g_per_l"
    if "\u4ea7\u7269\u5dee\u503c" in value:
        return "product_delta"
    if "\u65f6\u7a7a\u4ea7\u7387" in value:
        return "space_time_yield"
    if "\u4f53\u79ef" in value:
        return "volume_ml"
    return current


def _parse_liquid_long(
    workbook_path: Path,
    values_ws: Worksheet,
    formulas_ws: Worksheet,
    experiment_id: str,
    created_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    item_index = 1

    def label_at(row: int, col: int) -> str:
        value = _text(values_ws.cell(row, col).value)
        if value.startswith("#"):
            value = _text(formulas_ws.cell(row, col).value)
        return value

    def add_value(row: int, col: int, section: str, sample_label: str, sample_time: float) -> None:
        nonlocal item_index
        value = values_ws.cell(row, col).value
        formula_or_raw = formulas_ws.cell(row, col).value
        if value is None and formula_or_raw is None:
            return
        formula = formula_or_raw if isinstance(formula_or_raw, str) and formula_or_raw.startswith("=") else ""
        rows.append(
            {
                "id": f"{experiment_id}-{values_ws.title}-L{item_index:04d}",
                "experiment_id": experiment_id,
                "file_name": workbook_path.name,
                "sheet_name": values_ws.title,
                "section": section,
                "sample_label": sample_label,
                "sample_time_h": sample_time,
                "value": _number(value),
                "formula": formula,
                "source_cell": f"{get_column_letter(col)}{row}",
                "created_at": created_at,
            }
        )
        item_index += 1

    def numeric_rows(start_row: int) -> list[int]:
        result: list[int] = []
        for row in range(start_row, values_ws.max_row + 1):
            if _number(values_ws.cell(row, 1).value) is None:
                if result:
                    break
                continue
            result.append(row)
        return result

    def row_labels(label_row: int) -> dict[int, str]:
        return {
            col: label_at(label_row, col)
            for col in range(2, values_ws.max_column + 1)
            if label_at(label_row, col) and _number(label_at(label_row, col)) is None
        }

    def parse_wide_block(label_row: int, start_row: int, section: str, section_by_col: dict[int, str] | None = None) -> None:
        labels = row_labels(label_row)
        for row in numeric_rows(start_row):
            sample_time = _number(values_ws.cell(row, 1).value)
            if sample_time is None:
                continue
            for col, sample_label in labels.items():
                add_value(row, col, section_by_col.get(col, section) if section_by_col else section, sample_label, sample_time)

    if values_ws.max_row >= 2 and _number(values_ws.cell(2, 1).value) is not None:
        parse_wide_block(1, 2, "od600")

    if values_ws.max_row >= 22 and _number(values_ws.cell(22, 1).value) is not None:
        section_by_col = {}
        for idx, col in enumerate(col for col in range(2, values_ws.max_column + 1) if label_at(21, col)):
            section_by_col[col] = "extracellular_yield_g_per_l" if idx % 2 == 0 else "extracellular_lactose_g_per_l"
        parse_wide_block(21, 22, "extracellular_yield_g_per_l", section_by_col)

    for row in range(1, values_ws.max_row):
        sample_label = _text(values_ws.cell(row, 1).value)
        if not sample_label or _number(sample_label) is not None:
            continue
        headers = {}
        for col in range(2, values_ws.max_column + 1):
            header_text = _text(values_ws.cell(row + 1, col).value)
            if not any(token in header_text for token in ("\u4ea7\u91cf", "\u4e73\u7cd6\u91cf", "\u4e59\u9178\u91cf")):
                continue
            section = _section_name(header_text, "")
            if section:
                headers[col] = section
        if not headers:
            continue
        for data_row in numeric_rows(row + 2):
            sample_time = _number(values_ws.cell(data_row, 1).value)
            if sample_time is None:
                continue
            for col, section in headers.items():
                add_value(data_row, col, section, sample_label, sample_time)

    inactivated_header = next(
        (row for row in range(1, values_ws.max_row + 1) if "\u706d\u6d3b\u4ea7\u91cf" in _text(values_ws.cell(row, 1).value)),
        None,
    )
    if inactivated_header:
        label_row = inactivated_header + 1
        start_row = label_row + 1
        if _number(values_ws.cell(label_row, 1).value) is not None:
            start_row = label_row
        parse_wide_block(label_row, start_row, "inactivated_yield_g_per_l")

        product_delta_row = next(
            (row for row in range(start_row, values_ws.max_row + 1) if "\u4ea7\u7269\u5dee\u503c" in _text(values_ws.cell(row, 1).value)),
            None,
        )
        if product_delta_row:
            volume_rows = [row for row in range(start_row, product_delta_row) if _number(values_ws.cell(row, 1).value) is not None]
            if len(volume_rows) > 4:
                for row in volume_rows[-4:]:
                    sample_time = _number(values_ws.cell(row, 1).value)
                    if sample_time is None:
                        continue
                    for col, sample_label in row_labels(label_row).items():
                        add_value(row, col, "volume_ml", sample_label, sample_time)

    for marker, section in {
        "\u4ea7\u7269\u5dee\u503c": "product_delta",
        "\u65f6\u95f4\u5dee\u503c": "time_delta",
    }.items():
        marker_row = next((row for row in range(1, values_ws.max_row + 1) if marker in _text(values_ws.cell(row, 1).value)), None)
        if not marker_row:
            continue
        labels = row_labels(marker_row - 1) or row_labels(marker_row)
        for row in numeric_rows(marker_row + 1):
            sample_time = _number(values_ws.cell(row, 1).value)
            if sample_time is None:
                continue
            for col, sample_label in labels.items():
                add_value(row, col, section, sample_label, sample_time)

    for row in range(1, values_ws.max_row + 1):
        if "h~" not in _text(values_ws.cell(row + 1, 1).value):
            continue
        labels = row_labels(row)
        for data_row in range(row + 1, values_ws.max_row + 1):
            interval = _text(values_ws.cell(data_row, 1).value)
            if "h~" not in interval:
                break
            for col, sample_label in labels.items():
                add_value(data_row, col, "space_time_yield", sample_label, float(data_row))
        break
    return rows


def convert_excel_directory(
    excel_dir: str | Path = "data/excel",
    output_dir: str | Path = "data/csv_from_excel",
) -> ConversionResult:
    excel_path = Path(excel_dir)
    output_path = Path(output_dir)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    products: dict[str, dict[str, Any]] = {}
    strains: dict[str, dict[str, Any]] = {}
    experiments: list[dict[str, Any]] = []
    fermenter_runs: list[dict[str, Any]] = []
    time_series: list[dict[str, Any]] = []
    hplc_rows: list[dict[str, Any]] = []
    excel_cells: list[dict[str, Any]] = []
    supplemental_cells: list[dict[str, Any]] = []
    liquid_long: list[dict[str, Any]] = []
    skipped_sheets: list[dict[str, str]] = []
    used_run_ids: set[str] = set()

    for workbook_path in sorted(excel_path.glob("*.xlsx")):
        product_code = _parse_product_code(workbook_path)
        experiment_id = _experiment_id(workbook_path, product_code)
        experiment_date = _parse_date_from_filename(workbook_path)
        products.setdefault(product_code, {"code": product_code, "full_name": "", "created_at": created_at})
        experiments.append(
            {
                "id": experiment_id,
                "product_code": product_code,
                "experiment_date": experiment_date,
                "file_name": workbook_path.name,
                "recorder": "",
                "source_file_md5": _md5(workbook_path),
                "created_at": created_at,
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            workbook = load_workbook(workbook_path, read_only=False, data_only=True)
            formula_workbook = load_workbook(workbook_path, read_only=False, data_only=False)
        excel_cells.extend(_dump_workbook_cells(workbook_path, workbook, formula_workbook))

        fermentation_sheets: list[tuple[Worksheet, str]] = []
        for ws in workbook.worksheets:
            if not _is_fermentation_sheet(ws):
                if _looks_like_liquid_sheet(ws):
                    liquid_long.extend(_parse_liquid_long(workbook_path, ws, formula_workbook[ws.title], experiment_id, created_at))
                else:
                    skipped_sheets.append({"file_name": workbook_path.name, "sheet_name": ws.title, "reason": "not fermentation main table"})
                continue
            fermenter_label = _find_labeled_value(ws, "发酵罐编号") or ws.title
            fermentation_sheets.append((ws, fermenter_label))

        label_counts = pd.Series([label for _, label in fermentation_sheets]).value_counts().to_dict()
        for ws, fermenter_label in fermentation_sheets:
            batch_number = _batch_number(ws)
            strain = _strain_name(_find_labeled_value(ws, "发酵菌株名称"))
            run_suffix = _id_suffix(ws.title) if label_counts.get(fermenter_label, 0) > 1 else fermenter_label
            run_id = f"{experiment_id}-{run_suffix}"
            if run_id in used_run_ids:
                run_id = f"{experiment_id}-{_id_suffix(ws.title)}"
            used_run_ids.add(run_id)
            strains.setdefault(strain, {"name": strain, "created_at": created_at})
            fermenter_runs.append(
                {
                    "id": run_id,
                    "experiment_id": experiment_id,
                    "fermenter_label": fermenter_label,
                    "sheet_name": ws.title,
                    "strain_name": strain,
                    "batch_number": batch_number,
                    "inoculum_ratio": _find_labeled_value(ws, "种子液接种比例"),
                    "seed_culture_time": _find_labeled_value(ws, "种子液培养时间"),
                    "seed_od_value": _find_labeled_value(ws, "种子液最终 OD 值"),
                    "inoculation_time": _find_labeled_value(ws, "接种上罐时间"),
                    "fermentation_end_time": _find_labeled_value(ws, "发酵结束时间"),
                    "condition_notes": _condition_notes(ws),
                    "created_at": created_at,
                }
            )
            time_series.extend(_read_time_series(ws, run_id, created_at))
            supplemental_cells.extend(_supplemental_cells(workbook_path, ws, formula_workbook[ws.title], run_id))

    tables = {
        "product": pd.DataFrame(products.values(), columns=PRODUCT_COLUMNS),
        "strain": pd.DataFrame(strains.values(), columns=STRAIN_COLUMNS),
        "experiment": pd.DataFrame(experiments, columns=EXPERIMENT_COLUMNS),
        "fermenter_run": pd.DataFrame(fermenter_runs, columns=FERMENTER_RUN_COLUMNS),
        "time_series_data": pd.DataFrame(time_series, columns=TIME_SERIES_COLUMNS),
        "hplc_data": pd.DataFrame(hplc_rows, columns=HPLC_COLUMNS),
        "liquid_long_data": pd.DataFrame(liquid_long, columns=LIQUID_LONG_COLUMNS),
        "supplemental_cells": pd.DataFrame(supplemental_cells, columns=SUPPLEMENTAL_CELL_COLUMNS),
        "excel_cells": pd.DataFrame(excel_cells, columns=EXCEL_CELL_COLUMNS),
    }

    output_path.mkdir(parents=True, exist_ok=True)
    for table_name, frame in tables.items():
        frame.to_csv(output_path / f"{table_name}.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(skipped_sheets).to_csv(output_path / "skipped_sheets.csv", index=False, encoding="utf-8-sig")
    return ConversionResult(output_dir=output_path, tables=tables, skipped_sheets=skipped_sheets)


def _read_csv_table(directory: Path, table_name: str) -> pd.DataFrame:
    path = directory / f"{table_name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _stringify(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _values_equal(old_value: Any, new_value: Any, tolerance: float = 1e-6) -> bool:
    if pd.isna(old_value) and pd.isna(new_value):
        return True
    old_number = pd.to_numeric(pd.Series([old_value]), errors="coerce").iloc[0]
    new_number = pd.to_numeric(pd.Series([new_value]), errors="coerce").iloc[0]
    if not pd.isna(old_number) and not pd.isna(new_number):
        return abs(float(old_number) - float(new_number)) <= tolerance
    old_datetime = pd.to_datetime(pd.Series([old_value]), errors="coerce").iloc[0]
    new_datetime = pd.to_datetime(pd.Series([new_value]), errors="coerce").iloc[0]
    if not pd.isna(old_datetime) and not pd.isna(new_datetime):
        return old_datetime == new_datetime
    return _stringify(old_value).strip() == _stringify(new_value).strip()


def _key_text(row: pd.Series, key_columns: list[str]) -> str:
    return json.dumps({column: _stringify(row[column]) for column in key_columns}, ensure_ascii=False, sort_keys=True)


def _detailed_diff(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    key_columns: list[str],
    compare_columns: list[str] | None = None,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    if old_df.empty and new_df.empty:
        return pd.DataFrame(columns=["status", "key", "column", "old_value", "new_value"])
    missing_keys = [column for column in key_columns if column not in old_df.columns or column not in new_df.columns]
    if missing_keys:
        return pd.DataFrame(
            [
                {
                    "status": "invalid_key",
                    "key": ",".join(missing_keys),
                    "column": "",
                    "old_value": "",
                    "new_value": "",
                }
            ]
        )

    old = old_df.copy()
    new = new_df.copy()
    old["__key"] = old.apply(lambda row: _key_text(row, key_columns), axis=1)
    new["__key"] = new.apply(lambda row: _key_text(row, key_columns), axis=1)

    columns = sorted((set(old.columns) & set(new.columns)) - {"__key"}) if compare_columns is None else compare_columns
    columns = [column for column in columns if column not in key_columns and column in old.columns and column in new.columns]

    old_keys = set(old["__key"])
    new_keys = set(new["__key"])
    rows: list[dict[str, str]] = []

    for key in sorted(old_keys - new_keys):
        rows.append({"status": "old_only_row", "key": key, "column": "__row__", "old_value": "present", "new_value": "missing"})
    for key in sorted(new_keys - old_keys):
        rows.append({"status": "new_only_row", "key": key, "column": "__row__", "old_value": "missing", "new_value": "present"})

    old_indexed = old.drop_duplicates("__key", keep="first").set_index("__key")
    new_indexed = new.drop_duplicates("__key", keep="first").set_index("__key")
    duplicate_old = old[old.duplicated("__key", keep=False)]["__key"].unique().tolist()
    duplicate_new = new[new.duplicated("__key", keep=False)]["__key"].unique().tolist()
    for key in sorted(duplicate_old):
        rows.append({"status": "duplicate_old_key", "key": key, "column": "__key__", "old_value": "duplicated", "new_value": ""})
    for key in sorted(duplicate_new):
        rows.append({"status": "duplicate_new_key", "key": key, "column": "__key__", "old_value": "", "new_value": "duplicated"})

    for key in sorted(old_keys & new_keys):
        old_row = old_indexed.loc[key]
        new_row = new_indexed.loc[key]
        for column in columns:
            old_value = old_row[column]
            new_value = new_row[column]
            if not _values_equal(old_value, new_value, tolerance=tolerance):
                rows.append(
                    {
                        "status": "cell_diff",
                        "key": key,
                        "column": column,
                        "old_value": _stringify(old_value),
                        "new_value": _stringify(new_value),
                    }
                )

    return pd.DataFrame(rows, columns=["status", "key", "column", "old_value", "new_value"])


def _with_experiment_file(runs: pd.DataFrame, experiments: pd.DataFrame) -> pd.DataFrame:
    if runs.empty or experiments.empty:
        return runs.copy()
    experiment_cols = ["id", "file_name"]
    enriched = runs.merge(experiments[experiment_cols], left_on="experiment_id", right_on="id", how="left", suffixes=("", "_experiment"))
    return enriched.drop(columns=["id_experiment"], errors="ignore")


def _with_run_context(time_series: pd.DataFrame, runs: pd.DataFrame, experiments: pd.DataFrame) -> pd.DataFrame:
    if time_series.empty or runs.empty:
        return time_series.copy()
    run_context = _with_experiment_file(runs, experiments)[["id", "file_name", "sheet_name"]].rename(columns={"id": "fermenter_run_id"})
    enriched = time_series.merge(run_context, on="fermenter_run_id", how="left")
    enriched["_row_index"] = enriched.groupby(["file_name", "sheet_name"], dropna=False).cumcount() + 1
    return enriched


def write_detailed_diff_files(
    old_tables: dict[str, pd.DataFrame],
    new_tables: dict[str, pd.DataFrame],
    diff_dir: str | Path = "summary/supporting_reports/excel_csv_diffs",
) -> dict[str, dict[str, int]]:
    output_dir = Path(diff_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, int]] = {}

    exact_keys = {
        "product": ["code"],
        "strain": ["name"],
        "experiment": ["id"],
        "fermenter_run": ["id"],
        "time_series_data": ["id"],
        "hplc_data": ["id"],
    }
    ignored_columns = {"created_at", "source_file_md5"}

    for table_name, key_columns in exact_keys.items():
        old_df = old_tables.get(table_name, pd.DataFrame())
        new_df = new_tables.get(table_name, pd.DataFrame())
        compare_columns = sorted((set(old_df.columns) & set(new_df.columns)) - set(key_columns) - ignored_columns)
        diff = _detailed_diff(old_df, new_df, key_columns=key_columns, compare_columns=compare_columns, tolerance=1e-6)
        diff.to_csv(output_dir / f"{table_name}_by_primary_key.csv", index=False, encoding="utf-8-sig")
        summaries[f"{table_name}_by_primary_key"] = {
            "diff_rows": int(len(diff)),
            "cell_diffs": int((diff["status"] == "cell_diff").sum()) if not diff.empty else 0,
            "old_only_rows": int((diff["status"] == "old_only_row").sum()) if not diff.empty else 0,
            "new_only_rows": int((diff["status"] == "new_only_row").sum()) if not diff.empty else 0,
        }

    old_runs_natural = _with_experiment_file(old_tables["fermenter_run"], old_tables["experiment"])
    new_runs_natural = _with_experiment_file(new_tables["fermenter_run"], new_tables["experiment"])
    run_compare_columns = [
        "fermenter_label",
        "strain_name",
        "batch_number",
        "inoculum_ratio",
        "seed_culture_time",
        "seed_od_value",
        "inoculation_time",
        "fermentation_end_time",
        "condition_notes",
    ]
    run_diff = _detailed_diff(
        old_runs_natural,
        new_runs_natural,
        key_columns=["file_name", "sheet_name"],
        compare_columns=run_compare_columns,
        tolerance=1e-6,
    )
    run_diff.to_csv(output_dir / "fermenter_run_by_file_sheet.csv", index=False, encoding="utf-8-sig")
    summaries["fermenter_run_by_file_sheet"] = {
        "diff_rows": int(len(run_diff)),
        "cell_diffs": int((run_diff["status"] == "cell_diff").sum()) if not run_diff.empty else 0,
        "old_only_rows": int((run_diff["status"] == "old_only_row").sum()) if not run_diff.empty else 0,
        "new_only_rows": int((run_diff["status"] == "new_only_row").sum()) if not run_diff.empty else 0,
    }

    old_ts_natural = _with_run_context(old_tables["time_series_data"], old_tables["fermenter_run"], old_tables["experiment"])
    new_ts_natural = _with_run_context(new_tables["time_series_data"], new_tables["fermenter_run"], new_tables["experiment"])
    ts_compare_columns = [
        "fermentation_time_h",
        "temperature_c",
        "ph",
        "feed1_ml",
        "feed2_ml",
        "base_ml",
        "lactose_ml",
        "volume_ml",
        "od600",
        "yield_g_per_l",
        "lactose_g_per_l",
        "remarks",
    ]
    ts_diff = _detailed_diff(
        old_ts_natural,
        new_ts_natural,
        key_columns=["file_name", "sheet_name", "_row_index"],
        compare_columns=ts_compare_columns,
        tolerance=1e-6,
    )
    ts_diff.to_csv(output_dir / "time_series_data_by_file_sheet_row.csv", index=False, encoding="utf-8-sig")
    summaries["time_series_data_by_file_sheet_row"] = {
        "diff_rows": int(len(ts_diff)),
        "cell_diffs": int((ts_diff["status"] == "cell_diff").sum()) if not ts_diff.empty else 0,
        "old_only_rows": int((ts_diff["status"] == "old_only_row").sum()) if not ts_diff.empty else 0,
        "new_only_rows": int((ts_diff["status"] == "new_only_row").sum()) if not ts_diff.empty else 0,
    }
    return summaries


def _count_by_key(frame: pd.DataFrame, key: str) -> pd.Series:
    if frame.empty or key not in frame.columns:
        return pd.Series(dtype="int64")
    return frame.groupby(key).size().sort_index()


def _is_blank(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def _value_in_frame(frame: pd.DataFrame, value_columns: list[str], target: Any, tolerance: float = 1e-3) -> bool:
    if frame.empty:
        return False
    for column in value_columns:
        if column not in frame.columns:
            continue
        if frame[column].apply(lambda candidate: _values_equal(candidate, target, tolerance=tolerance)).any():
            return True
    return False


def _liquid_has_value(
    liquid: pd.DataFrame,
    target: Any,
    file_name: str,
    sample_time: Any | None = None,
    tolerance: float = 1e-3,
) -> bool:
    if liquid.empty or "value" not in liquid.columns:
        return False
    candidates = liquid[liquid["file_name"].astype(str) == file_name] if "file_name" in liquid.columns else liquid
    if candidates.empty:
        return False
    if sample_time is not None and "sample_time_h" in candidates.columns:
        target_time = pd.to_numeric(pd.Series([sample_time]), errors="coerce").iloc[0]
        if not pd.isna(target_time):
            times = pd.to_numeric(candidates["sample_time_h"], errors="coerce")
            time_candidates = candidates[(times - float(target_time)).abs() <= tolerance]
            if not time_candidates.empty and _value_in_frame(time_candidates, ["value"], target, tolerance=tolerance):
                return True
    return _value_in_frame(candidates, ["value"], target, tolerance=tolerance)


def _coverage_tags(
    *,
    target: Any,
    file_name: str,
    sheet_name: str,
    sample_time: Any | None,
    new_tables: dict[str, pd.DataFrame],
) -> str:
    tags: list[str] = []

    excel_cells = new_tables.get("excel_cells", pd.DataFrame())
    if not excel_cells.empty:
        same_file = excel_cells[excel_cells["file_name"].astype(str) == file_name] if "file_name" in excel_cells.columns else excel_cells
        same_sheet = same_file[same_file["sheet_name"].astype(str) == sheet_name] if "sheet_name" in same_file.columns else same_file
        if _value_in_frame(same_sheet, ["value"], target):
            tags.append("excel_cells_same_sheet")
        elif _value_in_frame(same_file, ["value"], target):
            tags.append("excel_cells_same_file")

    supplemental = new_tables.get("supplemental_cells", pd.DataFrame())
    if not supplemental.empty:
        same_file = supplemental[supplemental["file_name"].astype(str) == file_name] if "file_name" in supplemental.columns else supplemental
        same_sheet = same_file[same_file["sheet_name"].astype(str) == sheet_name] if "sheet_name" in same_file.columns else same_file
        if _value_in_frame(same_sheet, ["value"], target):
            tags.append("supplemental_same_sheet")
        elif _value_in_frame(same_file, ["value"], target):
            tags.append("supplemental_same_file")

    liquid = new_tables.get("liquid_long_data", pd.DataFrame())
    if _liquid_has_value(liquid, target, file_name, sample_time=sample_time):
        tags.append("liquid_same_file_time")
    elif _liquid_has_value(liquid, target, file_name):
        tags.append("liquid_same_file")

    return "|".join(tags) if tags else "not_found_in_new_outputs"


def audit_old_nonblank_value_coverage(
    old_tables: dict[str, pd.DataFrame],
    new_tables: dict[str, pd.DataFrame],
    diff_dir: str | Path = "summary/supporting_reports/excel_csv_diffs",
) -> dict[str, int]:
    """Audit whether old non-empty values missing from schema CSV still exist in new audit CSVs."""
    output_dir = Path(diff_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    coverage_counts: Counter[str] = Counter()

    old_ts = _with_run_context(old_tables["time_series_data"], old_tables["fermenter_run"], old_tables["experiment"])
    new_ts = _with_run_context(new_tables["time_series_data"], new_tables["fermenter_run"], new_tables["experiment"])
    ts_columns = [
        "fermentation_time_h",
        "temperature_c",
        "ph",
        "feed1_ml",
        "feed2_ml",
        "base_ml",
        "lactose_ml",
        "volume_ml",
        "od600",
        "yield_g_per_l",
        "lactose_g_per_l",
        "remarks",
    ]
    if not old_ts.empty:
        if not new_ts.empty:
            new_ts = new_ts.copy()
            new_ts["__natural_key"] = new_ts.apply(lambda row: _key_text(row, ["file_name", "sheet_name", "_row_index"]), axis=1)
            new_ts_by_key = new_ts.drop_duplicates("__natural_key", keep="first").set_index("__natural_key")
        else:
            new_ts_by_key = pd.DataFrame()

        for _, old_row in old_ts.iterrows():
            key = _key_text(old_row, ["file_name", "sheet_name", "_row_index"])
            new_row = None if new_ts_by_key.empty or key not in new_ts_by_key.index else new_ts_by_key.loc[key]
            for column in ts_columns:
                if column not in old_row.index or _is_blank(old_row[column]):
                    continue
                if new_row is not None and column in new_row.index and not _is_blank(new_row[column]):
                    continue
                coverage = _coverage_tags(
                    target=old_row[column],
                    file_name=_stringify(old_row.get("file_name", "")),
                    sheet_name=_stringify(old_row.get("sheet_name", "")),
                    sample_time=old_row.get("fermentation_time_h"),
                    new_tables=new_tables,
                )
                records.append(
                    {
                        "table": "time_series_data",
                        "file_name": _stringify(old_row.get("file_name", "")),
                        "sheet_name": _stringify(old_row.get("sheet_name", "")),
                        "row_index": _stringify(old_row.get("_row_index", "")),
                        "key": key,
                        "column": column,
                        "old_value": _stringify(old_row[column]),
                        "new_schema_value": "" if new_row is None or column not in new_row.index else _stringify(new_row[column]),
                        "reason": "missing_row_in_new_schema" if new_row is None else "blank_cell_in_new_schema",
                        "coverage": coverage,
                        "old_time_h": _stringify(old_row.get("fermentation_time_h", "")),
                    }
                )
                coverage_counts[coverage] += 1

    old_hplc = old_tables.get("hplc_data", pd.DataFrame())
    if not old_hplc.empty:
        old_hplc = old_hplc.merge(
            _with_experiment_file(old_tables["fermenter_run"], old_tables["experiment"])[["id", "file_name", "sheet_name"]],
            left_on="fermenter_run_id",
            right_on="id",
            how="left",
            suffixes=("", "_run"),
        )
        hplc_value_columns = [
            "sample_time_h",
            "extracellular_yield_g_per_l",
            "inactivated_yield_g_per_l",
            "extracellular_lactose_g_per_l",
            "extracellular_lactose_peak_area",
            "inactivated_lactose_g_per_l",
            "inactivated_lactose_peak_area",
            "extracellular_acetate_g_per_l",
            "inactivated_acetate_g_per_l",
        ]
        for _, old_row in old_hplc.iterrows():
            for column in hplc_value_columns:
                if column not in old_row.index or _is_blank(old_row[column]):
                    continue
                coverage = _coverage_tags(
                    target=old_row[column],
                    file_name=_stringify(old_row.get("file_name", "")),
                    sheet_name=_stringify(old_row.get("sheet_name", "")),
                    sample_time=old_row.get("sample_time_h"),
                    new_tables=new_tables,
                )
                records.append(
                    {
                        "table": "hplc_data",
                        "file_name": _stringify(old_row.get("file_name", "")),
                        "sheet_name": _stringify(old_row.get("sheet_name", "")),
                        "row_index": "",
                        "key": _stringify(old_row.get("id", "")),
                        "column": column,
                        "old_value": _stringify(old_row[column]),
                        "new_schema_value": "",
                        "reason": "missing_row_in_new_schema",
                        "coverage": coverage,
                        "old_time_h": _stringify(old_row.get("sample_time_h", "")),
                    }
                )
                coverage_counts[coverage] += 1

    detail = pd.DataFrame(
        records,
        columns=[
            "table",
            "file_name",
            "sheet_name",
            "row_index",
            "key",
            "column",
            "old_value",
            "new_schema_value",
            "reason",
            "coverage",
            "old_time_h",
        ],
    )
    detail.to_csv(output_dir / "old_nonblank_values_blank_or_missing_in_new_schema_with_coverage.csv", index=False, encoding="utf-8-sig")

    summary: Counter[str] = Counter()
    summary["old_nonblank_values_blank_or_missing_in_new_schema"] = len(detail)
    for table, count in detail["table"].value_counts().sort_index().items() if not detail.empty else []:
        summary[f"table::{table}"] = int(count)
    for coverage, count in detail["coverage"].value_counts().sort_index().items() if not detail.empty else []:
        summary[f"coverage::{coverage}"] = int(count)
    for column, count in detail["column"].value_counts().sort_index().items() if not detail.empty else []:
        summary[f"column::{column}"] = int(count)
    not_found = detail[detail["coverage"] == "not_found_in_new_outputs"] if not detail.empty else pd.DataFrame()
    for file_name, count in not_found["file_name"].value_counts().sort_index().items() if not not_found.empty else []:
        summary[f"not_found_file::{file_name}"] = int(count)

    summary_frame = pd.DataFrame([{"metric": key, "count": value} for key, value in summary.items()])
    summary_frame.to_csv(output_dir / "missing_value_coverage_summary_strict.csv", index=False, encoding="utf-8-sig")
    return dict(summary)


def compare_csv_directories(
    old_dir: str | Path = "data/csv",
    new_dir: str | Path = "data/csv_from_excel",
    report_path: str | Path = "summary/supporting_reports/excel_csv_comparison.md",
    diff_dir: str | Path = "summary/supporting_reports/excel_csv_diffs",
) -> str:
    old_path = Path(old_dir)
    new_path = Path(new_dir)
    table_names = [
        "product",
        "strain",
        "experiment",
        "fermenter_run",
        "time_series_data",
        "hplc_data",
        "liquid_long_data",
        "supplemental_cells",
        "excel_cells",
    ]
    old_tables = {name: _read_csv_table(old_path, name) for name in table_names}
    new_tables = {name: _read_csv_table(new_path, name) for name in table_names}
    detailed_summaries = write_detailed_diff_files(old_tables, new_tables, diff_dir=diff_dir)
    missing_value_summary = audit_old_nonblank_value_coverage(old_tables, new_tables, diff_dir=diff_dir)

    lines = [
        "# Excel 转换 CSV 与旧 CSV 对比报告",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 表级行数",
        "",
        "| 表 | 旧 CSV 行数 | Excel 转换行数 | 差异 |",
        "|---|---:|---:|---:|",
    ]
    for name in table_names:
        old_count = len(old_tables[name])
        new_count = len(new_tables[name])
        lines.append(f"| {name} | {old_count} | {new_count} | {new_count - old_count:+d} |")

    old_exp = old_tables["experiment"]
    new_exp = new_tables["experiment"]
    if not old_exp.empty and not new_exp.empty:
        old_files = set(old_exp["file_name"].astype(str))
        new_files = set(new_exp["file_name"].astype(str))
        old_only_files = [f"- {name}" for name in sorted(old_files - new_files)] or ["- 无"]
        new_only_files = [f"- {name}" for name in sorted(new_files - old_files)] or ["- 无"]
        lines.extend(["", "## 源文件差异", "", "旧 CSV 有但 Excel 目录没有："])
        lines.extend(old_only_files)
        lines.extend(["", "Excel 目录有但旧 CSV 没有："])
        lines.extend(new_only_files)

    old_runs = set(old_tables["fermenter_run"].get("id", pd.Series(dtype=str)).astype(str))
    new_runs = set(new_tables["fermenter_run"].get("id", pd.Series(dtype=str)).astype(str))
    lines.extend(
        [
            "",
            "## fermenter_run 主键差异",
            "",
            f"- 共同 run 数：{len(old_runs & new_runs)}",
            f"- 旧 CSV 独有 run 数：{len(old_runs - new_runs)}",
            f"- Excel 转换独有 run 数：{len(new_runs - old_runs)}",
        ]
    )
    for title, values in [("旧 CSV 独有 run 示例", old_runs - new_runs), ("Excel 转换独有 run 示例", new_runs - old_runs)]:
        sample = sorted(values)[:10]
        lines.extend(["", f"{title}："])
        lines.extend([f"- {value}" for value in sample] or ["- 无"])

    old_ts_counts = _count_by_key(old_tables["time_series_data"], "fermenter_run_id")
    new_ts_counts = _count_by_key(new_tables["time_series_data"], "fermenter_run_id")
    common_run_counts = sorted(set(old_ts_counts.index) & set(new_ts_counts.index))
    changed_counts = [
        (run_id, int(old_ts_counts.loc[run_id]), int(new_ts_counts.loc[run_id]))
        for run_id in common_run_counts
        if int(old_ts_counts.loc[run_id]) != int(new_ts_counts.loc[run_id])
    ]
    lines.extend(
        [
            "",
            "## time_series_data 行数差异",
            "",
            f"- 共同 run 中时间点数量不同的 run 数：{len(changed_counts)}",
            "",
            "| run_id | 旧 CSV 时间点 | Excel 转换时间点 | 差异 |",
            "|---|---:|---:|---:|",
        ]
    )
    for run_id, old_count, new_count in changed_counts[:30]:
        lines.append(f"| {run_id} | {old_count} | {new_count} | {new_count - old_count:+d} |")
    if not changed_counts:
        lines.append("| 无 | 0 | 0 | 0 |")

    compare_columns = [
        "fermentation_time_h",
        "temperature_c",
        "ph",
        "feed1_ml",
        "feed2_ml",
        "base_ml",
        "lactose_ml",
        "od600",
        "yield_g_per_l",
        "lactose_g_per_l",
    ]
    old_ts = old_tables["time_series_data"]
    new_ts = new_tables["time_series_data"]
    value_lines: list[str] = []
    if not old_ts.empty and not new_ts.empty and "id" in old_ts.columns and "id" in new_ts.columns:
        merged = old_ts.merge(new_ts, on="id", suffixes=("_old", "_new"))
        for column in compare_columns:
            old_col = f"{column}_old"
            new_col = f"{column}_new"
            if old_col not in merged.columns or new_col not in merged.columns:
                continue
            old_values = pd.to_numeric(merged[old_col], errors="coerce")
            new_values = pd.to_numeric(merged[new_col], errors="coerce")
            diff = (old_values - new_values).abs()
            unequal = int((diff.fillna(0) > 1e-9).sum())
            max_diff = float(diff.max()) if not diff.dropna().empty else 0.0
            value_lines.append(f"| {column} | {unequal} | {max_diff:.6g} |")
    lines.extend(["", "## 共同 time_series_data 主键的数值差异", "", "| 字段 | 不一致行数 | 最大绝对差 |", "|---|---:|---:|"])
    lines.extend(value_lines or ["| 无共同主键或无可比字段 | 0 | 0 |"])

    lines.extend(
        [
            "",
            "## 逐行逐单元格 diff 文件",
            "",
            f"详细差异已写入 `{Path(diff_dir).as_posix()}`。",
            "",
            "| 对比文件 | diff 行数 | 单元格差异 | 旧 CSV 独有行 | Excel 独有行 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, summary in detailed_summaries.items():
        lines.append(
            f"| {name}.csv | {summary['diff_rows']} | {summary['cell_diffs']} | {summary['old_only_rows']} | {summary['new_only_rows']} |"
        )

    total_missing_values = missing_value_summary.get("old_nonblank_values_blank_or_missing_in_new_schema", 0)
    not_found_values = missing_value_summary.get("coverage::not_found_in_new_outputs", 0)
    covered_values = total_missing_values - not_found_values
    lines.extend(
        [
            "",
            "## 旧 CSV 非空值缺失覆盖审计",
            "",
            "口径：旧 CSV 里非空的实验值，如果在新生成的 schema 主表中整行缺失或单元格为空，则继续检查 `liquid_long_data.csv`、`supplemental_cells.csv`、`excel_cells.csv` 是否仍能找到该值。",
            "",
            f"- 新 schema 主表中缺失或空白的旧非空值：{total_missing_values}",
            f"- 已在新审计/液相输出中找到：{covered_values}",
            f"- 当前新输出中仍未找到：{not_found_values}",
            "",
            "| 维度 | 数量 |",
            "|---|---:|",
        ]
    )
    for key, count in missing_value_summary.items():
        if key.startswith("table::") or key.startswith("not_found_file::"):
            lines.append(f"| {key} | {count} |")
    lines.extend(
        [
            "",
            "详细缺失覆盖已写入 `summary/supporting_reports/excel_csv_diffs/old_nonblank_values_blank_or_missing_in_new_schema_with_coverage.csv`；汇总已写入 `summary/supporting_reports/excel_csv_diffs/missing_value_coverage_summary_strict.csv`。",
        ]
    )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- Excel 转换结果以 `data/excel` 为权威源，输出在 `data/csv_from_excel`。",
            "- 主键 diff 使用表的 schema 主键；自然键 diff 使用 `file_name + sheet_name` 或 `file_name + sheet_name + 行序号`，用于避开 run_id 命名策略差异。",
            "- 液相 sheet 已输出到 `liquid_long_data.csv`；原始公式和所有非空单元格已输出到 `excel_cells.csv`，发酵主表之外的补充区域已输出到 `supplemental_cells.csv`。",
            "- `hplc_data` 仍保留 schema 空表；后续可在确认液相字段口径后，从 `liquid_long_data.csv` 派生标准 HPLC 宽表。",
            "- 对比中的 `created_at` 和 `source_file_md5` 不用于判断实验数据一致性。",
        ]
    )

    markdown = "\n".join(lines) + "\n"
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    return markdown
