"""Data ingestion, validation, and feature engineering for ExperimentAdvisor."""

from experiment_advisor.ingestion.features import engineer_features
from experiment_advisor.ingestion.loader import load_fermentation_data
from experiment_advisor.ingestion.pipeline import build_final_dataset
from experiment_advisor.ingestion.run_level import build_run_level_dataset, training_view
from experiment_advisor.ingestion.validator import validate
from experiment_advisor.ingestion.excel_schema_converter import (
    audit_old_nonblank_value_coverage,
    compare_csv_directories,
    convert_excel_directory,
    write_detailed_diff_files,
)

__all__ = [
    "build_final_dataset",
    "build_run_level_dataset",
    "audit_old_nonblank_value_coverage",
    "compare_csv_directories",
    "convert_excel_directory",
    "engineer_features",
    "load_fermentation_data",
    "training_view",
    "validate",
    "write_detailed_diff_files",
]
