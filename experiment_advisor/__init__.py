"""ExperimentAdvisor: historical-data cold-start Bayesian optimization."""

from experiment_advisor.analysis.diagnostics import estimate_noise, run_loocv
from experiment_advisor.ingestion.features import engineer_features
from experiment_advisor.ingestion.loader import load_fermentation_data
from experiment_advisor.ingestion.pipeline import build_final_dataset, build_run_level_dataset
from experiment_advisor.ingestion.run_level import MODEL_FEATURES
from experiment_advisor.ingestion.validator import validate
from experiment_advisor.recommendation.service import compare_recommenders, recommend_next
from experiment_advisor.report.reporter import generate_recommendation_report

__all__ = [
    "MODEL_FEATURES",
    "build_final_dataset",
    "build_run_level_dataset",
    "compare_recommenders",
    "engineer_features",
    "estimate_noise",
    "generate_recommendation_report",
    "load_fermentation_data",
    "recommend_next",
    "run_loocv",
    "validate",
]
