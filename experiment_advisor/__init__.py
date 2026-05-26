"""ExperimentAdvisor: historical-data cold-start Bayesian optimization."""

from experiment_advisor.analysis.diagnostics import estimate_noise, run_loocv
from experiment_advisor.analysis.offline_analyzer import run_offline_analysis
from experiment_advisor.ingestion.features import engineer_features
from experiment_advisor.ingestion.loader import load_fermentation_data
from experiment_advisor.ingestion.pipeline import build_final_dataset, build_run_level_dataset
from experiment_advisor.ingestion.validator import validate
from experiment_advisor.optimizer.bo_engine import BOEngine
from experiment_advisor.optimizer.xgp_bo import recommend_xgp_bo
from experiment_advisor.recommendation.service import compare_recommenders, recommend_next
from experiment_advisor.report.reporter import generate_recommendation_report
from experiment_advisor.space.parameter_space import build_search_space

__all__ = [
    "BOEngine",
    "build_final_dataset",
    "build_run_level_dataset",
    "build_search_space",
    "compare_recommenders",
    "engineer_features",
    "estimate_noise",
    "generate_recommendation_report",
    "load_fermentation_data",
    "recommend_next",
    "recommend_xgp_bo",
    "run_loocv",
    "run_offline_analysis",
    "validate",
]
