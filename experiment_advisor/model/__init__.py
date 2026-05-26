"""Surrogate model training and prediction."""

from experiment_advisor.model.predictor import predict_candidates
from experiment_advisor.model.registry import load_model_bundle, save_model_bundle
from experiment_advisor.model.trainer import train_surrogate_ensemble

__all__ = [
    "load_model_bundle",
    "predict_candidates",
    "save_model_bundle",
    "train_surrogate_ensemble",
]
