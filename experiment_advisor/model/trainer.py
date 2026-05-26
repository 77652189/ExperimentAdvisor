from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from experiment_advisor.ingestion.run_level import CONTROL_FEATURES, TARGET_COL, training_view


@dataclass
class ModelBundle:
    models: dict[str, Any]
    reference_models: dict[str, Any]
    feature_columns: list[str]
    target_col: str
    metrics: dict[str, dict[str, float]]
    reference_metrics: dict[str, dict[str, float]]
    model_info: dict[str, Any]


def _require_sklearn():
    try:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        from sklearn.linear_model import ElasticNetCV, RidgeCV
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError("Model training requires scikit-learn. Install dependencies with: pip install -r requirements.txt") from exc
    return {
        "ExtraTreesRegressor": ExtraTreesRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "ElasticNetCV": ElasticNetCV,
        "RidgeCV": RidgeCV,
        "mean_absolute_error": mean_absolute_error,
        "r2_score": r2_score,
        "LeaveOneOut": LeaveOneOut,
        "cross_val_predict": cross_val_predict,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
    }


def _xgboost_model():
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None
    return XGBRegressor(
        max_depth=2,
        n_estimators=50,
        subsample=0.8,
        min_child_weight=3,
        reg_alpha=1,
        reg_lambda=2,
        random_state=42,
        objective="reg:squarederror",
    )


def _default_features(df: pd.DataFrame, target_col: str) -> list[str]:
    candidates = [column for column in CONTROL_FEATURES if column in df.columns]
    if candidates:
        return candidates
    return [
        column
        for column in df.columns
        if column != target_col and pd.api.types.is_numeric_dtype(df[column])
    ]


def _metrics(model: Any, x: pd.DataFrame, y: pd.Series, helpers: dict[str, Any]) -> dict[str, float]:
    if len(x) < 3:
        return {"mae_loocv": float("nan"), "r2_loocv": float("nan")}
    try:
        predictions = helpers["cross_val_predict"](model, x, y, cv=helpers["LeaveOneOut"]())
        return {
            "mae_loocv": float(helpers["mean_absolute_error"](y, predictions)),
            "r2_loocv": float(helpers["r2_score"](y, predictions)),
        }
    except Exception:
        return {"mae_loocv": float("nan"), "r2_loocv": float("nan")}


def train_surrogate_ensemble(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    feature_cols: list[str] | None = None,
    include_reference_models: bool = False,
) -> ModelBundle:
    """训练小样本 surrogate ensemble，并锁定特征列顺序。

    默认 ensemble 只包含 Ridge、ElasticNet 和强正则 XGBoost。RandomForest
    与 ExtraTrees 在当前样本量下只作为可选参考模型，不参与保守推荐器的不确定性计算。
    """

    helpers = _require_sklearn()
    train_df = training_view(df, target_col) if "exclude_from_training" in df.columns else df.dropna(subset=[target_col])
    features = feature_cols or _default_features(train_df, target_col)
    clean = train_df[[*features, target_col]].dropna()
    if len(clean) < 5:
        raise ValueError("At least 5 complete training rows are required for surrogate ensemble")

    x = clean[features]
    y = clean[target_col].astype(float)
    models: dict[str, Any] = {
        "ridge": helpers["make_pipeline"](
            helpers["StandardScaler"](),
            helpers["RidgeCV"](alphas=[0.1, 1.0, 10.0, 100.0]),
        ),
        "elastic_net": helpers["make_pipeline"](
            helpers["StandardScaler"](),
            helpers["ElasticNetCV"](cv=min(5, len(clean)), random_state=42, max_iter=10000),
        ),
    }
    xgb = _xgboost_model()
    if xgb is not None:
        models["xgboost"] = xgb

    reference_models: dict[str, Any] = {
        "random_forest_reference": helpers["RandomForestRegressor"](
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
        ),
        "extra_trees_reference": helpers["ExtraTreesRegressor"](
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
        ),
    }

    metrics: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}
    for name, model in models.items():
        metrics[name] = _metrics(model, x, y, helpers)
        model.fit(x, y)
        fitted[name] = model

    reference_metrics: dict[str, dict[str, float]] = {}
    fitted_reference: dict[str, Any] = {}
    if include_reference_models:
        for name, model in reference_models.items():
            reference_metrics[name] = _metrics(model, x, y, helpers)
            model.fit(x, y)
            fitted_reference[name] = model

    return ModelBundle(
        models=fitted,
        reference_models=fitted_reference,
        feature_columns=features,
        target_col=target_col,
        metrics=metrics,
        reference_metrics=reference_metrics,
        model_info={
            "n_training_rows": int(len(clean)),
            "target_col": target_col,
            "feature_columns": features,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "model_names": list(fitted),
            "reference_model_names": list(fitted_reference),
            "uncertainty_note": "model_uncertainty is ensemble disagreement across ridge, elastic_net, and xgboost when available",
        },
    )
