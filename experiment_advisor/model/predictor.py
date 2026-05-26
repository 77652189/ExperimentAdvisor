from __future__ import annotations

import pandas as pd

from experiment_advisor.model.trainer import ModelBundle


def predict_candidates(model_bundle: ModelBundle, candidates: pd.DataFrame) -> pd.DataFrame:
    """批量预测候选点，返回 ensemble 均值、模型分歧标准差和每个模型的预测。"""

    missing = [column for column in model_bundle.feature_columns if column not in candidates.columns]
    if missing:
        raise ValueError(f"candidate data missing feature columns: {', '.join(missing)}")
    x = candidates[model_bundle.feature_columns]
    result = candidates.copy()
    prediction_columns: list[str] = []
    for name, model in model_bundle.models.items():
        column = f"pred_{name}"
        result[column] = model.predict(x)
        prediction_columns.append(column)
    result["predicted_yield"] = result[prediction_columns].mean(axis=1)
    result["model_uncertainty"] = result[prediction_columns].std(axis=1).fillna(0.0)
    return result
