from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _default_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    excluded = {target_col, "batch_id"}
    return [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]


def _loocv_r2(model_factory, x, y) -> float:
    from sklearn.metrics import r2_score
    from sklearn.model_selection import LeaveOneOut

    predictions = []
    actuals = []
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(x):
        model = model_factory()
        model.fit(x[train_idx], y[train_idx])
        predictions.append(float(model.predict(x[test_idx])[0]))
        actuals.append(float(y[test_idx][0]))
    return float(r2_score(actuals, predictions))


def run_offline_analysis(
    df: pd.DataFrame,
    target_col: str = "yield_g_per_l",
    feature_cols: list[str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    对历史数据做一次性离线分析，产出 Ridge/Lasso/XGBoost/SHAP 特征重要性报告。

    该结果只用于冷启动前解释与缩减 BO 搜索维度，不参与 BO 迭代循环。
    """

    try:
        import matplotlib.pyplot as plt
        import shap
        from sklearn.linear_model import LassoCV, RidgeCV
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError("Offline analysis requires scikit-learn, xgboost, shap, and matplotlib.") from exc

    features = feature_cols or _default_feature_cols(df, target_col)
    if target_col not in df.columns:
        raise ValueError(f"target column not found: {target_col}")
    if not features:
        raise ValueError("No numeric feature columns available for offline analysis")

    modeling = df[[*features, target_col]].dropna()
    if len(modeling) < 3:
        raise ValueError("At least 3 complete rows are required for offline analysis")

    x_raw = modeling[features].to_numpy(dtype=float)
    y = modeling[target_col].to_numpy(dtype=float)
    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw)

    ridge = RidgeCV(alphas=[0.1, 1, 10, 100], cv=LeaveOneOut()).fit(x, y)
    lasso = LassoCV(cv=LeaveOneOut(), random_state=42, max_iter=10000).fit(x, y)
    xgb = XGBRegressor(
        max_depth=2,
        n_estimators=50,
        subsample=0.8,
        min_child_weight=3,
        reg_alpha=1,
        reg_lambda=2,
        random_state=42,
        objective="reg:squarederror",
    ).fit(x, y)

    linear_r2 = _loocv_r2(lambda: RidgeCV(alphas=[0.1, 1, 10, 100]), x, y)
    xgb_r2 = _loocv_r2(
        lambda: XGBRegressor(
            max_depth=2,
            n_estimators=50,
            subsample=0.8,
            min_child_weight=3,
            reg_alpha=1,
            reg_lambda=2,
            random_state=42,
            objective="reg:squarederror",
        ),
        x,
        y,
    )

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(x)
    shap_importance_values = abs(shap_values).mean(axis=0)
    shap_importance = {
        feature: float(value)
        for feature, value in sorted(zip(features, shap_importance_values), key=lambda item: item[1], reverse=True)
    }
    lasso_selected = [feature for feature, coef in zip(features, lasso.coef_) if abs(float(coef)) > 1e-9]
    top_shap = list(shap_importance)[:6]
    recommended = [feature for feature in lasso_selected if feature in top_shap]

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(shap_values, pd.DataFrame(x, columns=features), show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=160)
        plt.close()

    result: dict[str, Any] = {
        "ridge_coefs": {feature: float(coef) for feature, coef in zip(features, ridge.coef_)},
        "lasso_selected": lasso_selected,
        "shap_importance": shap_importance,
        "recommended_bo_features": recommended,
        "linear_r2_loocv": float(linear_r2),
        "xgb_r2_loocv": float(xgb_r2),
    }
    if xgb_r2 < 0.2:
        result["low_signal_warning"] = True
    return result
