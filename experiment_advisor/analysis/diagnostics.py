from __future__ import annotations

from typing import Any

import pandas as pd


def _numeric_features(df: pd.DataFrame, target_col: str) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in {target_col, "batch_id"} and pd.api.types.is_numeric_dtype(df[column])
    ]


def estimate_noise(df: pd.DataFrame, target_col: str = "yield_g_per_l") -> float:
    """
    估计批次间随机噪声标准差。

    先寻找标准化欧氏距离 < 0.5 的相近批次对；若不足两对，退回到目标列标准差的 10%。
    """

    features = _numeric_features(df, target_col)
    clean = df[[*features, target_col]].dropna()
    if len(clean) < 2 or not features:
        series = pd.to_numeric(df[target_col], errors="coerce").dropna()
        return float(series.std(ddof=1) * 0.1) if len(series) > 1 else 0.0

    x = clean[features].astype(float)
    scaled = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
    y = clean[target_col].astype(float).to_numpy()
    diffs: list[float] = []
    values = scaled.to_numpy()
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            distance = float(((values[i] - values[j]) ** 2).sum() ** 0.5)
            if distance < 0.5:
                diffs.append(abs(float(y[i] - y[j])))
    if len(diffs) >= 2:
        return float(pd.Series(diffs).std(ddof=1) / (2 ** 0.5))
    return float(clean[target_col].std(ddof=1) * 0.1)


def run_loocv(
    df: pd.DataFrame,
    target_col: str = "yield_g_per_l",
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    使用 BoTorch SingleTaskGP 对历史数据做留一交叉验证。

    返回 {"rmse": float, "r2": float, "predictions": [(actual, predicted), ...]}。
    """

    try:
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import LeaveOneOut
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError("GP diagnostics require botorch, torch, gpytorch, and scikit-learn.") from exc

    features = feature_cols or _numeric_features(df, target_col)
    clean = df[[*features, target_col]].dropna()
    if len(clean) < 3:
        raise ValueError("At least 3 complete rows are required for GP LOO-CV")

    x = clean[features].to_numpy(dtype=float)
    y = clean[target_col].to_numpy(dtype=float)
    predictions: list[tuple[float, float]] = []
    noise_var = estimate_noise(clean, target_col) ** 2
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(x):
        scaler = StandardScaler()
        train_x = torch.tensor(scaler.fit_transform(x[train_idx]), dtype=torch.double)
        test_x = torch.tensor(scaler.transform(x[test_idx]), dtype=torch.double)
        train_y = torch.tensor(y[train_idx].reshape(-1, 1), dtype=torch.double)
        likelihood = GaussianLikelihood(noise_constraint=None)
        likelihood.noise = torch.tensor(max(noise_var, 1e-8), dtype=torch.double)
        model = SingleTaskGP(train_x, train_y, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        posterior = model.posterior(test_x)
        predictions.append((float(y[test_idx][0]), float(posterior.mean.detach().numpy()[0, 0])))

    actual = [item[0] for item in predictions]
    predicted = [item[1] for item in predictions]
    return {
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "predictions": predictions,
    }
