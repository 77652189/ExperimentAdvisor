from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from experiment_advisor.optimizer.search_space import SearchSpace


class _BoTorchGPPredictor:
    """Thin predict wrapper around a fitted BoTorch SingleTaskGP."""

    def __init__(self, model: SingleTaskGP, feature_cols: list[str]):
        self.model = model
        self.feature_cols = feature_cols

    def predict(self, X, return_std: bool = False):
        """
        X may be a pd.DataFrame or np.ndarray with shape (n, d).

        When return_std=True, returns (mean, std) as np.ndarray values.
        """

        if hasattr(X, "values"):
            X_arr = X[self.feature_cols].values if hasattr(X, "columns") else X.values
        else:
            X_arr = np.asarray(X)

        X_tensor = torch.tensor(X_arr, dtype=torch.double)
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            if return_std:
                std = posterior.variance.sqrt().squeeze(-1).cpu().numpy()
                return mean, std
        return mean


def recommend_standard_bo(
    df: pd.DataFrame,
    search_space: SearchSpace,
    top_k: int = 5,
    target_col: str = "yield_g_per_l",
    feature_cols: list[str] | None = None,
    seed: int = 0,
) -> dict:
    """Recommend a batch with BoTorch qLogNEI and an MLE-fitted SingleTaskGP.

    seed 固定随机种子以保证结果可复现。传入不同整数可探索多组推荐方案。
    """

    features = feature_cols or list(search_space.bounds)
    train = df[[*features, target_col]].dropna()
    if len(train) < 5:
        raise ValueError("At least 5 complete training rows are required")

    x = train[features]
    y = train[target_col].astype(float)

    # Also seed model construction. We seed again immediately before optimize_acqf
    # below, so any RNG consumed while fitting cannot perturb raw_samples.
    torch.manual_seed(seed)
    train_X = torch.tensor(x.values, dtype=torch.double)
    train_Y = torch.tensor(y.values, dtype=torch.double).unsqueeze(-1)

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=train_X.shape[-1]),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    bounds = torch.tensor(
        [
            [search_space.bounds[f][0] for f in features],
            [search_space.bounds[f][1] for f in features],
        ],
        dtype=torch.double,
    )

    # seed=seed 直接控制 Sobol 引擎的准随机序列，与 PyTorch 全局 RNG 无关
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]), seed=seed)
    acqf = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_X,
        sampler=sampler,
        prune_baseline=True,
    )

    # 紧接着 optimize_acqf 之前设种子，精确控制 raw_samples 的随机起始点
    # 不放在函数顶部是因为 fit_gpytorch_mll 会消耗不确定数量的随机数，
    # 导致 optimize_acqf 实际拿到的 RNG 状态不可预测
    torch.manual_seed(seed)
    candidates_tensor, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=top_k,
        num_restarts=10,
        raw_samples=512,
    )

    with torch.no_grad():
        posterior = model.posterior(candidates_tensor)
        pred_means = posterior.mean.squeeze(-1).cpu().numpy()
        pred_stds = posterior.variance.sqrt().squeeze(-1).cpu().numpy()

    wrapper = _BoTorchGPPredictor(model=model, feature_cols=features)
    recommendations = []
    for rank, idx in enumerate(range(top_k), start=1):
        params = {
            features[j]: float(candidates_tensor[idx, j].item())
            for j in range(len(features))
        }
        recommendations.append(
            {
                "method": "standard_bo_qnei",
                "rank": rank,
                "params": params,
                "predicted_yield": float(pred_means[idx]),
                "model_uncertainty": float(pred_stds[idx]),
                "uncertainty_type": "gp_posterior_std",
                "acquisition_score": float(pred_means[idx]),
            }
        )

    return {
        "recommendations": recommendations,
        "fitted_gp": wrapper,
        "feature_cols": features,
    }


def _fit_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    seed: int,
) -> SingleTaskGP:
    """Fit a SingleTaskGP with Normalize + Standardize transforms."""
    torch.manual_seed(seed)
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=train_X.shape[-1]),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model


def recommend_standard_bo_ei(
    df: pd.DataFrame,
    search_space: SearchSpace,
    top_k: int = 5,
    target_col: str = "yield_g_per_l",
    feature_cols: list[str] | None = None,
    seed: int = 0,
) -> dict:
    """Recommend a batch with sequential-greedy BoTorch EI.

    Sequentially optimises single-point EI top_k times.  After each step the
    chosen candidate is registered as a pending point via set_X_pending so that
    subsequent EI evaluations are repelled away from already-chosen locations,
    providing batch diversity without re-fitting the GP.
    """

    features = feature_cols or list(search_space.bounds)
    train = df[[*features, target_col]].dropna()
    if len(train) < 5:
        raise ValueError("At least 5 complete training rows are required")

    x = train[features]
    y = train[target_col].astype(float)

    train_X = torch.tensor(x.values, dtype=torch.double)
    train_Y = torch.tensor(y.values, dtype=torch.double).unsqueeze(-1)

    model = _fit_gp(train_X, train_Y, seed)

    bounds = torch.tensor(
        [
            [search_space.bounds[f][0] for f in features],
            [search_space.bounds[f][1] for f in features],
        ],
        dtype=torch.double,
    )

    # best_f: best observed yield in the original (untransformed) scale.
    best_f = float(train_Y.max())

    candidates: list[torch.Tensor] = []
    for step in range(top_k):
        # qLogExpectedImprovement with q=1 is numerically equivalent to analytic
        # LogEI but supports set_X_pending — required for sequential greedy.
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]), seed=seed + step)
        acqf = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler)
        if candidates:
            acqf.set_X_pending(torch.cat(candidates, dim=0))

        # Re-seed before each optimize_acqf call so raw_samples are reproducible
        # for this step regardless of previous steps' RNG consumption.
        torch.manual_seed(seed + step)
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        candidates.append(candidate)

    candidates_tensor = torch.cat(candidates, dim=0)  # shape (top_k, d)

    with torch.no_grad():
        posterior = model.posterior(candidates_tensor)
        pred_means = posterior.mean.squeeze(-1).cpu().numpy()
        pred_stds = posterior.variance.sqrt().squeeze(-1).cpu().numpy()

    wrapper = _BoTorchGPPredictor(model=model, feature_cols=features)
    recommendations = []
    for rank in range(1, top_k + 1):
        idx = rank - 1
        params = {
            features[j]: float(candidates_tensor[idx, j].item())
            for j in range(len(features))
        }
        recommendations.append(
            {
                "method": "standard_bo_ei",
                "rank": rank,
                "params": params,
                "predicted_yield": float(pred_means[idx]),
                "model_uncertainty": float(pred_stds[idx]),
                "uncertainty_type": "gp_posterior_std",
                "acquisition_score": float(pred_means[idx]),
            }
        )

    return {
        "recommendations": recommendations,
        "fitted_gp": wrapper,
        "feature_cols": features,
    }
