from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

df = pd.read_csv("data/final/run_level_modeling_dataset.csv")
train = df[df["exclude_from_training"] == False].copy()

MODEL_FEATURES = [
    "temperature_shift_time_h",
    "temperature_production_phase_c",
    "lactose_total_ml",
    "feed1_total_ml",
    "feed2_total_ml",
    "lactose_first_add_time_h",
]
TARGET = "yield_g_per_l"
REMOVED = ["fermentation_duration_h", "lactose_after_48h_ml"]

train = train[MODEL_FEATURES + [TARGET] + REMOVED].dropna()
print(f"Training rows: {len(train)}")

X = train[MODEL_FEATURES].values
y = train[TARGET].values

# LOO-CV with Ridge
loo = LeaveOneOut()
residuals = np.zeros(len(train))
scaler = StandardScaler()

for train_idx, test_idx in loo.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr = y[train_idx]
    y_te = y[test_idx]
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = Ridge(alpha=1.0)
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_te_s)[0]
    residuals[test_idx[0]] = y_te[0] - pred  # actual - predicted

mae = float(np.abs(residuals).mean())
r2 = float(1 - np.var(residuals) / np.var(y))
print(f"LOO-CV MAE : {mae:.3f} g/L")
print(f"LOO-CV R2  : {r2:.3f}")
print()

# Residual vs removed features
print("=== Residual vs REMOVED features ===")
for feat in REMOVED:
    feat_vals = train[feat].values
    r, p = stats.spearmanr(feat_vals, residuals)
    sig = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"{feat}")
    print(f"  Spearman r = {r:+.3f},  p = {p:.4f}  [{sig}]")
    if abs(r) > 0.2:
        if r > 0:
            print("  -> larger value = model under-predicts (residual positive)")
        else:
            print("  -> larger value = model over-predicts (residual negative)")
    print()

# Residual vs in-model features (control)
print("=== Residual vs IN-MODEL features (control) ===")
for feat in MODEL_FEATURES:
    feat_vals = train[feat].values
    r, p = stats.spearmanr(feat_vals, residuals)
    sig = " **" if p < 0.05 else ""
    print(f"  {feat:<42s}  r={r:+.3f}  p={p:.3f}{sig}")
