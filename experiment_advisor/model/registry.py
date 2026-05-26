from __future__ import annotations

import json
from pathlib import Path

from experiment_advisor.model.trainer import ModelBundle


def _require_joblib():
    try:
        import joblib
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError("Model registry requires joblib. Install dependencies with: pip install -r requirements.txt") from exc
    return joblib


def save_model_bundle(model_bundle: ModelBundle, output_dir: str | Path = "artifacts/model") -> None:
    """保存模型、feature_columns 和 model_info。"""

    joblib = _require_joblib()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle.models, out / "best_model.pkl")
    joblib.dump(model_bundle.feature_columns, out / "feature_columns.pkl")
    (out / "model_info.json").write_text(json.dumps(model_bundle.model_info, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model_bundle(model_dir: str | Path = "artifacts/model") -> ModelBundle:
    """加载模型包。metrics 不随 pkl 保存，恢复时为空 dict。"""

    joblib = _require_joblib()
    model_path = Path(model_dir)
    models = joblib.load(model_path / "best_model.pkl")
    feature_columns = joblib.load(model_path / "feature_columns.pkl")
    model_info = json.loads((model_path / "model_info.json").read_text(encoding="utf-8"))
    return ModelBundle(
        models=models,
        feature_columns=feature_columns,
        target_col=model_info.get("target_col", "yield_g_per_l"),
        metrics={},
        model_info=model_info,
    )
