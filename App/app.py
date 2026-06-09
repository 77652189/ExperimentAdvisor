from __future__ import annotations

import hashlib
import importlib
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_advisor.ingestion.run_level import TARGET_COL, add_model_derived_features, training_view
from experiment_advisor.optimizer.search_space import SearchSpace
from experiment_advisor.recommendation.pichia import (
    METHOD_COUPLING_DEFAULTS,
    PICHIA_PARAMETER_SPECS,
    PICHIA_TARGET_COL,
    analyze_pichia_doe_feedback,
    empty_pichia_template,
    pichia_template_columns,
    recommend_pichia_design,
    two_factor_design_matrix,
    two_factor_doe_levels,
    uniform_single_variable_levels,
)
import experiment_advisor.recommendation.service as recommendation_service
from experiment_advisor.recommendation.quality import evaluate_recommendation_quality
from experiment_advisor.report import generate_recommendation_report


def _configure_plot_fonts() -> None:
    """Prefer CJK-capable fonts so Chinese plot labels render correctly."""
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, *preferred_fonts]
            plt.rcParams["font.family"] = "sans-serif"
            break
    plt.rcParams["axes.unicode_minus"] = False


_configure_plot_fonts()

METHOD_LABELS = {
    "standard_bo_qnei": "标准 GP-BO（qNEI）",
}

METHOD_EXPLANATIONS = {
    "standard_bo_qnei": "BoTorch SingleTaskGP 直接拟合产量，并用 qNEI 联合优化下一批推荐。",
}

UNCERTAINTY_LABELS = {
    "gp_posterior_std": "GP 后验标准差",
}

RISK_LABELS = {"low": "低", "medium": "中", "high": "高"}
FLAG_LABELS = {
    "far_from_history": "远离历史实验",
    "near_search_boundary": "接近参数边界",
    "high_residual_uncertainty": "不确定性较高",
}

RECOMMENDATION_CACHE_VERSION = "raw-bo-v1"
RECOMMENDATION_CACHE_KEY = "recommendation_result_cache"

PICHIA_DATA_DIR = PROJECT_ROOT / "data" / "pichia"
PICHIA_UPLOAD_DIR = PICHIA_DATA_DIR / "uploads"
PICHIA_FINAL_DIR = PICHIA_DATA_DIR / "final"
PICHIA_TEMPLATE_DIR = PICHIA_DATA_DIR / "templates"
PICHIA_DEFAULT_DATASET_PATH = PICHIA_FINAL_DIR / "pichia_run_level_dataset.csv"
PICHIA_TEMPLATE_PATH = PICHIA_TEMPLATE_DIR / "pichia_run_level_template.csv"
PICHIA_BASELINE_SOURCES = ["同菌种历史最优", "同菌种最近成功实验", "亲本菌种历史最优", "手动输入"]
PICHIA_VARIABLE_LABELS = {name: spec.label for name, spec in PICHIA_PARAMETER_SPECS.items()}
PICHIA_VARIABLE_KEYS_BY_LABEL = {label: name for name, label in PICHIA_VARIABLE_LABELS.items()}


def _load_field_labels() -> dict[str, str]:
    dictionary_path = PROJECT_ROOT / "summary" / "supporting_reports" / "field_dictionary.csv"
    if not dictionary_path.exists():
        return {}
    dictionary = pd.read_csv(dictionary_path)
    return {
        str(row["field"]): str(row["zh_name"])
        for _, row in dictionary.iterrows()
        if pd.notna(row.get("field")) and pd.notna(row.get("zh_name"))
    }


FIELD_LABELS = _load_field_labels()


def _compare_recommenders(df: pd.DataFrame, top_k: int, seed: int, method: str = "ei") -> dict[str, Any]:
    service = importlib.reload(recommendation_service)
    return service.compare_recommenders(df, top_k=top_k, seed=seed, method=method)


def _recommendation_pool_size(top_k: int, multiplier: int = 3) -> int:
    return min(max(top_k * multiplier, top_k), 40)


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    frame = df.copy()
    frame.columns = frame.columns.map(str)
    frame = frame.reindex(sorted(frame.columns), axis=1)

    digest = hashlib.sha256()
    digest.update(RECOMMENDATION_CACHE_VERSION.encode("utf-8"))
    digest.update(str(frame.shape).encode("utf-8"))
    digest.update("\0".join(frame.columns).encode("utf-8"))
    try:
        row_hash = pd.util.hash_pandas_object(frame, index=True).values
    except TypeError:
        row_hash = pd.util.hash_pandas_object(frame.astype(str), index=True).values
    digest.update(row_hash.tobytes())
    return digest.hexdigest()


def _read_recommendation_cache(state: Any, dataset_fingerprint: str) -> dict[str, Any] | None:
    cache = state.get(RECOMMENDATION_CACHE_KEY)
    if not isinstance(cache, dict):
        return None
    if cache.get("version") != RECOMMENDATION_CACHE_VERSION:
        return None
    if cache.get("dataset_fingerprint") != dataset_fingerprint:
        return None
    if not cache.get("comparison"):
        return None
    return cache


def _write_recommendation_cache(
    state: Any,
    *,
    dataset_fingerprint: str,
    comparison: dict[str, Any],
    report_md: str,
    report_path: Path,
    run_settings: dict[str, Any],
) -> None:
    state[RECOMMENDATION_CACHE_KEY] = {
        "version": RECOMMENDATION_CACHE_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "comparison": comparison,
        "report_md": report_md,
        "report_path": str(report_path),
        "run_settings": run_settings,
    }


def _ensure_strategy_quality(comparison: dict[str, Any], df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    quality = comparison.get("strategy_quality_pool") or comparison.get("strategy_quality") or {}
    if quality:
        return quality

    selected = comparison.get("unfiltered_selected_recommendations") or comparison.get("selected_recommendations", [])
    search_bounds = comparison.get("search_space") or {}
    features = feature_cols or list(search_bounds)
    if not selected or not search_bounds or not features:
        return {}

    space = SearchSpace(bounds={name: tuple(bounds) for name, bounds in search_bounds.items()})
    quality = evaluate_recommendation_quality(
        selected,
        _training_data(df),
        space,
        feature_cols=features,
        target_col=TARGET_COL,
    )
    comparison["strategy_quality"] = quality
    return quality


def _select_without_soft_filters(
    comparison: dict[str, Any],
    df: pd.DataFrame,
    feature_cols: list[str],
    target_count: int,
) -> dict[str, Any]:
    selected_method = comparison.get("selected_method", "standard_bo_qnei")
    base = (
        comparison.get("unfiltered_selected_recommendations")
        or comparison.get("recommendations", {}).get(selected_method, [])
        or comparison.get("selected_recommendations", [])
    )
    selected = []
    for item in base[:target_count]:
        tagged = item.copy()
        tagged.pop("soft_filter_status", None)
        tagged.pop("history_range_violations", None)
        tagged.pop("history_range_violation_features", None)
        selected.append(tagged)

    comparison["unfiltered_selected_recommendations"] = base
    comparison["selected_recommendations"] = selected
    comparison["soft_filter"] = {
        "enabled": False,
        "n_before": len(base),
        "n_after": len(selected),
        "target_count": target_count,
    }

    search_bounds = comparison.get("search_space") or {}
    if search_bounds:
        space = SearchSpace(bounds={name: tuple(bounds) for name, bounds in search_bounds.items()})
        comparison["strategy_quality"] = evaluate_recommendation_quality(
            selected,
            _training_data(df),
            space,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
        )
    return comparison


def _history_sigma_ranges(df: pd.DataFrame, feature_cols: list[str], sigma: float) -> dict[str, tuple[float, float, float]]:
    train = _training_data(df)
    ranges: dict[str, tuple[float, float, float]] = {}
    for feature in feature_cols:
        if feature not in train.columns:
            continue
        values = pd.to_numeric(train[feature], errors="coerce").dropna()
        if len(values) < 2:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std <= 1e-12:
            continue
        ranges[feature] = (mean - sigma * std, mean + sigma * std, std)
    return ranges


def _history_range_violation(
    recommendation: dict[str, Any],
    ranges: dict[str, tuple[float, float, float]],
) -> tuple[int, float, list[str]]:
    count = 0
    total_excess = 0.0
    features = []
    for feature, value in recommendation.get("params", {}).items():
        if feature not in ranges or value is None:
            continue
        low, high, std = ranges[feature]
        numeric_value = float(value)
        if numeric_value < low:
            count += 1
            total_excess += (low - numeric_value) / std
            features.append(feature)
        elif numeric_value > high:
            count += 1
            total_excess += (numeric_value - high) / std
            features.append(feature)
    return count, total_excess, features


def _apply_soft_filters(
    comparison: dict[str, Any],
    df: pd.DataFrame,
    feature_cols: list[str],
    max_nearest_history_distance: float,
    max_boundary_risk: float,
    history_sigma: float,
    target_count: int | None = None,
) -> dict[str, Any]:
    quality = _ensure_strategy_quality(comparison, df, feature_cols)
    per_items = quality.get("per_recommendation") or []
    if not per_items:
        return comparison

    base = comparison.get("unfiltered_selected_recommendations") or comparison.get("selected_recommendations", [])
    per_by_rank = {item.get("rank"): item for item in per_items}
    sigma_ranges = _history_sigma_ranges(df, feature_cols, history_sigma)

    def failure_reasons(rec: dict[str, Any]) -> dict[str, Any]:
        quality_item = per_by_rank.get(rec.get("rank"), {})
        nearest = quality_item.get("nearest_history_distance")
        boundary = quality_item.get("boundary_risk")
        sigma_violation_count, _, sigma_features = _history_range_violation(rec, sigma_ranges)
        return {
            "nearest": nearest is not None and nearest > max_nearest_history_distance,
            "boundary": boundary is not None and boundary > max_boundary_risk,
            "history_range": sigma_violation_count > 0,
            "history_range_features": sigma_features,
        }

    def passes(rec: dict[str, Any]) -> bool:
        reasons = failure_reasons(rec)
        return not (reasons["nearest"] or reasons["boundary"] or reasons["history_range"])

    passed = [item for item in base if passes(item)]
    failed = [item for item in base if item.get("rank") not in {rec.get("rank") for rec in passed}]
    keep_count = target_count if target_count is not None else len(base)

    selected = []
    for item in passed[:keep_count]:
        tagged = item.copy()
        tagged["soft_filter_status"] = "通过"
        violation_count, _, violation_features = _history_range_violation(item, sigma_ranges)
        tagged["history_range_violations"] = violation_count
        tagged["history_range_violation_features"] = violation_features
        selected.append(tagged)

    failed_sigma = [
        item.get("rank")
        for item in failed
        if failure_reasons(item)["history_range"]
    ]
    failed_nearest = [
        item.get("rank")
        for item in failed
        if failure_reasons(item)["nearest"]
    ]
    failed_boundary = [
        item.get("rank")
        for item in failed
        if failure_reasons(item)["boundary"]
    ]
    comparison["unfiltered_selected_recommendations"] = base
    comparison["selected_recommendations"] = selected
    comparison["strategy_quality_pool"] = quality
    search_bounds = comparison.get("search_space") or {}
    if search_bounds:
        space = SearchSpace(bounds={name: tuple(bounds) for name, bounds in search_bounds.items()})
        comparison["strategy_quality"] = evaluate_recommendation_quality(
            selected,
            _training_data(df),
            space,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
        )
    comparison["soft_filter"] = {
        "enabled": True,
        "max_nearest_history_distance": max_nearest_history_distance,
        "max_boundary_risk": max_boundary_risk,
        "history_sigma": history_sigma,
        "n_before": len(base),
        "n_passed": len(passed),
        "n_after": len(selected),
        "target_count": keep_count,
        "failed_ranks": [item.get("rank") for item in failed],
        "failed_nearest_history_ranks": failed_nearest,
        "failed_boundary_risk_ranks": failed_boundary,
        "failed_history_range_ranks": failed_sigma,
        "failure_counts": {
            "nearest_history_distance": len(failed_nearest),
            "boundary_risk": len(failed_boundary),
            "history_range": len(failed_sigma),
        },
    }
    return comparison


def _load_default_dataset() -> pd.DataFrame:
    dataset_path = PROJECT_ROOT / "data" / "final" / "run_level_modeling_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"默认数据集不存在：{dataset_path}")
    return pd.read_csv(dataset_path)


def _training_data(df: pd.DataFrame) -> pd.DataFrame:
    if "exclude_from_training" in df.columns:
        return training_view(df, TARGET_COL)
    return df.dropna(subset=[TARGET_COL])


def _name(value: str) -> str:
    return FIELD_LABELS.get(value, value)


def _display_name(value: str) -> str:
    zh_name = _name(value)
    if zh_name == value:
        return value
    return f"{zh_name} ({value})"


def _display_dataframe(frame: pd.DataFrame, *, keep_english: bool = False) -> pd.DataFrame:
    if keep_english:
        return frame.rename(columns={column: _display_name(str(column)) for column in frame.columns})
    return frame.rename(columns={column: _name(str(column)) for column in frame.columns})


def _num(value: Any) -> Any:
    return round(float(value), 4) if isinstance(value, int | float) else value


def _flags(flags: list[str] | None) -> str:
    if not flags:
        return "无明显风险标记"
    return "；".join(FLAG_LABELS.get(flag, flag) for flag in flags)


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


def _drop_empty_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    keep = []
    for column in frame.columns:
        values = frame[column]
        if not values.map(_is_empty_value).all():
            keep.append(column)
    return frame[keep]


def _deduplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not frame.columns.duplicated().any():
        return frame
    counts: dict[str, int] = {}
    columns = []
    for column in frame.columns:
        name = str(column)
        counts[name] = counts.get(name, 0) + 1
        columns.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    result = frame.copy()
    result.columns = columns
    return result


def _candidate_table(method: str, items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        row = {"排序": item.get("rank"), "预测产量": _num(item.get("predicted_yield"))}
        if item.get("soft_filter_status"):
            row["软过滤状态"] = item.get("soft_filter_status")
        if item.get("history_range_violations") is not None:
            row["历史范围超限数"] = item.get("history_range_violations")
        if method == "standard_bo_qnei":
            row.update(
                {
                    "GP 后验标准差": _num(item.get("model_uncertainty")),
                    "qNEI 批量推荐得分": _num(item.get("acquisition_score")),
                }
            )
        for key, value in item.get("params", {}).items():
            row[_name(key)] = _num(value)
        rows.append(row)
    return _deduplicate_columns(_drop_empty_columns(pd.DataFrame(rows)))


def _overview(df: pd.DataFrame) -> None:
    train_df = _training_data(df)
    cols = st.columns(4)
    cols[0].metric("总 run 数", len(df))
    cols[1].metric("可训练 run 数", len(train_df))
    cols[2].metric("排除 run 数", len(df) - len(train_df))
    cols[3].metric("目标字段", _display_name(TARGET_COL))

    if not train_df.empty:
        y = pd.to_numeric(train_df[TARGET_COL], errors="coerce").dropna()
        stats = st.columns(4)
        stats[0].metric("历史最低产量", f"{y.min():.3g}")
        stats[1].metric("历史中位产量", f"{y.median():.3g}")
        stats[2].metric("历史最高产量", f"{y.max():.3g}")
        stats[3].metric("历史平均产量", f"{y.mean():.3g}")

    with st.expander("训练数据筛选说明", expanded=False):
        st.write("缺少产量、污染、异常或失败备注的 run 会保留在数据表中用于审计，但不会参与模型训练。")
        if "exclusion_reason" in df.columns:
            counts = df["exclusion_reason"].fillna("").replace("", "可训练").value_counts().reset_index()
            counts.columns = ["原因", "数量"]
            st.dataframe(counts, width="stretch", hide_index=True)

    with st.expander("数据预览", expanded=False):
        st.dataframe(_display_dataframe(df.head(30), keep_english=True), width="stretch")

    with st.expander("字段中英对照", expanded=False):
        dictionary_path = PROJECT_ROOT / "summary" / "supporting_reports" / "field_dictionary.csv"
        if dictionary_path.exists():
            dictionary = pd.read_csv(dictionary_path)
            run_dictionary = dictionary.loc[dictionary["table"] == "run_level_modeling_dataset"].copy()
            run_dictionary = run_dictionary.fillna("")
            run_dictionary = run_dictionary.rename(
                columns={
                    "field": "英文字段",
                    "zh_name": "中文名称",
                    "unit": "单位",
                    "role": "角色",
                    "description": "说明",
                }
            )
            st.dataframe(run_dictionary[["英文字段", "中文名称", "单位", "角色", "说明"]], width="stretch", hide_index=True)
        else:
            st.info("尚未生成字段字典，可运行 `python data/scripts/generate_field_dictionary.py`。")


def _nearest_history(df: pd.DataFrame, params: dict[str, float], top_n: int = 3) -> pd.DataFrame:
    features = [name for name in params if name in df.columns]
    history = _training_data(df).dropna(subset=features)
    if history.empty or not features:
        return pd.DataFrame()
    means = history[features].mean()
    stds = history[features].std(ddof=0).replace(0, 1.0)
    candidate = pd.Series(params)[features]
    distances = (((history[features] - means) / stds - ((candidate - means) / stds)) ** 2).sum(axis=1) ** 0.5
    nearest = history.assign(相似距离=distances).sort_values("相似距离").head(top_n)
    columns = [col for col in ["fermenter_run_id", "experiment_date", TARGET_COL, "相似距离", *features] if col in nearest.columns]
    renamed = nearest[columns].rename(columns={col: _name(col) for col in columns})
    return _deduplicate_columns(renamed)


def _method_block(method: str, items: list[dict[str, Any]]) -> None:
    st.subheader(METHOD_LABELS.get(method, method))
    st.caption(METHOD_EXPLANATIONS.get(method, ""))
    if items:
        st.dataframe(_candidate_table(method, items), width="stretch", hide_index=True)
    else:
        st.info("暂无结果。")


def _standard_bo_summary(top: dict[str, Any]) -> None:
    rows = [
        {"项目": "预测产量", "数值": _num(top.get("predicted_yield")), "说明": "标准 GP 模型对候选点的产量均值预测。"},
        {"项目": "GP 后验标准差", "数值": _num(top.get("model_uncertainty")), "说明": "标准 GP 对该候选点预测不确定性的估计。"},
        {"项目": "qNEI 批量推荐得分", "数值": _num(top.get("acquisition_score")), "说明": "qNEI 联合优化后候选点的模型均值展示值，批内点由联合采集函数生成。"},
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _standard_gp_slice_frame(
    df: pd.DataFrame,
    params: dict[str, float],
    search_space: dict[str, tuple[float, float]] | dict[str, list[float]],
    feature: str,
    fitted_gp: Any,
    feature_cols: list[str],
    n_points: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import numpy as np
    except Exception as exc:
        raise ImportError("绘制标准 GP 图需要 numpy。") from exc

    train = _training_data(df)[[*feature_cols, TARGET_COL]].dropna()
    if len(train) < 5:
        raise ValueError("至少需要 5 条完整训练数据才能绘制标准 GP 图。")

    # Fix other features at historical mean — makes the slice deterministic
    # (independent of which recommendation or seed was used).
    hist_means = {f: float(train[f].mean()) for f in feature_cols if f in train.columns}
    anchor = {**hist_means, **{k: v for k, v in params.items() if k not in feature_cols}}

    # X-axis spans the historical observed range, not the full search-space bounds.
    # If the recommended value falls outside, extend the range to include it.
    hist_min = float(train[feature].min())
    hist_max = float(train[feature].max())
    rec_val = float(params.get(feature, hist_means.get(feature, hist_min)))
    x_min = min(hist_min, rec_val)
    x_max = max(hist_max, rec_val)

    grid = pd.DataFrame([anchor] * n_points)
    grid[feature] = np.linspace(x_min, x_max, n_points)
    mean, std = fitted_gp.predict(grid[feature_cols], return_std=True)
    curve = pd.DataFrame(
        {
            feature: grid[feature],
            "posterior_mean": mean,
            "posterior_std": std,
            "lower_95": mean - 1.96 * std,
            "upper_95": mean + 1.96 * std,
        }
    )
    return curve, train


def _standard_gp_plot(
    df: pd.DataFrame,
    items: list[dict[str, Any]] | dict[str, Any],
    search_space: dict[str, Any],
    fitted_gp: Any,
    feature_cols: list[str],
) -> None:
    if fitted_gp is None:
        st.info("标准 GP 模型不可用，无法绘制后验切片图。")
        return

    # Accept either a single recommendation dict or a list
    if isinstance(items, dict):
        items = [items]
    if not items:
        st.info("当前推荐缺少可绘图参数。")
        return

    # Build rank labels for the switcher
    rank_options = {
        f"推荐 #{item.get('rank', i + 1)}（预测产量 {_num(item.get('predicted_yield', ''))} g/L）": item
        for i, item in enumerate(items)
    }

    st.markdown("### 标准 GP 后验切片图")
    st.caption(
        "其它参数固定在历史均值，只沿一个参数变化，横轴为历史实测范围。"
        "红虚线为所选推荐点的参数值。此图曲线与随机种子和采集函数无关，反映模型学到的稳定规律。"
    )

    col_rec, col_feat = st.columns([2, 3])
    with col_rec:
        selected_label = st.radio(
            "查看推荐",
            list(rank_options.keys()),
            index=0,
            key="standard_gp_slice_rank",
        )
    active_item = rank_options[selected_label]
    params = active_item.get("params", {})

    features = [f for f in feature_cols if f in search_space and f in params and f in df.columns]
    if not features:
        st.info("当前推荐缺少可绘图参数。")
        return

    with col_feat:
        feature = st.selectbox(
            "选择横轴参数",
            features,
            format_func=_display_name,
            key="standard_gp_slice_feature",
        )

    try:
        curve, train = _standard_gp_slice_frame(df, params, search_space, feature, fitted_gp, feature_cols)
    except Exception as exc:
        st.warning(f"无法绘制标准 GP 图：{exc}")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve[feature], curve["posterior_mean"], color="#2563eb", label="GP posterior mean")
    ax.fill_between(
        curve[feature],
        curve["lower_95"],
        curve["upper_95"],
        color="#93c5fd",
        alpha=0.35,
        label="95% posterior interval",
    )
    ax.scatter(train[feature], train[TARGET_COL], color="#111827", s=24, alpha=0.65, label="history")
    rec_val = float(params[feature])
    ax.axvline(rec_val, color="#dc2626", linestyle="--", linewidth=1.5, label=f"recommended value ({rec_val:.2f})")
    ax.set_xlabel(_display_name(feature))
    ax.set_ylabel(_display_name(TARGET_COL))
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)


def _pdp_curve(fitted_gp: Any, train: pd.DataFrame, feature_cols: list[str], feature: str, n_points: int = 50) -> pd.DataFrame:
    """Calculate a simple 1D PDP curve for explanation tables."""
    try:
        import numpy as np
    except Exception as exc:
        raise ImportError("计算偏依赖摘要需要 numpy。") from exc

    low = float(train[feature].quantile(0.05))
    high = float(train[feature].quantile(0.95))
    grid = np.linspace(low, high, n_points)
    values = []
    for value in grid:
        batch = train.copy()
        batch[feature] = value
        predictions = fitted_gp.predict(batch[feature_cols])
        values.append(float(np.mean(predictions)))
    return pd.DataFrame({feature: grid, "mean_prediction": values})


def _pdp_direction(curve: pd.DataFrame, feature: str) -> tuple[str, str]:
    """Convert a PDP curve into a plain-language trend label."""
    values = curve["mean_prediction"]
    x_values = curve[feature]
    effect = float(values.max() - values.min())
    if effect < 1.0:
        return "影响很弱", "当前数据下模型几乎不随该参数变化"

    start = float(values.iloc[0])
    end = float(values.iloc[-1])
    if end - start > effect * 0.35:
        return "偏高更好", "参数升高时，模型平均预测产量上升"
    if start - end > effect * 0.35:
        return "偏低更好", "参数升高时，模型平均预测产量下降"

    best_position = (float(x_values.iloc[values.idxmax()]) - float(x_values.min())) / max(float(x_values.max() - x_values.min()), 1e-9)
    if best_position < 0.35:
        return "低区间较好", "模型偏好的区域靠近历史低值端"
    if best_position > 0.65:
        return "高区间较好", "模型偏好的区域靠近历史高值端"
    return "中间区间较好", "模型偏好的区域靠近历史中间值"


def _pdp_summary(fitted_gp: Any, train: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Build a concise PDP interpretation table for non-ML readers."""
    rows = []
    for feature in feature_cols:
        curve = _pdp_curve(fitted_gp, train, feature_cols, feature)
        values = curve["mean_prediction"]
        x_values = curve[feature]
        best_index = int(values.idxmax())
        direction, explanation = _pdp_direction(curve, feature)
        rows.append(
            {
                "参数": _display_name(feature),
                "模型倾向": direction,
                "模型偏好值": _num(float(x_values.iloc[best_index])),
                "影响幅度(g/L)": _num(float(values.max() - values.min())),
                "直观解释": explanation,
            }
        )
    return pd.DataFrame(rows).sort_values("影响幅度(g/L)", ascending=False)


def _gp_pdp(df: pd.DataFrame, fitted_gp: Any, feature_cols: list[str]) -> None:
    """使用偏依赖图展示 GP 各特征的平均边际效应。"""
    import math as _math
    import numpy as np

    if fitted_gp is None:
        st.info("标准 GP 模型不可用，无法绘制偏依赖图。")
        return
    if not feature_cols:
        st.info("无可用特征列，无法绘制偏依赖图。")
        return

    train = _training_data(df)[feature_cols].dropna()
    if len(train) < 5:
        st.info("训练数据不足，无法绘制偏依赖图。")
        return

    with st.expander("这张图怎么读", expanded=True):
        st.write(
            "偏依赖图不是单次实验的真实曲线，而是模型在历史参数分布上做平均后的趋势。"
            "曲线向上表示该参数增大时，模型平均预测产量更高；曲线向下表示更低。"
            "黑色短线表示历史实验实际覆盖的位置，远离短线密集区域的结论要谨慎。"
        )
        st.write("下面的表把每条曲线翻译成更直接的工艺判断，按模型认为影响幅度从大到小排序。")
        try:
            st.dataframe(_pdp_summary(fitted_gp, train, feature_cols), width="stretch", hide_index=True)
        except Exception as exc:
            st.warning(f"偏依赖摘要计算失败：{exc}")

    st.markdown("### 各特征平均边际效应（1D 偏依赖图）")
    st.caption(
        "对每个特征值，将其余特征在训练数据的实际分布上取平均，"
        "反映该特征的边际效应。与切片图不同，平均过程自然携带特征间的协变关系。"
    )

    n_features = len(feature_cols)
    n_cols = min(3, n_features)
    n_rows = _math.ceil(n_features / n_cols)
    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten()

    for i, feature in enumerate(feature_cols):
        curve = _pdp_curve(fitted_gp, train, feature_cols, feature, n_points=50)
        ax = axes_flat[i]
        ax.plot(curve[feature], curve["mean_prediction"], color="#2563eb", linewidth=1.8)
        y_min = float(curve["mean_prediction"].min())
        y_max = float(curve["mean_prediction"].max())
        margin = (y_max - y_min) * 0.15 if y_max > y_min else 1.0
        rug_y = y_min - margin * 0.6
        ax.plot(train[feature].values, np.full(len(train), rug_y),
                "|", color="black", markersize=8, alpha=0.6)
        ax.set_ylim(rug_y - margin * 0.2, y_max + margin * 0.2)
        ax.set_title(_display_name(feature), fontsize=10)
        ax.set_xlabel(_display_name(feature))
        ax.set_ylabel("平均预测产量（g/L）")
        ax.grid(alpha=0.2)

    for ax in axes_flat[n_features:]:
        ax.set_visible(False)

    fig1.suptitle("GP 偏依赖图（平均边际效应）", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig1, clear_figure=True)

    key_a = "temperature_shift_time_h"
    key_b = "lactose_total_ml"
    if key_a in feature_cols and key_b in feature_cols:
        st.markdown("### 升温时机 × 乳糖总量 联动效应（2D 偏依赖图）")
        st.caption(
            "两个相关性最强的特征（EDA Spearman r=0.91）的联合偏依赖图。"
            "颜色越深表示该参数组合下 GP 预测产量越高。"
        )
        a_low = float(train[key_a].quantile(0.05))
        a_high = float(train[key_a].quantile(0.95))
        b_low = float(train[key_b].quantile(0.05))
        b_high = float(train[key_b].quantile(0.95))
        grid_a = np.linspace(a_low, a_high, 20)
        grid_b = np.linspace(b_low, b_high, 20)
        Z = np.zeros((len(grid_b), len(grid_a)))
        for i, val_a in enumerate(grid_a):
            for j, val_b in enumerate(grid_b):
                batch = train.copy()
                batch[key_a] = val_a
                batch[key_b] = val_b
                Z[j, i] = float(np.mean(fitted_gp.predict(batch[feature_cols])))

        in_window = (
            train[key_a].between(a_low, a_high)
            & train[key_b].between(b_low, b_high)
        )
        n_outside = int((~in_window).sum())
        if n_outside:
            st.caption(
                f"该 2D 图只显示 5%-95% 历史分位窗口，避免把少数边缘点拉大坐标轴并造成外推误读；"
                f"{n_outside} 个窗口外历史点未参与坐标轴缩放。"
            )

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        filled = ax2.contourf(grid_a, grid_b, Z, levels=10, cmap="viridis")
        plt.colorbar(filled, ax=ax2, label="预测产量（g/L）")
        contour_lines = ax2.contour(grid_a, grid_b, Z, levels=10,
                                    colors="black", linewidths=0.5, alpha=0.6)
        ax2.clabel(contour_lines, fmt="%.2f", fontsize=8)
        ax2.scatter(train.loc[in_window, key_a], train.loc[in_window, key_b],
                    marker="|", color="black", alpha=0.5, s=60, zorder=5)
        ax2.set_xlim(a_low, a_high)
        ax2.set_ylim(b_low, b_high)
        ax2.set_xlabel(_display_name(key_a))
        ax2.set_ylabel(_display_name(key_b))
        ax2.set_title("GP预测产量：升温时机 × 乳糖总量")
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)


def _loocv_scatter(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """LOO-CV 预测 vs 实测散点图，评估 GP 模型的泛化能力。"""
    try:
        import warnings
        import numpy as np
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        st.warning(f"LOO-CV 需要 scikit-learn：{exc}")
        return

    if not feature_cols:
        st.info("无可用特征列。")
        return

    feature_df = add_model_derived_features(_training_data(df))
    train = feature_df[feature_cols + [TARGET_COL]].dropna()
    if len(train) < 8:
        st.info("训练数据不足（至少需要 8 条），无法计算 LOO-CV。")
        return

    x = train[feature_cols]
    y = train[TARGET_COL].astype(float).values
    n_features = len(feature_cols)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(
            nu=2.5,
            length_scale=[1.0] * n_features,
            length_scale_bounds=[(1e-2, 1e2)] * n_features,
        )
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    )
    pipeline = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42),
    )

    with st.spinner("正在计算 LOO-CV（约需 5-15 秒）..."):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            y_pred = cross_val_predict(pipeline, x, y, cv=LeaveOneOut())

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y, y_pred, color="#2563eb", alpha=0.7, s=40, label="历史 run")
    lims = [float(min(np.min(y), np.min(y_pred)) - 5), float(max(np.max(y), np.max(y_pred)) + 5)]
    ax.plot(lims, lims, color="#dc2626", linestyle="--", linewidth=1.2, label="完美预测线")
    ax.set_xlabel("实测产量 (g/L)")
    ax.set_ylabel("LOO-CV 预测产量 (g/L)")
    ax.set_title(f"GP 泛化能力：LOO-CV  |  R²={r2:.3f}  |  MAE={mae:.2f} g/L")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    if r2 >= 0.6:
        st.success(f"R²={r2:.3f}，GP 对留出实验有较好的预测能力，推荐方向可参考。")
    elif r2 >= 0.3:
        st.warning(f"R²={r2:.3f}，GP 有一定预测能力，但不确定性较高，推荐应结合人工判断。")
    else:
        st.error(f"R²={r2:.3f}，GP 泛化能力较弱，推荐参数仅供参考，建议扩充数据后再优化。")

    st.caption(
        f"每个点代表一条历史 run：用其余 {len(train) - 1} 条数据训练 GP，预测该 run 的产量。"
        f"MAE={mae:.2f} g/L 表示平均预测误差。"
    )


def _nearest_history_validation(
    df: pd.DataFrame,
    params: dict[str, float],
    feature_cols: list[str],
    top_n: int = 5,
) -> None:
    """展示与推荐参数最相似的历史实验及其实际产量。"""
    try:
        import numpy as np
    except ImportError as exc:
        st.warning(f"最近邻验证需要 numpy：{exc}")
        return

    features = [feature for feature in feature_cols if feature in df.columns and feature in params]
    history = _training_data(df).dropna(subset=features + [TARGET_COL])
    if history.empty or not features:
        st.info("无足够历史数据进行最近邻对比。")
        return

    means = history[features].mean()
    stds = history[features].std(ddof=0).replace(0, 1.0)
    candidate = pd.Series(params)[features]
    norm_candidate = (candidate - means) / stds
    norm_history = (history[features] - means) / stds
    distances = ((norm_history - norm_candidate) ** 2).sum(axis=1) ** 0.5
    nearest = history.assign(_dist=distances).sort_values("_dist").head(top_n)

    actual_yields = nearest[TARGET_COL].values
    recommended_yield = params.get("predicted_yield")

    fig, ax = plt.subplots(figsize=(7, 4))
    run_labels = [
        str(run_id)[:12] if not pd.isna(run_id) else f"Run {index + 1}"
        for index, run_id in enumerate(nearest.get("fermenter_run_id", nearest.index))
    ]
    ax.bar(run_labels, actual_yields, color="#6b7280", alpha=0.8, label="历史实测产量")
    if recommended_yield is not None:
        ax.axhline(
            float(recommended_yield),
            color="#2563eb",
            linestyle="--",
            linewidth=1.5,
            label=f"推荐预测产量 ({float(recommended_yield):.1f} g/L)",
        )
    ax.set_ylabel("产量 (g/L)")
    ax.set_title(f"与推荐参数最相似的 {top_n} 条历史实验")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    mean_nearby = float(np.mean(actual_yields))
    max_nearby = float(np.max(actual_yields))
    dist_nearest = float(nearest["_dist"].iloc[0])

    if dist_nearest > 2.0:
        st.warning(
            f"最近邻距离={dist_nearest:.2f}（标准化），推荐参数远离历史实验区域，属于外推，需谨慎。"
        )
    else:
        st.success(
            f"最近邻距离={dist_nearest:.2f}，推荐参数有历史数据支撑。"
            f"相似实验平均产量 {mean_nearby:.1f} g/L，最高 {max_nearby:.1f} g/L。"
        )

    display_cols = [feature for feature in features if feature in nearest.columns] + [TARGET_COL]
    table = nearest[display_cols].copy().rename(columns={column: _display_name(column) for column in display_cols})
    table.insert(0, "相似距离", nearest["_dist"].round(3).values)
    if "fermenter_run_id" in nearest.columns:
        table.insert(0, "Run ID", nearest["fermenter_run_id"].values)
    st.dataframe(table, width="stretch", hide_index=True)


def _metric_value(metrics: dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    return _num(value) if isinstance(value, int | float) else value


def _strategy_quality_block(comparison: dict[str, Any], df: pd.DataFrame, selected: list[dict[str, Any]], feature_cols: list[str]) -> None:
    quality = comparison.get("strategy_quality") or {}
    if not quality:
        search_bounds = comparison.get("search_space") or {}
        features = feature_cols or list(search_bounds)
        if selected and search_bounds and features:
            try:
                space = SearchSpace(bounds={name: tuple(bounds) for name, bounds in search_bounds.items()})
                quality = evaluate_recommendation_quality(
                    selected,
                    _training_data(df),
                    space,
                    feature_cols=features,
                    target_col=TARGET_COL,
                )
                comparison["strategy_quality"] = quality
                st.caption("当前结果来自旧缓存，已基于现有推荐现场补算策略质量指标。")
            except Exception as exc:
                st.info(f"当前结果没有推荐策略质量指标，现场补算也失败：{exc}")
                return
        else:
            st.info("当前结果没有推荐策略质量指标。请重新运行推荐生成完整诊断。")
            return

    st.markdown("### Batch 多样性")
    diversity = quality.get("batch_diversity", {})
    diversity_cols = st.columns(4)
    diversity_cols[0].metric("最小两两距离", _metric_value(diversity, "min_pairwise_distance"))
    diversity_cols[1].metric("平均两两距离", _metric_value(diversity, "mean_pairwise_distance"))
    diversity_cols[2].metric("簇数量", diversity.get("cluster_count_threshold_0_10"))
    diversity_cols[3].metric("平均特征覆盖", _metric_value(diversity, "mean_feature_range_coverage"))

    coverage = diversity.get("feature_range_coverage") or {}
    if coverage:
        coverage_rows = [
            {"参数": _display_name(feature), "字段名": feature, "覆盖比例": _num(value)}
            for feature, value in coverage.items()
        ]
        st.dataframe(pd.DataFrame(coverage_rows), width="stretch", hide_index=True)

    st.markdown("### 历史支撑与边界风险")
    support = quality.get("history_support", {})
    boundary = quality.get("boundary_risk", {})
    risk_cols = st.columns(4)
    risk_cols[0].metric("平均最近邻距离", _metric_value(support, "mean_nearest_history_distance"))
    risk_cols[1].metric("最远最近邻距离", _metric_value(support, "max_nearest_history_distance"))
    risk_cols[2].metric("平均边界风险", _metric_value(boundary, "mean_boundary_risk"))
    risk_cols[3].metric("高边界风险数量", boundary.get("n_near_boundary_gt_0_8"))

    per_items = quality.get("per_recommendation") or []
    if per_items:
        rows = []
        for item in per_items:
            rows.append(
                {
                    "排序": item.get("rank"),
                    "类型": item.get("recommendation_type", "—"),
                    "预测产量": _num(item.get("predicted_yield")),
                    "GP 后验标准差": _num(item.get("model_uncertainty")),
                    "最近邻距离": _num(item.get("nearest_history_distance")),
                    "最近邻 Run": item.get("nearest_run_id", "—"),
                    "最近邻产量": _num(item.get("nearest_run_yield")),
                    "边界风险": _num(item.get("boundary_risk")),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("### 推荐参数的历史近邻")
    st.caption("为每个推荐组合分别展示最相似的历史实验实际产量，用于判断整批推荐是否落在有数据支撑的区域。")
    if selected:
        for item in selected:
            rank = item.get("rank", "?")
            predicted = item.get("predicted_yield")
            title = f"推荐 #{rank}"
            if predicted is not None:
                title = f"{title}｜预测产量 {_num(predicted)} g/L"
            with st.expander(title, expanded=rank == 1):
                selected_params = {**item.get("params", {}), "predicted_yield": predicted}
                _nearest_history_validation(df, selected_params, feature_cols)
    else:
        st.info("暂无推荐点，无法做历史近邻对比。")


def _metric_explanations() -> None:
    st.markdown("### 主推荐：标准 GP-BO（qNEI）")
    standard_rows = [
        {"指标": "预测产量", "如何产生": "GP 直接拟合历史产量后，对候选点输出 posterior mean。", "如何解读": "模型预测值，不是实测值。"},
        {"指标": "GP 后验标准差", "如何产生": "GP 对候选点预测分布的 posterior std。", "如何解读": "越高表示模型在该区域越不确定。"},
        {"指标": "qNEI 批量推荐", "如何产生": "BoTorch qNoisyExpectedImprovement 联合优化整批候选点。", "如何解读": "批内候选会考虑边际收益递减，减少 top-k 聚集。"},
        {"指标": "观测噪声", "如何产生": "qNEI 基于 noisy baseline 建模已观测数据。", "如何解读": "相比直接使用历史最大值的 EI，更不容易被噪声高点牵引。"},
    ]
    st.dataframe(pd.DataFrame(standard_rows), width="stretch", hide_index=True)


def _ensure_pichia_data_area() -> None:
    for directory in [PICHIA_UPLOAD_DIR, PICHIA_FINAL_DIR, PICHIA_TEMPLATE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    if not PICHIA_TEMPLATE_PATH.exists():
        empty_pichia_template().to_csv(PICHIA_TEMPLATE_PATH, index=False, encoding="utf-8-sig")


def _load_pichia_default_dataset() -> pd.DataFrame:
    _ensure_pichia_data_area()
    if not PICHIA_DEFAULT_DATASET_PATH.exists():
        return empty_pichia_template()
    return pd.read_csv(PICHIA_DEFAULT_DATASET_PATH)


def _load_pichia_uploaded_dataset(uploaded_file: Any) -> tuple[pd.DataFrame, Path]:
    _ensure_pichia_data_area()
    payload = uploaded_file.getvalue()
    safe_name = Path(uploaded_file.name).name or "pichia_uploaded.csv"
    target_path = PICHIA_UPLOAD_DIR / safe_name
    target_path.write_bytes(payload)
    return pd.read_csv(BytesIO(payload)), target_path


def _pichia_strain_options(df: pd.DataFrame) -> list[str]:
    if "strain_id" not in df.columns:
        return []
    values = df["strain_id"].dropna().astype(str).str.strip()
    return sorted(value for value in values.unique().tolist() if value)


def _pichia_overview(df: pd.DataFrame, data_path: Path | None) -> None:
    st.markdown("### 毕赤酵母历史数据")
    if data_path:
        st.caption(f"当前数据文件：{data_path}")
    st.caption(f"Pichia 上传目录：{PICHIA_UPLOAD_DIR}")

    strains = _pichia_strain_options(df)
    cols = st.columns(4)
    cols[0].metric("历史 run 数", len(df))
    cols[1].metric("菌种数", len(strains))
    cols[2].metric("有产量记录 run", int(pd.to_numeric(df.get(PICHIA_TARGET_COL, pd.Series(dtype=float)), errors="coerce").notna().sum()))
    cols[3].metric("推荐方式", "序贯 DOE + 基准点")

    missing = [name for name in pichia_template_columns() if name not in df.columns]
    if missing and not df.empty:
        st.warning(f"当前数据缺少模板字段：{', '.join(missing)}")
    if df.empty:
        st.info("当前 Pichia 历史数据为空。可上传 CSV，或使用手动基准点生成 2-4 罐早期探索方案。")
        return

    preview_cols = [column for column in pichia_template_columns() if column in df.columns]
    with st.expander("Pichia 数据预览", expanded=False):
        st.dataframe(df[preview_cols].head(30), width="stretch", hide_index=True)


def _pichia_manual_baseline_inputs() -> dict[str, float]:
    manual_baseline: dict[str, float] = {}
    with st.expander("手动基准点", expanded=True):
        columns = st.columns(2)
        for index, (name, spec) in enumerate(PICHIA_PARAMETER_SPECS.items()):
            with columns[index % 2]:
                midpoint = _num((spec.lower + spec.upper) / 2.0)
                number_format = "%.2f" if spec.step < 0.1 else "%.1f" if spec.step < 1 else "%.0f"
                manual_baseline[name] = float(
                    st.number_input(
                        f"{spec.label}{f' ({spec.unit})' if spec.unit else ''}",
                        min_value=float(spec.lower),
                        max_value=float(spec.upper),
                        value=float(midpoint),
                        step=float(spec.step),
                        format=number_format,
                        key=f"pichia_manual_{name}",
                    )
                )
    return manual_baseline


def _pichia_recommendation_frame(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in result.get("recommendations", []):
        row = {"排序": item.get("rank")}
        for name, spec in PICHIA_PARAMETER_SPECS.items():
            row[f"{spec.label}{f' ({spec.unit})' if spec.unit else ''}"] = _num(item.get("params", {}).get(name))
        rows.append(row)
    return _drop_empty_columns(pd.DataFrame(rows))


def _pichia_baseline_frame(result: dict[str, Any]) -> pd.DataFrame:
    baseline = result.get("baseline") or {}
    rows = []
    for name, spec in PICHIA_PARAMETER_SPECS.items():
        rows.append(
            {
                "参数": spec.label,
                "字段": name,
                "基准值": _num(baseline.get(name)),
                "硬约束下限": spec.lower,
                "硬约束上限": spec.upper,
                "单位": spec.unit,
            }
        )
    return pd.DataFrame(rows)


def _pichia_exploration_metrics(result: dict[str, Any]) -> None:
    metrics = result.get("exploration_metrics") or {}
    st.markdown("### 探索反馈指标")
    overall = pd.DataFrame(
        [
            {"指标": "平均每个推荐改变变量数", "数值": _num(metrics.get("average_changed_variables"))},
            {"指标": "单变量性", "数值": _num(metrics.get("univariate_score"))},
            {"指标": "联合探索性", "数值": _num(metrics.get("joint_exploration_score"))},
            {"指标": "批内平均距离", "数值": _num(metrics.get("mean_pairwise_distance"))},
            {"指标": "批内多样性", "数值": metrics.get("batch_diversity")},
        ]
    )
    st.dataframe(overall, width="stretch", hide_index=True)

    variable_rows = []
    for row in metrics.get("variables", []):
        rec_min = row.get("min")
        rec_max = row.get("max")
        variable_rows.append(
            {
                "变量": row.get("label"),
                "变化次数": f"{row.get('changed_count')}/{row.get('n_recommendations')}",
                "推荐变化范围": "未变化" if rec_min is None else f"{_num(rec_min)} - {_num(rec_max)}",
                "允许范围": f"{_num(row.get('allowed_lower'))} - {_num(row.get('allowed_upper'))}",
                "覆盖比例": _num(row.get("coverage")),
            }
        )
    st.dataframe(pd.DataFrame(variable_rows), width="stretch", hide_index=True)


def _pichia_information_gain_block(result: dict[str, Any]) -> bool:
    info = result.get("information_gain") or {}
    if not info:
        return False
    st.markdown("### DOE 信息增益")
    rows = [
        {"项目": "设计类型", "说明": info.get("design_type")},
        {"项目": "主动变量", "说明": " × ".join(info.get("active_variable_labels", []))},
        {"项目": "可估计主效应", "说明": "是" if info.get("can_estimate_main_effects") else "否"},
        {"项目": "可估计交互效应", "说明": "是" if info.get("can_estimate_interaction") else "否"},
        {"项目": "可估计项", "说明": "；".join(info.get("estimable_terms", []))},
        {"项目": "相比单变量", "说明": info.get("vs_single_variable")},
        {"项目": "下一轮反馈规则", "说明": info.get("next_round_rule")},
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    return True


def _pichia_feedback_input_frame(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    active_variables = (result.get("doe") or {}).get("active_variables") or []
    for item in result.get("recommendations", []):
        row = {"排序": item.get("rank"), "实测产量": None}
        for variable in active_variables:
            spec = PICHIA_PARAMETER_SPECS[variable]
            row[spec.label] = _num(item.get("params", {}).get(variable))
        rows.append(row)
    columns = ["排序", *[PICHIA_PARAMETER_SPECS[name].label for name in active_variables], "实测产量"]
    return pd.DataFrame(rows, columns=columns)


def _pichia_suggested_baseline_frame(feedback: dict[str, Any]) -> pd.DataFrame:
    baseline = feedback.get("suggested_baseline") or {}
    rows = []
    for name, spec in PICHIA_PARAMETER_SPECS.items():
        rows.append(
            {
                "参数": spec.label,
                "建议基准值": _num(baseline.get(name)),
                "单位": spec.unit,
            }
        )
    return pd.DataFrame(rows)


def _pichia_doe_feedback_block(result: dict[str, Any]) -> None:
    if result.get("mode") != "pichia_sequential_doe":
        return

    st.markdown("### 实验结果反馈")
    st.caption("实验完成后回填每罐实测产量，用于计算主效应、交互效应和下一轮 DOE 建议。")
    threshold = st.number_input(
        "有意义产量差阈值",
        min_value=0.0,
        max_value=1000.0,
        value=0.05,
        step=0.01,
        help="低于该阈值的产量差视为无明显信号，避免把实验噪声解释成方向。",
        key="pichia_feedback_threshold",
    )

    feedback_frame = _pichia_feedback_input_frame(result)
    if feedback_frame.empty:
        st.info("当前结果不是可反馈的 DOE 设计。")
        return
    disabled_columns = [column for column in feedback_frame.columns if column != "实测产量"]
    edited = st.data_editor(
        feedback_frame,
        width="stretch",
        hide_index=True,
        disabled=disabled_columns,
        key="pichia_feedback_yields",
    )
    observed_yields = {
        int(row["排序"]): row.get("实测产量")
        for _, row in edited.iterrows()
    }
    feedback = analyze_pichia_doe_feedback(
        result,
        observed_yields,
        practical_threshold=float(threshold),
    )

    if feedback.get("status") == "insufficient":
        st.info(feedback.get("message", "请先回填实测产量。"))
        return

    cols = st.columns(3)
    cols[0].metric("已回填罐数", f"{feedback.get('completed_count')}/{feedback.get('total_count')}")
    cols[1].metric("最高产排序", feedback.get("best_rank"))
    cols[2].metric("最高实测产量", _num(feedback.get("best_yield")))

    effect_rows = []
    for item in feedback.get("main_effects", []):
        effect_rows.append(
            {
                "变量": item.get("label"),
                "低水平均值": _num(item.get("low_mean")),
                "高水平均值": _num(item.get("high_mean")),
                "主效应": _num(item.get("effect")),
                "判断": item.get("direction"),
                "样本数": f"{item.get('n_low')}/{item.get('n_high')}",
            }
        )
    st.dataframe(pd.DataFrame(effect_rows), width="stretch", hide_index=True)

    interaction = feedback.get("interaction_effect") or {}
    if interaction.get("can_estimate"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "项目": "交互效应",
                        "效应值": _num(interaction.get("effect")),
                        "判断": interaction.get("direction"),
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("当前回填不足 4 个 DOE 组合，暂不能估计交互效应。")

    st.success(feedback.get("suggestion"))
    with st.expander("建议下一轮基准点", expanded=False):
        st.dataframe(_pichia_suggested_baseline_frame(feedback), width="stretch", hide_index=True)


def _pichia_result_block(result: dict[str, Any]) -> None:
    baseline_meta = result.get("baseline_meta") or {}
    if baseline_meta.get("warning"):
        st.warning(str(baseline_meta["warning"]))

    st.markdown("### 基准点")
    cols = st.columns(4)
    cols[0].metric("菌种 ID", result.get("strain_id") or "未填写")
    cols[1].metric("亲本菌种", result.get("parent_strain_id") or "无")
    cols[2].metric("基准来源", baseline_meta.get("source", ""))
    cols[3].metric("每个推荐改变变量数", result.get("changed_variable_count"))
    st.dataframe(_pichia_baseline_frame(result), width="stretch", hide_index=True)

    st.markdown("### 推荐组合")
    recommendations = result.get("recommendations", [])
    if not recommendations:
        st.info("暂无 Pichia 推荐结果。")
        return
    st.dataframe(_pichia_recommendation_frame(result), width="stretch", hide_index=True)

    if not _pichia_information_gain_block(result):
        _pichia_exploration_metrics(result)
    _pichia_doe_feedback_block(result)


def _pichia_design_page() -> None:
    _ensure_pichia_data_area()
    st.caption("当前模式：毕赤酵母序贯 DOE。此模式不使用大肠杆菌 BO 模型，围绕基准点设计可解释的下一批实验。")

    with st.sidebar:
        st.header("毕赤酵母数据入口")
        source = st.radio(
            "选择数据来源",
            ["使用 data/pichia/final/pichia_run_level_dataset.csv", "上传 Pichia run-level CSV"],
            key="pichia_data_source",
        )
        uploaded_file = None
        if source == "上传 Pichia run-level CSV":
            uploaded_file = st.file_uploader("上传 Pichia run-level CSV", type=["csv"], key="pichia_uploaded_csv")
        template_csv = empty_pichia_template().to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "下载 Pichia 数据模板",
            data=template_csv,
            file_name="pichia_run_level_template.csv",
            mime="text/csv",
            key="download_pichia_template",
        )

    try:
        if source == "上传 Pichia run-level CSV":
            if uploaded_file is None:
                df = empty_pichia_template()
                data_path = None
            else:
                df, data_path = _load_pichia_uploaded_dataset(uploaded_file)
        else:
            df = _load_pichia_default_dataset()
            data_path = PICHIA_DEFAULT_DATASET_PATH if PICHIA_DEFAULT_DATASET_PATH.exists() else None
    except Exception as exc:
        st.error(f"Pichia 数据加载失败：{exc}")
        return

    _pichia_overview(df, data_path)
    pichia_fingerprint = _dataset_fingerprint(df)
    strain_options = _pichia_strain_options(df)

    with st.sidebar:
        st.header("实验上下文")
        if strain_options:
            strain_choice = st.selectbox("菌种 ID", [*strain_options, "手动输入新菌种"], key="pichia_strain_choice")
            strain_id = (
                st.text_input("新菌种 ID", key="pichia_strain_manual").strip()
                if strain_choice == "手动输入新菌种"
                else strain_choice
            )
            parent_choice = st.selectbox("亲本菌种 ID", ["无", *strain_options, "手动输入"], key="pichia_parent_choice")
            if parent_choice == "手动输入":
                parent_strain_id = st.text_input("亲本菌种 ID（手动）", key="pichia_parent_manual").strip() or None
            elif parent_choice == "无":
                parent_strain_id = None
            else:
                parent_strain_id = parent_choice
        else:
            strain_id = st.text_input("菌种 ID", key="pichia_strain_manual_empty").strip()
            parent_strain_id = st.text_input("亲本菌种 ID（可选）", key="pichia_parent_manual_empty").strip() or None

    st.markdown("### 实验设计设置")
    settings_col, input_col = st.columns([0.9, 1.1])
    with settings_col:
        method = st.selectbox("实验方法", list(METHOD_COUPLING_DEFAULTS), index=0, key="pichia_experiment_method")
        n_recommendations = st.slider("推荐罐数", min_value=2, max_value=4, value=4, key="pichia_n_recommendations")
        baseline_source = st.selectbox(
            "基准点来源",
            PICHIA_BASELINE_SOURCES,
            index=3 if df.empty else 0,
            key="pichia_baseline_source",
        )
        if method == "序贯 DOE（2因子）":
            coupling = 0.5
            seed = 0
            st.caption("4 罐为完整 2×2 DOE，可同时估计两个主效应和一次交互效应。")
        elif method == "单变量验证":
            coupling = 0.0
            seed = 0
            st.caption("单变量验证按探索区间等距取点，不使用随机种子。")
        else:
            coupling = st.slider(
                "探索耦合度",
                min_value=0.0,
                max_value=1.0,
                value=float(METHOD_COUPLING_DEFAULTS[method]),
                step=0.01,
                help="0.0 接近单变量法；1.0 为所有可探索变量联合变化。",
                key=f"pichia_coupling_{method}",
            )
            seed = st.number_input("设计随机种子", min_value=0, max_value=9999, value=0, step=1, key="pichia_seed")

    with input_col:
        manual_baseline = _pichia_manual_baseline_inputs() if baseline_source == "手动输入" else {}
        variable_labels = list(PICHIA_VARIABLE_KEYS_BY_LABEL)
        doe_variables = None
        doe_bounds = None
        single_variable = None
        single_variable_bounds = None
        intensities = {}
        if method == "序贯 DOE（2因子）":
            default_a = "生长期 pH"
            default_b = "生产期 pH"
            factor_a_label = st.selectbox(
                "主动变量 A",
                variable_labels,
                index=variable_labels.index(default_a) if default_a in variable_labels else 0,
                key="pichia_doe_factor_a",
            )
            factor_b_options = [label for label in variable_labels if label != factor_a_label]
            factor_b_label = st.selectbox(
                "主动变量 B",
                factor_b_options,
                index=factor_b_options.index(default_b) if default_b in factor_b_options else 0,
                key="pichia_doe_factor_b",
            )
            doe_variables = [
                PICHIA_VARIABLE_KEYS_BY_LABEL[factor_a_label],
                PICHIA_VARIABLE_KEYS_BY_LABEL[factor_b_label],
            ]
            doe_bounds = {}
            for variable in doe_variables:
                spec = PICHIA_PARAMETER_SPECS[variable]
                st.markdown(f"**{spec.label} DOE 水平**")
                bound_cols = st.columns(2)
                number_format = "%.2f" if spec.step < 0.1 else "%.1f" if spec.step < 1 else "%.0f"
                with bound_cols[0]:
                    lower = st.number_input(
                        "低水平",
                        min_value=float(spec.lower),
                        max_value=float(spec.upper),
                        value=float(spec.lower),
                        step=float(spec.step),
                        format=number_format,
                        key=f"pichia_doe_lower_{variable}",
                    )
                with bound_cols[1]:
                    upper = st.number_input(
                        "高水平",
                        min_value=float(spec.lower),
                        max_value=float(spec.upper),
                        value=float(spec.upper),
                        step=float(spec.step),
                        format=number_format,
                        key=f"pichia_doe_upper_{variable}",
                    )
                doe_bounds[variable] = (float(lower), float(upper))

            try:
                levels = {
                    variable: two_factor_doe_levels(variable, *doe_bounds[variable])
                    for variable in doe_variables
                }
                preview_rows = []
                for rank, (code_a, code_b) in enumerate(two_factor_design_matrix(int(n_recommendations)), start=1):
                    preview_rows.append(
                        {
                            "排序": rank,
                            PICHIA_PARAMETER_SPECS[doe_variables[0]].label: levels[doe_variables[0]]["high" if code_a > 0 else "low"],
                            PICHIA_PARAMETER_SPECS[doe_variables[1]].label: levels[doe_variables[1]]["high" if code_b > 0 else "low"],
                        }
                    )
                st.dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)
            except ValueError as exc:
                st.warning(str(exc))
            variables = doe_variables
        elif method == "单变量验证":
            single_label = st.selectbox("单变量验证对象", variable_labels, key="pichia_single_variable_label")
            single_variable = PICHIA_VARIABLE_KEYS_BY_LABEL[single_label]
            spec = PICHIA_PARAMETER_SPECS[single_variable]
            bound_cols = st.columns(2)
            number_format = "%.2f" if spec.step < 0.1 else "%.1f" if spec.step < 1 else "%.0f"
            with bound_cols[0]:
                single_lower = st.number_input(
                    f"{spec.label}探索下限",
                    min_value=float(spec.lower),
                    max_value=float(spec.upper),
                    value=float(spec.lower),
                    step=float(spec.step),
                    format=number_format,
                    key=f"pichia_single_lower_{single_variable}",
                )
            with bound_cols[1]:
                single_upper = st.number_input(
                    f"{spec.label}探索上限",
                    min_value=float(spec.lower),
                    max_value=float(spec.upper),
                    value=float(spec.upper),
                    step=float(spec.step),
                    format=number_format,
                    key=f"pichia_single_upper_{single_variable}",
                )
            single_variable_bounds = (float(single_lower), float(single_upper))
            levels = uniform_single_variable_levels(
                single_variable,
                int(n_recommendations),
                lower=float(single_lower),
                upper=float(single_upper),
            )
            st.caption(
                "本次单变量水平："
                + "，".join(str(_num(value)) for value in levels)
                + "；其他参数保持基准点不变。"
            )
            variables = [single_variable]
        else:
            selected_labels = st.multiselect(
                "可探索变量",
                variable_labels,
                default=variable_labels,
                key="pichia_variables",
            )
            variables = [PICHIA_VARIABLE_KEYS_BY_LABEL[label] for label in selected_labels]
            with st.expander("每个变量探索强度", expanded=True):
                for label in selected_labels:
                    key = PICHIA_VARIABLE_KEYS_BY_LABEL[label]
                    intensities[key] = st.selectbox(
                        f"{label}",
                        ["低", "中", "高"],
                        index=1,
                        key=f"pichia_intensity_{key}",
                    )

    run_button = st.button("生成 Pichia 推荐", type="primary", width="stretch")

    if not strain_id:
        st.warning("请先在左侧填写或选择菌种 ID。")
        return

    if run_button:
        try:
            result = recommend_pichia_design(
                df,
                strain_id=strain_id,
                parent_strain_id=parent_strain_id,
                baseline_source=baseline_source,
                manual_baseline=manual_baseline if baseline_source == "手动输入" else None,
                variables=variables,
                intensities=intensities,
                coupling=float(coupling),
                doe_variables=doe_variables,
                doe_bounds=doe_bounds,
                single_variable=single_variable,
                single_variable_bounds=single_variable_bounds,
                n_recommendations=int(n_recommendations),
                seed=int(seed),
                target_col=PICHIA_TARGET_COL,
            )
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state["pichia_design_result"] = result
        st.session_state["pichia_design_settings"] = {
            "dataset_fingerprint": pichia_fingerprint,
            "strain_id": strain_id,
            "parent_strain_id": parent_strain_id,
            "method": method,
            "baseline_source": baseline_source,
            "coupling": float(coupling),
            "doe_variables": doe_variables,
            "doe_bounds": doe_bounds,
            "single_variable": single_variable,
            "single_variable_bounds": single_variable_bounds,
            "n_recommendations": int(n_recommendations),
            "seed": int(seed),
        }
        st.success("Pichia 推荐已生成")
    else:
        settings = st.session_state.get("pichia_design_settings") or {}
        result = (
            st.session_state.get("pichia_design_result")
            if settings.get("dataset_fingerprint") == pichia_fingerprint
            else None
        )
        if result is None:
            st.info("设置左侧参数后点击“生成 Pichia 推荐”。当前阶段推荐用于设计可执行探索实验，不用于宣称最优点。")
            return
        st.caption("当前显示上一次生成的 Pichia 推荐；点击左侧按钮会按当前设置重新生成。")

    _pichia_result_block(result)


def main() -> None:
    st.set_page_config(page_title="发酵工艺优化推荐系统", layout="wide")
    st.title("发酵工艺优化推荐系统")
    with st.sidebar:
        mode = st.radio("推荐模式", ["大肠杆菌 BO", "毕赤酵母早期实验设计"], key="recommendation_mode")

    if mode == "毕赤酵母早期实验设计":
        _pichia_design_page()
        return

    st.caption("主方法：标准 GP-BO（qNEI），使用 BoTorch 联合优化批量推荐。")

    with st.sidebar:
        st.header("数据入口")
        source = st.radio("选择数据来源", ["使用 data/final/run_level_modeling_dataset.csv", "上传 run-level CSV"])
        bo_method = st.radio(
            "推荐方法",
            options=["EI（顺序贪心）", "qNEI（批量联合）"],
            index=0,
            help=(
                "EI：单点期望改善，顺序贪心生成批次，结果稳定可解释，推荐作为主方法。\n\n"
                "qNEI：联合批量优化，理论上多样性更好，但对随机种子敏感，推荐值偶有偏离历史分布。"
            ),
            key="bo_method",
        )
        _method_arg = "ei" if bo_method.startswith("EI") else "qnei"
        top_k = st.slider("推荐数量", min_value=1, max_value=10, value=5)
        seed = st.number_input(
            "随机种子",
            min_value=0,
            max_value=9999,
            value=0,
            step=1,
            help="固定种子保证每次运行结果一致。修改为不同整数可探索多组推荐方案。",
            key="bo_seed",
        )
        enable_soft_filter = st.checkbox(
            "启用软过滤",
            value=False,
            help="开启后先生成更大的候选池，再按最近邻距离、边界风险和历史合理范围筛出主推荐。",
            key="enable_soft_filter",
        )
        candidate_pool_multiplier = 1
        max_nearest_history_distance = 2.0
        max_boundary_risk = 0.8
        history_sigma = 2.0
        if enable_soft_filter:
            candidate_pool_multiplier = st.slider(
                "候选池倍数",
                min_value=1,
                max_value=5,
                value=3,
                help="先生成 推荐数量×倍数 的候选池，再用软过滤筛出主推荐。候选池越大越可能凑够通过项，但运行会更慢。",
                key="candidate_pool_multiplier",
            )
            max_nearest_history_distance = st.number_input(
                "最大最近邻距离",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help="软过滤阈值。推荐点到最近历史实验的标准化距离超过该值时，不进入主推荐。",
                key="max_nearest_history_distance",
            )
            max_boundary_risk = st.number_input(
                "最大边界风险",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="软过滤阈值。推荐点过于靠近搜索空间边界时，不进入主推荐。",
                key="max_boundary_risk",
            )
            history_sigma = st.number_input(
                "历史合理范围 σ 倍数",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="软过滤阈值。每个参数以历史均值±k个标准差作为合理范围，比绝对min/max更能抵抗异常批次。",
                key="history_sigma",
            )
        run_button = st.button("运行推荐", type="primary", width="stretch")
        st.divider()
        st.markdown("### 默认决策")
        st.write("默认采用标准 GP-BO（qNEI）作为主推荐；qNEI 联合优化整批候选点并处理观测噪声。")

    uploaded_file = st.file_uploader("上传已整理好的 run-level CSV", type=["csv"]) if source == "上传 run-level CSV" else None

    try:
        df = _load_default_dataset() if source == "使用 data/final/run_level_modeling_dataset.csv" else pd.read_csv(uploaded_file) if uploaded_file else None
    except Exception as exc:
        st.error(f"数据加载失败：{exc}")
        return

    if df is None:
        st.info("请上传 run-level CSV，或选择默认数据目录。")
        return
    if TARGET_COL not in df.columns:
        st.error(f"数据缺少目标字段 `{TARGET_COL}`。")
        return

    _overview(df)
    dataset_fingerprint = _dataset_fingerprint(df)
    report_path = PROJECT_ROOT / "summary" / "recommendation_report.md"
    cache_message = ""
    if run_button:
        with st.spinner("正在训练标准 GP-BO（qNEI）并生成推荐..."):
            pool_size = _recommendation_pool_size(top_k, candidate_pool_multiplier) if enable_soft_filter else int(top_k)
            comparison = _compare_recommenders(df, top_k=pool_size, seed=int(seed), method=_method_arg)
            comparison["requested_top_k"] = int(top_k)
            comparison["recommendation_pool_size"] = int(pool_size)
            comparison["soft_filter_enabled"] = bool(enable_soft_filter)
            feature_cols_for_filter = comparison.get("model_info", {}).get("standard_bo_feature_cols", [])
            if enable_soft_filter:
                comparison = _apply_soft_filters(
                    comparison,
                    df,
                    feature_cols_for_filter,
                    max_nearest_history_distance=float(max_nearest_history_distance),
                    max_boundary_risk=float(max_boundary_risk),
                    history_sigma=float(history_sigma),
                    target_count=int(top_k),
                )
            else:
                comparison = _select_without_soft_filters(
                    comparison,
                    df,
                    feature_cols_for_filter,
                    target_count=int(top_k),
                )
            run_settings = {
                "top_k": int(top_k),
                "pool_size": int(pool_size),
                "seed": int(seed),
                "method": _method_arg,
                "soft_filter_enabled": bool(enable_soft_filter),
                "candidate_pool_multiplier": int(candidate_pool_multiplier),
                "max_nearest_history_distance": float(max_nearest_history_distance),
                "max_boundary_risk": float(max_boundary_risk),
                "history_sigma": float(history_sigma),
            }
            comparison["run_settings"] = run_settings
            report_md = generate_recommendation_report(comparison, output_path=report_path)
            _write_recommendation_cache(
                st.session_state,
                dataset_fingerprint=dataset_fingerprint,
                comparison=comparison,
                report_md=report_md,
                report_path=report_path,
                run_settings=run_settings,
            )
        st.session_state["recommendation_comparison"] = comparison
        st.session_state["recommendation_report_md"] = report_md
        st.session_state["recommendation_report_path"] = str(report_path)
        cache_message = "已重新训练并更新缓存。"
    else:
        cached = _read_recommendation_cache(st.session_state, dataset_fingerprint)
        if cached:
            comparison = cached["comparison"]
            report_md = cached.get("report_md", "")
            report_path = Path(cached.get("report_path", str(report_path)))
            st.session_state["recommendation_comparison"] = comparison
            st.session_state["recommendation_report_md"] = report_md
            st.session_state["recommendation_report_path"] = str(report_path)
            cache_message = "已自动读取缓存结果；只有点击“运行推荐”才会重新训练。"
        else:
            st.info("当前数据没有可用推荐缓存。点击左侧“运行推荐”后，系统会训练模型、生成候选点，并缓存本次结果。")
            return

    selected_method = comparison.get("selected_method", "standard_bo_qnei")
    selected = comparison.get("selected_recommendations", [])
    decision = comparison.get("decision", {})
    fitted_gp = comparison.get("model_info", {}).get("fitted_standard_bo_gp")
    feature_cols = comparison.get("model_info", {}).get("standard_bo_feature_cols", [])
    model_feature_cols = comparison.get("model_info", {}).get("standard_bo_model_feature_cols", feature_cols)
    selected = comparison.get("selected_recommendations", [])
    if not report_md:
        report_md = generate_recommendation_report(comparison, output_path=report_path)
        st.session_state["recommendation_report_md"] = report_md

    st.success("推荐已生成")
    if cache_message:
        st.caption(cache_message)
    run_settings = comparison.get("run_settings") or {}
    if run_settings:
        st.caption(
            "当前结果参数：方法 {method}，推荐数 {top_k}，候选池 {pool_size}，随机种子 {seed}，软过滤 {soft_filter}。".format(
                method=run_settings.get("method"),
                top_k=run_settings.get("top_k"),
                pool_size=run_settings.get("pool_size"),
                seed=run_settings.get("seed"),
                soft_filter="启用" if run_settings.get("soft_filter_enabled") else "未启用",
            )
        )
    cols = st.columns(2)
    cols[0].metric("主推荐方法", METHOD_LABELS.get(selected_method, selected_method))
    cols[1].metric("训练 run 数", comparison.get("n_training_rows"))
    st.info(decision.get("reason", "默认采用标准 GP-BO（qNEI）。"))
    soft_filter = comparison.get("soft_filter") or {}
    if soft_filter and not soft_filter.get("enabled", True):
        st.caption(
            "软过滤未启用：直接显示 BO 候选池前 {after}/{target} 个推荐。".format(
                after=soft_filter.get("n_after"),
                target=soft_filter.get("target_count"),
            )
        )
    elif soft_filter:
        st.caption(
            "软过滤：最近邻距离 <= {dist}，边界风险 <= {risk}，历史合理范围 ±{sigma}σ；"
            "候选池 {before} 个，通过 {passed} 个，当前显示 {after}/{target} 个推荐。".format(
                dist=_num(soft_filter.get("max_nearest_history_distance")),
                risk=_num(soft_filter.get("max_boundary_risk")),
                sigma=_num(soft_filter.get("history_sigma")),
                passed=soft_filter.get("n_passed"),
                after=soft_filter.get("n_after"),
                before=soft_filter.get("n_before"),
                target=soft_filter.get("target_count"),
            )
        )
        failure_counts = soft_filter.get("failure_counts") or {}
        if failure_counts:
            st.caption(
                "软过滤失败原因计数（可重叠）：最近邻距离 {nearest}，边界风险 {boundary}，历史合理范围 {history_range}。".format(
                    nearest=failure_counts.get("nearest_history_distance", 0),
                    boundary=failure_counts.get("boundary_risk", 0),
                    history_range=failure_counts.get("history_range", 0),
                )
            )
        if soft_filter.get("failed_nearest_history_ranks"):
            st.caption(f"最近邻距离超限推荐排序：{soft_filter.get('failed_nearest_history_ranks')}")
        if soft_filter.get("failed_boundary_risk_ranks"):
            st.caption(f"边界风险超限推荐排序：{soft_filter.get('failed_boundary_risk_ranks')}")
        if soft_filter.get("failed_history_range_ranks"):
            st.caption(f"历史合理范围外推荐排序：{soft_filter.get('failed_history_range_ranks')}")
        if soft_filter.get("target_count") and soft_filter.get("n_after", 0) < soft_filter.get("target_count"):
            st.warning("当前候选池中通过软过滤的推荐不足目标数量；建议放宽阈值、提高候选池倍数，或更换随机种子。")

    tabs = st.tabs(["主推荐", "代理模型验证", "推荐策略质量", "GP 偏依赖图", "指标说明", "Markdown 报告"])
    with tabs[0]:
        _method_block(selected_method, selected)
        if selected:
            _standard_bo_summary(selected[0])
            _standard_gp_plot(df, selected, comparison.get("search_space", {}), fitted_gp, feature_cols)
    with tabs[1]:
        st.markdown("### LOO-CV 模型泛化能力")
        st.caption("验证 GP 模型是否能预测它没见过的实验结果。这是判断推荐可信度的核心依据。")
        _loocv_scatter(df, model_feature_cols)
    with tabs[2]:
        _strategy_quality_block(comparison, df, selected, feature_cols)
    with tabs[3]:
        st.caption("偏依赖图反映特征的平均边际效应，自然携带历史数据中的特征协变关系，优于固定其他参数的条件切片图。")
        _gp_pdp(df, fitted_gp, feature_cols)
    with tabs[4]:
        _metric_explanations()
    with tabs[5]:
        st.markdown(report_md)
        st.caption(f"报告已写入：{report_path}")


if __name__ == "__main__":
    main()
