from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_advisor.ingestion.run_level import TARGET_COL, training_view
from experiment_advisor.recommendation.service import compare_recommenders
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
    "standard_bo_ei": "标准 GP-BO（EI）",
    "xgp_bo_ei": "XGBoost + GP 残差 BO（EI）",
}

METHOD_EXPLANATIONS = {
    "standard_bo_ei": "Gaussian Process 直接拟合产量，并用 EI 选择预期改进较大的候选。",
    "xgp_bo_ei": "候选参考方法。XGBoost 预测产量均值，GP 只拟合 XGBoost 残差，用于补充查看非线性模型下的建议。",
}

UNCERTAINTY_LABELS = {
    "gp_posterior_std": "GP 后验标准差",
    "xgp_gp_residual_std": "GP 残差后验标准差",
}

RISK_LABELS = {"low": "低", "medium": "中", "high": "高"}
FLAG_LABELS = {
    "far_from_history": "远离历史实验",
    "near_search_boundary": "接近参数边界",
    "high_residual_uncertainty": "残差不确定性较高",
}


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
        if method == "standard_bo_ei":
            row.update(
                {
                    "GP 后验标准差": _num(item.get("model_uncertainty")),
                    "EI 推荐得分": _num(item.get("acquisition_score")),
                }
            )
        else:
            row.update(
                {
                    "XGBoost 均值": _num(item.get("xgb_prediction")),
                    "GP 残差修正": _num(item.get("gp_residual_mean")),
                    "不确定性": _num(item.get("model_uncertainty")),
                    "不确定性定义": UNCERTAINTY_LABELS.get(item.get("uncertainty_type"), item.get("uncertainty_type")),
                    "历史距离": _num(item.get("history_distance")),
                    "边界风险": _num(item.get("boundary_risk")),
                    "风险等级": RISK_LABELS.get(item.get("risk_level"), item.get("risk_level")),
                    "风险标记": _flags(item.get("quality_flags")),
                    "推荐得分": _num(item.get("acquisition_score")),
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
        {"项目": "EI 推荐得分", "数值": _num(top.get("acquisition_score")), "说明": "Expected Improvement，越高表示相对当前历史最好值越值得尝试。"},
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

    low, high = search_space[feature]
    grid = pd.DataFrame([params] * n_points)
    grid[feature] = np.linspace(float(low), float(high), n_points)
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
    top: dict[str, Any],
    search_space: dict[str, Any],
    fitted_gp: Any,
    feature_cols: list[str],
) -> None:
    if fitted_gp is None:
        st.info("标准 GP 模型不可用，无法绘制后验切片图。")
        return

    params = top.get("params", {})
    features = [feature for feature in feature_cols if feature in search_space and feature in params and feature in df.columns]
    if not features:
        st.info("当前推荐缺少可绘图参数。")
        return

    st.markdown("### 标准 GP 后验切片图")
    st.caption("固定主推荐点的其它参数，只沿一个参数变化，展示标准 GP 的 posterior mean 和 95% 区间。")
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
    ax.axvline(float(params[feature]), color="#dc2626", linestyle="--", linewidth=1.5, label="recommended value")
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
    try:
        import math as _math
        import numpy as np
        from sklearn.inspection import PartialDependenceDisplay
    except ImportError as exc:
        st.warning(f"偏依赖图需要 scikit-learn >= 1.0：{exc}")
        return

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

    PartialDependenceDisplay.from_estimator(
        fitted_gp,
        train,
        features=list(range(n_features)),
        feature_names=[_display_name(col) for col in feature_cols],
        kind="average",
        grid_resolution=50,
        ax=axes_flat[:n_features],
        random_state=42,
    )
    for index, ax in enumerate(axes_flat[:n_features]):
        ax.set_title(_display_name(feature_cols[index]), fontsize=10)
        ax.set_xlabel(_display_name(feature_cols[index]))
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
        idx_a = feature_cols.index(key_a)
        idx_b = feature_cols.index(key_b)

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        PartialDependenceDisplay.from_estimator(
            fitted_gp,
            train,
            features=[(idx_a, idx_b)],
            feature_names=[_display_name(col) for col in feature_cols],
            kind="average",
            grid_resolution=20,
            ax=[ax2],
            random_state=42,
        )
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

    train = _training_data(df)[feature_cols + [TARGET_COL]].dropna()
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


def _xgp_summary(top: dict[str, Any]) -> None:
    rows = [
        {"项目": "XGBoost 均值预测", "数值": _num(top.get("xgb_prediction")), "说明": "学习非线性产量结构。"},
        {"项目": "GP 残差修正", "数值": _num(top.get("gp_residual_mean")), "说明": "对 XGBoost 未解释残差做局部修正。"},
        {"项目": "最终预测产量", "数值": _num(top.get("predicted_yield")), "说明": "XGBoost 均值 + GP 残差修正。"},
        {"项目": "GP 残差后验标准差", "数值": _num(top.get("model_uncertainty")), "说明": "残差模型的不确定性。"},
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _gp_health(items: list[dict[str, Any]]) -> None:
    health = (items[0].get("gp_health") if items else None) or {}
    if not health:
        st.info("当前结果没有残差 GP 健康诊断。")
        return

    cols = st.columns(4)
    cols[0].metric("残差均值", _num(health.get("residual_mean")))
    cols[1].metric("残差标准差", _num(health.get("residual_std")))
    cols[2].metric("最大绝对残差", _num(health.get("residual_max_abs")))
    cols[3].metric("不确定性种类数", health.get("candidate_uncertainty_unique_rounded"))

    gp_features = health.get("gp_feature_cols") or []
    st.write("GP 实际使用的特征")
    st.dataframe(pd.DataFrame({"显示名": [_name(col) for col in gp_features], "字段名": gp_features}), width="stretch", hide_index=True)

    if health.get("candidate_uncertainty_degenerate"):
        st.error("候选点不确定性几乎完全相同，残差 GP 可能退化。")
    else:
        st.success("候选点不确定性可以区分不同候选，残差 GP 未表现为常数退化。")

    st.dataframe(
        pd.DataFrame(
            [
                {"指标": "候选不确定性最小值", "数值": _num(health.get("candidate_uncertainty_min"))},
                {"指标": "候选不确定性最大值", "数值": _num(health.get("candidate_uncertainty_max"))},
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    warnings = health.get("warnings") or []
    if warnings:
        st.warning("GP 训练出现警告，建议谨慎解释残差不确定性。")
        for message in warnings:
            st.caption(message)

    with st.expander("GP kernel 详情", expanded=False):
        st.code(str(health.get("kernel", "")), language="text")


def _metric_explanations() -> None:
    st.markdown("### 主推荐：标准 GP-BO（EI）")
    standard_rows = [
        {"指标": "预测产量", "如何产生": "GP 直接拟合历史产量后，对候选点输出 posterior mean。", "如何解读": "模型预测值，不是实测值。"},
        {"指标": "GP 后验标准差", "如何产生": "GP 对候选点预测分布的 posterior std。", "如何解读": "越高表示模型在该区域越不确定。"},
        {"指标": "EI 推荐得分", "如何产生": "Expected Improvement，综合预测均值、后验标准差和当前历史最好值。", "如何解读": "越高表示越有可能带来改进。"},
    ]
    st.dataframe(pd.DataFrame(standard_rows), width="stretch", hide_index=True)

    st.markdown("### 候选参考：XGBoost + GP 残差 BO（EI）")
    xgp_rows = [
        {"指标": "XGBoost 均值", "如何产生": "XGBoost 使用全部搜索特征预测产量。", "如何解读": "反映非线性模型学到的主要产量结构。"},
        {"指标": "GP 残差修正", "如何产生": "GP 只拟合 XGBoost 的训练残差。", "如何解读": "正值表示局部上调预测，负值表示局部下调预测。"},
        {"指标": "GP 残差后验标准差", "如何产生": "残差 GP 的 posterior std。", "如何解读": "只表示残差模型不确定性，不是湿实验置信区间。"},
        {"指标": "历史距离", "如何产生": "候选点到最近历史实验的标准化距离。", "如何解读": "越高越像外推，工艺采纳风险越高。"},
        {"指标": "边界风险", "如何产生": "候选点靠近搜索空间上下限的程度。", "如何解读": "越高越靠边界，需要检查是否可执行。"},
    ]
    st.dataframe(pd.DataFrame(xgp_rows), width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="发酵工艺优化推荐系统", layout="wide")
    st.title("发酵工艺优化推荐系统")
    st.caption("主方法：标准 GP-BO（EI）。候选参考：XGBoost + GP 残差 BO（EI）。")

    with st.sidebar:
        st.header("数据入口")
        source = st.radio("选择数据来源", ["使用 data/final/run_level_modeling_dataset.csv", "上传 run-level CSV"])
        top_k = st.slider("推荐数量", min_value=1, max_value=10, value=5)
        run_button = st.button("运行推荐", type="primary", width="stretch")
        st.divider()
        st.markdown("### 默认决策")
        st.write("默认采用标准 GP-BO（EI）作为主推荐；XGBoost + GP 残差 BO（EI）只作为候选参考。")

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
    if run_button:
        with st.spinner("正在训练标准 GP-BO、XGP 候选模型并生成推荐..."):
            comparison = compare_recommenders(df, top_k=top_k)
            report_path = PROJECT_ROOT / "summary" / "recommendation_report.md"
            report_md = generate_recommendation_report(comparison, output_path=report_path)
        st.session_state["recommendation_comparison"] = comparison
        st.session_state["recommendation_report_md"] = report_md
        st.session_state["recommendation_report_path"] = str(report_path)
    elif "recommendation_comparison" in st.session_state:
        comparison = st.session_state["recommendation_comparison"]
        report_md = st.session_state.get("recommendation_report_md", "")
        report_path = Path(st.session_state.get("recommendation_report_path", PROJECT_ROOT / "summary" / "recommendation_report.md"))
    else:
        st.info("点击左侧“运行推荐”后，系统会训练模型、生成候选点，并输出诊断信息。")
        return

    selected_method = comparison.get("selected_method", "standard_bo_ei")
    selected = comparison.get("selected_recommendations", [])
    xgp_items = comparison.get("recommendations", {}).get("xgp_bo_ei", [])
    decision = comparison.get("decision", {})
    fitted_gp = comparison.get("model_info", {}).get("fitted_standard_bo_gp")
    feature_cols = comparison.get("model_info", {}).get("standard_bo_feature_cols", [])

    st.success("推荐已生成")
    cols = st.columns(3)
    cols[0].metric("主推荐方法", METHOD_LABELS.get(selected_method, selected_method))
    cols[1].metric("需要人工审议", "是" if decision.get("needs_human_review") else "否")
    cols[2].metric("训练 run 数", comparison.get("n_training_rows"))
    st.info(decision.get("reason", "默认采用标准 GP-BO（EI）。"))

    tabs = st.tabs(["主推荐", "XGP 候选", "推荐验证", "GP 偏依赖图", "残差 GP 健康", "指标说明", "Markdown 报告"])
    with tabs[0]:
        _method_block(selected_method, selected)
        if selected:
            _standard_bo_summary(selected[0])
            _standard_gp_plot(df, selected[0], comparison.get("search_space", {}), fitted_gp, feature_cols)
            nearest = _nearest_history(df, selected[0].get("params", {}))
            if not nearest.empty:
                st.write("最相似的历史实验")
                st.dataframe(nearest, width="stretch", hide_index=True)
    with tabs[1]:
        _method_block("xgp_bo_ei", xgp_items)
        if xgp_items:
            _xgp_summary(xgp_items[0])
    with tabs[2]:
        st.markdown("### LOO-CV 模型泛化能力")
        st.caption("验证 GP 模型是否能预测它没见过的实验结果。这是判断推荐可信度的核心依据。")
        _loocv_scatter(df, feature_cols)

        st.divider()

        st.markdown("### 推荐参数的历史近邻")
        st.caption("与推荐参数最相似的历史实验实际产量，验证推荐是否落在有数据支撑的高产区域。")
        if selected:
            selected_params = {**selected[0].get("params", {}), "predicted_yield": selected[0].get("predicted_yield")}
            _nearest_history_validation(df, selected_params, feature_cols)
    with tabs[3]:
        st.caption("偏依赖图反映特征的平均边际效应，自然携带历史数据中的特征协变关系，优于固定其他参数的条件切片图。")
        _gp_pdp(df, fitted_gp, feature_cols)
    with tabs[4]:
        _gp_health(xgp_items)
    with tabs[5]:
        _metric_explanations()
    with tabs[6]:
        st.markdown(report_md)
        st.caption(f"报告已写入：{report_path}")


if __name__ == "__main__":
    main()
