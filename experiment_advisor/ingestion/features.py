from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始批次数据基础上追加发酵阶段聚合和机理代理特征。

    原始列不会被删除。所有派生特征统一使用 feat_ 前缀；分母为 0 时返回 NaN。
    """

    result = df.copy()
    feed_amount = pd.to_numeric(result["feed_amount"], errors="coerce")
    feed_time = pd.to_numeric(result["feed_time"], errors="coerce")
    induction_time = pd.to_numeric(result["induction_time"], errors="coerce")
    inducer_dose = pd.to_numeric(result["inducer_dose"], errors="coerce")

    # 诱导前生长时长：诱导越晚，细胞量积累窗口通常越长。
    result["feat_pre_induction_duration"] = induction_time
    # 补料到诱导的间隔：表示碳源补充后到诱导表达之间的缓冲时间。
    result["feat_feed_to_induction_interval"] = induction_time - feed_time
    # 诱导后补料代理：缺少分时段补料时，用诱导时间比例估算诱导后的可用补料量。
    result["feat_post_induction_feed"] = feed_amount * (1 - induction_time / (induction_time + 24))
    # 诱导前平均补料速率代理：补料总量除以诱导前时长。
    result["feat_feed_rate_proxy"] = _safe_divide(feed_amount, induction_time)
    # 诱导时机相对比例：把诱导时间压缩到 0-1 风格的尺度，便于模型学习。
    result["feat_inducer_timing_ratio"] = _safe_divide(induction_time, induction_time + 24)
    # 碳负荷代理：补料总量与速率共同刻画碳源压力。
    result["feat_carbon_load_proxy"] = feed_amount * result["feat_feed_rate_proxy"]
    # 诱导剂强度代理：保留诱导剂用量对后续分析的显式数值入口。
    result["feat_inducer_dose_proxy"] = inducer_dose
    return result
