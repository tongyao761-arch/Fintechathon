"""Ten point-in-time price and volume features for the first formal baseline."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.data.data_contract import (
    DataContractError,
    assert_sorted_by_panel_key,
    require_columns,
)


FEATURE_COLUMNS = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "gap_1d",
    "intraday_ret",
    "high_low_range",
    "volatility_5d",
    "volatility_20d",
    "volume_ratio_5d",
    "amount_ratio_5d",
)
RAW_FEATURE_COLUMNS = ("open", "high", "low", "close", "vol", "amount")
FEATURE_DEFINITIONS: Mapping[str, str] = {
    "ret_1d": "close / close.shift(1) - 1",
    "ret_5d": "close / close.shift(5) - 1",
    "ret_20d": "close / close.shift(20) - 1",
    "gap_1d": "open / close.shift(1) - 1",
    "intraday_ret": "close / open - 1",
    "high_low_range": "high / low - 1",
    "volatility_5d": "ret_1d.rolling(5, min_periods=5).std(ddof=1)",
    "volatility_20d": "ret_1d.rolling(20, min_periods=20).std(ddof=1)",
    "volume_ratio_5d": (
        "vol / vol.shift(1).rolling(5, min_periods=5).mean()"
    ),
    "amount_ratio_5d": (
        "amount / amount.shift(1).rolling(5, min_periods=5).mean()"
    ),
}


def _rolling_mean(
    values: pd.Series,
    group_keys: pd.Series,
    window: int,
) -> pd.Series:
    result = (
        values.groupby(group_keys, sort=False, observed=True)
        .rolling(window, min_periods=window)
        .mean()
    )
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def _rolling_std(
    values: pd.Series,
    group_keys: pd.Series,
    window: int,
) -> pd.Series:
    result = (
        values.groupby(group_keys, sort=False, observed=True)
        .rolling(window, min_periods=window)
        .std(ddof=1)
    )
    result.index = result.index.droplevel(0)
    return result.reindex(values.index)


def build_baseline_v1_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed baseline features without mutating or dropping input rows."""
    require_columns(
        panel,
        ("ts_code", "trade_date", *RAW_FEATURE_COLUMNS),
        name="baseline feature panel",
    )
    if not panel.index.is_unique:
        raise DataContractError("baseline feature panel index must be unique")
    assert_sorted_by_panel_key(panel, name="baseline feature panel")

    grouped = panel.groupby("ts_code", sort=False, observed=True)
    previous_close_1d = grouped["close"].shift(1)
    ret_1d = panel["close"] / previous_close_1d - 1.0
    previous_vol = grouped["vol"].shift(1)
    previous_amount = grouped["amount"].shift(1)

    features = pd.DataFrame(
        {
            "ret_1d": ret_1d,
            "ret_5d": panel["close"] / grouped["close"].shift(5) - 1.0,
            "ret_20d": panel["close"] / grouped["close"].shift(20) - 1.0,
            "gap_1d": panel["open"] / previous_close_1d - 1.0,
            "intraday_ret": panel["close"] / panel["open"] - 1.0,
            "high_low_range": panel["high"] / panel["low"] - 1.0,
            "volatility_5d": _rolling_std(
                ret_1d,
                panel["ts_code"],
                5,
            ),
            "volatility_20d": _rolling_std(
                ret_1d,
                panel["ts_code"],
                20,
            ),
            "volume_ratio_5d": panel["vol"]
            / _rolling_mean(previous_vol, panel["ts_code"], 5),
            "amount_ratio_5d": panel["amount"]
            / _rolling_mean(previous_amount, panel["ts_code"], 5),
        },
        index=panel.index,
        columns=FEATURE_COLUMNS,
    )
    # A finite float64 result can overflow during the final float32 cast.
    with np.errstate(over="ignore", invalid="ignore"):
        features = features.astype("float32")
    features = features.replace([np.inf, -np.inf], np.nan)
    if len(features) != len(panel) or not features.index.equals(panel.index):
        raise AssertionError("feature construction changed panel rows or index")
    return features
