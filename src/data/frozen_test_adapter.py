"""X-only bridge to the frozen float32 feature contract; no test labels."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.baseline_panel import PRICE_COLUMNS, RAW_NUMERIC_COLUMNS
from src.data.data_contract import DataContractError, X_COLUMNS, validate_panel


def adapt_test_x(raw):
    validate_panel(raw, kind="test")
    if "y_ret_1d" in raw or "is_trainable" in raw:
        raise DataContractError("test adapter accepts X only")
    result = raw.loc[:, X_COLUMNS].copy()
    dates = pd.Series(result.trade_date.unique()).astype(str)
    if not dates.str.fullmatch(r"\d{8}").all() or pd.to_datetime(
        dates, format="%Y%m%d", errors="coerce"
    ).isna().any():
        raise DataContractError("invalid test calendar dates")
    prices = result[PRICE_COLUMNS].astype("float64")
    result["is_price_valid"] = (
        np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
        & prices.high.ge(prices.max(axis=1)) & prices.low.le(prices.min(axis=1))
    ).astype("int8")
    result[RAW_NUMERIC_COLUMNS] = result[RAW_NUMERIC_COLUMNS].astype("float32")
    result["ts_code"] = result.ts_code.astype(str)
    result["trade_date"] = result.trade_date.astype("int32")
    result[["flag_limit_up", "flag_limit_down"]] = result[["flag_limit_up", "flag_limit_down"]].astype("int8")
    result["test_row_id"] = np.arange(len(raw), dtype="int64")
    return result


def load_test_x(path: Path):
    # Match the baseline loader's original float64 parse before the float32 cast.
    raw = pd.read_csv(path, dtype={"ts_code": str, "trade_date": "int32",
        **{c: "float64" for c in RAW_NUMERIC_COLUMNS}})
    return raw, adapt_test_x(raw)


def bridge_x(history, test):
    if int(history.trade_date.max()) >= int(test.trade_date.min()):
        raise DataContractError("history/test dates overlap")
    past = history.loc[:, [*X_COLUMNS, "is_price_valid"]].copy()
    past["ts_code"] = past.ts_code.astype(str)
    past["history_row_id"] = np.arange(len(past), dtype="int64")
    past["test_row_id"] = -1
    future = test.copy()
    future["history_row_id"] = -1
    joined = pd.concat([past, future], ignore_index=True)
    joined["ts_code"] = joined.ts_code.astype("category")
    return joined.sort_values(["ts_code", "trade_date"], kind="stable").reset_index(drop=True)


def training_boundary(history, test):
    first_test = int(test.trade_date.min())
    dates = sorted(int(d) for d in history.trade_date.unique() if d < first_test)
    if len(dates) < 2 or dates[-1] != int(history.trade_date.max()):
        raise DataContractError("cannot determine training boundary")
    if dates[0] // 10000 != 2018 or dates[-1] // 10000 != 2024 or first_test // 10000 != 2025:
        raise DataContractError("unexpected training/test years")
    return dict(train_start=dates[0], train_end=dates[-2], purge_date=dates[-1],
        first_test_date=first_test, last_test_date=int(test.trade_date.max()))
