"""Fixed ten-feature skeleton plus 24 X-only candidates; no row filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.data_contract import DataContractError, require_columns
from src.features.baseline_v1 import (
    FEATURE_COLUMNS as BASE_COLUMNS,
    FEATURE_DEFINITIONS as BASE_DEFINITIONS,
    build_baseline_v1_features,
)

RANK_SOURCES = ("ret_1d", "ret_5d", "ret_20d", "amount_ratio_5d",
                "volatility_20d", "log_mean_amount_20d")
GROUPS = {
    "A": ("ret_2d", "ret_10d", "ret_60d"),
    "B": ("close_location", "upper_shadow", "lower_shadow", "body_ratio"),
    "C": ("bias_20d", "bias_60d", "price_position_20d", "price_position_60d"),
    "D": ("amount_ratio_20d", "log_mean_amount_20d", "volatility_60d"),
    "E": ("flag_limit_up", "flag_limit_down", "limit_up_count_5d", "limit_down_count_5d"),
    "F": tuple(f"{source}_rank_pct" for source in RANK_SOURCES),
}
NEW_COLUMNS = tuple(column for group in GROUPS.values() for column in group)
FEATURE_COLUMNS = BASE_COLUMNS + NEW_COLUMNS
FEATURE_DEFINITIONS = {
    **BASE_DEFINITIONS,
    **{f"ret_{n}d": f"C / C.shift({n}) - 1" for n in (2, 10, 60)},
    "close_location": "(C-L)/(H-L)",
    "upper_shadow": "(H-max(O,C))/C",
    "lower_shadow": "(min(O,C)-L)/C",
    "body_ratio": "(C-O)/(H-L)",
    **{f"bias_{n}d": f"C/close.rolling({n}, min_periods={n}).mean()-1" for n in (20, 60)},
    **{f"price_position_{n}d": f"(C-low.rolling({n}).min())/(high.rolling({n}).max()-low.rolling({n}).min()); min_periods={n}"
       for n in (20, 60)},
    "amount_ratio_20d": "A/amount.shift(1).rolling(20, min_periods=20).mean()",
    "log_mean_amount_20d": "log1p(amount.rolling(20, min_periods=20).mean())",
    "volatility_60d": "baseline ret_1d.rolling(60, min_periods=60).std(ddof=1)",
    "flag_limit_up": "raw current flag_limit_up",
    "flag_limit_down": "raw current flag_limit_down",
    "limit_up_count_5d": "flag_limit_up.rolling(5, min_periods=5).sum()",
    "limit_down_count_5d": "flag_limit_down.rolling(5, min_periods=5).sum()",
    **{f"{source}_rank_pct": f"same-date {source}.rank(method='average', pct=True); price valid and finite source only"
       for source in RANK_SOURCES},
}


def select_features(*, groups=(), include=(), exclude=()) -> tuple[str, ...]:
    """Select new columns independently, always retaining the ten-feature order."""
    if len(set(groups)) != len(groups) or len(set(include)) != len(include) or len(set(exclude)) != len(exclude):
        raise ValueError("duplicate feature/group selection")
    if set(groups) - set(GROUPS):
        raise ValueError("unknown feature group")
    if (set(include) | set(exclude)) - set(NEW_COLUMNS):
        raise ValueError("only the contracted 24 new features may be selected/excluded")
    selected = set(include).union(*(GROUPS[group] for group in groups))
    if set(exclude) - selected:
        raise ValueError("excluded feature was not selected")
    selected -= set(exclude)
    return BASE_COLUMNS + tuple(column for column in NEW_COLUMNS if column in selected)


def _finite32(values: pd.Series) -> pd.Series:
    with np.errstate(over="ignore", invalid="ignore"):
        return values.astype("float32").replace([np.inf, -np.inf], np.nan)


def _divide(numerator, denominator):
    return numerator / denominator.where(denominator.ne(0))


def build_features34(panel: pd.DataFrame, columns=FEATURE_COLUMNS) -> pd.DataFrame:
    """Compute before masks. is_price_valid must be the loader's X-only flag.

    New arithmetic uses float64 intermediates then finite float32 model columns.
    Ranks use those finalized source columns, including unselected intermediates.
    Baseline columns call the frozen builder unchanged.
    """
    columns = tuple(columns)
    if columns != BASE_COLUMNS + tuple(c for c in NEW_COLUMNS if c in columns):
        raise ValueError("columns must be a canonical selection with the fixed ten-feature skeleton")
    baseline = build_baseline_v1_features(panel)
    if columns == BASE_COLUMNS:
        return baseline
    require_columns(panel, ("flag_limit_up", "flag_limit_down", "is_price_valid"), name="features34 panel")
    if not panel.is_price_valid.isin((0, 1)).all():
        raise DataContractError("is_price_valid must be the X-only binary price validity flag")
    for flag in ("flag_limit_up", "flag_limit_down"):
        if not panel[flag].dropna().isin((0, 1)).all():
            raise DataContractError(f"{flag} must be binary or missing")
    keys = panel.ts_code
    needed = set(columns) - set(BASE_COLUMNS)
    needed |= {source for source in RANK_SOURCES if f"{source}_rank_pct" in needed}
    raw = {c: panel[c].astype("float64").replace([np.inf, -np.inf], np.nan)
           for c in ("open", "high", "low", "close", "amount", "flag_limit_up", "flag_limit_down")}
    o, h, l, c, a = (raw[name] for name in ("open", "high", "low", "close", "amount"))
    values = {}

    def rolling(series, window, operation):
        result = getattr(series.groupby(keys, sort=False, observed=True)
                         .rolling(window, min_periods=window), operation)()
        result.index = result.index.droplevel(0)
        return result.reindex(panel.index)

    for n in (2, 10, 60):
        if f"ret_{n}d" in needed:
            values[f"ret_{n}d"] = _divide(c, c.groupby(keys, sort=False, observed=True).shift(n)) - 1
    if "close_location" in needed:
        values["close_location"] = _divide(c-l, h-l)
    if "upper_shadow" in needed:
        # skipna=False: either missing O/C makes the shadow missing.
        values["upper_shadow"] = _divide(h-pd.concat([o, c], axis=1).max(axis=1, skipna=False), c)
    if "lower_shadow" in needed:
        values["lower_shadow"] = _divide(pd.concat([o, c], axis=1).min(axis=1, skipna=False)-l, c)
    if "body_ratio" in needed:
        values["body_ratio"] = _divide(c-o, h-l)
    for n in (20, 60):
        if f"bias_{n}d" in needed:
            values[f"bias_{n}d"] = _divide(c, rolling(c, n, "mean")) - 1
        if f"price_position_{n}d" in needed:
            minimum, maximum = rolling(l, n, "min"), rolling(h, n, "max")
            values[f"price_position_{n}d"] = _divide(c-minimum, maximum-minimum)
    if "amount_ratio_20d" in needed:
        values["amount_ratio_20d"] = _divide(a, rolling(a.groupby(keys, sort=False, observed=True).shift(1), 20, "mean"))
    if "log_mean_amount_20d" in needed:
        with np.errstate(divide="ignore", invalid="ignore"):
            values["log_mean_amount_20d"] = np.log1p(rolling(a, 20, "mean"))
    if "volatility_60d" in needed:
        # pandas rolling std defaults to ddof=1, as in the frozen baseline.
        values["volatility_60d"] = rolling(baseline.ret_1d.astype("float64"), 60, "std")
    for direction in ("up", "down"):
        flag = f"flag_limit_{direction}"
        if flag in needed:
            values[flag] = raw[flag]
        if f"limit_{direction}_count_5d" in needed:
            values[f"limit_{direction}_count_5d"] = rolling(raw[flag], 5, "sum")
    values = {name: _finite32(series) for name, series in values.items()}
    for source in RANK_SOURCES:
        name = f"{source}_rank_pct"
        if name in needed:
            series = baseline[source] if source in BASE_COLUMNS else values[source]
            eligible = panel.is_price_valid.eq(1) & np.isfinite(series)
            values[name] = _finite32(series.where(eligible).groupby(panel.trade_date, sort=False)
                                     .rank(method="average", pct=True))
    result = pd.concat([baseline, pd.DataFrame(values, index=panel.index)], axis=1).loc[:, columns]
    if not result.index.equals(panel.index) or len(result) != len(panel):
        raise AssertionError("features34 changed rows/index")
    return result
