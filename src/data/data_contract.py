"""Invariant checks for the competition's stock-by-date panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

KEY_COLUMNS = ("ts_code", "trade_date")
X_COLUMNS = (
    "ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
    "flag_limit_up", "flag_limit_down",
)
TRAIN_COLUMNS = X_COLUMNS + ("y_ret_1d",)


class DataContractError(ValueError):
    """Raised when a data frame cannot safely enter the shared pipeline."""


@dataclass(frozen=True)
class PanelSummary:
    rows: int
    dates: int
    stocks: int
    min_date: int
    max_date: int


def require_columns(df: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise DataContractError(f"{name} is missing required columns: {sorted(missing)}")


def assert_unique_keys(df: pd.DataFrame, *, name: str) -> None:
    require_columns(df, KEY_COLUMNS, name=name)
    if df.loc[:, KEY_COLUMNS].isna().any().any():
        raise DataContractError(f"{name} contains missing primary-key values")
    duplicated = df.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, list(KEY_COLUMNS)].head(5).to_dict("records")
        raise DataContractError(f"{name} has duplicate primary keys; examples: {examples}")


def assert_sorted_by_panel_key(df: pd.DataFrame, *, name: str) -> None:
    """Require the stable, documented order used before time-series features."""
    assert_unique_keys(df, name=name)
    actual = df.loc[:, KEY_COLUMNS].reset_index(drop=True)
    expected = actual.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    if not actual.equals(expected):
        raise DataContractError(f"{name} must be sorted by ts_code, trade_date")


def validate_panel(
    df: pd.DataFrame,
    *,
    kind: str,
    require_sorted: bool = False,
    allow_column_subset: bool = False,
) -> PanelSummary:
    """Validate only structural invariants; raw missing observations are permitted."""
    if kind not in {"train", "test"}:
        raise ValueError("kind must be 'train' or 'test'")
    expected = TRAIN_COLUMNS if kind == "train" else X_COLUMNS
    require_columns(df, ("ts_code", "trade_date") if allow_column_subset else expected, name=kind)
    assert_unique_keys(df, name=kind)
    if require_sorted:
        assert_sorted_by_panel_key(df, name=kind)
    dates = pd.to_numeric(df["trade_date"], errors="coerce")
    if dates.isna().any():
        raise DataContractError(f"{kind} trade_date contains non-numeric values")
    available_flags = [column for column in ("flag_limit_up", "flag_limit_down") if column in df]
    if available_flags and not df[available_flags].isin([0, 1]).all().all():
        raise DataContractError(f"{kind} limit flags must be encoded as 0 or 1")
    return PanelSummary(
        rows=len(df),
        dates=int(df["trade_date"].nunique()),
        stocks=int(df["ts_code"].nunique()),
        min_date=int(dates.min()),
        max_date=int(dates.max()),
    )


def assert_finite_predictions(df: pd.DataFrame, *, column: str = "pred") -> None:
    require_columns(df, (column,), name="prediction")
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise DataContractError(f"prediction contains {bad} non-finite values in {column}")
