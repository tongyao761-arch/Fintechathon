"""Small explicit helpers used in feature reviews."""

from __future__ import annotations

import pandas as pd


def assert_no_future_dates(feature_df: pd.DataFrame, decision_dates: pd.Series, *, feature_date_column: str = "trade_date") -> None:
    """A feature stamped later than its prediction date is a hard leakage failure."""
    if len(feature_df) != len(decision_dates):
        raise ValueError("feature and decision date lengths differ")
    if (feature_df[feature_date_column].to_numpy() > decision_dates.to_numpy()).any():
        raise ValueError("feature contains information later than its decision date")


def assert_fit_on_train_only(fit_dates: pd.Series, train_mask: pd.Series) -> None:
    if len(fit_dates) != len(train_mask) or bool((~train_mask).any() and fit_dates[~train_mask].notna().any()):
        raise ValueError("a fitted transformation may only consume training rows")
