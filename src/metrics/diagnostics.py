"""Non-official summaries derived from a scored result."""

from __future__ import annotations

import pandas as pd


def monthly_score_diagnostics(details: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for metric, key in (("ic", "daily_ic"), ("excess", "daily_excess"), ("turnover", "daily_turnover")):
        frame = details[key].copy()
        frame["month"] = frame.trade_date.astype(str).str.slice(0, 6)
        value = metric if metric in frame else metric
        frames.append(frame.groupby("month")[value].mean().rename(metric))
    return pd.concat(frames, axis=1).reset_index()
