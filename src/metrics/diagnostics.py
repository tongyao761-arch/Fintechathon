"""Non-official summaries derived from a scored result."""

from __future__ import annotations

import pandas as pd

from src.data.data_contract import DataContractError, assert_unique_keys


def monthly_score_diagnostics(details: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for metric, key in (("ic", "daily_ic"), ("excess", "daily_excess"), ("turnover", "daily_turnover")):
        frame = details[key].copy()
        frame["month"] = frame.trade_date.astype(str).str.slice(0, 6)
        value = metric if metric in frame else metric
        frames.append(frame.groupby("month")[value].mean().rename(metric))
    result = pd.concat(frames, axis=1).reset_index()
    result["annual_excess"] = result.excess * 252
    result["ic_contribution"] = result.ic * 0.4
    result["excess_contribution"] = result.annual_excess * 0.3
    result["stability_contribution"] = (1 - result.turnover) * 0.3
    result["score"] = result[["ic_contribution", "excess_contribution", "stability_contribution"]].sum(axis=1, min_count=3)
    return result


def baseline_diagnostics(pred: pd.DataFrame, truth: pd.DataFrame,
                         quality: pd.DataFrame, details: dict) -> tuple[dict, dict]:
    """Observe missing samples without changing the official scoring universe.

    Label availability is diagnostic only. The alternative turnover filter uses
    current-row price validity, never the future label's availability.
    """
    keys = ["ts_code", "trade_date"]
    assert_unique_keys(quality, name="diagnostic quality")
    joined = pred.merge(truth, on=keys, validate="one_to_one")
    joined = joined.merge(quality, on=keys, how="outer", validate="one_to_one", indicator=True)
    if len(joined) != len(pred) or not joined["_merge"].eq("both").all():
        raise DataContractError("diagnostic keys differ from prediction keys")
    top_lookup = {}
    for kind, key in (("turnover", "daily_top_sets"), ("return", "daily_return_top_sets")):
        top_lookup[kind] = {r.trade_date: set(r.top_codes.split(","))
                            for r in details[key].itertuples(index=False)}
    rows, diagnostic_turnover = [], []
    previous_date, previous = None, None
    for date, group in joined.groupby("trade_date", sort=True):
        for kind, lookup in top_lookup.items():
            if date not in lookup:
                continue
            top = group[group.ts_code.isin(lookup[date])]
            count = len(top)
            row = {"trade_date": date, "top_type": kind, "top_count": count,
                   "missing_label_count": int(top.y_ret_1d.isna().sum()),
                   "invalid_price_count": int(top.is_price_valid.eq(0).sum()),
                   "all_features_missing_count": int(top.baseline_features_all_missing.sum())}
            for column in ("missing_label", "invalid_price", "all_features_missing"):
                row[column + "_fraction"] = row[column + "_count"] / count
            rows.append(row)
        eligible = group[group.flag_limit_up.eq(0) & group.is_price_valid.eq(1)]
        if len(eligible) < 100:
            previous_date, previous = None, None
            continue
        top = eligible.sort_values("pred", ascending=False).iloc[:len(eligible) // 10]
        current = set(top.ts_code)
        if previous is not None:
            diagnostic_turnover.append({"trade_date": date, "previous_trade_date": previous_date,
                                        "turnover": 1 - len(previous & current) / len(previous | current)})
        previous_date, previous = date, current
    daily = pd.DataFrame(rows)
    alternative = pd.DataFrame(diagnostic_turnover, columns=["trade_date", "previous_trade_date", "turnover"])
    summary = {"note": "Diagnostics only; predictions and official scores are unchanged.",
               "price_valid_only_turnover": None if alternative.empty else float(alternative.turnover.mean()),
               "top_groups": {}}
    for kind, frame in daily.groupby("top_type"):
        totals = {c: int(frame[c].sum()) for c in (
            "top_count", "missing_label_count", "invalid_price_count", "all_features_missing_count")}
        for column in ("missing_label", "invalid_price", "all_features_missing"):
            totals[column + "_fraction"] = totals[column + "_count"] / totals["top_count"]
        summary["top_groups"][kind] = totals
    return summary, {"daily_missing_diagnostics": daily, "daily_price_valid_turnover": alternative,
                     "monthly_metrics": monthly_score_diagnostics(details)}
