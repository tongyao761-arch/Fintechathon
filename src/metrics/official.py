"""Strict local wrapper around the official evaluation algorithm."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.data_contract import DataContractError, assert_finite_predictions, assert_unique_keys, require_columns

KEYS = ["ts_code", "trade_date"]


def _validate_prediction_keys(pred_df: pd.DataFrame, truth_df: pd.DataFrame, x_df: pd.DataFrame) -> None:
    for name, frame in (("prediction", pred_df), ("truth", truth_df), ("features", x_df)):
        assert_unique_keys(frame, name=name)
    pred_keys = pd.MultiIndex.from_frame(pred_df[KEYS])
    for name, frame in (("truth", truth_df), ("features", x_df)):
        other = pd.MultiIndex.from_frame(frame[KEYS])
        # Ordering is deliberately not part of the key contract: callers may
        # score a model output in model order, then restore CSV order only when
        # exporting a submission.
        if len(pred_keys) != len(other) or len(pred_keys.difference(other)) or len(other.difference(pred_keys)):
            pred_only = len(pred_keys.difference(other))
            other_only = len(other.difference(pred_keys))
            raise DataContractError(f"prediction keys and {name} keys differ (prediction-only={pred_only}, {name}-only={other_only})")


def _check_daily_prediction_variation(pred_df: pd.DataFrame) -> None:
    constant_dates = pred_df.groupby("trade_date", sort=False)["pred"].nunique(dropna=False)
    bad = constant_dates[constant_dates < 2]
    if not bad.empty:
        raise DataContractError(f"daily constant predictions are not scoreable; dates: {bad.index[:5].tolist()}")


def score_official(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    x_df: pd.DataFrame,
    *,
    validate_keys: bool = True,
    return_details: bool = False,
) -> dict[str, Any]:
    """Compute the official score while refusing the official script's silent failures.

    Metric loops intentionally mirror ``赛题五/evaluate.py``.  Details are additional
    diagnostics and never change the scalar score.
    """
    require_columns(pred_df, [*KEYS, "pred"], name="prediction")
    require_columns(truth_df, [*KEYS, "y_ret_1d"], name="truth")
    require_columns(x_df, [*KEYS, "flag_limit_up"], name="features")
    assert_finite_predictions(pred_df)
    if not np.isfinite(truth_df["y_ret_1d"].dropna().to_numpy(dtype=float)).all():
        raise DataContractError("truth contains non-finite nonmissing labels")
    if not x_df["flag_limit_up"].isin((0, 1)).all():
        raise DataContractError("flag_limit_up must contain only 0 or 1")
    if validate_keys:
        _validate_prediction_keys(pred_df, truth_df, x_df)
        _check_daily_prediction_variation(pred_df)
    df = pred_df[KEYS + ["pred"]].merge(truth_df[KEYS + ["y_ret_1d"]], on=KEYS, how="inner", validate="one_to_one")
    df = df.merge(x_df[KEYS + ["flag_limit_up"]], on=KEYS, how="inner", validate="one_to_one")

    ic_rows, excess_rows, turnover_rows, top_sets, return_top_sets = [], [], [], [], []
    previous_date, previous = None, None
    for date, group in df.groupby("trade_date", sort=True):
        valid_ic = group.dropna(subset=["y_ret_1d"])
        if len(valid_ic) >= 30:
            if valid_ic["pred"].nunique() < 2 or valid_ic["y_ret_1d"].nunique() < 2:
                raise DataContractError(f"{date}: constant prediction or label in IC sample")
            ic = float(spearmanr(valid_ic["pred"], valid_ic["y_ret_1d"])[0])
            if not np.isfinite(ic):
                raise DataContractError(f"{date}: non-finite daily IC")
            ic_rows.append({"trade_date": date, "ic": ic, "n": len(valid_ic)})

        valid_ret = group[(group.flag_limit_up == 0) & group.y_ret_1d.notna()].copy()
        if len(valid_ret) >= 100:
            valid_ret = valid_ret.sort_values("pred", ascending=False).reset_index(drop=True)
            n_top = max(len(valid_ret) // 10, 1)
            top_ret = float(valid_ret.y_ret_1d.iloc[:n_top].mean())
            market_ret = float(valid_ret.y_ret_1d.mean())
            if not np.isfinite([top_ret, market_ret, top_ret - market_ret]).all():
                raise DataContractError(f"{date}: non-finite daily return")
            excess_rows.append({"trade_date": date, "top_ret": top_ret, "market_ret": market_ret, "excess": top_ret - market_ret, "n": len(valid_ret)})
            return_top_sets.append((date, frozenset(valid_ret.ts_code.iloc[:n_top])))

        valid_turnover = group[group.flag_limit_up == 0].copy()
        if len(valid_turnover) >= 100:
            valid_turnover = valid_turnover.sort_values("pred", ascending=False)
            codes = frozenset(valid_turnover.ts_code.iloc[:max(len(valid_turnover) // 10, 1)])
            top_sets.append((date, codes))
            if previous is not None:
                turnover_rows.append({"trade_date": date, "previous_trade_date": previous_date, "turnover": 1.0 - len(previous & codes) / len(previous | codes)})
            previous_date, previous = date, codes
        else:
            # The official evaluator resets continuity on an ineligible date.
            previous_date, previous = None, None

    ic_df, excess_df, turnover_df = pd.DataFrame(ic_rows), pd.DataFrame(excess_rows), pd.DataFrame(turnover_rows)
    if len(ic_df) < 2 or excess_df.empty or turnover_df.empty:
        raise DataContractError("not enough valid daily observations to calculate all official metrics")
    ic_mean = float(ic_df.ic.mean())
    ic_std = float(ic_df.ic.std(ddof=1))
    annual_excess = float(excess_df.excess.mean() * 252)
    mean_turnover = float(turnover_df.turnover.mean())
    result: dict[str, Any] = {
        "ic_mean": ic_mean, "ic_std": ic_std, "icir": ic_mean / ic_std if ic_std > 0 else 0.0,
        "ic_positive_ratio": float((ic_df.ic > 0).mean()), "annual_excess": annual_excess,
        "top1_annual_ret": float(excess_df.top_ret.mean() * 252), "mean_turnover": mean_turnover,
        "final_score": float(ic_mean * 0.4 + annual_excess * 0.3 + (1 - mean_turnover) * 0.3),
    }
    if not np.isfinite(list(result.values())).all():
        raise DataContractError("non-finite scoring result")
    if return_details:
        result["details"] = {"daily_ic": ic_df, "daily_excess": excess_df, "daily_turnover": turnover_df, "daily_top_sets": pd.DataFrame({"trade_date": [d for d, _ in top_sets], "top_codes": [",".join(sorted(s)) for _, s in top_sets]})}
        result["details"]["daily_return_top_sets"] = pd.DataFrame({"trade_date": [d for d, _ in return_top_sets], "top_codes": [",".join(sorted(s)) for _, s in return_top_sets]})
    return result
