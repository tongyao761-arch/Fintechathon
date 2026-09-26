"""Validation and deterministic export for competition submissions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_contract import DataContractError, assert_finite_predictions, assert_unique_keys, require_columns

SUBMISSION_COLUMNS = ["ts_code", "trade_date", "pred"]


def validate_submission(pred_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    if list(pred_df.columns) != SUBMISSION_COLUMNS:
        raise DataContractError(f"submission columns must be exactly {SUBMISSION_COLUMNS}; got {list(pred_df.columns)}")
    require_columns(pred_df, SUBMISSION_COLUMNS, name="submission")
    require_columns(test_df, ["ts_code", "trade_date"], name="test panel")
    assert_unique_keys(pred_df, name="submission")
    assert_unique_keys(test_df, name="test panel")
    assert_finite_predictions(pred_df)
    pred_keys = pd.MultiIndex.from_frame(pred_df[["ts_code", "trade_date"]])
    test_keys = pd.MultiIndex.from_frame(test_df[["ts_code", "trade_date"]])
    # Prediction frames may arrive sorted by date/model batch rather than raw
    # test-file order; export_submission restores that order after this check.
    if len(pred_keys) != len(test_keys) or len(pred_keys.difference(test_keys)) or len(test_keys.difference(pred_keys)):
        raise DataContractError(f"submission keys differ from test panel (submission-only={len(pred_keys.difference(test_keys))}, test-only={len(test_keys.difference(pred_keys))})")
    daily = pred_df.groupby("trade_date").pred.agg(["min", "max", "mean", "std", "nunique"])
    top_missing = 0
    if "close" in test_df:
        joined = pred_df.merge(test_df[["ts_code", "trade_date", "close"]], on=["ts_code", "trade_date"], validate="one_to_one")
        for _, group in joined.groupby("trade_date"):
            top = group.nlargest(max(len(group) // 10, 1), "pred")
            top_missing += int(top.close.isna().sum())
    return {"rows": len(pred_df), "dates": int(pred_df.trade_date.nunique()), "daily_prediction_summary": daily, "top_decile_missing_close": top_missing}


def export_submission(pred_df: pd.DataFrame, test_df: pd.DataFrame, output_path: Path) -> dict:
    """Validate and restore immutable test CSV order; do not impute or drop any row."""
    report = validate_submission(pred_df, test_df)
    order = test_df[["ts_code", "trade_date"]].copy()
    order["row_id"] = test_df["row_id"] if "row_id" in test_df else np.arange(len(test_df))
    result = order.merge(pred_df[SUBMISSION_COLUMNS], on=["ts_code", "trade_date"], validate="one_to_one").sort_values("row_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result[SUBMISSION_COLUMNS].to_csv(output_path, index=False)
    return report
