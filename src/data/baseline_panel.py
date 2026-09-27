"""Rebuild the baseline input contract from unmodified competition data."""

from __future__ import annotations

import csv
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from .data_contract import DataContractError, TRAIN_COLUMNS, assert_sorted_by_panel_key

PRICE_COLUMNS = ["open", "high", "low", "close"]
RAW_NUMERIC_COLUMNS = [*PRICE_COLUMNS, "vol", "amount"]


def load_raw_baseline_panel(path: Path) -> pd.DataFrame:
    """Keep scoring labels float64; quality flags use only current-row prices.

    The caller explicitly converts training labels to float32. Missing prices
    and features are never filled and no source row is removed or reordered.
    """
    panel = pd.read_csv(path, usecols=list(TRAIN_COLUMNS), dtype={
        "ts_code": "category", "trade_date": "int32",
        **{column: "float64" for column in RAW_NUMERIC_COLUMNS},
        "y_ret_1d": "float64",
    })
    assert_sorted_by_panel_key(panel, name="raw baseline panel")
    for column in ("flag_limit_up", "flag_limit_down"):
        if not panel[column].isin((0, 1)).all():
            raise DataContractError(f"{column} must contain only 0 or 1")
        panel[column] = panel[column].astype("int8")
    prices = panel[PRICE_COLUMNS]
    valid = (np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
             & panel.high.ge(prices.max(axis=1))
             & panel.low.le(prices.min(axis=1)))
    panel["is_price_valid"] = valid.astype("int8")
    panel["is_trainable"] = (valid & panel.y_ret_1d.notna()).astype("int8")
    panel[RAW_NUMERIC_COLUMNS] = panel[RAW_NUMERIC_COLUMNS].astype("float32")
    return panel


def compare_clean_input(panel: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Migration gate: compare the complete old model input, including labels."""
    current = panel.loc[:, clean.columns].copy()
    current["y_ret_1d"] = current["y_ret_1d"].astype("float32")
    pd.testing.assert_frame_equal(current, clean, check_exact=True)
    return {"status": "passed", "rows": len(panel), "columns": list(clean.columns),
            "keys_flags_raw_float32_inputs_training_labels_equal": True}


def extract_truth_files(source: Path, targets: dict[Path, tuple[int, int]]) -> dict[str, int]:
    """Extract keys and original label tokens in one pass, without float conversion.

    Files are exclusive-create. Exact key coverage is checked by the scoring
    pipeline against each validation panel before these files can be accepted.
    """
    counts = {str(path): 0 for path in targets}
    with ExitStack() as stack:
        writers = []
        for path, (start, end) in targets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("x", encoding="utf-8", newline=""))
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["ts_code", "trade_date", "y_ret_1d"])
            writers.append((path, start, end, writer))
        handle = stack.enter_context(source.open(encoding="utf-8-sig", newline=""))
        reader = csv.reader(handle)
        header = next(reader)
        code_i, date_i, label_i = [header.index(c) for c in ("ts_code", "trade_date", "y_ret_1d")]
        for row in reader:
            date = int(row[date_i])
            for path, start, end, writer in writers:
                if start <= date <= end:
                    writer.writerow([row[code_i], row[date_i], row[label_i]])
                    counts[str(path)] += 1
    return counts
