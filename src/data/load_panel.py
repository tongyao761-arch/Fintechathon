"""The single supported entry point for raw competition data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_contract import X_COLUMNS, TRAIN_COLUMNS, validate_panel


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_data_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "赛题五" / "赛题五数据"


def raw_path(kind: str, root: Path | None = None) -> Path:
    if kind == "train":
        return raw_data_dir(root) / "训练集.csv"
    if kind == "test":
        return raw_data_dir(root) / "测试集_X.csv"
    raise ValueError("kind must be 'train' or 'test'")


def load_panel(
    kind: str,
    *,
    columns: Iterable[str] | None = None,
    root: Path | None = None,
    add_row_id: bool = True,
    sort_for_features: bool = False,
) -> pd.DataFrame:
    """Read raw CSV without mutating it.

    ``row_id`` is the immutable original test-file order and must be retained through
    feature generation; it is how final submissions are restored to required order.
    """
    allowed = TRAIN_COLUMNS if kind == "train" else X_COLUMNS
    usecols = list(columns) if columns is not None else list(allowed)
    missing_keys = set(("ts_code", "trade_date")) - set(usecols)
    if missing_keys:
        raise ValueError(f"columns must include primary keys: {sorted(missing_keys)}")
    df = pd.read_csv(raw_path(kind, root), usecols=usecols)
    if kind == "test" and add_row_id:
        df.insert(0, "row_id", range(len(df)))
    validate_panel(df, kind=kind, require_sorted=False, allow_column_subset=columns is not None)
    if sort_for_features:
        df = df.sort_values(["ts_code", "trade_date"], kind="stable").reset_index(drop=True)
    return df
