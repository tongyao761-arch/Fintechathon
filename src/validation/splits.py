"""Frozen expanding-window time splits with an explicit one-day purge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    name: str
    train_start: int
    train_end: int
    purge_date: int
    valid_start: int
    valid_end: int

    def masks(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        if "trade_date" not in df:
            raise KeyError("data frame needs trade_date")
        date = pd.to_numeric(df.trade_date, errors="raise")
        train = date.between(self.train_start, self.train_end)
        valid = date.between(self.valid_start, self.valid_end)
        if bool((train & valid).any()) or bool((date.eq(self.purge_date) & (train | valid)).any()):
            raise AssertionError("time split violates purge or non-overlap invariant")
        return train, valid

    def describe(self, df: pd.DataFrame, label_column: str = "y_ret_1d") -> dict:
        train, valid = self.masks(df)
        result = {"name": self.name, "train_rows": int(train.sum()), "valid_rows": int(valid.sum()), "purge_rows": int(df.trade_date.eq(self.purge_date).sum())}
        if label_column in df:
            result["train_labeled_rows"] = int(df.loc[train, label_column].notna().sum())
            result["valid_labeled_rows"] = int(df.loc[valid, label_column].notna().sum())
        return result


def split_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "splits.yaml"


def load_splits(path: Path | None = None) -> dict[str, TimeSplit]:
    # JSON is valid YAML; JSON keeps this critical configuration dependency-free.
    specs = json.loads((path or split_config_path()).read_text(encoding="utf-8"))
    return {name: TimeSplit(name=name, **values) for name, values in specs.items()}


def get_split(name: str, path: Path | None = None) -> TimeSplit:
    try:
        return load_splits(path)[name]
    except KeyError as exc:
        raise KeyError(f"unknown split {name!r}; use one of {sorted(load_splits(path))}") from exc
