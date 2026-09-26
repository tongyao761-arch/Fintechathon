"""Chunked, reproducible audit of the original CSV files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .load_panel import project_root, raw_path

CHUNK_SIZE = 250_000
NUMERIC_NONNEGATIVE = ("open", "high", "low", "close", "vol", "amount")


def _as_int_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def profile_csv(path: Path, *, has_label: bool) -> dict:
    usecols = ["ts_code", "trade_date", *NUMERIC_NONNEGATIVE, "flag_limit_up", "flag_limit_down"]
    if has_label:
        usecols.append("y_ret_1d")
    rows = 0
    missing = Counter()
    dates = Counter()
    stocks = Counter()
    flags = Counter()
    invalid = Counter()
    adjacent_duplicate_keys = 0
    sort_violations = 0
    label_pairs_checked = 0
    label_mismatches = 0
    label_abs_error_max = 0.0
    previous: pd.DataFrame | None = None

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE):
        rows += len(chunk)
        for col in usecols:
            missing[col] += int(chunk[col].isna().sum())
        dates.update(chunk["trade_date"].value_counts().astype(int).to_dict())
        stocks.update(chunk["ts_code"].value_counts().astype(int).to_dict())
        flags.update({"flag_limit_up=0": int((chunk.flag_limit_up == 0).sum()), "flag_limit_up=1": int((chunk.flag_limit_up == 1).sum())})
        flags.update({"flag_limit_down=0": int((chunk.flag_limit_down == 0).sum()), "flag_limit_down=1": int((chunk.flag_limit_down == 1).sum())})
        flags["both_limit_flags=1"] += int(((chunk.flag_limit_up == 1) & (chunk.flag_limit_down == 1)).sum())
        for col in ("open", "high", "low", "close"):
            values = chunk[col].to_numpy(dtype=float)
            invalid[f"{col}<=0"] += int(((values <= 0) & np.isfinite(values)).sum())
            invalid[f"{col}_nonfinite"] += int((~np.isfinite(values)).sum())
        for col in ("vol", "amount"):
            values = chunk[col].to_numpy(dtype=float)
            invalid[f"{col}<0"] += int(((values < 0) & np.isfinite(values)).sum())
            invalid[f"{col}_nonfinite"] += int((~np.isfinite(values)).sum())
        invalid["high<low"] += int((chunk.high < chunk.low).sum())
        invalid["high<max(open,close)"] += int((chunk.high < chunk[["open", "close"]].max(axis=1)).sum())
        invalid["low>min(open,close)"] += int((chunk.low > chunk[["open", "close"]].min(axis=1)).sum())
        invalid["flag_not_0_or_1"] += int((~chunk[["flag_limit_up", "flag_limit_down"]].isin([0, 1]).all(axis=1)).sum())

        combined = chunk if previous is None else pd.concat([previous, chunk], ignore_index=True)
        key_pairs = combined[["ts_code", "trade_date"]]
        duplicate = key_pairs.duplicated(keep=False)
        adjacent_duplicate_keys += int(duplicate.iloc[:-1].sum())
        stock = combined.ts_code.astype(str).to_numpy()
        date = combined.trade_date.to_numpy()
        in_order = (stock[1:] > stock[:-1]) | ((stock[1:] == stock[:-1]) & (date[1:] > date[:-1]))
        sort_violations += int((~in_order).sum())
        if has_label:
            same_stock = combined.ts_code.eq(combined.ts_code.shift(-1))
            expected = combined.close.shift(-1) / combined.close - 1
            valid = same_stock & combined.y_ret_1d.notna() & expected.notna()
            valid.iloc[-1] = False  # the final row is checked with the next chunk
            errors = (combined.loc[valid, "y_ret_1d"] - expected.loc[valid]).abs()
            label_pairs_checked += len(errors)
            if len(errors):
                label_mismatches += int((errors > 1e-10).sum())
                label_abs_error_max = max(label_abs_error_max, float(errors.max()))
        previous = combined.tail(1).copy()

    result = {
        "path": str(path), "rows": rows, "stocks": len(stocks), "dates": len(dates),
        "date_range": [int(min(dates)), int(max(dates))],
        "date_rows_min_median_max": [int(min(dates.values())), float(np.median(list(dates.values()))), int(max(dates.values()))],
        "stock_rows_min_median_max": [int(min(stocks.values())), float(np.median(list(stocks.values()))), int(max(stocks.values()))],
        "missing": _as_int_dict(missing), "flags": _as_int_dict(flags), "invalid": _as_int_dict(invalid),
        "adjacent_duplicate_keys": adjacent_duplicate_keys, "sort_violations": sort_violations,
        "label_pairs_checked": label_pairs_checked, "label_mismatches_gt_1e_10": label_mismatches,
        "label_abs_error_max": label_abs_error_max,
        "first_dates": sorted(map(int, dates))[:5], "last_dates": sorted(map(int, dates))[-5:],
    }
    return result


def render_markdown(report: dict) -> str:
    lines = ["# 原始数据审计报告", "", "本报告由 `python -m src.data.profile_raw` 从原始 CSV 只读生成。", ""]
    lines += ["| 数据集 | 行数 | 股票数 | 交易日数 | 日期范围 |", "|---|---:|---:|---:|---|"]
    for name in ("train", "test"):
        item = report[name]
        lines.append(f"| {name} | {item['rows']:,} | {item['stocks']:,} | {item['dates']:,} | {item['date_range'][0]} 至 {item['date_range'][1]} |")
    lines += ["", "## 核心结论", ""]
    train, test = report["train"], report["test"]
    lines += [
        f"- 训练集标签可复核连续样本 {train['label_pairs_checked']:,} 对；误差大于 1e-10 的记录为 {train['label_mismatches_gt_1e_10']:,}。",
        f"- 主键相邻重复：训练 {train['adjacent_duplicate_keys']}，测试 {test['adjacent_duplicate_keys']}；排序违例：训练 {train['sort_violations']}，测试 {test['sort_violations']}。",
        f"- 训练集 OHLC 缺失：{train['missing']['open']:,} 行；训练标签缺失：{train['missing']['y_ret_1d']:,} 行。缺失是面板状态的一部分，不能从测试键中删除。",
        f"- 同时标记涨停和跌停：训练 {train['flags']['both_limit_flags=1']} 行，测试 {test['flags']['both_limit_flags=1']} 行；保留原值，官方口径仅据 `flag_limit_up` 过滤。",
        f"- 训练末日 {report['cross_boundary_labels']['train_last_date']} 的非空标签中，有 {report['cross_boundary_labels']['labels_checked']:,} 条与测试首日 {report['cross_boundary_labels']['test_first_date']} 收盘价一致；因此任何以 2024/2025 为边界的切分都必须 purge 训练末尾一天。",
        "- 特征只能使用当日及历史信息；不允许后向填充、未来 `shift(-1)` 或用验证/测试期拟合统计量。",
    ]
    return "\n".join(lines) + "\n"


def cross_boundary_label_check(train_path: Path, test_path: Path) -> dict:
    """Prove whether the terminal train labels already consume test-period prices."""
    train_dates = pd.read_csv(train_path, usecols=["trade_date"])
    test_dates = pd.read_csv(test_path, usecols=["trade_date"])
    last_train, first_test = int(train_dates.trade_date.max()), int(test_dates.trade_date.min())
    train = pd.read_csv(train_path, usecols=["ts_code", "trade_date", "close", "y_ret_1d"])
    test = pd.read_csv(test_path, usecols=["ts_code", "trade_date", "close"])
    boundary = train[train.trade_date.eq(last_train)].merge(
        test[test.trade_date.eq(first_test)], on="ts_code", suffixes=("_train", "_test"), validate="one_to_one"
    )
    expected = boundary.close_test / boundary.close_train - 1
    valid = boundary.y_ret_1d.notna() & expected.notna()
    error = (boundary.loc[valid, "y_ret_1d"] - expected.loc[valid]).abs()
    return {
        "train_last_date": last_train, "test_first_date": first_test, "labels_checked": int(valid.sum()),
        "mismatches_gt_1e_10": int((error > 1e-10).sum()), "max_abs_error": float(error.max()) if len(error) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the untouched competition CSV files.")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "artifacts")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"train": profile_csv(raw_path("train"), has_label=True), "test": profile_csv(raw_path("test"), has_label=False)}
    report["cross_boundary_labels"] = cross_boundary_label_check(raw_path("train"), raw_path("test"))
    train_codes = set(pd.read_csv(raw_path("train"), usecols=["ts_code"])["ts_code"].unique())
    test_codes = set(pd.read_csv(raw_path("test"), usecols=["ts_code"])["ts_code"].unique())
    report["stock_universe"] = {"train_only": len(train_codes - test_codes), "test_only": len(test_codes - train_codes), "intersection": len(train_codes & test_codes)}
    (args.output_dir / "data_profile.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "data_profile.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'data_profile.json'} and {args.output_dir / 'data_profile.md'}")


if __name__ == "__main__":
    main()
