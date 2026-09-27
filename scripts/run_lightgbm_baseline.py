"""Run the fixed raw-field LightGBM baseline on both frozen time splits."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_contract import assert_unique_keys, require_columns
from src.features.baseline_v1 import (
    FEATURE_COLUMNS,
    FEATURE_DEFINITIONS,
    RAW_FEATURE_COLUMNS,
    build_baseline_v1_features,
)
from src.metrics.official import score_official
from src.validation.splits import TimeSplit, get_split


DATA_PATH = ROOT / "赛题五" / "赛题五数据" / "训练集_clean.csv"
OUTPUT_PATH = ROOT / "artifacts" / "lightgbm_baseline_v1" / "summary.json"
EXCLUDED_COLUMNS = [
    "ts_code",
    "trade_date",
    *RAW_FEATURE_COLUMNS,
    "flag_limit_up",
    "flag_limit_down",
    "y_ret_1d",
    "is_price_valid",
    "is_trainable",
    "row_id",
]
REQUIRED_COLUMNS = [
    "ts_code",
    "trade_date",
    *RAW_FEATURE_COLUMNS,
    "flag_limit_up",
    "flag_limit_down",
    "y_ret_1d",
    "is_price_valid",
    "is_trainable",
]
SPLIT_NAMES = ["primary_2023", "oos_2024"]
RANDOM_SEED = 20260927
MODEL_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 100,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "data_random_seed": RANDOM_SEED,
    "feature_fraction_seed": RANDOM_SEED,
    "bagging_seed": RANDOM_SEED,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": -1,
    "verbosity": -1,
}


class PeakMemoryMonitor:
    """Sample this process's resident memory while the baseline is running."""

    def __init__(self, interval_seconds: float = 0.25) -> None:
        self._process = psutil.Process()
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_rss_bytes = self._process.memory_info().rss

    def _sample(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.peak_rss_bytes = max(
                self.peak_rss_bytes, self._process.memory_info().rss
            )

    def __enter__(self) -> "PeakMemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_rss_bytes = max(
            self.peak_rss_bytes, self._process.memory_info().rss
        )


def load_clean_panel(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read only the columns needed by this baseline with compact dtypes."""
    if not path.is_file():
        raise FileNotFoundError(f"clean training data not found: {path}")
    dtypes = {
        "ts_code": "category",
        "trade_date": "int32",
        "open": "float32",
        "high": "float32",
        "low": "float32",
        "close": "float32",
        "vol": "float32",
        "amount": "float32",
        "flag_limit_up": "int8",
        "flag_limit_down": "int8",
        "y_ret_1d": "float32",
        "is_price_valid": "int8",
        "is_trainable": "int8",
    }
    panel = pd.read_csv(
        path,
        usecols=REQUIRED_COLUMNS,
        dtype=dtypes,
        memory_map=True,
    )
    require_columns(panel, REQUIRED_COLUMNS, name="clean training panel")
    assert_unique_keys(panel, name="clean training panel")
    for column in ("flag_limit_up", "flag_limit_down", "is_price_valid", "is_trainable"):
        if not bool(panel[column].isin((0, 1)).all()):
            raise ValueError(f"{column} must contain only 0 or 1")
    return panel


def _load_official_evaluator():
    spec = importlib.util.spec_from_file_location(
        "competition_evaluate",
        ROOT / "赛题五" / "evaluate.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load supplied competition evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def _compare_with_official_evaluator(
    pred: pd.DataFrame,
    truth: pd.DataFrame,
    x: pd.DataFrame,
    local_metrics: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fintechathon_lgbm_eval_") as temp_dir:
        fixture = Path(temp_dir)
        submission_path = fixture / "submission.csv"
        pred.to_csv(submission_path, index=False, float_format="%.17g")
        truth.to_csv(
            fixture / "测试集_Y.csv",
            index=False,
            float_format="%.17g",
        )
        x.to_csv(fixture / "测试集_X.csv", index=False)
        official_raw = _load_official_evaluator()(str(submission_path), str(fixture))
    official = {key: float(value) for key, value in official_raw.items()}
    differences = {
        key: float(local_metrics[key] - official[key]) for key in local_metrics
    }
    max_abs_difference = max(abs(value) for value in differences.values())
    if max_abs_difference > 1e-12:
        raise AssertionError(
            "local scorer diverges from supplied official evaluator: "
            f"{differences}"
        )
    return {
        "official_metrics": official,
        "local_minus_official": differences,
        "max_abs_difference": max_abs_difference,
    }


def split_masks(panel: pd.DataFrame, split: TimeSplit) -> tuple[pd.Series, pd.Series]:
    """Return the eligible training mask and the complete validation mask."""
    train_period, valid_mask = split.masks(panel)
    train_mask = (
        train_period
        & panel["is_trainable"].eq(1)
        & panel["y_ret_1d"].notna()
    )
    labels = panel.loc[train_mask, "y_ret_1d"].to_numpy(dtype=np.float64)
    if not np.isfinite(labels).all():
        raise ValueError(
            f"{split.name} training labels contain "
            f"{int((~np.isfinite(labels)).sum())} non-finite values"
        )
    if not bool(train_mask.any()):
        raise ValueError(f"{split.name} has no eligible training samples")
    if not bool(valid_mask.any()):
        raise ValueError(f"{split.name} has no validation rows")
    return train_mask, valid_mask


def run_split(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    split_name: str,
) -> dict[str, Any]:
    split = get_split(split_name)
    split_started = time.perf_counter()
    train_period, _ = split.masks(panel)
    train_mask, valid_mask = split_masks(panel, split)

    train_x = features.loc[train_mask, FEATURE_COLUMNS]
    train_y = panel.loc[train_mask, "y_ret_1d"]
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(train_x, train_y, feature_name=FEATURE_COLUMNS)
    if tuple(model.feature_name_) != FEATURE_COLUMNS:
        raise AssertionError(
            f"model feature order changed: {model.feature_name_!r}"
        )
    del train_x, train_y

    valid_x = features.loc[valid_mask, FEATURE_COLUMNS]
    prediction_values = model.predict(valid_x)
    del valid_x, model
    if len(prediction_values) != int(valid_mask.sum()):
        raise AssertionError("validation prediction row count changed")
    if not np.isfinite(prediction_values).all():
        raise ValueError(
            f"{split_name} predictions contain "
            f"{int((~np.isfinite(prediction_values)).sum())} non-finite values"
        )
    prediction_sha256 = hashlib.sha256(
        np.asarray(prediction_values, dtype="<f8").tobytes()
    ).hexdigest()

    valid = panel.loc[
        valid_mask,
        ["ts_code", "trade_date", "y_ret_1d", "flag_limit_up"],
    ].copy()
    pred = valid[["ts_code", "trade_date"]].copy()
    pred["pred"] = prediction_values
    truth = valid[["ts_code", "trade_date", "y_ret_1d"]].copy()
    truth["y_ret_1d"] = truth["y_ret_1d"].astype("float64")
    x = valid[["ts_code", "trade_date", "flag_limit_up"]]
    metrics_with_details = score_official(
        pred,
        truth,
        x,
        return_details=True,
    )
    metrics_with_details.pop("details")
    official_comparison = _compare_with_official_evaluator(
        pred,
        truth,
        x,
        metrics_with_details,
    )

    elapsed = time.perf_counter() - split_started
    return {
        "split_name": split.name,
        "dates": {
            "train_start": split.train_start,
            "train_end": split.train_end,
            "purge_date": split.purge_date,
            "valid_start": split.valid_start,
            "valid_end": split.valid_end,
        },
        "split_train_rows": int(train_period.sum()),
        "train_samples": int(train_mask.sum()),
        "purge_rows": int(panel["trade_date"].eq(split.purge_date).sum()),
        "valid_prediction_rows": len(pred),
        "prediction_coverage": 1.0,
        "prediction_sha256": prediction_sha256,
        "metrics": metrics_with_details,
        "official_comparison": official_comparison,
        "elapsed_seconds": elapsed,
    }


def environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "architecture": platform.architecture()[0],
        "lightgbm": lgb.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def main() -> None:
    started_at = datetime.now().astimezone()
    total_started = time.perf_counter()
    with PeakMemoryMonitor() as memory_monitor:
        panel = load_clean_panel()
        raw_rows = len(panel)
        raw_dates = int(panel["trade_date"].nunique())
        raw_stocks = int(panel["ts_code"].nunique())
        features = build_baseline_v1_features(panel)
        if len(features) != raw_rows or not features.index.equals(panel.index):
            raise AssertionError("feature construction changed panel alignment")
        feature_values = features.to_numpy(dtype=np.float32, copy=False)
        feature_infinite_counts = {
            column: int(np.isinf(feature_values[:, position]).sum())
            for position, column in enumerate(FEATURE_COLUMNS)
        }
        if any(feature_infinite_counts.values()):
            raise ValueError(
                f"formal baseline features contain infinity: {feature_infinite_counts}"
            )
        feature_missing_counts = {
            column: int(features[column].isna().sum())
            for column in FEATURE_COLUMNS
        }
        results = [run_split(panel, features, name) for name in SPLIT_NAMES]
    total_elapsed = time.perf_counter() - total_started
    finished_at = datetime.now().astimezone()

    summary = {
        "experiment_id": "lightgbm_ten_features_v1",
        "purpose": (
            "First formal ten-feature baseline using frozen time splits and the "
            "official scoring contract."
        ),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "data": {
            "file": str(DATA_PATH),
            "file_size_bytes": DATA_PATH.stat().st_size,
            "raw_rows": raw_rows,
            "dates": raw_dates,
            "stocks": raw_stocks,
        },
        "features": list(FEATURE_COLUMNS),
        "feature_definitions": dict(FEATURE_DEFINITIONS),
        "feature_rows": len(features),
        "feature_index_preserved": features.index.equals(panel.index),
        "feature_missing_counts": feature_missing_counts,
        "feature_infinite_counts": feature_infinite_counts,
        "excluded_columns": EXCLUDED_COLUMNS,
        "random_seed": RANDOM_SEED,
        "model": "lightgbm.LGBMRegressor",
        "model_params": MODEL_PARAMS,
        "splits": results,
        "environment": environment_versions(),
        "resources": {
            "elapsed_seconds": total_elapsed,
            "peak_process_rss_mb": memory_monitor.peak_rss_bytes / (1024**2),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
