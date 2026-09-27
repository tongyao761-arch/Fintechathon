"""Run and retain the fixed ten-feature baseline with strict official acceptance."""

from __future__ import annotations

import hashlib
import argparse
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
from src.data.baseline_panel import load_raw_baseline_panel, compare_clean_input, extract_truth_files
from src.features.baseline_v1 import (
    FEATURE_COLUMNS,
    FEATURE_DEFINITIONS,
    RAW_FEATURE_COLUMNS,
    build_baseline_v1_features,
)
from src.metrics.official import score_official
from src.metrics.diagnostics import baseline_diagnostics
from src.validation.experiment import experiment_run, provenance, sha256_file, write_json
from src.validation.splits import TimeSplit, get_split


DATA_PATH = ROOT / "赛题五" / "赛题五数据" / "训练集_clean.csv"
OUTPUT_PATH = ROOT / "artifacts" / "lightgbm_baseline_v1" / "summary.json"
RAW_DATA_PATH = ROOT / "赛题五" / "赛题五数据" / "训练集.csv"
EXPERIMENT_ROOT = ROOT / "artifacts" / "experiments"
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
    return compare_metrics(local_metrics, official_raw)


def compare_metrics(local_metrics: dict, official_raw: dict) -> dict:
    """Never allow NaN/inf to pass a tolerance comparison."""
    if not local_metrics or set(local_metrics) != set(official_raw):
        raise AssertionError("local and official metric names differ or are empty")
    official = {key: float(value) for key, value in official_raw.items()}
    for name, values in (("local", local_metrics), ("official", official)):
        if not np.isfinite(list(values.values())).all():
            raise AssertionError(f"{name} metrics contain non-finite values")
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


def score_saved_inputs(fixture: Path) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """Both implementations score precisely the same default-CSV-parsed inputs."""
    pred = pd.read_csv(fixture / "submission.csv")
    truth = pd.read_csv(fixture / "测试集_Y.csv")
    x = pd.read_csv(fixture / "测试集_X.csv")
    local = score_official(pred, truth, x, return_details=True)
    scalars = {k: v for k, v in local.items() if k != "details"}
    official = _load_official_evaluator()(str(fixture / "submission.csv"), str(fixture))
    return local, compare_metrics(scalars, official), pred, truth


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
    *,
    output_dir: Path,
    reference: dict,
) -> dict[str, Any]:
    split = get_split(split_name)
    split_started = time.perf_counter()
    train_period, _ = split.masks(panel)
    train_mask, valid_mask = split_masks(panel, split)
    if not features.index.equals(panel.index):
        raise AssertionError("feature index is not aligned with the panel")

    train_x = features.loc[train_mask, FEATURE_COLUMNS]
    train_y = panel.loc[train_mask, "y_ret_1d"].astype("float32")
    if not np.isfinite(train_y.to_numpy()).all():
        raise ValueError("training labels overflow float32")
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(train_x, train_y, feature_name=FEATURE_COLUMNS)
    if tuple(model.feature_name_) != FEATURE_COLUMNS:
        raise AssertionError(
            f"model feature order changed: {model.feature_name_!r}"
        )
    del train_x, train_y

    valid_x = features.loc[valid_mask, FEATURE_COLUMNS]
    prediction_values = model.predict(valid_x)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=False)
    model_path = model_dir / "lightgbm.txt"
    model.booster_.save_model(str(model_path))
    reloaded = lgb.Booster(model_file=str(model_path))
    if not np.array_equal(prediction_values, reloaded.predict(valid_x)):
        raise AssertionError("saved model reload changed predictions")
    del valid_x, model, reloaded
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
    if prediction_sha256 != reference["prediction_sha256"]:
        raise AssertionError(f"{split_name}: original prediction hash changed; investigate before acceptance")
    if int(train_mask.sum()) != reference["train_samples"] or int(valid_mask.sum()) != reference["valid_prediction_rows"]:
        raise AssertionError(f"{split_name}: frozen sample counts changed")

    valid = panel.loc[
        valid_mask,
        ["ts_code", "trade_date", "y_ret_1d", "flag_limit_up"],
    ].copy()
    pred = valid[["ts_code", "trade_date"]].copy()
    pred["pred"] = prediction_values
    x = valid[["ts_code", "trade_date", "flag_limit_up"]]
    pred.to_parquet(output_dir / "predictions.parquet", index=False)
    if not np.array_equal(pd.read_parquet(output_dir / "predictions.parquet").pred.to_numpy(), prediction_values):
        raise AssertionError("saved predictions changed values")
    fixture = output_dir / "evaluate_input"
    pred.to_csv(fixture / "submission.csv", index=False, float_format="%.17g")
    x.to_csv(fixture / "测试集_X.csv", index=False)
    metrics_with_details, official_comparison, scored_pred, truth = score_saved_inputs(fixture)
    aligned_truth = valid[["ts_code", "trade_date", "y_ret_1d"]].merge(
        truth, on=["ts_code", "trade_date"], validate="one_to_one", suffixes=("_raw", "_fixture"))
    if len(aligned_truth) != len(valid) or not np.array_equal(
        aligned_truth.y_ret_1d_raw, aligned_truth.y_ret_1d_fixture, equal_nan=True
    ):
        raise AssertionError("extracted truth differs from original float64 labels")
    details = metrics_with_details.pop("details")
    quality = panel.loc[valid_mask, ["ts_code", "trade_date", "flag_limit_up", "is_price_valid"]].copy()
    quality["baseline_features_all_missing"] = features.loc[valid_mask].isna().all(axis=1)
    diagnostics, diagnostic_tables = baseline_diagnostics(scored_pred, truth, quality, details)
    for name, frame in {**details, **diagnostic_tables}.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.17g")
    write_json(output_dir / "diagnostics.json", diagnostics)
    contributions = {"ic": metrics_with_details["ic_mean"] * 0.4,
                     "excess": metrics_with_details["annual_excess"] * 0.3,
                     "stability": (1 - metrics_with_details["mean_turnover"]) * 0.3}

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
        "model_reload_predictions_equal": True,
        "original_prediction_hash_matches": True,
        "metrics": metrics_with_details,
        "score_contributions": contributions,
        "diagnostics": diagnostics,
        "metrics_minus_v1": {key: value - reference["metrics"][key]
                             for key, value in metrics_with_details.items()},
        "file_sha256": {str(p.relative_to(output_dir)).replace("\\", "/"): sha256_file(p)
                        for p in sorted(output_dir.rglob("*")) if p.is_file()},
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="baseline_v1_1")
    parser.add_argument("--split", choices=[*SPLIT_NAMES, "all"], default="primary_2023")
    parser.add_argument("--verify-clean", action="store_true",
                        help="Require full equality with the legacy clean model input")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    names = SPLIT_NAMES if args.split == "all" else [args.split]
    reference = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    references = {s["split_name"]: s for s in reference["splits"]}
    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    with experiment_run(EXPERIMENT_ROOT, args.experiment_id) as output:
        print(f"Experiment directory: {output}", flush=True)
        metadata = provenance(ROOT, RAW_DATA_PATH)
        metadata["reference_summary_sha256"] = sha256_file(OUTPUT_PATH)
        write_json(output / "provenance.json", metadata)
        with PeakMemoryMonitor() as memory_monitor:
            panel = load_raw_baseline_panel(RAW_DATA_PATH)
            migration = {"status": "not_requested", "requires_clean_file": False}
            if args.verify_clean:
                clean = load_clean_panel()
                migration = compare_clean_input(panel, clean)
                for name in names:
                    old_masks, new_masks = split_masks(clean, get_split(name)), split_masks(panel, get_split(name))
                    if not all(a.equals(b) for a, b in zip(old_masks, new_masks)):
                        raise AssertionError(f"{name}: migration changed split masks")
                migration["split_masks_equal"] = True
                migration["clean_sha256"] = sha256_file(DATA_PATH)
                del clean
            write_json(output / "input_comparison.json", migration)
            print("Raw input loaded and migration checks finished", flush=True)
            features = build_baseline_v1_features(panel)
            if not features.index.equals(panel.index):
                raise AssertionError("feature construction changed panel alignment")
            infinite = {c: int(np.isinf(features[c]).sum()) for c in FEATURE_COLUMNS}
            if any(infinite.values()):
                raise ValueError(f"features contain infinity: {infinite}")
            targets = {}
            for name in names:
                split = get_split(name)
                targets[output / name / "evaluate_input" / "测试集_Y.csv"] = (split.valid_start, split.valid_end)
            extract_truth_files(RAW_DATA_PATH, targets)
            print("Original truth tokens extracted; training begins", flush=True)
            results = []
            for name in names:
                result = run_split(panel, features, name, output_dir=output / name, reference=references[name])
                results.append(result)
                write_json(output / name / "summary.json", result)
                print(json.dumps({"split": name, "metrics": result["metrics"],
                                  "prediction_hash_matches_v1": True}), flush=True)
        # Detect input edits during the run; a mixed data snapshot is not acceptable.
        if sha256_file(RAW_DATA_PATH) != metadata["data"]["sha256"]:
            raise AssertionError("raw data changed during experiment")
        summary = {
            "experiment_id": args.experiment_id, "baseline_version": "baseline_v1_1",
            "run_id": output.name, "started_at": started_at.isoformat(),
            "finished_at": datetime.now().astimezone().isoformat(),
            "data": {**metadata["data"], "raw_rows": len(panel),
                     "dates": int(panel.trade_date.nunique()), "stocks": int(panel.ts_code.nunique())},
            "input_comparison": migration,
            "features": list(FEATURE_COLUMNS), "feature_definitions": dict(FEATURE_DEFINITIONS),
            "feature_missing_counts": {c: int(features[c].isna().sum()) for c in FEATURE_COLUMNS},
            "feature_infinite_counts": infinite, "feature_index_preserved": True,
            "excluded_columns": EXCLUDED_COLUMNS, "random_seed": RANDOM_SEED,
            "model": "lightgbm.LGBMRegressor", "model_params": MODEL_PARAMS,
            "environment": environment_versions(), "splits": results,
            "reference_summary_sha256": metadata["reference_summary_sha256"],
            "provenance_sha256": sha256_file(output / "provenance.json"),
            "resources": {"elapsed_seconds": time.perf_counter() - started,
                          "peak_process_rss_mb": memory_monitor.peak_rss_bytes / (1024**2),
                          "logical_cpu_count": psutil.cpu_count(logical=True),
                          "physical_cpu_count": psutil.cpu_count(logical=False)},
        }
        write_json(output / "summary.json", summary)
    print(f"Accepted experiment: {output}", flush=True)


if __name__ == "__main__":
    main()
