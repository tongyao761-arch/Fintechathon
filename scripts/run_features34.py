"""Independent features34 framework runner; current authorization is step 1 only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_lightgbm_baseline import (
    MODEL_PARAMS, RAW_DATA_PATH, EXPERIMENT_ROOT, PeakMemoryMonitor,
    split_masks, score_saved_inputs, environment_versions, compare_metrics,
)
from src.data.baseline_panel import load_raw_baseline_panel, extract_truth_files
from src.features.baseline_v1 import FEATURE_COLUMNS as BASE_COLUMNS
from src.features.features34 import build_features34, select_features, FEATURE_COLUMNS, FEATURE_DEFINITIONS
from src.metrics.diagnostics import baseline_diagnostics
from src.validation.experiment import experiment_run, provenance, sha256_file, write_json
from src.validation.splits import get_split

def run_split(
    panel: pd.DataFrame,
    features: pd.DataFrame,
    split_name: str,
    *,
    output_dir: Path,
    reference: dict,
    columns: tuple[str, ...],
) -> dict[str, Any]:
    split = get_split(split_name)
    split_started = time.perf_counter()
    train_period, _ = split.masks(panel)
    train_mask, valid_mask = split_masks(panel, split)
    if not features.index.equals(panel.index):
        raise AssertionError("feature index is not aligned with the panel")

    train_x = features.loc[train_mask, columns]
    train_y = panel.loc[train_mask, "y_ret_1d"].astype("float32")
    if not np.isfinite(train_y.to_numpy()).all():
        raise ValueError("training labels overflow float32")
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    model.fit(train_x, train_y, feature_name=columns)
    if tuple(model.feature_name_) != columns:
        raise AssertionError(
            f"model feature order changed: {model.feature_name_!r}"
        )
    del train_x, train_y

    valid_x = features.loc[valid_mask, columns]
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
    if columns == BASE_COLUMNS and prediction_sha256 != reference["prediction_sha256"]:
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
    quality["baseline_features_all_missing"] = features.loc[valid_mask, BASE_COLUMNS].isna().all(axis=1)
    diagnostics, diagnostic_tables = baseline_diagnostics(scored_pred, truth, quality, details)
    candidate_quality = quality.copy()
    candidate_quality["baseline_features_all_missing"] = features.loc[valid_mask, columns].isna().all(axis=1)
    candidate_diagnostics, candidate_tables = baseline_diagnostics(scored_pred, truth, candidate_quality, details)
    diagnostics["all_features_missing_scope"] = "fixed ten-feature skeleton"
    diagnostics["candidate_features_top_groups"] = candidate_diagnostics["top_groups"]
    diagnostic_tables["daily_candidate_missing_diagnostics"] = candidate_tables["daily_missing_diagnostics"]
    for name, frame in {**details, **diagnostic_tables}.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.17g")
    write_json(output_dir / "diagnostics.json", diagnostics)
    contributions = {"ic": metrics_with_details["ic_mean"] * 0.4,
                     "excess": metrics_with_details["annual_excess"] * 0.3,
                     "stability": (1 - metrics_with_details["mean_turnover"]) * 0.3}
    if columns == BASE_COLUMNS:
        compare_metrics(metrics_with_details, reference["metrics"])
        if diagnostics["top_groups"] != reference["diagnostics"]["top_groups"] or diagnostics["price_valid_only_turnover"] != reference["diagnostics"]["price_valid_only_turnover"]:
            raise AssertionError("ten-feature diagnostic reproduction differs from baseline_v1_1")
    stats = feature_statistics(features, train_mask=train_mask, valid_mask=valid_mask)
    stats.to_csv(output_dir / "feature_missing_statistics.csv", index=False, float_format="%.17g")

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
        "original_prediction_hash_matches": prediction_sha256 == reference["prediction_sha256"],
        "metrics": metrics_with_details,
        "score_contributions": contributions,
        "diagnostics": diagnostics,
        "metrics_minus_baseline_v1_1": {key: value - reference["metrics"][key]
                             for key, value in metrics_with_details.items()},
        "file_sha256": {str(p.relative_to(output_dir)).replace("\\", "/"): sha256_file(p)
                        for p in sorted(output_dir.rglob("*")) if p.is_file()},
        "official_comparison": official_comparison,
        "elapsed_seconds": elapsed,
    }


def feature_statistics(features, *, train_mask=None, valid_mask=None):
    rows = []
    scopes = {"full_panel": None}
    if train_mask is not None:
        scopes.update(train_eligible=train_mask, validation_all_keys=valid_mask)
    for scope, mask in scopes.items():
        frame = features if mask is None else features.loc[mask]
        for column in frame:
            values = frame[column].to_numpy()
            rows.append({"scope": scope, "feature": column, "rows": len(frame),
                         "missing": int(np.isnan(values).sum()), "infinite": int(np.isinf(values).sum()),
                         "finite": int(np.isfinite(values).sum()),
                         "missing_fraction": float(np.isnan(values).mean())})
    return pd.DataFrame(rows)


def resolve_selection(config, candidate, split_name):
    if config["stage"] != "step1_framework_only":
        raise ValueError("this runner currently authorizes only step 1")
    spec = config["experiments"][candidate]
    if set(spec) - {"groups", "include", "exclude"}:
        raise ValueError("candidate config contains unsupported fields; model tuning is prohibited")
    columns = select_features(**spec)
    if split_name not in config["allowed_runs"].get(candidate, []):
        raise ValueError("run outside current step-1 authorization")
    if columns not in (BASE_COLUMNS, FEATURE_COLUMNS):
        raise ValueError("step 1 permits only baseline10 or full34, not feature selection")
    if split_name == "oos_2024" and columns != BASE_COLUMNS:
        raise ValueError("2024 is authorized only for ten-feature reproduction in step 1")
    return columns


def check_frozen(manifest):
    for relative, expected in manifest.items():
        if sha256_file(ROOT / relative) != expected:
            raise AssertionError(f"frozen file changed: {relative}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/features34.json")
    parser.add_argument("--candidate", default="baseline10")
    parser.add_argument("--split", choices=["primary_2023", "oos_2024"], default="primary_2023")
    parser.add_argument("--experiment-id", default="features34_step1")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat()
    output = None
    try:
        with experiment_run(EXPERIMENT_ROOT, args.experiment_id) as output:
            print(f"Experiment directory: {output}", flush=True)
            config = json.loads(args.config.read_text(encoding="utf-8"))
            columns = resolve_selection(config, args.candidate, args.split)
            write_json(output / "config.json", {"requested": config, "candidate": args.candidate,
                       "split": args.split, "features": list(columns), "model_params": MODEL_PARAMS,
                       "purpose": "framework validation only; no feature-selection conclusion"})
            reference_path = ROOT / config["reference_summary"]
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            references = {s["split_name"]: s for s in reference["splits"]}
            manifest_path = ROOT / config["frozen_manifest"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            check_frozen(manifest)
            metadata = provenance(ROOT, RAW_DATA_PATH)
            metadata["candidate_config_sha256"] = sha256_file(args.config)
            metadata["reference_summary_sha256"] = sha256_file(reference_path)
            metadata["frozen_manifest_sha256"] = sha256_file(manifest_path)
            write_json(output / "provenance.json", metadata)
            if metadata["git"]["branch"] != "ivor-work":
                raise AssertionError("features34 runs require local ivor-work branch")
            if metadata["data"]["sha256"] != reference["data"]["sha256"] or MODEL_PARAMS != reference["model_params"]:
                raise AssertionError("data/model parameters differ from frozen baseline_v1_1")
            with PeakMemoryMonitor() as memory:
                panel = load_raw_baseline_panel(RAW_DATA_PATH)
                print("Full historical panel loaded; computing selected features before masks", flush=True)
                features = build_features34(panel, columns)
                if tuple(features.columns) != columns or not features.index.equals(panel.index) or len(panel) != len(features):
                    raise AssertionError("feature contract changed columns/rows/index")
                stats = feature_statistics(features)
                if stats.infinite.any() or not stats.finite.gt(0).all():
                    raise AssertionError("feature contains infinity or has no finite observations")
                stats.to_csv(output / "feature_missing_statistics.csv", index=False, float_format="%.17g")
                split = get_split(args.split)
                split_output = output / args.split
                extract_truth_files(RAW_DATA_PATH, {split_output / "evaluate_input/测试集_Y.csv": (split.valid_start, split.valid_end)})
                print("Original float64 truth extracted; training begins", flush=True)
                result = run_split(panel, features, args.split, output_dir=split_output,
                                   reference=references[args.split], columns=columns)
            check_frozen(manifest)
            if sha256_file(RAW_DATA_PATH) != metadata["data"]["sha256"]:
                raise AssertionError("raw data changed during run")
            current_metadata = provenance(ROOT, RAW_DATA_PATH)
            if current_metadata["source_sha256"] != metadata["source_sha256"] or sha256_file(args.config) != metadata["candidate_config_sha256"]:
                raise AssertionError("source/config changed during run")
            write_json(split_output / "summary.json", result)
            summary = {"stage": "step1_framework_only", "candidate": args.candidate,
                       "run_id": output.name, "started_at": started_at,
                       "finished_at": datetime.now().astimezone().isoformat(),
                       "features": list(columns), "feature_count": len(columns),
                       "feature_definitions": {c: FEATURE_DEFINITIONS[c] for c in columns},
                       "feature_index_preserved": True, "panel_rows": len(panel),
                       "model_params": MODEL_PARAMS, "environment": environment_versions(),
                       "reference_summary_sha256": metadata["reference_summary_sha256"],
                       "provenance_sha256": sha256_file(output / "provenance.json"),
                       "config_sha256": sha256_file(output / "config.json"),
                       "frozen_files_unchanged": True, "splits": [result],
                       "purpose": "framework validation only; no feature-selection conclusion",
                       "resources": {"elapsed_seconds": time.perf_counter()-started,
                                     "peak_process_rss_mb": memory.peak_rss_bytes/(1024**2)}}
            write_json(output / "summary.json", summary)
        state = json.loads((output / "status.json").read_text(encoding="utf-8"))
        state["elapsed_seconds"] = time.perf_counter()-started
        write_json(output / "status.json", state)
        print(json.dumps({"accepted": str(output), "candidate": args.candidate,
                          "split": args.split, "metrics": result["metrics"]}), flush=True)
    except BaseException:
        if output is not None:
            state = json.loads((output / "status.json").read_text(encoding="utf-8"))
            state["elapsed_seconds"] = time.perf_counter()-started
            write_json(output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
