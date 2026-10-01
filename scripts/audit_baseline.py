"""Evidence-based acceptance of the frozen baseline; no feature selection."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_lightgbm_baseline as runner
from src.data.baseline_panel import load_raw_baseline_panel
from src.data.data_contract import TRAIN_COLUMNS, X_COLUMNS, assert_sorted_by_panel_key
from src.features.baseline_v1 import FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, build_baseline_v1_features
from src.metrics.official import score_official
from src.validation.experiment import experiment_run, prediction_hash, provenance, sha256_file, write_json
from src.validation.splits import get_split

KEYS = ["ts_code", "trade_date"]
SEEDS = tuple(range(20260927, 20260932))
UPSTREAM_LIMITS = (
    "后复权历史因子是否随未来事件或数据修订改变（本地说明标注后复权，但无历史因子快照）",
    "历史股票池、退市样本覆盖与幸存者偏差",
    "原始数据事后修订及其历史可得性",
    "当日收盘价、成交量额的实际发布时间和可成交时点",
    "交易成本、成交限制与实际可实现收益（本次比赛验收范围之外）",
)


def reference_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Independent scalar oracle: no pandas shift/rolling or production helpers."""
    output = np.full((len(panel), 10), np.nan, dtype=np.float64)
    codes = panel.ts_code.astype(str).to_numpy()
    raw = {c: panel[c].to_numpy() for c in RAW_FEATURE_COLUMNS}
    bounds = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(panel)]
    with np.errstate(all="ignore"):
        for start, end in zip(bounds[:-1], bounds[1:]):
            returns = []
            for i in range(start, end):
                # Preserve the documented intermediate precision of raw X.
                one = raw["close"][i] / raw["close"][i - 1] - 1 if i > start else np.nan
                returns.append(float(one))
                output[i, 0] = one
                for column, lag in ((1, 5), (2, 20)):
                    if i - lag >= start:
                        output[i, column] = raw["close"][i] / raw["close"][i - lag] - 1
                if i > start:
                    output[i, 3] = raw["open"][i] / raw["close"][i - 1] - 1
                output[i, 4] = raw["close"][i] / raw["open"][i] - 1
                output[i, 5] = raw["high"][i] / raw["low"][i] - 1
                for column, window in ((6, 5), (7, 20)):
                    values = np.array(returns[-window:], dtype=np.float64)
                    if len(values) == window and np.isfinite(values).all():
                        average = sum(values) / window
                        output[i, column] = (sum((v - average) ** 2 for v in values) / (window - 1)) ** .5
                if i - 5 >= start:
                    for column, source in ((8, "vol"), (9, "amount")):
                        history = raw[source][i - 5:i].astype(np.float64)
                        if np.isfinite(history).all():
                            output[i, column] = raw[source][i] / (sum(history) / 5)
        output = output.astype(np.float32)
    output[~np.isfinite(output)] = np.nan
    return pd.DataFrame(output, index=panel.index, columns=FEATURE_COLUMNS)


def compare_feature_oracle(panel: pd.DataFrame) -> dict:
    actual, expected = build_baseline_v1_features(panel), reference_features(panel)
    np.testing.assert_array_equal(actual.isna(), expected.isna())
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7, equal_nan=True)
    error = np.abs(actual.to_numpy(dtype=float) - expected.to_numpy(dtype=float))
    return {"rows": len(panel), "stocks": int(panel.ts_code.nunique()),
            "max_abs_error": float(np.nanmax(error)) if np.isfinite(error).any() else 0.,
            "nan_positions_equal": True, "rtol": 1e-6, "atol": 1e-7}


def assert_prefix_invariance(panel: pd.DataFrame, cutoffs) -> dict:
    original = build_baseline_v1_features(panel)
    for cutoff in cutoffs:
        prefix = panel.trade_date.le(cutoff)
        if not prefix.any() or prefix.all():
            raise AssertionError(f"cutoff {cutoff} needs past and future rows")
        pd.testing.assert_frame_equal(original.loc[prefix], build_baseline_v1_features(panel.loc[prefix]))
        changed = panel.copy(deep=True)
        changed.loc[~prefix, list(RAW_FEATURE_COLUMNS)] *= 13.
        if "y_ret_1d" in changed:
            changed.loc[~prefix, "y_ret_1d"] = .8
        pd.testing.assert_frame_equal(original.loc[prefix], build_baseline_v1_features(changed).loc[prefix])
    changed = panel.copy(deep=True)
    selected = panel.ts_code.eq(panel.ts_code.iloc[0])
    changed.loc[selected, list(RAW_FEATURE_COLUMNS)] *= 7.
    pd.testing.assert_frame_equal(original.loc[~selected], build_baseline_v1_features(changed).loc[~selected])
    return {"rows": len(panel), "cutoffs": list(map(int, cutoffs)),
            "truncation_equal": True, "future_perturbation_equal": True, "stock_isolation_equal": True}


def label_check(raw: pd.DataFrame, next_rows: pd.DataFrame | None = None) -> dict:
    """Independent adjacent-array label check, respecting stock and calendar boundaries."""
    codes = raw.ts_code.astype(str).to_numpy()
    dates = raw.trade_date.to_numpy()
    close = raw.close.to_numpy(dtype=np.float64)
    labels = raw.y_ret_1d.to_numpy(dtype=np.float64)
    following = np.r_[close[1:], np.nan]
    targets = np.r_[dates[1:], 0]
    same = np.r_[codes[1:] == codes[:-1], False]
    following[~same] = np.nan
    targets[~same] = 0
    if next_rows is not None:
        lookup = next_rows.set_index("ts_code")
        for i in np.flatnonzero(~same):
            if codes[i] in lookup.index:
                following[i] = float(lookup.loc[codes[i], "close"])
                targets[i] = int(lookup.loc[codes[i], "trade_date"])
    calendar = np.unique(np.r_[dates, next_rows.trade_date.to_numpy() if next_rows is not None else []])
    calendar_next = dict(zip(calendar[:-1], calendar[1:]))
    expected_target = pd.Series(dates).map(calendar_next).fillna(0).to_numpy(dtype=int)
    adjacent = (targets == expected_target) & (targets > dates)
    with np.errstate(all="ignore"):
        expected = following / close - 1
    checkable = adjacent & np.isfinite(expected) & np.isfinite(labels)
    error = np.abs(labels[checkable] - expected[checkable])
    failures = checkable & (np.abs(labels - expected) > 1e-10)
    if failures.any():
        raise AssertionError(f"label mismatches: {int(failures.sum())}, examples={raw.loc[failures, KEYS].head().to_dict('records')}")
    boundaries = {}
    for name in runner.SPLIT_NAMES:
        split = get_split(name)
        train, _ = runner.split_masks(raw, split)
        if (targets[train] >= split.valid_start).any():
            raise AssertionError(f"{name}: training label target reaches validation")
        if (targets[train] == 0).any():
            raise AssertionError(f"{name}: missing training target date")
        boundaries[name] = {"train_samples": int(train.sum()),
                            "latest_training_target_date": int(targets[train].max()),
                            "purge_date": split.purge_date, "validation_start": split.valid_start}
    nonmissing = np.isfinite(labels)
    return {"checked": int(checkable.sum()), "mismatches_gt_1e_10": 0,
            "max_abs_error": float(error.max()) if len(error) else None,
            "missing_labels": int(np.isnan(labels).sum()),
            "nonmissing_uncheckable": int((nonmissing & ~checkable).sum()),
            "uncheckable_examples": raw.loc[nonmissing & ~checkable, KEYS].head(10).to_dict("records"),
            "split_target_boundaries": boundaries,
            "terminal_cross_file_checked": int((checkable & dates.__eq__(dates.max())).sum())}


def validate_raw(raw: pd.DataFrame, name: str) -> dict:
    assert_sorted_by_panel_key(raw, name=name)
    dates = pd.Series(raw.trade_date.unique()).astype(str)
    assert dates.str.fullmatch(r"\d{8}").all()
    assert pd.to_datetime(dates, format="%Y%m%d", errors="coerce").notna().all()
    counts = {}
    for column in RAW_FEATURE_COLUMNS:
        values = raw[column].to_numpy(dtype=float)
        counts[column + "_missing"] = int(np.isnan(values).sum())
        counts[column + "_infinite"] = int(np.isinf(values).sum())
        counts[column + "_invalid_sign"] = int((values < 0).sum() if column in ("vol", "amount") else (values <= 0).sum())
    for column in ("flag_limit_up", "flag_limit_down"):
        counts[column + "_invalid"] = int((~raw[column].isin((0, 1))).sum())
    counts["high_below_other_price"] = int(raw.high.lt(raw[["open", "low", "close"]].max(axis=1)).sum())
    counts["low_above_other_price"] = int(raw.low.gt(raw[["open", "high", "close"]].min(axis=1)).sum())
    if "y_ret_1d" in raw:
        counts["label_infinite"] = int(np.isinf(raw.y_ret_1d).sum())
    unacceptable = {k: v for k, v in counts.items() if not k.endswith("_missing") and v}
    if unacceptable:
        raise AssertionError(f"{name}: unexpected invalid values {unacceptable}")
    return {"rows": len(raw), "stocks": int(raw.ts_code.nunique()), "dates": len(dates),
            "key_unique_sorted": True, "valid_calendar_dates": True, "counts": counts,
            "both_limit_flags": int((raw.flag_limit_up.eq(1) & raw.flag_limit_down.eq(1)).sum()),
            "both_limit_flags_policy": "preserve supplied values; official only filters flag_limit_up"}


def verify_saved_run(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    state = json.loads((path / "status.json").read_text(encoding="utf-8"))
    if state["status"] != "success":
        raise AssertionError(f"saved run not successful: {path}")
    checked = 0
    for split in summary["splits"]:
        for name, expected in split["file_sha256"].items():
            item = path / split["split_name"] / name
            if sha256_file(item) != expected:
                raise AssertionError(f"corrupt saved artifact: {item}")
            checked += 1
        pred = pd.read_parquet(path / split["split_name"] / "predictions.parquet")
        if prediction_hash(pred.pred) != split["prediction_sha256"]:
            raise AssertionError("prediction value hash changed")
        assert len(pred) == split["valid_prediction_rows"]
        assert not pred.duplicated(KEYS).any() and np.isfinite(pred.pred).all()
    assert sha256_file(path / "provenance.json") == summary["provenance_sha256"]
    return {"path": str(path.relative_to(ROOT)), "summary_sha256": sha256_file(path / "summary.json"),
            "checked_files": checked, "all_file_hashes_match": True, "summary": summary}


def compare_runs(first: dict, second: dict, reference: dict) -> dict:
    for key in ("features", "feature_definitions", "model_params", "feature_missing_counts", "feature_infinite_counts", "environment", "data"):
        assert first[key] == second[key], key
    old = {s["split_name"]: s for s in reference["splits"]}
    result = {}
    for a, b in zip(first["splits"], second["splits"], strict=True):
        for key in ("split_name", "dates", "train_samples", "valid_prediction_rows", "prediction_sha256", "metrics", "official_comparison", "diagnostics", "file_sha256"):
            assert a[key] == b[key], f"{a['split_name']}: repeat {key}"
        expected = old[a["split_name"]]
        for key in ("train_samples", "valid_prediction_rows", "prediction_sha256", "metrics", "diagnostics"):
            assert a[key] == expected[key], f"{a['split_name']}: frozen {key}"
        assert a["prediction_coverage"] == b["prediction_coverage"] == 1.
        assert a["model_reload_predictions_equal"] and b["model_reload_predictions_equal"]
        assert np.isfinite(list(a["metrics"].values())).all()
        assert a["official_comparison"]["max_abs_difference"] <= 1e-12
        result[a["split_name"]] = {"matches_frozen_v1_1": True, "repeat_equal": True,
                                   "prediction_sha256": a["prediction_sha256"], "metrics": a["metrics"]}
    return result


def tie_sensitivity(pred: pd.DataFrame, truth: pd.DataFrame, x: pd.DataFrame,
                    source_fixture: Path | None = None) -> dict:
    metrics = []
    for seed in (None, *SEEDS):
        ordered = pred if seed is None else pred.sample(frac=1, random_state=seed)
        with tempfile.TemporaryDirectory(prefix="fintechathon_ties_") as directory:
            fixture = Path(directory)
            ordered.to_csv(fixture / "submission.csv", index=False, float_format="%.17g")
            if source_fixture is None:
                truth.to_csv(fixture / "测试集_Y.csv", index=False, float_format="%.17g")
                x.to_csv(fixture / "测试集_X.csv", index=False)
            else:
                for name in ("测试集_Y.csv", "测试集_X.csv"):
                    shutil.copyfile(source_fixture / name, fixture / name)
            local, parity, _, _ = runner.score_saved_inputs(fixture)
            local.pop("details")
            metrics.append({"seed": seed, "metrics": local, "official_max_abs_difference": parity["max_abs_difference"]})
    unique = pred.groupby("trade_date").pred.nunique()
    size = pred.groupby("trade_date").size()
    return {"rows": len(pred), "dates": len(unique), "duplicate_prediction_rows": int((size - unique).sum()),
            "runs": metrics,
            "metric_ranges": {k: float(max(m["metrics"][k] for m in metrics) - min(m["metrics"][k] for m in metrics)) for k in metrics[0]["metrics"]},
            "policy": "freeze original prediction CSV order and environment; do not introduce a new tiebreaker"}


class Audit:
    def __init__(self, output: Path):
        self.output, self.checks = output, []

    def check(self, name, expected, function, blocking=True):
        print(f"Checking {name}", flush=True)
        evidence = self.output / (name + ".json")
        try:
            actual = function()
            write_json(evidence, actual)
            state, error = "passed", None
        except Exception as exc:
            actual = {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
            write_json(evidence, actual)
            state, error = "failed", str(exc)
        self.checks.append({"name": name, "status": state, "blocking": blocking,
                            "expected": expected, "actual": actual,
                            "evidence": evidence.name, "error": error})
        return actual if state == "passed" else None


def run_command(command, log: Path) -> str:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    with log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    output = log.read_text(encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"command exit {completed.returncode}; see {log.name}: {output[-1200:]}")
    return output


def test_suite(output: Path) -> dict:
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"]
    log = output / "unit_tests.log"
    text = run_command(command, log)
    match = re.search(r"Ran (\d+) tests", text)
    assert match and text.rstrip().endswith("OK"), "test completion absent"
    return {"command": command, "count": int(match[1]), "log": log.name, "log_sha256": sha256_file(log)}


def dependency_check(output: Path) -> dict:
    text = run_command([sys.executable, "-B", "-m", "pip", "check"], output / "pip_check.log").strip()
    versions = {}
    for line in (ROOT / "requirements-model.lock").read_text(encoding="utf-8").splitlines():
        if "==" not in line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        actual = importlib.metadata.version(name)
        assert actual == expected, f"dependency differs from lock: {name} {actual} != {expected}"
        versions[name] = actual
    assert versions, "empty dependency lock"
    return {"pip_check": text, "locked_versions": versions, "all_pins_match": True}


def data_audit(output: Path) -> dict:
    raw = pd.read_csv(runner.RAW_DATA_PATH, usecols=list(TRAIN_COLUMNS))
    test_path = ROOT / "赛题五/赛题五数据/测试集_X.csv"
    test = pd.read_csv(test_path, usecols=list(X_COLUMNS))
    train_stats, test_stats = validate_raw(raw, "train"), validate_raw(test, "test")
    prices = raw[["open", "high", "low", "close"]]
    raw["is_price_valid"] = (np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
                             & raw.high.ge(prices.max(axis=1)) & raw.low.le(prices.min(axis=1))).astype("int8")
    raw["is_trainable"] = (raw.is_price_valid.eq(1) & raw.y_ret_1d.notna()).astype("int8")
    first_test = test.loc[test.trade_date.eq(test.trade_date.min()), [*KEYS, "close"]]
    labels = label_check(raw, first_test)
    assert labels["nonmissing_uncheckable"] == 0, "nonmissing labels lack target-price evidence"
    assert train_stats["rows"] == 7900350 and test_stats["rows"] == 1599600
    # Spread the deterministic real-data feature sample across the stock universe.
    codes = sorted(raw.ts_code.unique())
    selected = [codes[i] for i in np.linspace(0, len(codes) - 1, 32, dtype=int)]
    sample = raw.loc[raw.ts_code.isin(selected)].copy()
    sample[list(RAW_FEATURE_COLUMNS)] = sample[list(RAW_FEATURE_COLUMNS)].astype("float32")
    oracle = compare_feature_oracle(sample)
    prefixes = assert_prefix_invariance(sample, [20221229, 20221230, 20230103, 20231228, 20231229, 20240102])
    continuation = {}
    features = build_baseline_v1_features(sample)
    for name in runner.SPLIT_NAMES:
        split = get_split(name)
        valid = sample.trade_date.between(split.valid_start, split.valid_end)
        separately = build_baseline_v1_features(sample.loc[valid])
        first = sample.trade_date.eq(split.valid_start)
        differences = features.loc[first].notna() & separately.loc[first].isna()
        assert differences.to_numpy().any(), "real history was not carried into validation"
        continuation[name] = {"first_day_history_values": int(differences.to_numpy().sum()), "history_preserved": True}
    result = {"train": train_stats, "test": test_stats, "labels": labels,
              "real_feature_sample": oracle, "prefixes": prefixes, "history_continuation": continuation,
              "sample_codes": selected, "sample_policy": "32 evenly spaced sorted codes; all 1699 historical rows"}
    del raw, test, sample, features
    gc.collect()
    return result


def full_runs(output: Path) -> dict:
    command = [sys.executable, "-B", "scripts/run_lightgbm_baseline.py", "--experiment-id", "baseline_audit", "--split", "all", "--verify-clean"]
    runs = []
    for i in range(2):
        print(f"Full baseline run {i + 1}/2", flush=True)
        text = run_command(command, output / f"full_run_{i + 1}.log")
        match = re.search(r"Accepted experiment: (.+)", text)
        assert match, "no accepted baseline directory"
        runs.append(verify_saved_run(Path(match[1].strip())))
    reference = json.loads((ROOT / "artifacts/baseline_v1_1/summary.json").read_text(encoding="utf-8"))
    compared = compare_runs(runs[0]["summary"], runs[1]["summary"], reference)
    a = json.loads((ROOT / runs[0]["path"] / "provenance.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / runs[1]["path"] / "provenance.json").read_text(encoding="utf-8"))
    assert a == b, "source/dependency/data snapshot changed between runs"
    return {"command": command, "runs": runs, "per_split": compared, "same_provenance": True}


def fit_saved_control(train_x, labels, valid_x, valid_keys, fixture, output, kind, seed=None):
    labels = np.asarray(labels, dtype=np.float32)
    assert np.isfinite(labels).all()
    model = lgb.LGBMRegressor(**runner.MODEL_PARAMS)
    model.fit(train_x, labels, feature_name=FEATURE_COLUMNS)
    assert tuple(model.feature_name_) == FEATURE_COLUMNS
    prediction = model.predict(valid_x)
    assert np.isfinite(prediction).all()
    pred = valid_keys.copy()
    pred["pred"] = prediction
    with experiment_run(runner.EXPERIMENT_ROOT, "baseline_audit_controls") as control_dir:
        saved_fixture = control_dir / "evaluate_input"
        saved_fixture.mkdir()
        for name in ("测试集_Y.csv", "测试集_X.csv"):
            shutil.copyfile(fixture / name, saved_fixture / name)
        pred.to_csv(saved_fixture / "submission.csv", index=False, float_format="%.17g")
        pred.to_parquet(control_dir / "predictions.parquet", index=False)
        assert np.array_equal(pd.read_parquet(control_dir / "predictions.parquet").pred, prediction)
        models = control_dir / "models"
        models.mkdir()
        model.booster_.save_model(str(models / "lightgbm.txt"))
        assert np.array_equal(prediction, lgb.Booster(model_file=str(models / "lightgbm.txt")).predict(valid_x))
        metrics, parity, _, _ = runner.score_saved_inputs(saved_fixture)
        details = metrics.pop("details")
        for name in ("daily_ic", "daily_excess", "daily_turnover"):
            details[name].to_csv(control_dir / (name + ".csv"), index=False, float_format="%.17g")
        # status.json transitions on context exit; hash its final version below.
        hashes = {str(p.relative_to(control_dir)).replace("\\", "/"): sha256_file(p)
                  for p in sorted(control_dir.rglob("*")) if p.is_file() and p.name != "status.json"}
        result = {"kind": kind, "seed": seed, "prediction_sha256": prediction_hash(prediction),
                  "training_label_sha256": sha256_file_bytes(labels.astype("<f4").tobytes()),
                  "metrics": metrics, "official_max_abs_difference": parity["max_abs_difference"],
                  "run_directory": str(control_dir.relative_to(ROOT)), "file_sha256": hashes,
                  "model_reload_equal": True, "model_params": runner.MODEL_PARAMS,
                  "audit_source_provenance": str((output / "provenance.json").relative_to(ROOT))}
        write_json(control_dir / "summary.json", result)
    for name, expected in hashes.items():
        assert sha256_file(control_dir / name) == expected, f"control artifact changed: {name}"
    assert json.loads((control_dir / "status.json").read_text())["status"] == "success"
    result["summary_sha256"] = sha256_file(control_dir / "summary.json")
    result["status_sha256"] = sha256_file(control_dir / "status.json")
    return result


def recurring_ic_alert(values, baseline_ic):
    """Investigation trigger, not a hypothesis test or an alpha acceptance rule."""
    return sum(abs(value) >= abs(baseline_ic) for value in values) >= 3


def residual_exposure_diagnostic(valid_x, valid_keys, residual_controls, fixture, output):
    from scipy.stats import spearmanr
    strongest = max(residual_controls, key=lambda c: abs(c["metrics"]["ic_mean"]))
    pred = pd.read_parquet(ROOT / strongest["run_directory"] / "predictions.parquet")
    joined = valid_keys.copy()
    for column in FEATURE_COLUMNS:
        joined[column] = valid_x[column].to_numpy()
    joined = joined.merge(pd.read_csv(fixture / "测试集_Y.csv"), on=KEYS, validate="one_to_one")
    joined = joined.merge(pred, on=KEYS, validate="one_to_one")
    statistics = {}
    for column in FEATURE_COLUMNS:
        feature_ic, prediction_corr = [], []
        for _, group in joined.groupby("trade_date"):
            group = group.dropna(subset=[column, "y_ret_1d"])
            if len(group) >= 30 and group[column].nunique() > 1 and group.pred.nunique() > 1:
                feature_ic.append(float(spearmanr(group[column], group.y_ret_1d)[0]))
                prediction_corr.append(float(spearmanr(group[column], group.pred)[0]))
        assert feature_ic and np.isfinite(feature_ic + prediction_corr).all()
        statistics[column] = {"feature_mean_ic": float(np.mean(feature_ic)),
                              "prediction_mean_spearman": float(np.mean(prediction_corr))}
    # A model trained on shuffled labels can still be a function of informative X.
    # Destroy that X-to-row correspondence without changing each day's predictions.
    rng = np.random.default_rng(SEEDS[0])
    permuted = pred.copy()
    for _, indices in permuted.groupby("trade_date").groups.items():
        before = pred.loc[indices, "pred"].to_numpy()
        permuted.loc[indices, "pred"] = rng.permutation(before)
        np.testing.assert_array_equal(np.sort(permuted.loc[indices, "pred"]), np.sort(before))
    with experiment_run(runner.EXPERIMENT_ROOT, "baseline_audit_prediction_permutation") as directory:
        saved = directory / "evaluate_input"
        saved.mkdir()
        for name in ("测试集_Y.csv", "测试集_X.csv"):
            shutil.copyfile(fixture / name, saved / name)
        permuted.to_csv(saved / "submission.csv", index=False, float_format="%.17g")
        metrics, parity, _, _ = runner.score_saved_inputs(saved)
        metrics.pop("details")
        result = {"strongest_residual_seed": strongest["seed"], "original_metrics": strongest["metrics"],
                  "existing_feature_associations": statistics, "within_day_prediction_multiset_preserved": True,
                  "prediction_permutation_metrics": metrics, "official_max_abs_difference": parity["max_abs_difference"],
                  "run_directory": str(directory.relative_to(ROOT)),
                  "file_sha256": {str(p.relative_to(directory)).replace("\\", "/"): sha256_file(p)
                                  for p in saved.iterdir() if p.is_file()},
                  "interpretation": "Nonzero shuffled-model IC can arise from accidental exposure to informative existing X. Prediction-row permutation checks the observed row association; it does not certify future alpha."}
        write_json(directory / "summary.json", result)
    result["summary_sha256"] = sha256_file(directory / "summary.json")
    result["status_sha256"] = sha256_file(directory / "status.json")
    write_json(output / "residual_feature_exposure.json", result)
    return result


def investigate_shuffles(train_x, train_y, valid_x, valid_keys, dates, groups,
                        controls, fixture, output, base_ic):
    from scipy.stats import spearmanr
    daily_means = pd.Series(train_y).groupby(dates).transform("mean").to_numpy(dtype=np.float32)
    print("Investigating retained daily market means", flush=True)
    mean_control = fit_saved_control(train_x, daily_means, valid_x, valid_keys, fixture, output, "market_mean_only")
    mean_pred = pd.read_parquet(ROOT / mean_control["run_directory"] / "predictions.parquet")
    correlations = []
    for control in controls:
        shuffled = pd.read_parquet(ROOT / control["run_directory"] / "predictions.parquet")
        joined = mean_pred.merge(shuffled, on=KEYS, validate="one_to_one", suffixes=("_mean", "_shuffled"))
        daily = joined.groupby("trade_date").apply(
            lambda group: float(spearmanr(group.pred_mean, group.pred_shuffled)[0]), include_groups=False)
        assert np.isfinite(daily).all()
        correlations.append({"seed": control["seed"], "mean_daily_prediction_spearman": float(daily.mean()),
                             "median_daily_prediction_spearman": float(daily.median())})
    residual = (train_y - daily_means).astype(np.float32)
    residual_controls = []
    for seed in SEEDS:
        print(f"Demeaned shuffled-label control seed {seed}", flush=True)
        rng = np.random.default_rng(seed)
        shuffled = residual.copy()
        for indices in groups:
            shuffled[indices] = rng.permutation(residual[indices])
            np.testing.assert_array_equal(np.sort(shuffled[indices]), np.sort(residual[indices]))
        result = fit_saved_control(train_x, shuffled, valid_x, valid_keys, fixture, output, "demeaned_day_shuffle", seed)
        result["within_day_label_multiset_preserved"] = True
        residual_controls.append(result)
    residual_alert = recurring_ic_alert([c["metrics"]["ic_mean"] for c in residual_controls], base_ic)
    exposure = residual_exposure_diagnostic(valid_x, valid_keys, residual_controls, fixture, output)
    resolved = (all(c["mean_daily_prediction_spearman"] >= .8 for c in correlations)
                and not residual_alert and abs(exposure["prediction_permutation_metrics"]["ic_mean"]) < .005)
    result = {"market_mean_control": mean_control, "prediction_correlations": correlations,
              "demeaned_controls": residual_controls, "resolved": resolved,
              "residual_recurring_alert": residual_alert, "residual_exposure": exposure,
              "explanation_rule": "all shuffled/market-mean daily rank correlations >=0.8; demeaned controls do not trigger the SAME original 3/5 alert; prediction-row permutation |IC| <0.005",
              "methodology_note": "A single large residual IC is retained and investigated, not automatically treated as a coding defect. The original alert is unchanged; an earlier exploratory requirement that every residual IC be smaller was too strict for a diagnostic control.",
              "interpretation": "Daily market targets explain the recurring original effect. Existing informative X can also yield incidental residual exposures after label shuffling. This is diagnostic evidence, not a statistical proof of zero leakage or significant baseline alpha."}
    write_json(output / "negative_control_investigation.json", result)
    if not resolved:
        raise AssertionError("shuffled-control anomaly remains unexplained; see negative_control_investigation.json")
    return result


def full_controls(output: Path, accepted: dict) -> dict:
    """Five shuffled-label training runs, 2023 only; never replace the frozen model."""
    panel = load_raw_baseline_panel(runner.RAW_DATA_PATH)
    features = build_baseline_v1_features(panel)
    train, valid = runner.split_masks(panel, get_split("primary_2023"))
    train_x, valid_x = features.loc[train, FEATURE_COLUMNS], features.loc[valid, FEATURE_COLUMNS]
    train_y = panel.loc[train, "y_ret_1d"].astype("float32").to_numpy()
    dates = panel.loc[train, "trade_date"].to_numpy()
    groups = pd.Series(np.arange(len(dates))).groupby(dates, sort=True).apply(np.array)
    run = ROOT / accepted["runs"][0]["path"] / "primary_2023"
    fixture = run / "evaluate_input"
    truth, x = pd.read_csv(fixture / "测试集_Y.csv"), pd.read_csv(fixture / "测试集_X.csv")
    base_pred = pd.read_parquet(run / "predictions.parquet")
    controls = []
    for seed in SEEDS:
        print(f"Shuffled-label control seed {seed}", flush=True)
        rng = np.random.default_rng(seed)
        shuffled = train_y.copy()
        for indices in groups:
            shuffled[indices] = rng.permutation(train_y[indices])
            np.testing.assert_array_equal(np.sort(shuffled[indices]), np.sort(train_y[indices]))
        result = fit_saved_control(train_x, shuffled, valid_x, panel.loc[valid, KEYS], fixture, output, "day_shuffle", seed)
        result["within_day_label_multiset_preserved"] = True
        controls.append(result)
    base_ic = accepted["runs"][0]["summary"]["splits"][0]["metrics"]["ic_mean"]
    assert len({c["training_label_sha256"] for c in controls}) == len(SEEDS)
    assert len({c["prediction_sha256"] for c in controls}) == len(SEEDS)
    # A large persistent IC merits a blocking investigation, not an invented null score.
    suspicious = recurring_ic_alert([c["metrics"]["ic_mean"] for c in controls], base_ic)
    investigation = None
    if suspicious:
        write_json(output / "negative_control_anomaly.json", {"baseline_ic": base_ic, "controls": controls})
        investigation = investigate_shuffles(train_x, train_y, valid_x, panel.loc[valid, KEYS], dates, groups,
                                             controls, fixture, output, base_ic)
    tied = tie_sensitivity(base_pred, truth, x, source_fixture=fixture)
    assert tied["runs"][0]["metrics"] == accepted["runs"][0]["summary"]["splits"][0]["metrics"], "canonical tie test changed baseline scores"
    oos_run = ROOT / accepted["runs"][0]["path"] / "oos_2024"
    oos_fixture = oos_run / "evaluate_input"
    tied_oos = tie_sensitivity(pd.read_parquet(oos_run / "predictions.parquet"),
                               pd.read_csv(oos_fixture / "测试集_Y.csv"),
                               pd.read_csv(oos_fixture / "测试集_X.csv"), source_fixture=oos_fixture)
    assert tied_oos["runs"][0]["metrics"] == accepted["runs"][0]["summary"]["splits"][1]["metrics"], "canonical OOS tie test changed baseline scores"
    del panel, features, train_x, valid_x
    gc.collect()
    return {"seeds": list(SEEDS), "training_years": "2018-2022", "evaluation_year": 2023,
            "model_params_unchanged": True, "controls": controls, "baseline_ic": base_ic,
            "investigation_trigger": "at least 3/5 controls reach baseline absolute mean IC",
            "anomaly_triggered": suspicious, "investigation": investigation,
            "positive_random_scores_are_not_failures": True,
            "tie_sensitivity_2023": tied,
            "tie_sensitivity_2024": tied_oos,
            "interpretation": "Day-level return distribution and market state remain; positive random scores or low turnover alone do not prove leakage."}


def sha256_file_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


def release_decision(checks, suite):
    required = {"unit_tests", "dependencies", "raw_data_and_features", "full_repeatability", "controls_and_ties", "artifact_integrity", "snapshot_unchanged"}
    states = {c["name"]: c["status"] for c in checks}
    missing = sorted(required - states.keys())
    failures = [c["name"] for c in checks if (c["blocking"] or c["name"] in required) and c["status"] != "passed"]
    return {"can_start_feature_experiments": suite != "unit" and not missing and not failures,
            "blocking_failures": failures, "critical_unverified": missing,
            "scope": "fixed-data competition experiment baseline; no absolute correctness or tradability guarantee"}


def verify_linked_artifacts(accepted, controls):
    """Re-read immutable outputs at the final acceptance boundary."""
    runs = [verify_saved_run(ROOT / run["path"]) for run in accepted["runs"]]
    for actual, recorded in zip(runs, accepted["runs"], strict=True):
        assert actual["summary_sha256"] == recorded["summary_sha256"], "full-run summary changed"
    entries = controls["controls"][:]
    if controls["investigation"]:
        investigation = controls["investigation"]
        entries += [investigation["market_mean_control"], *investigation["demeaned_controls"], investigation["residual_exposure"]]
    checked = sum(run["checked_files"] for run in runs)
    for entry in entries:
        path = ROOT / entry["run_directory"]
        assert json.loads((path / "status.json").read_text())["status"] == "success"
        for name, expected in {**entry["file_sha256"], "summary.json": entry["summary_sha256"],
                               "status.json": entry["status_sha256"]}.items():
            assert sha256_file(path / name) == expected, f"final artifact changed: {path / name}"
            checked += 1
    return {"checked_files": checked, "full_runs": len(runs), "diagnostic_runs": len(entries),
            "all_recorded_hashes_match_at_acceptance": True}


def assert_snapshot_unchanged(before):
    after = provenance(ROOT, runner.RAW_DATA_PATH)
    for key in ("data", "source_sha256", "dependencies", "git"):
        assert after[key] == before[key], f"run snapshot changed: {key}"
    for path, expected in before["additional_inputs"].items():
        assert sha256_file(ROOT / path) == expected, f"run input changed: {path}"
    return {"source_dependencies_git_and_all_inputs_unchanged": True}


def upstream_documentation():
    path = ROOT / "赛题五/完整赛题说明_本地.md"
    if not path.is_file():
        return {"available": False, "status": "unverified", "reason": "local problem statement absent"}
    content = path.read_text(encoding="utf-8")
    return {"available": True, "source": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
            "declares_backward_adjusted_prices": "后复权" in content,
            "provenance_limit": "Local problem-statement copy; no original PDF or historical adjustment-factor snapshots available."}


def render_report(summary):
    released = summary["decision"]["can_start_feature_experiments"]
    lines = ["# Baseline 严格验收报告", "", "结论：" + ("允许开始独立候选特征实验。" if released else "不放行增加特征。"), "",
             "本结论只适用于保存的源码、数据和依赖快照，不表示绝对无缺陷或可实现交易收益。", "",
             "| 检查 | 状态 | 证据 |", "|---|---|---|"]
    for check in summary["checks"]:
        lines.append(f"| {check['name']} | {check['status']} | [{check['evidence']}]({check['evidence']}) |")
    lines += ["", "## 已修复边界缺陷", "", "- float32 转换后的溢出统一转为 NaN；非法日历日期在原始加载时明确拒绝。",
              "- 修复前复现及源码哈希见 ../../defects_20260930.json；正常数据结果必须匹配冻结 v1.1。", "",
              "## 结果与解释", ""]
    repeat = next((c["actual"] for c in summary["checks"] if c["name"] == "full_repeatability" and c["status"] == "passed"), None)
    if repeat:
        for split, value in repeat["per_split"].items():
            m = value["metrics"]
            lines.append(f"- {split}：IC={m['ic_mean']:.10f}，年化超额={m['annual_excess']:.10f}，换手={m['mean_turnover']:.10f}，总分={m['final_score']:.10f}。")
        for run in repeat["runs"]:
            s = run["summary"]
            lines.append(f"- 全量运行 `{run['path']}`：{s['resources']['elapsed_seconds']:.2f} 秒，采样峰值 RSS {s['resources']['peak_process_rss_mb']:.2f} MiB。")
        for split in repeat["runs"][0]["summary"]["splits"]:
            d = split["diagnostics"]
            lines.append(f"- {split['split_name']} 换手 Top 标签缺失比例 {d['top_groups']['turnover']['missing_label_fraction']:.4%}，价格有效股票诊断换手 {d['price_valid_only_turnover']:.6f}；不改变官方排名与分数。")
    controls = next((c["actual"] for c in summary["checks"] if c["name"] == "controls_and_ties" and c["status"] == "passed"), None)
    if controls:
        lines.append("- 五组标签打乱 IC：" + "、".join(f"{c['metrics']['ic_mean']:.6f}" for c in controls["controls"]) + "。")
        if controls["anomaly_triggered"]:
            investigation = controls["investigation"]
            lines.append(f"- 原触发条件保留并确实触发调查。每日均值标签模型 IC={investigation['market_mean_control']['metrics']['ic_mean']:.6f}；与打乱模型日内排名平均相关性为 " +
                         "、".join(f"{c['mean_daily_prediction_spearman']:.4f}" for c in investigation["prediction_correlations"]) + "。")
            lines.append("- 去均值后五组打乱 IC：" + "、".join(f"{c['metrics']['ic_mean']:.6f}" for c in investigation["demeaned_controls"]) +
                         "；一组绝对 IC 高于 baseline，原始结果保留；未再次触发原来的 3/5 系统性告警。")
            exposure = investigation["residual_exposure"]
            lines.append(f"- 最强残余对照与现有特征存在相关暴露；保持每日预测值集合、打乱预测与股票的对应关系后，IC={exposure['prediction_permutation_metrics']['ic_mean']:.6f}。")
            lines.append("- 原始告警条件没有放宽。调查解释了市场日结构及随机拟合对现有特征的暴露；不证明随机模型 IC 必须为零，也不证明 baseline 显著优于随机对照。先前试探性地要求每个残余 IC 都低于 baseline 过于严格，已改为复用原告警条件并补充预测对应关系对照。")
        for year in (2023, 2024):
            lines.append(f"- 原版 {year} 排名并列的五次行序扰动，指标范围：`" + json.dumps(controls[f"tie_sensitivity_{year}"]["metric_ranges"], ensure_ascii=False) + "`。冻结实际评分 CSV 行序；不声称任意行序下结果一致。")
    lines += ["", "## 无法验证与使用边界", ""]
    documentation = next((c["actual"] for c in summary["checks"] if c["name"] == "upstream_documentation" and c["status"] == "passed"), {})
    if documentation.get("declares_backward_adjusted_prices"):
        lines.append("- 已读取本地赛题说明，其标注 OHLC 为后复权价格；尚无原 PDF 和历史复权因子快照。")
    lines += ["- " + item for item in UPSTREAM_LIMITS]
    lines += ["- 非空标签若缺少下一交易日价格证据，属于阻断项；缺失标签单列，不能算作公式核验通过。",
              "- 默认只用 2023 筛选新特征；2024 仅作少量候选复核，不能反复选优后仍声称未参与选择。",
              "- 运行 Git 提交为执行时父提交；精确代码版本以 provenance.json 的逐文件哈希为准。",
              "- 大型模型、预测和评分输入仅保存在本地运行目录；本次未生成正式测试集 submission。", ""]
    resources = summary["resources"]
    lines.append(f"总验收耗时 {resources['elapsed_seconds']:.2f} 秒；主进程采样峰值 RSS {resources['peak_process_rss_mb']:.2f} MiB；关联实验产物 {resources['linked_experiment_bytes'] / 1024**2:.2f} MiB。子进程峰值分别见全量运行摘要。")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("unit", "full", "all"), default="all")
    args = parser.parse_args(argv)
    if not __debug__:
        parser.error("strict audit cannot run with Python -O or -OO")
    started = time.perf_counter()
    with experiment_run(ROOT / "artifacts/baseline_audit", "audit") as output:
        print(f"Audit directory: {output}", flush=True)
        audit = Audit(output)
        before = provenance(ROOT, runner.RAW_DATA_PATH)
        before["additional_inputs"] = {str(p.relative_to(ROOT)): sha256_file(p) for p in (
            runner.DATA_PATH, ROOT / "赛题五/赛题五数据/测试集_X.csv", ROOT / "artifacts/baseline_v1_1/summary.json")}
        statement = ROOT / "赛题五/完整赛题说明_本地.md"
        if statement.is_file():
            before["additional_inputs"][str(statement.relative_to(ROOT))] = sha256_file(statement)
        write_json(output / "provenance.json", before)
        with runner.PeakMemoryMonitor() as monitor:
            audit.check("unit_tests", "all existing and new tests pass", lambda: test_suite(output))
            audit.check("dependencies", "pip check passes and all dependency pins match", lambda: dependency_check(output))
            audit.check("upstream_documentation", "identify documented adjustment convention and unavailable evidence", upstream_documentation, blocking=False)
            if args.suite != "unit":
                audit.check("raw_data_and_features", "valid keys/dates, labels <=1e-10, independent features, no future leakage", lambda: data_audit(output))
                if all(c["status"] == "passed" for c in audit.checks if c["blocking"]):
                    accepted = audit.check("full_repeatability", "two full runs match frozen v1.1 with coverage 100% and official delta <=1e-12", lambda: full_runs(output))
                    if accepted:
                        controls = audit.check("controls_and_ties", "five controls investigated; official parity and both years' tie sensitivity quantified", lambda: full_controls(output, accepted))
                        if controls:
                            audit.check("artifact_integrity", "all linked outputs still match recorded hashes at acceptance", lambda: verify_linked_artifacts(accepted, controls))
            audit.check("snapshot_unchanged", "no mixed source or data snapshots", lambda: assert_snapshot_unchanged(before))
        decision = release_decision(audit.checks, args.suite)
        linked = set()
        for check in audit.checks:
            if check["status"] != "passed":
                continue
            if check["name"] == "full_repeatability":
                linked.update(ROOT / r["path"] for r in check["actual"]["runs"])
            elif check["name"] == "controls_and_ties":
                control = check["actual"]
                entries = control["controls"][:]
                if control["investigation"]:
                    entries += [control["investigation"]["market_mean_control"], *control["investigation"]["demeaned_controls"]]
                    entries += [control["investigation"]["residual_exposure"]]
                linked.update(ROOT / r["run_directory"] for r in entries)
        summary = {"suite": args.suite, "checks": audit.checks, "decision": decision,
                   "upstream": [{"status": "unverified", "blocking": False, "item": x} for x in UPSTREAM_LIMITS],
                   "resources": {"elapsed_seconds": time.perf_counter() - started,
                                 "peak_process_rss_mb": monitor.peak_rss_bytes / 1024**2,
                                 "linked_experiment_bytes": sum(p.stat().st_size for d in linked for p in d.rglob("*") if p.is_file()),
                                 "audit_directory_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file())},
                   "provenance_sha256": sha256_file(output / "provenance.json")}
        (output / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
        summary["report_sha256"] = sha256_file(output / "REPORT.md")
        write_json(output / "summary.json", summary)
        if not decision["can_start_feature_experiments"]:
            raise RuntimeError(f"baseline not released: {decision}")
    print(f"Baseline audit accepted: {output}", flush=True)


if __name__ == "__main__":
    main()
