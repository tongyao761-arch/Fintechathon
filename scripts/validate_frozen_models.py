"""Fixed 2+12 run validation, independent of all frozen training entrypoints."""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.run_lightgbm_baseline import (MODEL_PARAMS, RAW_DATA_PATH, PeakMemoryMonitor,
    compare_metrics, environment_versions, score_saved_inputs, split_masks)
from scripts.run_features34 import check_frozen
from scripts.run_features34_step3 import validate_dates
from scripts.run_features34_step4 import verify_artifact, protection_snapshot
from src.data.baseline_panel import load_raw_baseline_panel, extract_truth_files
from src.data.frozen_test_adapter import bridge_x, load_test_x, training_boundary
from src.features.baseline_v1 import FEATURE_COLUMNS as BASE_COLUMNS
from src.features.features34 import build_features34, FEATURE_COLUMNS, FEATURE_DEFINITIONS
from src.metrics.diagnostics import baseline_diagnostics
from src.validation.experiment import (create_run_directory, prediction_hash, provenance,
    sha256_file, write_json)
from src.validation.submission import export_submission, validate_submission

CONFIG = ROOT / "configs/frozen_models_validation.json"
TEST_PATH = ROOT / "赛题五/赛题五数据/测试集_X.csv"
IDS = ["S4R_lean31_minus4", "S4R_full34_minus3"]
SPLITS = ["dev_2021", "dev_2022", "primary_2023"]
OWN = ["src/data/frozen_test_adapter.py", "scripts/validate_frozen_models.py",
    "configs/frozen_models_validation.json", "tests/test_frozen_models_validation.py"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv(frame, path):
    frame.to_csv(path, index=False, float_format="%.17g")


def validate_config(cfg):
    if (cfg["candidates"] != IDS or cfg["splits"] != SPLITS
        or cfg["changes"] != {"B1": {"num_leaves": 15}, "B2": {"reg_lambda": 10.0}}
        or cfg["new_training_runs"] != {"A": 2, "B": 12}
        or cfg["freeze"] != "artifacts/features34_step5/FROZEN_CANDIDATES.json"
        or cfg["stage"] != "frozen_test_adaptation_and_model_sensitivity"):
        raise ValueError("outside the authorized fixed 2+12 matrix")


def parameters(frozen, probe=None):
    if frozen != MODEL_PARAMS:
        raise AssertionError("frozen model parameters differ from actual constants")
    result = dict(frozen)
    if probe is not None:
        changes = {"B1": {"num_leaves": 15}, "B2": {"reg_lambda": 10.0}}
        if probe not in changes:
            raise ValueError("unknown probe")
        result.update(changes[probe])
    return result


def prepare(output):
    cfg = read(CONFIG); validate_config(cfg)
    freeze = read(ROOT / cfg["freeze"])
    reg6 = read(ROOT / "artifacts/features34_step6/registration.json")
    acc6 = read(ROOT / "artifacts/features34_step6/acceptance.json")
    if not acc6["accepted"] or reg6["freeze_sha256"] != sha256_file(ROOT / cfg["freeze"]):
        raise AssertionError("step6/freeze contract mismatch")
    for name, digest in acc6["evidence_sha256"].items():
        if sha256_file(ROOT / "artifacts/features34_step6" / name) != digest:
            raise AssertionError(f"step6 evidence changed: {name}")
    for source in (freeze["source"], reg6["source"]):
        check_frozen(source["source_sha256"])
    check_frozen(read(ROOT / "docs/features34/FROZEN_REFERENCE.json"))
    meta = provenance(ROOT, RAW_DATA_PATH)
    if meta["git"]["branch"] != "ivor-work" or meta["data"] != freeze["source"]["data"]:
        raise AssertionError("branch/data changed")
    if meta["dependencies"] != freeze["source"]["dependencies"]:
        raise AssertionError("dependencies changed")
    for record in freeze["candidates"]:
        if record["feature_count"] != len(record["features"]):
            raise AssertionError("frozen column count mismatch")
        if record["feature_definitions"] != {c: FEATURE_DEFINITIONS[c] for c in record["features"]}:
            raise AssertionError("frozen formula mismatch")
        parameters(record["model_params"])
    if [r["candidate"] for r in freeze["candidates"]] != IDS:
        raise AssertionError("frozen candidate identity mismatch")
    # Read actual step6 index, not merely a prose declaration of acceptance.
    references = list(freeze["development_runs"])
    repeat_index = read(ROOT / "artifacts/features34_step6/run_index.json")["runs"]
    for rec in repeat_index:
        d = ROOT / rec["directory"]; item = rec["item"]
        cols = next(r["features"] for r in freeze["candidates"] if r["candidate"] == item["candidate"])
        if sha256_file(d / "summary.json") != rec["summary_sha256"]:
            raise AssertionError("step6 indexed summary changed")
        verify_artifact(d, item["split"], tuple(cols))
    for rec in references:
        d = ROOT / rec["directory"]
        cols = BASE_COLUMNS if rec["candidate"] == "baseline10" else tuple(next(
            r["features"] for r in freeze["candidates"] if r["candidate"] == rec["candidate"]))
        if sha256_file(d / "summary.json") != rec["summary_sha256"]:
            raise AssertionError("reference indexed summary changed")
        sm = verify_artifact(d, rec["split"], cols)
        prior = read(d / "provenance.json")
        if prior["data"] != meta["data"] or prior["dependencies"] != meta["dependencies"]:
            raise AssertionError("reference data/dependency mismatch")
        # Old runner authorization metadata differs across stages. Frozen executable
        # data, features, training masks, metrics and diagnostics must match exactly.
        for name in ["src/data/baseline_panel.py", "src/features/features34.py",
                     "src/features/baseline_v1.py", "src/metrics/official.py",
                     "src/metrics/diagnostics.py", "scripts/run_lightgbm_baseline.py",
                     "configs/features34_step3.json", "configs/splits.yaml",
                     "赛题五/evaluate.py"]:
            if name in prior["source_sha256"] and sha256_file(ROOT / name) != prior["source_sha256"][name]:
                raise AssertionError(f"reference executed dependency differs: {name}")
        rec["model_sha256"] = sha256_file(d / rec["split"] / "models/lightgbm.txt")
        rec["predictions_sha256"] = sha256_file(d / rec["split"] / "predictions.parquet")
        rec["parameters"] = sm["model_params"]
        rec["historical_runner_sha256"] = prior["source_sha256"].get("scripts/run_features34.py")
        rec["current_runner_sha256"] = sha256_file(ROOT/"scripts/run_features34.py")
        rec["runner_source_note"] = "Earlier stages extended runner authorization/metadata; current frozen sources verified; saved model predictions and scores independently reproduced before reuse."
    snapshot = protection_snapshot()
    snapshot["existing_untracked"] = {n:h for n,h in snapshot["existing_untracked"].items()
        if n not in OWN and not n.startswith("artifacts/frozen_models_validation/")}
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    snapshot["tracked"] = {n:sha256_file(ROOT/n) for n in tracked if n and (ROOT/n).is_file()}
    write_json(output / "preserved.json", snapshot)
    meta["test_data"] = dict(path=TEST_PATH.relative_to(ROOT).as_posix(),
        bytes=TEST_PATH.stat().st_size, sha256=sha256_file(TEST_PATH))
    write_json(output / "provenance.json", meta)
    for name in meta["source_sha256"]:
        dest = output / "executed_sources" / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    registration = dict(config=cfg, config_sha256=sha256_file(CONFIG),
        freeze_sha256=sha256_file(ROOT/cfg["freeze"]), registered_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        references=references, repeats=repeat_index,
        matrix_A=[dict(candidate=n, parameters=parameters(freeze["candidates"][0]["model_params"]),
            training="2018-2024 eligible labels; purge last observed history date; X retained") for n in IDS],
        matrix_B=[dict(candidate=n, probe=p, split=s, change=cfg["changes"][p],
            parameters=parameters(freeze["candidates"][0]["model_params"], p),
            split_spec=read(ROOT/"configs/features34_step3.json")["splits"][s])
            for n in IDS for p in cfg["changes"] for s in SPLITS],
        provenance_sha256=sha256_file(output/"provenance.json"))
    write_json(output / "registration.json", registration)
    write_json(output / "registration_hash.json", dict(sha256=sha256_file(output/"registration.json")))
    write_json(output / "run_index.json", dict(runs=[]))
    write_json(output / "attempts.json", dict(attempts=[]))
    write_json(output / "failures.json", dict(failures=[]))
    print("Sources, indexed artifacts and protections verified; fixed matrix registered.", flush=True)


def stable(output):
    reg = read(output / "registration.json")
    if read(output/"registration_hash.json")["sha256"] != sha256_file(output/"registration.json"):
        raise AssertionError("registered matrix changed")
    if sha256_file(CONFIG) != reg["config_sha256"] or sha256_file(ROOT/reg["config"]["freeze"]) != reg["freeze_sha256"]:
        raise AssertionError("configuration/freeze changed")
    meta = read(output/"provenance.json")
    current_sources = dict(meta["source_sha256"])
    repair_path = output/"repair_registration.json"
    if repair_path.exists():
        repair = read(repair_path)
        if repair["original_registration_sha256"] != sha256_file(output/"registration.json"):
            raise AssertionError("repair changed original matrix registration")
        if set(repair["repaired_source_sha256"]) != {"scripts/validate_frozen_models.py"}:
            raise AssertionError("repair outside independent runner")
        current_sources.update(repair["repaired_source_sha256"])
        for name, digest in repair["repaired_source_sha256"].items():
            if sha256_file(output/"repaired_sources"/name) != digest:
                raise AssertionError("repair execution snapshot changed")
    check_frozen(current_sources)
    for n, h in meta["source_sha256"].items():
        if sha256_file(output/"executed_sources"/n) != h:
            raise AssertionError("executed source snapshot changed")
    snap = read(output/"preserved.json")
    for group in snap.values():
        for name, digest in group.items():
            if sha256_file(Path(name) if Path(name).is_absolute() else ROOT/name) != digest:
                raise AssertionError(f"preexisting file/protection changed: {name}")
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "ivor-work":
        raise AssertionError("branch changed")


def monthly_missing(panel, features, columns, candidate, scope):
    rows = []
    months = panel.trade_date.astype(str).str[:6]
    for month, indices in months.groupby(months, sort=True).groups.items():
        frame = features.loc[indices, columns]
        for c in columns:
            values = frame[c].to_numpy()
            rows.append(dict(candidate=candidate, scope=scope, month=month, feature=c, rows=len(values),
                missing=int(np.isnan(values).sum()), infinite=int(np.isinf(values).sum()),
                nonfinite=int((~np.isfinite(values)).sum()), missing_fraction=float(np.isnan(values).mean()),
                all_inputs_missing_fraction=float(frame.isna().all(axis=1).mean())))
    return pd.DataFrame(rows)


def test_diagnostics(pred, quality, destination):
    joined = pred.merge(quality, on=["ts_code", "trade_date"], validate="one_to_one")
    rows, sets, turns, valid_turns = [], [], [], []
    previous = previous_valid = None
    previous_date = previous_valid_date = None
    for date, group in joined.groupby("trade_date", sort=True):
        eligible = group[group.flag_limit_up.eq(0)]
        if len(eligible) < 100:
            previous = None
        else:
            top = eligible.sort_values("pred", ascending=False).iloc[:max(len(eligible)//10, 1)]
            current = set(top.ts_code)
            sets.append(dict(trade_date=int(date), top_codes=",".join(sorted(current))))
            rows.append(dict(trade_date=int(date), top_count=len(top),
                invalid_price_count=int(top.is_price_valid.eq(0).sum()),
                baseline_all_missing_count=int(top.baseline_all_missing.sum()),
                candidate_all_missing_count=int(top.candidate_all_missing.sum())))
            if previous is not None:
                turns.append(dict(trade_date=int(date), previous_trade_date=int(previous_date),
                    turnover=1-len(previous & current)/len(previous | current)))
            previous, previous_date = current, date
        valid = group[group.flag_limit_up.eq(0) & group.is_price_valid.eq(1)]
        if len(valid) < 100:
            previous_valid = None
        else:
            current = set(valid.sort_values("pred", ascending=False).ts_code.iloc[:len(valid)//10])
            if previous_valid is not None:
                valid_turns.append(dict(trade_date=int(date), previous_trade_date=int(previous_valid_date),
                    turnover=1-len(previous_valid & current)/len(previous_valid | current)))
            previous_valid, previous_valid_date = current, date
    daily = pd.DataFrame(rows); turnover = pd.DataFrame(turns); alternative = pd.DataFrame(valid_turns)
    for name, frame in dict(daily_top_sets=pd.DataFrame(sets), daily_top_diagnostics=daily,
        daily_turnover=turnover, daily_price_valid_turnover=alternative).items():
        csv(frame, destination / (name+".csv"))
    daily["month"] = daily.trade_date.astype(str).str[:6]
    monthly = daily.groupby("month").sum(numeric_only=True).drop(columns="trade_date")
    monthly["invalid_price_fraction"] = monthly.invalid_price_count/monthly.top_count
    for name, frame in (("official_turnover", turnover), ("price_valid_turnover", alternative)):
        frame["month"] = frame.trade_date.astype(str).str[:6]
        monthly[name] = frame.groupby("month").turnover.mean()
    distribution = pred.assign(month=pred.trade_date.astype(str).str[:6]).groupby("month").pred.agg(
        ["count", "mean", "std", "min", "max", "median"])
    for q in (.01, .05, .95, .99):
        distribution[f"q{q}"] = pred.assign(month=pred.trade_date.astype(str).str[:6]).groupby("month").pred.quantile(q)
    csv(distribution.reset_index(), destination/"prediction_distribution.csv")
    csv(monthly.reset_index(), destination/"monthly_test_diagnostics.csv")
    return dict(official_prediction_turnover=float(turnover.turnover.mean()),
        price_valid_turnover=float(alternative.turnover.mean()),
        top_invalid_price_fraction=float(daily.invalid_price_count.sum()/daily.top_count.sum()),
        top_baseline_all_missing_fraction=float(daily.baseline_all_missing_count.sum()/daily.top_count.sum()),
        top_candidate_all_missing_fraction=float(daily.candidate_all_missing_count.sum()/daily.top_count.sum()),
        true_ic=None, true_excess=None, true_complete_score=None)


def fit(output, item, x, y, valid_x, params):
    attempts = read(output/"attempts.json")
    attempts["attempts"].append(dict(item=item, parameters=params, started_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        training_x_hash=prediction_hash(x.to_numpy().ravel()), training_y_hash=prediction_hash(y), status="started"))
    write_json(output/"attempts.json", attempts)
    model = lgb.LGBMRegressor(**params)
    model.fit(x, y, feature_name=list(x.columns))
    if any(model.get_params().get(k) != v for k,v in params.items()) or model.feature_name_ != list(x.columns):
        raise AssertionError("actual model inputs/parameters mismatch")
    if model.booster_.num_trees() != 100:
        raise AssertionError("tree budget changed")
    d = output/item["id"]; d.mkdir(exist_ok=False)
    model.booster_.save_model(str(d/"lightgbm.txt"))
    values = model.predict(valid_x)
    reload = lgb.Booster(model_file=str(d/"lightgbm.txt"))
    if reload.feature_name() != list(x.columns) or not np.array_equal(values, reload.predict(valid_x)):
        raise AssertionError("reload changed predictions/columns")
    if not np.isfinite(values).all():
        raise AssertionError("nonfinite predictions")
    write_json(d/"config.json", dict(item=item, parameters=params, features=list(x.columns),
        training_samples=len(x), prediction_rows=len(valid_x), actual_booster_params=model.booster_.params,
        model_reload_equal=True, num_trees=reload.num_trees()))
    attempts["attempts"][-1]["status"] = "fit_and_reload_passed"
    write_json(output/"attempts.json", attempts)
    return d, values, reload


def record(output, item, destination, result):
    write_json(destination/"summary.json", result)
    write_json(destination/"files.json", {p.relative_to(destination).as_posix():sha256_file(p)
        for p in destination.rglob("*") if p.is_file()})
    index = read(output/"run_index.json")
    index["runs"].append(dict(item=item, directory=destination.relative_to(ROOT).as_posix(),
        summary_sha256=sha256_file(destination/"summary.json"), files_sha256=sha256_file(destination/"files.json")))
    write_json(output/"run_index.json", index)


def stage_a(output, history, hist_features, cols_union, freeze):
    raw_test, test = load_test_x(TEST_PATH)
    boundary = training_boundary(history, test); write_json(output/"boundary.json", boundary)
    joined = bridge_x(history, test)
    print(f"A: observed boundary {boundary}; calculating bridged X.", flush=True)
    features = build_features34(joined, cols_union)
    past = joined.history_row_id.ge(0)
    indices = joined.loc[past].sort_values("history_row_id").index
    pd.testing.assert_frame_equal(features.loc[indices].reset_index(drop=True), hist_features, check_exact=True)
    test_mask = joined.test_row_id.ge(0)
    test_indices = joined.loc[test_mask].sort_values("test_row_id").index
    tf = features.loc[test_indices].reset_index(drop=True)
    tq = joined.loc[test_indices, ["ts_code", "trade_date", "flag_limit_up", "is_price_valid"]].reset_index(drop=True)
    tq["ts_code"] = tq.ts_code.astype(str)
    pd.testing.assert_frame_equal(tq[["ts_code", "trade_date"]], raw_test[["ts_code", "trade_date"]], check_dtype=False)
    cutoff = int(sorted(test.trade_date.unique())[test.trade_date.nunique()//2])
    prefix = joined.trade_date.le(cutoff)
    truncated = build_features34(joined.loc[prefix], cols_union)
    pd.testing.assert_frame_equal(features.loc[prefix], truncated, check_exact=True)
    # Change every later row, including flags and ranking pool membership.
    changed = joined.copy(); future = ~prefix
    for c in ("open", "high", "low", "close", "vol", "amount"):
        changed.loc[future, c] = changed.loc[future, c]*np.float32(1.37)
    changed.loc[future, "close"] = np.nan
    changed.loc[future, "is_price_valid"] = 0
    for c in ("flag_limit_up", "flag_limit_down"):
        changed.loc[future, c] = 1-changed.loc[future, c]
    perturbed = build_features34(changed, cols_union)
    pd.testing.assert_frame_equal(features.loc[prefix], perturbed.loc[prefix], check_exact=True)
    earlier_indices = joined.index[test_mask & prefix]
    prefix_features = [truncated.loc[earlier_indices].copy(), perturbed.loc[earlier_indices].copy()]
    original_prefix = features.loc[earlier_indices].copy()
    del changed, perturbed, truncated, features, joined; gc.collect()
    train = history.trade_date.between(boundary["train_start"], boundary["train_end"]) & history.is_trainable.eq(1) & history.y_ret_1d.notna()
    if train[history.trade_date.eq(boundary["purge_date"])].any():
        raise AssertionError("cross-boundary label admitted")
    labels = history.loc[train, "y_ret_1d"].astype("float32")
    if not np.isfinite(labels).all():
        raise AssertionError("training labels nonfinite")
    missing = []
    for candidate in freeze["candidates"]:
        name = candidate["candidate"]; columns = candidate["features"]
        item = dict(id="A_"+name, phase="A", candidate=name)
        print(f"Training {item['id']}; {int(train.sum())} eligible rows.", flush=True)
        began = time.perf_counter()
        d, values, model = fit(output, item, hist_features.loc[train, columns], labels,
            tf.loc[:, columns], parameters(candidate["model_params"]))
        for altered in prefix_features:
            if not np.array_equal(model.predict(original_prefix.loc[:, columns]), model.predict(altered.loc[:, columns])):
                raise AssertionError("future X changed earlier fixed-model predictions")
        pred = raw_test[["ts_code", "trade_date"]].copy(); pred["pred"] = values
        pred.to_parquet(d/"predictions.parquet", index=False)
        report = export_submission(pred, raw_test, d/"adaptation_validation_predictions.csv")
        csv(report.pop("daily_prediction_summary").reset_index(), d/"daily_prediction_summary.csv")
        saved = pd.read_csv(d/"adaptation_validation_predictions.csv")
        validate_submission(saved, raw_test)
        pd.testing.assert_frame_equal(saved[["ts_code", "trade_date"]], raw_test[["ts_code", "trade_date"]], check_dtype=False)
        # CSV is exported through the existing interface; parser round-trip can
        # differ by one ulp, so use Parquet as exact prediction truth.
        if not np.allclose(saved.pred, values, rtol=0, atol=1e-15):
            raise AssertionError("CSV changed predictions materially")
        quality = tq.copy()
        quality["baseline_all_missing"] = tf.loc[:, BASE_COLUMNS].isna().all(axis=1)
        quality["candidate_all_missing"] = tf.loc[:, columns].isna().all(axis=1)
        diagnostics = test_diagnostics(saved, quality, d)
        missing.extend([monthly_missing(history, hist_features, columns, name, "historical_all_X"),
            monthly_missing(tq, tf, columns, name, "test_all_X")])
        result = dict(candidate=name, phase="A", boundary=boundary, training_samples=int(train.sum()),
            prediction_rows=len(pred), prediction_coverage=1., prediction_sha256=prediction_hash(values),
            all_predictions_finite=True, model_reload_equal=True, columns=columns, validation=report,
            diagnostics=diagnostics, elapsed_seconds=time.perf_counter()-began)
        record(output, item, d, result)
        del model; gc.collect()
    table = pd.concat(missing, ignore_index=True)
    csv(table, output/"monthly_feature_missing.csv")
    reference = table[table.scope.eq("historical_all_X") & table.month.str.startswith("2024")].groupby(["candidate", "feature"]).agg(missing=("missing", "sum"), rows=("rows", "sum"))
    reference["historical_2024_missing_fraction"] = reference.missing/reference.rows
    drift = table[table.scope.eq("test_all_X")].merge(reference[["historical_2024_missing_fraction"]], on=["candidate", "feature"], validate="many_to_one")
    drift["missing_fraction_minus_2024"] = drift.missing_fraction-drift.historical_2024_missing_fraction
    csv(drift, output/"test_missing_vs_2024.csv")
    write_json(output/"stage_A_acceptance.json", dict(accepted=True, runs=2, training_features_exactly_unchanged=True,
        prefix_cutoff=cutoff, truncated_and_perturbed_features_exactly_equal=True,
        both_fixed_models_prefix_predictions_equal=True, no_test_labels_or_trainability=True,
        boundary=boundary, all_test_keys_original_order_finite=True))
    print("A accepted: historical/prefix features exact; two fixed-model predictions causal, finite and complete.", flush=True)


def verify_references(output, history, features, splits):
    reports = []
    for rec in read(output/"registration.json")["references"]:
        d = ROOT/rec["directory"]; s = rec["split"]; name = rec["candidate"]
        sm = read(d/"summary.json"); result = sm["splits"][0]
        train, valid = split_masks(history, splits[s]); columns = sm["features"]
        if result["dates"] != {k:v for k,v in splits[s].__dict__.items() if k != "name"} or result["train_samples"] != int(train.sum()):
            raise AssertionError("reference split/eligibility mismatch")
        saved = pd.read_parquet(d/s/"predictions.parquet")
        keys = history.loc[valid, ["ts_code", "trade_date"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved[["ts_code", "trade_date"]], keys, check_dtype=False, check_categorical=False)
        model = lgb.Booster(model_file=str(d/s/"models/lightgbm.txt"))
        if not np.array_equal(model.predict(features.loc[valid, columns]), saved.pred.to_numpy()):
            raise AssertionError("reference actual-X model prediction mismatch")
        metrics, parity, pred, truth = score_saved_inputs(d/s/"evaluate_input")
        compare_metrics(result["metrics"], {k:v for k,v in metrics.items() if k != "details"})
        pd.testing.assert_frame_equal(truth, history.loc[valid, ["ts_code", "trade_date", "y_ret_1d"]].reset_index(drop=True),
            check_dtype=False, check_categorical=False, check_exact=True)
        reports.append(dict(**rec, model_reproduced=True, all_keys_and_raw_truth_exact=True,
            official_max_difference=parity["max_abs_difference"]))
        print(f"Reused verified reference: {name} {s}", flush=True)
    write_json(output/"reuse_verification.json", dict(references=reports, new_training_runs=0))


def stage_b(output, history, features, splits, freeze):
    if not read(output/"stage_A_acceptance.json")["accepted"]:
        raise AssertionError("stage A must pass first")
    stable(output)  # Matrix and rules were hashed before any B result.
    verify_references(output, history, features, splits)
    for item in read(output/"registration.json")["matrix_B"]:
        item = {**item, "phase": "B", "id": "_".join([item["probe"], item["candidate"], item["split"]])}
        if any(r["item"]["id"] == item["id"] for r in read(output/"run_index.json")["runs"]):
            raise AssertionError("this continuation expects no completed B runs; refuse any repeated training")
        name, s = item["candidate"], item["split"]
        columns = next(r["features"] for r in freeze["candidates"] if r["candidate"] == name)
        train, valid = split_masks(history, splits[s])
        print(f"Training {item['id']}", flush=True)
        began = time.perf_counter()
        labels = history.loc[train, "y_ret_1d"].astype("float32")
        if not np.isfinite(labels).all(): raise AssertionError("nonfinite labels")
        d, values, model = fit(output, item, features.loc[train, columns], labels,
            features.loc[valid, columns], item["parameters"])
        del model
        fixture = d/"evaluate_input"
        extract_truth_files(RAW_DATA_PATH, {fixture/"测试集_Y.csv": (splits[s].valid_start, splits[s].valid_end)})
        pred = history.loc[valid, ["ts_code", "trade_date"]].copy(); pred["pred"] = values
        pred.to_parquet(d/"predictions.parquet", index=False)
        csv(pred, fixture/"submission.csv")
        csv(history.loc[valid, ["ts_code", "trade_date", "flag_limit_up"]], fixture/"测试集_X.csv")
        metrics, parity, scored, truth = score_saved_inputs(fixture); details = metrics.pop("details")
        pd.testing.assert_frame_equal(truth, history.loc[valid, ["ts_code", "trade_date", "y_ret_1d"]].reset_index(drop=True),
            check_exact=True, check_dtype=False, check_categorical=False)
        quality = history.loc[valid, ["ts_code", "trade_date", "flag_limit_up", "is_price_valid"]].copy()
        quality["baseline_features_all_missing"] = features.loc[valid, BASE_COLUMNS].isna().all(axis=1)
        diag, tables = baseline_diagnostics(scored, truth, quality, details)
        quality["baseline_features_all_missing"] = features.loc[valid, columns].isna().all(axis=1)
        cd, ct = baseline_diagnostics(scored, truth, quality, details)
        tables["daily_candidate_missing_diagnostics"] = ct["daily_missing_diagnostics"]
        for filename, frame in {**details, **tables}.items(): csv(frame, d/(filename+".csv"))
        result = dict(candidate=name, probe=item["probe"], split=s, phase="B", columns=columns,
            training_samples=int(train.sum()), prediction_rows=len(pred), prediction_coverage=1.,
            prediction_sha256=prediction_hash(values), model_reload_equal=True,
            metrics=metrics, official_comparison=parity, diagnostics=diag, candidate_diagnostics=cd,
            elapsed_seconds=time.perf_counter()-began)
        record(output, item, d, result)
        print(f"Completed {item['id']}: {metrics['final_score']:.12f}", flush=True)
        gc.collect()


def comparisons(output):
    refs = {(r["candidate"], r["split"]): ROOT/r["directory"] for r in read(output/"registration.json")["references"]}
    annual, monthly, missing = [], [], []
    for rec in read(output/"run_index.json")["runs"]:
        if rec["item"]["phase"] != "B": continue
        item = rec["item"]; d = ROOT/rec["directory"]; r = read(d/"summary.json")
        n, s, p = item["candidate"], item["split"], item["probe"]
        row = dict(candidate=n, probe=p, year=int(s[-4:]), **r["metrics"])
        for label, reference in (("original", refs[(n,s)]), ("baseline10", refs[("baseline10",s)])):
            original = read(reference/"summary.json")["splits"][0]["metrics"]
            for k in ("final_score", "ic_mean", "annual_excess", "mean_turnover"):
                row[k+"_minus_"+label] = r["metrics"][k]-original[k]
            before = pd.read_csv(reference/s/"monthly_metrics.csv", float_precision="round_trip")
            after = pd.read_csv(d/"monthly_metrics.csv", float_precision="round_trip")
            both = after.merge(before, on="month", suffixes=("", "_before"), validate="one_to_one")
            for b in both.to_dict("records"):
                monthly.append(dict(candidate=n, probe=p, year=int(s[-4:]), reference=label, month=int(b["month"]),
                    score=b["score"], score_before=b["score_before"], delta_score=b["score"]-b["score_before"],
                    delta_ic=b["ic"]-b["ic_before"], delta_annual_excess=b["annual_excess"]-b["annual_excess_before"],
                    delta_turnover=b["turnover"]-b["turnover_before"]))
        row["price_valid_turnover"] = r["diagnostics"]["price_valid_only_turnover"]
        row.update({"top_"+k:v for k,v in r["diagnostics"]["top_groups"]["turnover"].items()})
        row["top_candidate_all_missing_fraction"] = r["candidate_diagnostics"]["top_groups"]["turnover"]["all_features_missing_fraction"]
        annual.append(row)
        for role, path, split_dir in (("probe", d, ""), ("original", refs[(n,s)], s), ("baseline10", refs[("baseline10",s)], s)):
            for kind in ("daily_missing_diagnostics", "daily_candidate_missing_diagnostics"):
                source = path/split_dir/(kind+".csv")
                if not source.exists(): continue
                frame = pd.read_csv(source); frame["month"] = frame.trade_date.astype(str).str[:6]
                for (month, top_type), group in frame.groupby(["month", "top_type"]):
                    m = dict(candidate=n, probe=p, role=role, feature_scope="candidate" if "candidate" in kind else "baseline10",
                        month=month, top_type=top_type)
                    for key in ("missing_label", "invalid_price", "all_features_missing"):
                        m[key+"_fraction"] = float(group[key+"_count"].sum()/group.top_count.sum())
                    m["top_count"] = int(group.top_count.sum()); missing.append(m)
            turnover = pd.read_csv(path/split_dir/"daily_price_valid_turnover.csv")
            turnover["month"] = turnover.trade_date.astype(str).str[:6]
            for month, g in turnover.groupby("month"):
                missing.append(dict(candidate=n, probe=p, role=role, month=month, top_type="price_valid_turnover",
                    price_valid_turnover=float(g.turnover.mean())))
    a = pd.DataFrame(annual); m = pd.DataFrame(monthly)
    summaries = []
    for (n,p), g in a.groupby(["candidate", "probe"], sort=False):
        row = dict(candidate=n, probe=p, worst_score_year=int(g.loc[g.final_score.idxmin(), "year"]),
            worst_score=float(g.final_score.min()))
        for ref in ("original", "baseline10"):
            delta = g["final_score_minus_"+ref]
            months = m[m.candidate.eq(n) & m.probe.eq(p) & m.reference.eq(ref)]
            worst = months.loc[months.delta_score.idxmin()]
            row.update({"mean_delta_"+ref:float(delta.mean()), "worst_year_delta_"+ref:float(delta.min()),
                "worst_year_"+ref:int(g.loc[delta.idxmin(), "year"]),
                "mixed_signs_"+ref:bool(delta.lt(0).any() and delta.gt(0).any()),
                "negative_months_"+ref:int(months.delta_score.lt(0).sum()),
                "material_negative_months_"+ref:int(months.delta_score.lt(-.005).sum()),
                "worst_month_"+ref:int(worst.month), "worst_month_delta_"+ref:float(worst.delta_score)})
            row["degradation_months_"+ref] = ";".join(str(int(x)) for x in months.loc[months.delta_score.lt(0), "month"])
        summaries.append(row)
    for name, frame in dict(annual_comparison=a, monthly_comparison=m,
        monthly_missing_diagnostics=pd.DataFrame(missing), sensitivity_summary=pd.DataFrame(summaries)).items():
        csv(frame, output/(name+".csv"))
    if len(a) != 12 or len(m) != 288:
        raise AssertionError("comparison matrix incomplete")


def run(output):
    stable(output)
    cfg = read(CONFIG); freeze = read(ROOT/cfg["freeze"])
    history = load_raw_baseline_panel(RAW_DATA_PATH)
    splits = validate_dates(history.trade_date.unique(), read(ROOT/"configs/features34_step3.json"))
    columns = tuple(c for c in FEATURE_COLUMNS if any(c in r["features"] for r in freeze["candidates"]))
    print("Computing full historical X before any masks.", flush=True)
    features = build_features34(history, columns)
    if (output/"stage_A_acceptance.json").exists():
        if not read(output/"stage_A_acceptance.json")["accepted"]:
            raise AssertionError("cannot reuse failed stage A")
        a_runs = read(output/"run_index.json")["runs"]
        if len(a_runs) != 2 or {r["item"]["id"] for r in a_runs} != {"A_"+n for n in IDS}:
            raise AssertionError("continuation requires exactly the two accepted A runs")
        for rec in a_runs:
            d = ROOT/rec["directory"]
            if sha256_file(d/"summary.json") != rec["summary_sha256"] or sha256_file(d/"files.json") != rec["files_sha256"]:
                raise AssertionError("accepted A artifacts changed")
            for name, h in read(d/"files.json").items():
                if sha256_file(d/name) != h: raise AssertionError("accepted A file changed")
        print("Continuing after record-only failure; accepted A reused, no A retraining.", flush=True)
    else:
        stage_a(output, history, features, columns, freeze)
    stage_b(output, history, features, splits, freeze)
    comparisons(output)
    stable(output)
    for path, digest in ((RAW_DATA_PATH, read(output/"provenance.json")["data"]["sha256"]),
                         (TEST_PATH, read(output/"provenance.json")["test_data"]["sha256"])):
        if sha256_file(path) != digest: raise AssertionError("raw data changed during run")
    write_json(output/"runs_complete.json", dict(runs=14, stage_A=2, stage_B=12, accepted_pending_audit=True))


def audit(output):
    stable(output)
    index = read(output/"run_index.json")["runs"]
    if len(index) != 14 or len(read(output/"attempts.json")["attempts"]) != 14:
        raise AssertionError("training budget mismatch")
    if read(output/"tests_result.json")["exit_code"] != 0 or read(output/"dependency_result.json")["exit_code"] != 0:
        raise AssertionError("tests/dependencies failed")
    reg = read(output/"registration.json")
    expected = {"A_"+n for n in IDS} | {"_".join([i["probe"],i["candidate"],i["split"]]) for i in reg["matrix_B"]}
    if {r["item"]["id"] for r in index} != expected: raise AssertionError("run identities mismatch")
    history = load_raw_baseline_panel(RAW_DATA_PATH)
    freeze = read(ROOT/reg["config"]["freeze"])
    columns = tuple(c for c in FEATURE_COLUMNS if any(c in r["features"] for r in freeze["candidates"]))
    features = build_features34(history, columns)
    splits = validate_dates(history.trade_date.unique(), read(ROOT/"configs/features34_step3.json"))
    raw_test, test = load_test_x(TEST_PATH)
    joined = bridge_x(history, test); bridged = build_features34(joined, columns)
    indices = joined[joined.test_row_id.ge(0)].sort_values("test_row_id").index
    test_features = bridged.loc[indices].reset_index(drop=True)
    quality_test = joined.loc[indices, ["ts_code","trade_date","flag_limit_up","is_price_valid"]].reset_index(drop=True)
    quality_test["ts_code"] = quality_test.ts_code.astype(str)
    del bridged, joined; gc.collect()
    verified = []
    for rec in index:
        d = ROOT/rec["directory"]
        if sha256_file(d/"summary.json") != rec["summary_sha256"] or sha256_file(d/"files.json") != rec["files_sha256"]:
            raise AssertionError("indexed output changed")
        for name, h in read(d/"files.json").items():
            if sha256_file(d/name) != h: raise AssertionError("saved output changed")
        cfg = read(d/"config.json"); params = parameters(MODEL_PARAMS, rec["item"].get("probe"))
        if cfg["parameters"] != params: raise AssertionError("probe params mismatch")
        model = lgb.Booster(model_file=str(d/"lightgbm.txt"))
        if model.feature_name() != cfg["features"] or model.num_trees() != 100:
            raise AssertionError("saved model inputs/trees mismatch")
        saved = pd.read_parquet(d/"predictions.parquet")
        if saved.duplicated(["ts_code", "trade_date"]).any() or not np.isfinite(saved.pred).all():
            raise AssertionError("saved prediction invalid")
        sm = read(d/"summary.json")
        if prediction_hash(saved.pred) != sm["prediction_sha256"]: raise AssertionError("saved prediction values changed")
        if rec["item"]["phase"] == "B":
            split = rec["item"]["split"]; train, valid = split_masks(history, splits[split])
            pd.testing.assert_frame_equal(saved[["ts_code","trade_date"]],
                history.loc[valid, ["ts_code","trade_date"]].reset_index(drop=True), check_dtype=False, check_categorical=False)
            if not np.array_equal(model.predict(features.loc[valid, cfg["features"]]), saved.pred.to_numpy()):
                raise AssertionError("independent historical model reproduction failed")
            if sm["training_samples"] != int(train.sum()): raise AssertionError("training count changed")
            metrics, comparison, scored, truth = score_saved_inputs(d/"evaluate_input")
            pd.testing.assert_frame_equal(truth, history.loc[valid, ["ts_code","trade_date","y_ret_1d"]].reset_index(drop=True),
                check_exact=True, check_dtype=False, check_categorical=False)
            details = metrics.pop("details"); compare_metrics(sm["metrics"], metrics)
            if comparison["max_abs_difference"] > 1e-12: raise AssertionError("official parity failed")
            quality = history.loc[valid, ["ts_code","trade_date","flag_limit_up","is_price_valid"]].copy()
            quality["baseline_features_all_missing"] = features.loc[valid, BASE_COLUMNS].isna().all(axis=1)
            diag, tables = baseline_diagnostics(scored, truth, quality, details)
            quality["baseline_features_all_missing"] = features.loc[valid, cfg["features"]].isna().all(axis=1)
            cd, ct = baseline_diagnostics(scored, truth, quality, details)
            if diag != sm["diagnostics"] or cd != sm["candidate_diagnostics"]:
                raise AssertionError("independent historical diagnostics mismatch")
            tables["daily_candidate_missing_diagnostics"] = ct["daily_missing_diagnostics"]
            for name, frame in {**details, **tables}.items():
                saved_table = pd.read_csv(d/(name+".csv"), float_precision="round_trip")
                # CSV columns (e.g. month) have inferred numeric dtype; compare
                # an identically serialized/parsed reference, without tolerance.
                import io
                normalized = pd.read_csv(io.StringIO(frame.to_csv(index=False, float_format="%.17g")), float_precision="round_trip")
                pd.testing.assert_frame_equal(normalized, saved_table, check_exact=True, check_dtype=False)
        else:
            pd.testing.assert_frame_equal(saved[["ts_code","trade_date"]], raw_test[["ts_code","trade_date"]], check_dtype=False)
            if not np.array_equal(model.predict(test_features.loc[:, cfg["features"]]), saved.pred.to_numpy()):
                raise AssertionError("independent test model reproduction failed")
            exported = pd.read_csv(d/"adaptation_validation_predictions.csv")
            validate_submission(exported, raw_test)
            pd.testing.assert_frame_equal(exported[["ts_code","trade_date"]], raw_test[["ts_code","trade_date"]], check_dtype=False)
            if not np.allclose(exported.pred, saved.pred, rtol=0, atol=1e-15): raise AssertionError("export changed predictions")
            q = quality_test.copy()
            q["baseline_all_missing"] = test_features.loc[:, BASE_COLUMNS].isna().all(axis=1)
            q["candidate_all_missing"] = test_features.loc[:, cfg["features"]].isna().all(axis=1)
            check_dir = output/("audit_"+rec["item"]["id"]); check_dir.mkdir(exist_ok=False)
            diag = test_diagnostics(exported, q, check_dir)
            if diag != sm["diagnostics"]: raise AssertionError("independent test diagnostics mismatch")
            for f in check_dir.iterdir():
                if sha256_file(f) != sha256_file(d/f.name): raise AssertionError("test diagnostic tables mismatch")
        verified.append(dict(id=rec["item"]["id"], model_reproduced_from_raw_X=True,
            prediction_keys_finite_and_values_exact=True, diagnostics_recomputed=True))
        print(f"Audited {rec['item']['id']}", flush=True)
    comparisons(output)
    stable(output)
    evidence = {p.relative_to(output).as_posix():sha256_file(p) for p in output.rglob("*")
        if p.is_file() and "executed_sources" not in p.parts and p.name not in ("acceptance.json", "status.json", "audit.log")}
    write_json(output/"acceptance.json", dict(accepted=True, new_training_runs=14, stage_A=2, stage_B=12,
        frozen_models_not_replaced=True, test_true_returns_unverified=True, no_final_competition_submission=True,
        sources_data_protections_prior_files_unchanged=True, official_parity_tolerance=1e-12,
        verified_runs=verified, evidence_sha256=evidence))
    print("Accepted: fixed 14-run matrix, actual saved models/predictions and official rescoring verified.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run", "resume", "audit"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve() if args.output else None
    if args.phase == "prepare":
        if output is not None: raise ValueError("prepare always creates a unique directory")
        output = create_run_directory(ROOT/"artifacts", "frozen_models_validation")
    elif output is None or not output.is_dir(): raise ValueError("existing registered --output required")
    if not output.is_relative_to(ROOT/"artifacts/frozen_models_validation"):
        raise ValueError("output must stay inside this task's evidence root")
    began = time.perf_counter()
    command = dict(argv=[sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]], cwd=str(ROOT),
        started_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat())
    print(str(output), flush=True)
    try:
        with PeakMemoryMonitor() as memory, (output/(args.phase+".log")).open("x", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                {"prepare":prepare, "run":run, "resume":run, "audit":audit}[args.phase](output)
        command.update(exit_code=0, elapsed_seconds=time.perf_counter()-began,
            peak_rss_mb=memory.peak_rss_bytes/1024**2)
        write_json(output/(args.phase+"_command.json"), command)
        write_json(output/"status.json", dict(status="success" if args.phase == "audit" else args.phase+"_complete"))
        print(json.dumps(command), flush=True)
    except BaseException as exc:
        command.update(exit_code=1, elapsed_seconds=time.perf_counter()-began,
            error_type=type(exc).__name__, error=str(exc))
        write_json(output/(args.phase+"_command.json"), command)
        failures = read(output/"failures.json") if (output/"failures.json").exists() else dict(failures=[])
        failures["failures"].append(command); write_json(output/"failures.json", failures)
        write_json(output/"status.json", dict(status="failed", phase=args.phase, error=str(exc)))
        traceback.print_exc(); raise


if __name__ == "__main__":
    main()
