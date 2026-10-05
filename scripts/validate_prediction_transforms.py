"""Fixed F1/S1 on original frozen predictions. No training or model inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.diagnose_frozen_predictions import (
    aggregate_daily, aggregate_top, contributions, daily_metrics,
    metric_difference, official_evaluator, table, top_diagnostics,
)
from src.data.baseline_panel import load_raw_baseline_panel
from src.metrics.official import score_official
from src.validation.experiment import prediction_hash, sha256_file as _sha256_file

CONFIG = ROOT / "configs/model_optimization_step2.json"
HANDOFF1 = ROOT / "docs/model_optimization/STEP1_HANDOFF.json"
FREEZE = ROOT / "artifacts/features34_step5/FROZEN_CANDIDATES.json"
RAW = ROOT / "赛题五/赛题五数据/训练集.csv"
REPORT = ROOT / "docs/model_optimization/STEP2_REPORT.md"
HANDOFF = ROOT / "docs/model_optimization/STEP2_HANDOFF.json"
OWN = ["scripts/validate_prediction_transforms.py", "configs/model_optimization_step2.json",
       "tests/test_prediction_transforms.py"]
CONTROLS = ["S4R_lean31_minus4", "S4R_full34_minus3", "baseline10"]
SPLITS = ["dev_2021", "dev_2022", "primary_2023"]
KEYS = ["ts_code", "trade_date"]
METRICS = ["ic_mean", "annual_excess", "mean_turnover", "final_score",
           "ic_contribution", "excess_contribution", "stability_contribution", "price_valid_turnover"]


def sha256_file(path):
    return _sha256_file(Path(path))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    # All JSON writes are exclusive; an existing experiment cannot be overwritten.
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def csv(frame, path):
    frame.to_csv(path, index=False, float_format="%.17g", mode="x", encoding="utf-8")


def load_transitions(path):
    # Empty entered/exited code lists mean an empty set, not a missing value.
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False)


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def verify_hashes(mapping, base=ROOT):
    for path, expected in mapping.items():
        require(sha256_file(base / path) == expected, f"changed file: {base / path}")


def validated(frame):
    require(list(frame.columns) == KEYS + ["pred"], "unexpected prediction columns")
    require(not frame[KEYS].isna().any().any() and not frame.duplicated(KEYS).any(), "invalid/duplicate keys")
    dates = frame.trade_date.astype(str)
    require(dates.str.fullmatch(r"\d{8}").all() and pd.to_datetime(dates, format="%Y%m%d", errors="coerce").notna().all(), "invalid dates")
    require(np.isfinite(frame.pred.to_numpy(dtype=float)).all(), "nonfinite predictions")


def blend(a, b):
    validated(a)
    validated(b)
    idx = pd.MultiIndex.from_frame(a[KEYS])
    other = pd.MultiIndex.from_frame(b[KEYS])
    require(len(idx) == len(other) and not len(idx.difference(other)), "fusion keys differ")
    aligned = b.set_index(KEYS).pred.reindex(idx).to_numpy(dtype=float)
    out = a.copy()
    out["pred"] = .5 * a.pred.to_numpy(dtype=float) + .5 * aligned
    validated(out)
    return out


def smooth(a, calendar, model_ids):
    """Look up exactly (same stock, previous market day, same model)."""
    validated(a)
    dates = np.asarray(calendar, dtype=np.int64)
    require(len(dates) > 0 and np.all(np.diff(dates) > 0), "calendar must be strictly increasing")
    require(np.isin(a.trade_date, dates).all(), "prediction date outside calendar")
    ids = np.asarray(model_ids, dtype=object)
    require(len(ids) == len(a) and not pd.isna(ids).any(), "invalid model identity")
    previous = dict(zip(dates[1:], dates[:-1]))
    prev_dates = a.trade_date.map(previous).fillna(-1).astype(np.int64)
    current_index = pd.MultiIndex.from_arrays([a.ts_code, a.trade_date, ids])
    previous_index = pd.MultiIndex.from_arrays([a.ts_code, prev_dates, ids])
    lag = pd.Series(a.pred.to_numpy(dtype=float), index=current_index).reindex(previous_index).to_numpy()
    # Each annual validation is independent; never borrow last year's model output.
    usable = np.isfinite(lag) & (prev_dates.to_numpy() // 10000 == a.trade_date.to_numpy() // 10000)
    out = a.copy()
    original = a.pred.to_numpy(dtype=float)
    values = original.copy()
    values[usable] = .8 * original[usable] + .2 * lag[usable]
    out["pred"] = values
    validated(out)
    trace = a[KEYS].copy()
    trace["previous_market_date"] = prev_dates.to_numpy()
    trace["used_previous_original"] = usable
    trace["previous_original_pred"] = lag
    trace["model_sha256"] = ids
    return out, trace


def validate_config(config):
    require(config["splits"] == SPLITS and config["controls"] == CONTROLS, "fixed sources/splits changed")
    require(config["experiments"] == {
        "F1": {"formula": "0.5 * original27 + 0.5 * original31", "weights": [.5, .5]},
        "S1": {"formula": "0.8 * original27_today + 0.2 * original27_previous_market_day", "weights": [.8, .2]},
    }, "only the authorized F1/S1 matrix is supported")
    require(config["comparison"]["risk_flags"] == {
        "annual_delta_vs27_below": -.005, "monthly_delta_below": -.005,
        "severe_monthly_delta_below": -.02, "small_mean_absolute_delta_below": .003,
    }, "fixed risk flags changed")
    require(config["scope"] == dict(prediction_comparisons=6, new_training_runs=0,
        evaluate_2024=False, official_test_selection=False, stack_transforms=False,
        extra_weights=False, replace_frozen_candidate=False, final_submission=False,
        interval_estimation=False), "scope changed")


def gate():
    """Check final STEP1 evidence before registering/scoring anything new."""
    h = read(HANDOFF1)
    require(h["accepted"] and h["stage"] == "step1_complete_stop", "STEP1 incomplete")
    checked = 0
    for pk, hk in [("report", "report_sha256"), ("freeze_path", "freeze_sha256"),
        ("acceptance_path", "acceptance_sha256"), ("delivery_acceptance_path", "delivery_acceptance_sha256"),
        ("input_manifest_path", "input_manifest_sha256"),
        ("independent_diagnostics_acceptance_path", "independent_diagnostics_acceptance_sha256")]:
        require(sha256_file(h[pk]) == h[hk], f"STEP1 handoff mismatch: {pk}")
        checked += 1
    for key in ["acceptance_path", "delivery_acceptance_path", "independent_diagnostics_acceptance_path"]:
        require(read(h[key])["accepted"], f"STEP1 unresolved acceptance: {key}")
    delivery = read(h["delivery_acceptance_path"])
    verify_hashes(delivery["all_output_sha256"], Path(h["artifact_directory"]))
    checked += len(delivery["all_output_sha256"])
    for rec in h["historical_inputs"]:
        for key, path in rec["paths"].items():
            require(sha256_file(path) == rec["sha256"][key], f"STEP1 source mismatch: {path}")
            checked += 1
        for entry in rec["executed_sources"].values():
            require(sha256_file(entry["snapshot"]) == entry["sha256"], "historical executed source changed")
            checked += 1
    frozen = read(FREEZE)
    inputs = [r for r in h["historical_inputs"] if r["split"] in SPLITS]
    require(len(inputs) == 9 and {(r["candidate"], r["split"]) for r in inputs} ==
        {(c, s) for c in CONTROLS for s in SPLITS}, "incomplete original development controls")
    for rec in inputs:
        index = next(r for r in frozen["development_runs"] if (r["candidate"], r["split"]) == (rec["candidate"], rec["split"]))
        require(rec["directory"] == index["directory"] and rec["summary_sha256"] == index["summary_sha256"], "not an original frozen run")
        if rec["candidate"] != "baseline10":
            candidate = next(c for c in frozen["candidates"] if c["candidate"] == rec["candidate"])
            require(rec["features"] == candidate["features"] and len(rec["features"]) == candidate["feature_count"], "frozen feature mismatch")
    return h, inputs, checked


def preserved_files():
    names = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT).decode("utf-8").split("\0")
    names += [".git/config", ".git/hooks/pre-push"]
    return {n: sha256_file(ROOT / n) for n in sorted(set(names)) if n and n not in OWN and (ROOT / n).is_file()}


def load_input(rec, panel):
    pq = pd.read_parquet(rec["paths"]["predictions"])
    pq["ts_code"] = pq.ts_code.astype(str)
    validated(pq)
    pred = pd.read_csv(rec["paths"]["score_prediction"])
    truth = pd.read_csv(rec["paths"]["score_truth"])
    x = pd.read_csv(rec["paths"]["score_x"])
    spec = read(ROOT / "configs/features34_step3.json")["splits"][rec["split"]]
    require(spec == rec["split_spec"], "old split mismatch")
    raw = panel[panel.trade_date.between(spec["valid_start"], spec["valid_end"])].copy().reset_index(drop=True)
    raw["ts_code"] = raw.ts_code.astype(str)
    for frame in [pq, pred, truth, x]:
        require(not frame.duplicated(KEYS).any(), "duplicate source keys")
        pd.testing.assert_frame_equal(frame[KEYS], raw[KEYS], check_dtype=False, check_exact=True)
    pd.testing.assert_series_equal(truth.y_ret_1d, raw.y_ret_1d, check_dtype=False, check_exact=True)
    pd.testing.assert_series_equal(x.flag_limit_up, raw.flag_limit_up, check_dtype=False, check_exact=True)
    require(np.allclose(pred.pred, pq.pred, rtol=0, atol=1e-15), "historical scoring prediction mismatch")
    summary = read(rec["paths"]["summary"])
    sm = next(s for s in summary["splits"] if s["split_name"] == rec["split"])
    verify_hashes(sm["file_sha256"], ROOT / rec["directory"] / rec["split"])
    require(summary["features"] == rec["features"] and prediction_hash(pq.pred) == sm["prediction_sha256"], "original model output mismatch")
    train = panel.trade_date.between(spec["train_start"], spec["train_end"]) & panel.is_trainable.eq(1)
    require(int(train.sum()) == sm["train_samples"] and len(raw) == sm["valid_prediction_rows"], "split eligibility changed")
    model = Path(rec["paths"]["model"]).read_text(encoding="utf-8")
    feature_names = next(line for line in model.splitlines() if line.startswith("feature_names=")).split("=", 1)[1].split()
    require(feature_names == rec["features"] and model.count("\nTree=") == 100, "saved model columns/tree count mismatch")
    return pq, pred, truth, x, raw, sm


def prepare():
    require(not REPORT.exists() and not HANDOFF.exists(), "existing STEP2 report/handoff; do not overwrite")
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "ivor-work", "ivor-work required")
    config = read(CONFIG)
    validate_config(config)
    h, inputs, checked = gate()
    preserved = preserved_files()
    panel = load_raw_baseline_panel(RAW)
    require(sha256_file(RAW) == read(FREEZE)["source"]["data"]["sha256"], "raw data changed")
    calendar = sorted(int(d) for d in panel.trade_date.unique() if d < 20240000)
    preflight = []
    for rec in inputs:
        pq, pred, truth, x, raw, sm = load_input(rec, panel)
        preflight.append(dict(candidate=rec["candidate"], split=rec["split"], rows=len(pq),
            model_sha256=rec["sha256"]["model"], columns=len(rec["features"]),
            keys_raw_order_labels_flags_train_counts_verified=True,
            parquet_vs_historical_csv_max_abs=float(np.max(np.abs(pq.pred - pred.pred)))))
    output = ROOT / "artifacts/model_optimization/step2" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "preserved.json", preserved)
    write(output / "market_calendar.json", calendar)
    write(output / "preflight.json", dict(accepted=True, step1_hashes_verified=checked, inputs=preflight,
        unresolved_source_scoring_implementation_issues=[], raw_sha256=sha256_file(RAW)))
    dependencies = [*OWN, "scripts/diagnose_frozen_predictions.py", "src/metrics/official.py",
        "src/data/baseline_panel.py", "src/validation/experiment.py", "赛题五/evaluate.py",
        "赛题五/完整赛题说明_本地.md", "AGENTS.md", "AGENTS.override.md", "configs/features34_step3.json",
        "docs/features34/RESULTS.md", "docs/features34/FROZEN_MODEL_VALIDATION_REPORT.md",
        "docs/model_optimization/STEP1_REPORT.md", "docs/model_optimization/STEP1_HANDOFF.json",
        "artifacts/features34_step5/FROZEN_CANDIDATES.json"]
    source = {n: sha256_file(ROOT / n) for n in dependencies if (ROOT / n).is_file()}
    for name in source:
        dest = output / "executed_sources" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    matrix = [dict(experiment=e, split=s, year=int(s[-4:]),
        original27_model_sha256=next(r["sha256"]["model"] for r in inputs if r["candidate"] == CONTROLS[0] and r["split"] == s),
        original31_model_sha256=next(r["sha256"]["model"] for r in inputs if r["candidate"] == CONTROLS[1] and r["split"] == s))
        for e in ["F1", "S1"] for s in SPLITS]
    registration = dict(registered_at=datetime.now(timezone.utc).isoformat(), config=config, matrix=matrix,
        inputs=inputs, step1_handoff_sha256=sha256_file(HANDOFF1), freeze_sha256=sha256_file(FREEZE),
        raw_sha256=sha256_file(RAW), calendar_sha256=sha256_file(output / "market_calendar.json"),
        source_sha256=source, parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        branch="ivor-work", preserved_sha256=sha256_file(output / "preserved.json"),
        prediction_precision="Formula on original float64 parquet; save float64 parquet and %.17g CSV. Official default CSV parse explicitly audited; no invalid-sample ranking intervention.")
    write(output / "registration.json", registration)
    write(output / "registration_sha256.json", dict(sha256=sha256_file(output / "registration.json")))
    write(output / "environment.json", dict(python=sys.executable, version=sys.version,
        packages={n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scipy", "pyarrow", "pytest"]}))
    print(str(output), flush=True)


def verify_registration(output):
    reg = read(output / "registration.json")
    require(sha256_file(output / "registration.json") == read(output / "registration_sha256.json")["sha256"], "registration changed")
    validate_config(reg["config"])
    current_sources = dict(reg["source_sha256"])
    if (output / "repair_registration.json").exists():
        import ast
        repair = read(output / "repair_registration.json")
        require(sha256_file(output / "repair_registration.json") == read(output / "repair_registration_sha256.json")["sha256"], "repair registration changed")
        require(repair["original_registration_sha256"] == sha256_file(output / "registration.json"), "repair detached from original matrix")
        require(set(repair["repaired_source_sha256"]) <= set(OWN), "repair outside task sources")
        verify_hashes(repair["repaired_source_sha256"], output / "repaired_sources")
        current_sources.update(repair["repaired_source_sha256"])
        def functions(path):
            tree = ast.parse(Path(path).read_text(encoding="utf-8"))
            return {f.name: ast.dump(f, include_attributes=False) for f in tree.body if isinstance(f, ast.FunctionDef)}
        old = functions(output / "executed_sources" / OWN[0])
        new = functions(ROOT / OWN[0])
        for name in ["blend", "smooth", "load_input", "run", "compare", "risk_summary", "validate_config"]:
            require(old[name] == new[name], f"repair changed experimental implementation: {name}")
    verify_hashes(current_sources)
    verify_hashes(reg["source_sha256"], output / "executed_sources")
    require(sha256_file(RAW) == reg["raw_sha256"] and sha256_file(output / "market_calendar.json") == reg["calendar_sha256"], "data/calendar changed")
    require(len(reg["matrix"]) == 6 and {(r["experiment"], r["split"]) for r in reg["matrix"]} == {(e, s) for e in ["F1", "S1"] for s in SPLITS}, "matrix changed")
    require(sha256_file(output / "preserved.json") == reg["preserved_sha256"], "preservation manifest changed")
    verify_hashes(read(output / "preserved.json"))
    require(subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "ivor-work", "branch changed")
    return reg


def compare(frame, keys):
    rows = []
    for experiment in ["F1", "S1"]:
        a = frame[frame.candidate.eq(experiment)].set_index(keys).sort_index()
        for control in CONTROLS:
            b = frame[frame.candidate.eq(control)].set_index(keys).sort_index()
            require(a.index.equals(b.index), "comparison period mismatch")
            delta = a[METRICS] - b[METRICS]
            delta.columns = [c + "_delta" for c in METRICS]
            delta = delta.reset_index().assign(after=experiment, before=control)
            rows.append(delta)
    return pd.concat(rows, ignore_index=True)


def risk_summary(annual, monthly):
    rows = []
    for (a, b), g in annual.groupby(["after", "before"]):
        m = monthly[monthly.after.eq(a) & monthly.before.eq(b)]
        w = g.loc[g.final_score_delta.idxmin()]
        wm = m.loc[m.final_score_delta.idxmin()]
        mean = float(g.final_score_delta.mean())
        rows.append(dict(after=a, before=b, mean_delta=mean, small_change=abs(mean) < .003,
            worst_year=int(w.year), worst_year_delta=float(w.final_score_delta),
            mixed_annual_directions=bool(g.final_score_delta.gt(0).any() and g.final_score_delta.lt(0).any()),
            negative_years=",".join(g.loc[g.final_score_delta.lt(0), "year"].astype(str)),
            annual_risk_vs27=bool(b == CONTROLS[0] and g.final_score_delta.lt(-.005).any()),
            negative_months=int(m.final_score_delta.lt(0).sum()), risk_months=int(m.final_score_delta.lt(-.005).sum()),
            severe_months=int(m.final_score_delta.lt(-.02).sum()), worst_month=int(wm.month), worst_month_delta=float(wm.final_score_delta)))
    return pd.DataFrame(rows)


def run(output):
    reg = verify_registration(output)
    panel = load_raw_baseline_panel(RAW)
    calendar = read(output / "market_calendar.json")
    annual, dailies, qualities, members, transitions, parities, formulas = [], [], [], [], [], [], []
    for split in SPLITS:
        cache = {}
        for rec in [r for r in reg["inputs"] if r["split"] == split]:
            cache[rec["candidate"]] = (rec, load_input(rec, panel))
        a = cache[CONTROLS[0]][1][0]
        b = cache[CONTROLS[1]][1][0]
        p1 = blend(a, b)
        p2, trace = smooth(a, calendar, [cache[CONTROLS[0]][0]["sha256"]["model"]] * len(a))
        year = int(split[-4:])
        for name in [*CONTROLS, "F1", "S1"]:
            if name in CONTROLS:
                rec, (pq, pred, truth, x, raw, sm) = cache[name]
                score_path = Path(rec["paths"]["score_prediction"])
            else:
                rec, (_, _, truth, x, raw, _) = cache[CONTROLS[0]]
                transformed = p1 if name == "F1" else p2
                directory = output / name / split
                directory.mkdir(parents=True, exist_ok=False)
                transformed.to_parquet(directory / "predictions.parquet", index=False)
                score_path = directory / "evaluation_predictions.csv"
                csv(transformed, score_path)
                pred = pd.read_csv(score_path)
                pd.testing.assert_frame_equal(pred[KEYS], a[KEYS], check_dtype=False, check_exact=True)
                require(np.max(np.abs(pred.pred - transformed.pred)) <= 1e-15, "new CSV precision loss")
                if name == "S1":
                    trace.to_parquet(directory / "lag_trace.parquet", index=False)
                formulas.append(dict(experiment=name, split=split, rows=len(pred),
                    prediction_sha256=prediction_hash(transformed.pred), file_sha256=sha256_file(directory / "predictions.parquet"),
                    score_csv_sha256=sha256_file(score_path), csv_max_abs=float(np.max(np.abs(pred.pred - transformed.pred))),
                    smoothing_used_previous=int(trace.used_previous_original.sum()) if name == "S1" else None,
                    smoothing_fallback=int((~trace.used_previous_original).sum()) if name == "S1" else None))
            result = score_official(pred, truth, x, return_details=True)
            details = result.pop("details")
            reference = official_evaluator()(str(score_path), str(Path(rec["paths"]["score_truth"]).parent))
            diff = metric_difference(result, reference)
            require(max(abs(v) for v in diff.values()) <= 1e-12, "official parity failed")
            old_diff = metric_difference(result, sm["metrics"]) if name in CONTROLS else None
            if old_diff is not None:
                require(max(abs(v) for v in old_diff.values()) <= 1e-12, "control no longer matches frozen score")
            parities.append(dict(candidate=name, split=split, official_difference=diff, historical_difference=old_diff,
                score_path=str(score_path), truth_path=rec["paths"]["score_truth"], x_path=rec["paths"]["score_x"]))
            joined = pred.merge(truth, on=KEYS, validate="one_to_one").merge(raw[KEYS + ["flag_limit_up", "is_price_valid"]], on=KEYS, validate="one_to_one")
            q, mem, trans, valid, _ = top_diagnostics(joined, details)
            day = daily_metrics(details).merge(valid, on="trade_date", how="left", validate="one_to_one")
            for frame in [q, mem, trans, day]:
                frame["candidate"], frame["year"] = name, year
                frame["month"] = frame.trade_date // 100
            annual.append(dict(candidate=name, year=year, split=split, **result,
                price_valid_turnover=float(valid.price_valid_turnover.mean())))
            dailies.append(day)
            qualities.append(q)
            members.append(mem)
            transitions.append(trans)
            print(f"Scored {name} {split}: {result['final_score']:.12f}", flush=True)
    a = contributions(pd.DataFrame(annual))
    d = pd.concat(dailies, ignore_index=True)
    m = pd.concat([aggregate_daily(g, "month").assign(candidate=n, year=y) for (n, y), g in d.groupby(["candidate", "year"], sort=False)], ignore_index=True)
    m = m.merge(d.groupby(["candidate", "month"]).price_valid_turnover.mean().reset_index(), on=["candidate", "month"], validate="one_to_one")
    q = pd.concat(qualities, ignore_index=True)
    ad, md = compare(a, ["year"]), compare(m, ["year", "month"])
    ad["annual_risk_vs27"] = ad.before.eq(CONTROLS[0]) & ad.final_score_delta.lt(-.005)
    md["risk_below_minus005"] = md.final_score_delta.lt(-.005)
    md["severe_below_minus02"] = md.final_score_delta.lt(-.02)
    tables = dict(annual_metrics=a, monthly_metrics=m, daily_metrics=d, annual_comparison=ad, monthly_comparison=md,
        risk_summary=risk_summary(ad, md), negative_months=md[md.final_score_delta.lt(0)],
        daily_top_quality=q, annual_top_quality=aggregate_top(q, "year"), monthly_top_quality=aggregate_top(q, "month"),
        daily_top_members=pd.concat(members, ignore_index=True), daily_top_transitions=pd.concat(transitions, ignore_index=True))
    for name, frame in tables.items():
        csv(frame, output / (name + ".csv"))
    write(output / "official_parity.json", parities)
    write(output / "transform_manifest.json", formulas)
    write(output / "run_complete.json", dict(complete=True, new_training_runs=0, new_prediction_comparisons=6,
        official_scored_controls=9, official_scored_transforms=6, evaluated_2024=0, evaluated_official_test=0,
        annual_rows=len(a), monthly_rows=len(m), monthly_comparison_rows=len(md)))


def independent_smoothing(a, calendar, model_hash):
    # Separate audit implementation: sorted stock rows, adjacent observed record
    # accepted only when its date is the immediately previous market date.
    order = np.lexsort((a.trade_date.to_numpy(), a.ts_code.to_numpy()))
    sub = a.iloc[order]
    original = sub.pred.to_numpy(dtype=float)
    date = sub.trade_date.to_numpy()
    code = sub.ts_code.to_numpy()
    previous = dict(zip(calendar[1:], calendar[:-1]))
    expected_previous = sub.trade_date.map(previous).fillna(-1).to_numpy()
    usable = np.zeros(len(sub), dtype=bool)
    usable[1:] = (code[1:] == code[:-1]) & (date[:-1] == expected_previous[1:]) & (date[:-1] // 10000 == date[1:] // 10000)
    values = original.copy()
    indices = np.flatnonzero(usable)
    values[indices] = .8 * original[indices] + .2 * original[indices - 1]
    restored = np.empty(len(sub), dtype=float)
    restored[order] = values
    return restored


def audit(output):
    reg = verify_registration(output)
    require(read(output / "tests_result.json")["exit_code"] == 0, "tests did not pass")
    panel = load_raw_baseline_panel(RAW)
    calendar = read(output / "market_calendar.json")
    require(calendar == sorted(int(d) for d in panel.trade_date.unique() if d < 20240000), "calendar not raw market dates")
    annual = pd.read_csv(output / "annual_metrics.csv", float_precision="round_trip")
    daily = pd.read_csv(output / "daily_metrics.csv", float_precision="round_trip")
    month = pd.read_csv(output / "monthly_metrics.csv", float_precision="round_trip")
    quality = pd.read_csv(output / "daily_top_quality.csv", float_precision="round_trip")
    membership = pd.read_csv(output / "daily_top_members.csv")
    transition = load_transitions(output / "daily_top_transitions.csv")
    verified = []
    for split in SPLITS:
        cache = {r["candidate"]: (r, load_input(r, panel)) for r in reg["inputs"] if r["split"] == split}
        a, b = cache[CONTROLS[0]][1][0], cache[CONTROLS[1]][1][0]
        pd.testing.assert_frame_equal(a[KEYS], b[KEYS], check_exact=True)
        expected = {"F1": .5 * a.pred.to_numpy() + .5 * b.pred.to_numpy(),
            "S1": independent_smoothing(a, calendar, cache[CONTROLS[0]][0]["sha256"]["model"])}
        for name in [*CONTROLS, "F1", "S1"]:
            if name in CONTROLS:
                rec, (_, pred, truth, x, raw, sm) = cache[name]
                score_path = Path(rec["paths"]["score_prediction"])
            else:
                rec, (_, _, truth, x, raw, _) = cache[CONTROLS[0]]
                dest = output / name / split
                pq = pd.read_parquet(dest / "predictions.parquet")
                validated(pq)
                pd.testing.assert_frame_equal(pq[KEYS], a[KEYS], check_exact=True)
                np.testing.assert_array_equal(pq.pred, expected[name])
                score_path = dest / "evaluation_predictions.csv"
                pred = pd.read_csv(score_path)
                parsed_exact = pd.read_csv(score_path, float_precision="round_trip")
                np.testing.assert_array_equal(parsed_exact.pred, pq.pred)
                require(np.max(np.abs(pred.pred - pq.pred)) <= 1e-15, "CSV changed predictions")
                # Full-size future truncation and perturbation, before scoring.
                cutoff = sorted(a.trade_date.unique())[len(a.trade_date.unique()) // 2]
                prefix = a.trade_date.le(cutoff)
                changed_a, changed_b = a.copy(), b.copy()
                changed_a.loc[~prefix, "pred"] = 999 + changed_a.loc[~prefix, "pred"] * -11
                changed_b.loc[~prefix, "pred"] = -999 + changed_b.loc[~prefix, "pred"] * 17
                if name == "F1":
                    altered = blend(changed_a, changed_b)
                    truncated = blend(a[prefix].copy(), b[prefix].copy())
                else:
                    model = rec["sha256"]["model"]
                    altered, _ = smooth(changed_a, calendar, [model] * len(a))
                    truncated, _ = smooth(a[prefix].copy(), [d for d in calendar if d <= cutoff], [model] * int(prefix.sum()))
                    trace = pd.read_parquet(dest / "lag_trace.parquet")
                    _, reconstructed = smooth(a, calendar, [model] * len(a))
                    pd.testing.assert_frame_equal(trace, reconstructed, check_exact=True)
                np.testing.assert_array_equal(altered.loc[prefix, "pred"], pq.loc[prefix, "pred"])
                np.testing.assert_array_equal(truncated.pred, pq.loc[prefix, "pred"])
            result = score_official(pred, truth, x, return_details=True)
            details = result.pop("details")
            reference = official_evaluator()(str(score_path), str(Path(rec["paths"]["score_truth"]).parent))
            diff = metric_difference(result, reference)
            require(max(abs(v) for v in diff.values()) <= 1e-12, "audit official parity failed")
            year = int(split[-4:])
            saved = annual[annual.candidate.eq(name) & annual.year.eq(year)].iloc[0]
            for k, v in result.items():
                require(abs(saved[k] - v) <= 1e-12, "annual report mismatch")
            joined = pred.merge(truth, on=KEYS, validate="one_to_one").merge(raw[KEYS + ["flag_limit_up", "is_price_valid"]], on=KEYS, validate="one_to_one")
            q, mem, trans, valid, _ = top_diagnostics(joined, details)
            computed = daily_metrics(details).merge(valid, on="trade_date", how="left", validate="one_to_one")
            for frame, old in [(computed, daily), (q, quality), (mem, membership), (trans, transition)]:
                existing = old[old.candidate.eq(name) & old.year.eq(year)].reset_index(drop=True)
                pd.testing.assert_frame_equal(frame, existing[frame.columns], check_dtype=False, check_exact=True)
            computed["month"] = computed.trade_date // 100
            grouped = aggregate_daily(computed, "month")
            existing = month[month.candidate.eq(name) & month.year.eq(year)].reset_index(drop=True)
            pd.testing.assert_frame_equal(grouped, existing[grouped.columns], check_dtype=False, check_exact=True)
            require(abs(valid.price_valid_turnover.mean() - saved.price_valid_turnover) <= 1e-12, "price turnover mismatch")
            for k in ["ic_mean", "annual_excess", "mean_turnover", "final_score"]:
                require(abs(aggregate_daily(computed.assign(year=year), "year")[k].iloc[0] - saved[k]) <= 1e-12, "annual aggregation mismatch")
            verified.append(dict(candidate=name, split=split, official_difference=diff,
                keys_formula_causality_verified=name in ["F1", "S1"], monthly_top_and_quality_recomputed=True))
            print(f"Audit {name} {split} passed", flush=True)
    for name, frame, keys in [("annual_comparison", annual, ["year"]), ("monthly_comparison", month, ["year", "month"])]:
        computed = compare(frame, keys)
        saved = pd.read_csv(output / (name + ".csv"), float_precision="round_trip")
        pd.testing.assert_frame_equal(computed, saved[computed.columns], check_dtype=False, check_exact=True)
        np.testing.assert_allclose(computed.final_score_delta, computed.ic_contribution_delta + computed.excess_contribution_delta + computed.stability_contribution_delta, rtol=0, atol=1e-15)
    recomputed_risk = risk_summary(compare(annual, ["year"]), compare(month, ["year", "month"]))
    old_risk = pd.read_csv(output / "risk_summary.csv", float_precision="round_trip", keep_default_na=False)
    pd.testing.assert_frame_equal(recomputed_risk, old_risk, check_dtype=False, check_exact=True)
    verify_registration(output)
    write(output / "acceptance.json", dict(accepted=True, verified=verified, actual_transformed_prediction_runs=6,
        new_training_runs=0, evaluated_2024=0, evaluated_official_test=0, final_submission=False,
        formula_independently_recomputed=True, full_size_future_truncation_and_perturbation_passed=True,
        unchanged_inputs_old_artifacts_and_push_protection=True, registration_sha256=sha256_file(output / "registration.json")))


def process(output, phase, command):
    started = time.perf_counter()
    at = datetime.now(timezone.utc).isoformat()
    log = output / (phase + ".log")
    with log.open("xb") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    receipt = dict(command=command, cwd=str(ROOT), started_at=at, exit_code=result.returncode,
        elapsed_seconds=time.perf_counter() - started, log=str(log), log_sha256=sha256_file(log))
    write(output / (phase + "_result.json"), receipt)
    print(f"{phase}: exit={result.returncode}, elapsed={receipt['elapsed_seconds']:.2f}s", flush=True)
    require(result.returncode == 0, f"{phase} failed; see {log}")


def execute(output):
    verify_registration(output)
    process(output, "tests", [sys.executable, "-X", "utf8", "-B", "-m", "pytest",
        "tests/test_prediction_transforms.py", "tests/test_frozen_prediction_diagnostics.py",
        "tests/test_official_score.py", "-q", "-p", "no:cacheprovider"])
    for phase in ["run", "audit"]:
        process(output, phase, [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), phase, "--output", str(output)])
    print("Fixed comparisons and audit complete; finalize after saving evidence-based interpretation.", flush=True)


def finalize(output):
    require(read(output / "acceptance.json")["accepted"], "cannot finalize unaccepted run")
    verify_registration(output)
    require(not REPORT.exists() and not HANDOFF.exists(), "STEP2 docs already exist")
    a = pd.read_csv(output / "annual_metrics.csv", float_precision="round_trip")
    ad = pd.read_csv(output / "annual_comparison.csv", float_precision="round_trip")
    md = pd.read_csv(output / "monthly_comparison.csv", float_precision="round_trip")
    risk = pd.read_csv(output / "risk_summary.csv", keep_default_na=False)
    q = pd.read_csv(output / "annual_top_quality.csv")
    conclusions = []
    for name in ["F1", "S1"]:
        g = ad[ad.after.eq(name) & ad.before.eq(CONTROLS[0])]
        r = risk[risk.after.eq(name) & risk.before.eq(CONTROLS[0])].iloc[0]
        conclusions.append(f"{name}相对原27列：三年平均分差{r.mean_delta:+.9f}，最差年份{int(r.worst_year)}为{r.worst_year_delta:+.9f}；36个月中负分差{int(r.negative_months)}个、低于−0.005共{int(r.risk_months)}个、低于−0.02共{int(r.severe_months)}个，最差{int(r.worst_month)}为{r.worst_month_delta:+.9f}。平均IC差{g.ic_mean_delta.mean():+.9f}、年化超额差{g.annual_excess_delta.mean():+.9f}、官方换手差{g.mean_turnover_delta.mean():+.9f}、价格有效换手差{g.price_valid_turnover_delta.mean():+.9f}。")
    # Recommendation is filled after the fixed comparisons, without changing rules.
    narrative_path = output / "interpretation.json"
    require(narrative_path.exists(), "save evidence-based interpretation before final delivery")
    narrative = read(narrative_path)
    reg = read(output / "registration.json")
    lines = ["# 冻结27/31列预测的固定融合与因果平滑验证", "", narrative["conclusion"], "",
        *conclusions, "", "## 来源、矩阵与边界", "",
        f"第一步来源门核对{read(output / 'preflight.json')['step1_hashes_verified']}项哈希，无未解决来源、评分或实现问题。权威候选为{FREEZE}中的S4R_lean31_minus4（27列）与S4R_full34_minus3（31列），共9项原年度对照；完整目录和哈希见registration.json。未使用旧lean27/lean31或B1/B2模型。", "",
        f"事前登记SHA-256：`{sha256_file(output / 'registration.json')}`。唯一目录：`{output}`。", "",
        "仅2021/2022/2023既有开发切分，F1=0.5×原27+0.5×原31；S1=0.8×原27当日+0.2×原27前一市场交易日。6项变化比较，训练和模型推断均0次。两实验独立，未叠加或追加权重、强度、年份。", "",
        "S1从完整原始市场交易日历精确查找同股票、同模型、紧邻前日的原预测。验证首日、年度/模型边界、前日不存在时回退当日原值；月份连续。无递归、后向填充、跨缺失日借用或未来读取。原行顺序及全部键保留，全部预测有限。lag_trace.parquet逐行记录匹配与回退。", "",
        "公式使用原冻结Parquet的float64预测，保存float64 Parquet及17位CSV；官方评分读取新CSV和原封存标签/标记CSV，对照直接使用旧CSV。CSV默认解析的末位差≤1e-15，逐项留档；round_trip读取逐值等于输出Parquet。没有按标签或价格有效性调整排序。", "",
        "## 年度指标和对照分差", "", table(a, ["candidate", "year", "final_score", "ic_mean", "annual_excess", "mean_turnover", "price_valid_turnover"]), "",
        table(ad, ["after", "before", "year", "final_score_delta", "ic_mean_delta", "annual_excess_delta", "mean_turnover_delta", "ic_contribution_delta", "excess_contribution_delta", "stability_contribution_delta"]), "",
        "## 跨年与月度风险", "",
        "年度相对原27分差<−0.005标记风险；所有对照月度分差<−0.005标记、<−0.02突出；三年平均分差绝对值<0.003标记小幅变化。均为描述性标记，不是统计显著性或自动采用门槛。跨年方向冲突指同时有正、负年度分差，任何负年另行列出。", "",
        table(risk, ["after", "before", "mean_delta", "small_change", "worst_year", "worst_year_delta", "mixed_annual_directions", "negative_years", "annual_risk_vs27", "negative_months", "risk_months", "severe_months", "worst_month", "worst_month_delta"]), "",
        "以下列出全部36个月×2方案对三个对照的分差；完整IC、收益、换手分差在monthly_comparison.csv。月初换手保留上月前一有效日，年度首日无换手。全年各指标按自身有效日均值计分，不平均月分替代全年；月度年化超额是当月日均×252，不是当月累计资金收益。", ""]
    pivot = md.pivot(index=["after", "month"], columns="before", values="final_score_delta").reset_index()
    lines += [table(pivot, ["after", "month", *CONTROLS]), "", "所有负月份另存negative_months.csv，其中明显退化如下：", "",
        table(md[md.final_score_delta.lt(-.005)], ["after", "before", "month", "final_score_delta", "ic_contribution_delta", "excess_contribution_delta", "stability_contribution_delta", "severe_below_minus02"]), "",
        "## Top质量及收益代价", "",
        "官方收益Top排除涨停和缺标签；官方换手Top只排除涨停，因此缺标签、价格无效成员仍可入选。价格有效换手仅用当日OHLC有限、正数、高低价一致性与非涨停过滤，不用未来标签；只作诊断。质量占比按入选股票—日期次数加权，保存年度/月度/逐日计数、三种完整Top名单和相邻转换。", "",
        table(q, ["candidate", "year", "top_type", "top_count", "missing_label_fraction", "invalid_price_fraction"]), "",
        narrative["fusion"], "", narrative["smoothing"], "", narrative["risk"], "",
        "## 验收、执行与限制", "",
        "15项年度评分（9个原对照+6个变化）均与未修改赛题五/evaluate.py核对；独立验收再次读取全部来源和实际输出、使用另一个股票排序/相邻日期实现复算S1，重建官方评分、逐日/月度表、Top和质量表，核对分项及分差。6项完整预测执行未来截断和扰动检查，之前输出逐值不变。必要单元测试覆盖键重排/缺键/重键、公式、同股票、月份/年度/模型边界、缺失交易日、非递归、有限值和因果性。", "",
        "未评估任何新方案2024或官方测试分数，未用测试期选权重；未训练、重推断、改变冻结入口或输出最终比赛submission。原始数据、特征、名单、标签、资格、缺失策略、模型、旧评分器/切分、历史产物和push保护核验不变。", "",
        "未做区间估计，不能声称显著改善。2021—2023已有大量历史开发与选择；即使计算固定预测区间，也不能覆盖历史选择偏差。本步因果性验收仅证明输出不读取未来预测，不能证明收益的因果效应。未验证交易成本、容量、未来收益或实盘可执行性。", "",
        f"解释器：{sys.executable}。命令、完整日志、退出码和耗时见各*_result.json；源/输出哈希见registration.json、transform_manifest.json、delivery_acceptance.json。测试、运行、验收均成功后仅提交相关代码、配置、测试和小型报告；大型本地产物不提交。", "",
        "```powershell", r".\.venv\Scripts\python.exe -X utf8 -B scripts/validate_prediction_transforms.py prepare",
        f'.\\.venv\\Scripts\\python.exe -X utf8 -B scripts/validate_prediction_transforms.py execute --output "{output}"', "```", "",
        "验收完成后停止；保留27列主候选和31列备选，不自动替换。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    write(output / "delivery_acceptance.json", dict(accepted=True, acceptance_sha256=sha256_file(output / "acceptance.json"),
        report_sha256=sha256_file(REPORT), output_sha256={str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()},
        stop_after_step2=True, new_training_runs=0))
    write(HANDOFF, dict(stage="step2_complete_stop", accepted=True, artifact_directory=str(output),
        report=str(REPORT), report_sha256=sha256_file(REPORT), matrix=reg["matrix"], rules=reg["config"],
        registration_path=str(output / "registration.json"), registration_sha256=sha256_file(output / "registration.json"),
        acceptance_path=str(output / "acceptance.json"), acceptance_sha256=sha256_file(output / "acceptance.json"),
        delivery_acceptance_path=str(output / "delivery_acceptance.json"), delivery_acceptance_sha256=sha256_file(output / "delivery_acceptance.json"),
        actual_outputs=read(output / "transform_manifest.json"), sources=reg["inputs"], source_sha256=reg["source_sha256"],
        recommendation=narrative, annual_metrics=a.to_dict("records"), comparison_summary=risk.to_dict("records"),
        commands={p.stem: read(p) for p in output.glob("*_result.json")}, new_training_runs=0,
        evaluated_2024=0, evaluated_official_test=0, frozen_candidate_replaced=False, final_submission=False,
        limitations=["No interval estimates or significance claim", "Historical development/selection bias remains", "Official turnover includes invalid/missing-label rows", "No future/after-cost executable return validation"], next_action="Stop; no further training or experiment authorized."))
    print("STEP2 accepted and sealed; stop.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "execute", "run", "audit", "finalize"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            prepare()
        else:
            require(args.output is not None, "--output required")
            output = args.output.resolve()
            require(output.is_relative_to(ROOT / "artifacts/model_optimization/step2") and output.is_dir(), "invalid output directory")
            {"execute": execute, "run": run, "audit": audit, "finalize": finalize}[args.phase](output)
    except Exception:
        if args.output is not None and args.output.is_dir():
            write(args.output / ("failure_" + args.phase + "_" + datetime.now(timezone.utc).strftime("%H%M%S%f") + ".json"),
                dict(accepted=False, phase=args.phase, traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
