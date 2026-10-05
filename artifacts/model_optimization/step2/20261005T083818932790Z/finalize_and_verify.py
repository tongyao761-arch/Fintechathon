"""Close finalization logs before validating and sealing the final handoff."""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.validate_prediction_transforms import (
    HANDOFF, REPORT, read, write, require, sha256_file, verify_registration,
)

out = Path(__file__).resolve().parent
command = [sys.executable, "-X", "utf8", "-B", "scripts/validate_prediction_transforms.py",
           "finalize", "--output", str(out)]
started = time.perf_counter()
at = datetime.now(timezone.utc).isoformat()
result = subprocess.run(command, cwd=ROOT, capture_output=True)
log = out / "finalize.log"
log.write_bytes(result.stdout + result.stderr)
write(out / "finalize_result.json", dict(command=command, cwd=str(ROOT), started_at=at,
    exit_code=result.returncode, elapsed_seconds=time.perf_counter() - started,
    log=str(log), log_sha256=sha256_file(log)))
require(result.returncode == 0, "finalize failed; see finalize.log")

h = read(HANDOFF)
h["source_repairs"] = read(out / "repair_registration.json")
h["final_source_sha256"] = {**h["source_sha256"], **h["source_repairs"]["repaired_source_sha256"]}
h["actual_output_paths"] = read(out / "actual_output_paths.json")
h["commands"]["finalize_result"] = read(out / "finalize_result.json")
h["tests"] = dict(passed=35, result_path=str(out / "repair_tests_localtemp_result.json"),
    result_sha256=sha256_file(out / "repair_tests_localtemp_result.json"))
h["failures"] = read(out / "failure_summary.json")
h["execution_record_limitations"] = read(out / "interpretation.json")["execution_note"]
HANDOFF.write_text(json.dumps(h, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

reg = verify_registration(out)
sealed = read(out / "delivery_acceptance.json")
for name, digest in sealed["output_sha256"].items():
    require(sha256_file(out / name) == digest, f"sealed output changed: {name}")
require(sha256_file(REPORT) == sealed["report_sha256"] == h["report_sha256"], "report hash mismatch")
require(sha256_file(out / "acceptance.json") == h["acceptance_sha256"], "acceptance mismatch")
require(read(out / "repair_tests_localtemp_result.json")["exit_code"] == 0, "repaired tests incomplete")
require(read(out / "audit_repaired_result.json")["exit_code"] == 0, "repaired audit incomplete")

# Verify reported aggregations against complete daily evidence.
daily = pd.read_csv(out / "daily_metrics.csv", float_precision="round_trip")
monthly = pd.read_csv(out / "monthly_metrics.csv", float_precision="round_trip")
q = pd.read_csv(out / "daily_top_quality.csv", float_precision="round_trip")
for period in ["year", "month"]:
    actual = q.groupby(["candidate", period, "top_type"]).agg(
        top_count=("top_count", "sum"), missing_label_count=("missing_label_count", "sum"),
        invalid_price_count=("invalid_price_count", "sum"), observations=("trade_date", "size")).reset_index()
    actual["missing_label_fraction"] = actual.missing_label_count / actual.top_count
    actual["invalid_price_fraction"] = actual.invalid_price_count / actual.top_count
    saved = pd.read_csv(out / ("annual_top_quality.csv" if period == "year" else "monthly_top_quality.csv"), float_precision="round_trip")
    pd.testing.assert_frame_equal(actual, saved, check_dtype=False, check_exact=True)
prices = daily.groupby(["candidate", "month"]).price_valid_turnover.mean().reset_index()
joined = prices.merge(monthly[["candidate", "month", "price_valid_turnover"]], on=["candidate", "month"], validate="one_to_one")
np.testing.assert_array_equal(joined.price_valid_turnover_x, joined.price_valid_turnover_y)
ad = pd.read_csv(out / "annual_comparison.csv", float_precision="round_trip")
md = pd.read_csv(out / "monthly_comparison.csv", float_precision="round_trip")
require(ad.annual_risk_vs27.equals(ad.before.eq("S4R_lean31_minus4") & ad.final_score_delta.lt(-.005)), "annual flag mismatch")
require(md.risk_below_minus005.equals(md.final_score_delta.lt(-.005)), "monthly flag mismatch")
require(md.severe_below_minus02.equals(md.final_score_delta.lt(-.02)), "severe flag mismatch")
negative = pd.read_csv(out / "negative_months.csv", float_precision="round_trip")
pd.testing.assert_frame_equal(md[md.final_score_delta.lt(0)].reset_index(drop=True), negative, check_exact=True)
require(len(ad) == 18 and len(md) == 216 and len(monthly) == 180, "matrix table size mismatch")
complete = read(out / "run_complete.json")
require(complete["new_prediction_comparisons"] == 6 and complete["new_training_runs"] == 0,
        "actual experiment budget mismatch")
write(out / "final_delivery_verification.json", dict(accepted=True,
    verified_at=datetime.now(timezone.utc).isoformat(), report_sha256=sha256_file(REPORT),
    handoff_sha256=sha256_file(HANDOFF), final_source_sha256=h["final_source_sha256"],
    registration_sha256=sha256_file(out / "registration.json"),
    original_delivery_sha256=sha256_file(out / "delivery_acceptance.json"),
    all_output_sha256={str(p.relative_to(out)): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()},
    all_monthly_price_turnover_and_top_quality_and_risk_flags_verified=True,
    preserved_files=len(read(out / "preserved.json")), tests_passed=35,
    new_training_runs=0, new_prediction_comparisons=6, stop_after_local_commit=True))
print("Final delivery verified; report and handoff sealed; 35 tests, 6 transforms, 0 training.")
