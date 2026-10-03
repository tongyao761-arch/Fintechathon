"""Run the fixed 2023 group matrix serially, verify reuse, and retain comparisons."""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_features34 import check_frozen, resolve_selection
from scripts.run_lightgbm_baseline import MODEL_PARAMS, RAW_DATA_PATH, environment_versions
from src.data.baseline_panel import load_raw_baseline_panel
from src.validation.experiment import provenance, sha256_file, prediction_hash, write_json

CONFIG = ROOT / "configs/features34_step2.json"
OUTPUT = ROOT / "artifacts/features34_step2"
ORDER = ["baseline10", *[f"10+{g}" for g in "ABCDEF"], "full34", *[f"34-{g}" for g in "ABCDEF"]]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def function_ast(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))


def verify_run(directory, candidate, config, current, reference):
    summary, state, metadata, saved_config = [read_json(directory / name) for name in
        ("summary.json", "status.json", "provenance.json", "config.json")]
    assert state["status"] == "success"
    columns = list(resolve_selection(config, candidate, "primary_2023"))
    assert summary["candidate"] == saved_config["candidate"] == candidate
    assert saved_config["split"] == "primary_2023"
    assert summary["features"] == saved_config["features"] == columns
    assert summary["model_params"] == saved_config["model_params"] == MODEL_PARAMS
    assert summary["environment"] == environment_versions()
    assert metadata["data"] == current["data"] and metadata["dependencies"] == current["dependencies"]
    assert summary["reference_summary_sha256"] == current["reference_summary_sha256"]
    assert metadata["reference_summary_sha256"] == current["reference_summary_sha256"]
    assert metadata["frozen_manifest_sha256"] == sha256_file(ROOT / config["frozen_manifest"])
    assert summary["provenance_sha256"] == sha256_file(directory / "provenance.json")
    assert summary["config_sha256"] == sha256_file(directory / "config.json")
    assert summary["panel_rows"] == 7900350 and summary["feature_index_preserved"]
    assert summary["frozen_files_unchanged"]
    assert len(summary["splits"]) == 1
    result = summary["splits"][0]
    assert result == read_json(directory / "primary_2023/summary.json")
    assert result["split_name"] == "primary_2023" and result["dates"] == reference["dates"]
    for key in ("train_samples", "split_train_rows", "purge_rows", "valid_prediction_rows"):
        assert result[key] == reference[key], key
    assert result["prediction_coverage"] == 1 and result["model_reload_predictions_equal"]
    assert np.isfinite(list(result["metrics"].values())).all()
    assert result["official_comparison"]["max_abs_difference"] <= 1e-12
    split_dir = directory / "primary_2023"
    for relative, expected in result["file_sha256"].items():
        assert sha256_file(split_dir / relative) == expected, relative
    pred = pd.read_parquet(split_dir / "predictions.parquet")
    assert len(pred) == 1125300 and not pred.duplicated(["ts_code", "trade_date"]).any()
    assert np.isfinite(pred.pred).all() and prediction_hash(pred.pred) == result["prediction_sha256"]
    if candidate == "baseline10":
        assert result["prediction_sha256"] == reference["prediction_sha256"]
        assert result["metrics"] == reference["metrics"]
    x = pd.read_csv(split_dir / "evaluate_input/测试集_X.csv", usecols=["ts_code", "trade_date"])
    pd.testing.assert_frame_equal(pred[["ts_code", "trade_date"]], x, check_dtype=False, check_categorical=False)
    assert lgb.Booster(model_file=str(split_dir / "models/lightgbm.txt")).feature_name() == columns
    stats = pd.read_csv(split_dir / "feature_missing_statistics.csv")
    assert set(stats.scope) == {"full_panel", "train_eligible", "validation_all_keys"}
    assert not stats.infinite.any() and stats.finite.gt(0).all()
    assert len(pd.read_csv(split_dir / "monthly_metrics.csv")) == 12
    return summary


def verify_reuse(directory, candidate, config, current, reference, accepted):
    summary = verify_run(directory, candidate, config, current, reference)
    assert sha256_file(directory / "summary.json") == accepted["summary_sha256"]
    old = read_json(directory / "provenance.json")
    assert old["candidate_config_sha256"] == sha256_file(ROOT / "configs/features34.json")
    old_spec = read_json(directory / "config.json")["requested"]["experiments"][candidate]
    assert old_spec == config["experiments"][candidate]
    comparisons = {}
    for relative, expected in old["source_sha256"].items():
        if relative == "scripts/run_features34.py":
            continue
        assert current["source_sha256"][relative] == expected, relative
        comparisons[relative] = expected
    executed = ROOT / "artifacts/features34_step1/executed_sources/run_features34.py"
    assert sha256_file(executed) == old["source_sha256"]["scripts/run_features34.py"]
    # The authorization and summary labels changed. Actual training, scoring,
    # diagnostics and feature-statistic function bodies must remain identical.
    functions = ["run_split", "feature_statistics", "check_frozen"]
    for name in functions:
        assert function_ast(executed, name) == function_ast(ROOT / "scripts/run_features34.py", name), name
    old_main = next(n for n in ast.parse(executed.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
    new_main = next(n for n in ast.parse((ROOT / "scripts/run_features34.py").read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
    # Normalize only the three changed output metadata literals in main.
    class NormalizePurpose(ast.NodeTransformer):
        def visit_Dict(self, node):
            self.generic_visit(node)
            for i, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value in ("stage", "purpose"):
                    node.values[i] = ast.Constant(value="authorization-metadata")
            return node
    assert ast.dump(NormalizePurpose().visit(old_main)) == ast.dump(NormalizePurpose().visit(new_main))
    return summary, {"candidate": candidate, "directory": str(directory.relative_to(ROOT)).replace("\\", "/"),
        "summary_sha256": sha256_file(directory / "summary.json"),
        "same_selection_model_data_dependencies_split": True,
        "unchanged_source_sha256": comparisons, "unchanged_function_asts": functions,
        "main_ast_equal_after_stage_purpose_metadata_normalization": True,
        "old_runner_sha256": sha256_file(executed),
        "current_runner_sha256": sha256_file(ROOT / "scripts/run_features34.py"),
        "reason": "Only authorization and output stage/purpose metadata changed; executable feature/train/score/diagnostic logic is identical.",
        "disk_artifact_hashes_predictions_keys_model_features_verified": True}


def comparison_tables(entries):
    baseline = entries["baseline10"][1]["splits"][0]
    full = entries["full34"][1]["splits"][0]
    rows, monthly = [], []
    for i, name in enumerate(ORDER, 1):
        directory, summary, mode = entries[name]
        result = summary["splits"][0]
        m, d = result["metrics"], result["diagnostics"]
        row = {"experiment_id": f"G{i:02d}", "candidate": name, "result_mode": mode,
            "directory": str(directory.relative_to(ROOT)).replace("\\", "/"),
            "features": ";".join(summary["features"]), "feature_count": len(summary["features"]),
            "train_samples": result["train_samples"], "valid_prediction_rows": result["valid_prediction_rows"],
            "prediction_coverage": result["prediction_coverage"], **m,
            "score_minus_baseline10": m["final_score"] - baseline["metrics"]["final_score"],
            "score_minus_full34": m["final_score"] - full["metrics"]["final_score"],
            "top_missing_label_fraction": d["top_groups"]["turnover"]["missing_label_fraction"],
            "top_invalid_price_fraction": d["top_groups"]["turnover"]["invalid_price_fraction"],
            "top_baseline_all_missing_fraction": d["top_groups"]["turnover"]["all_features_missing_fraction"],
            "top_candidate_all_missing_fraction": d["candidate_features_top_groups"]["turnover"]["all_features_missing_fraction"],
            "price_valid_only_turnover": d["price_valid_only_turnover"],
            "prediction_sha256": result["prediction_sha256"],
            "official_max_abs_difference": result["official_comparison"]["max_abs_difference"],
            "elapsed_seconds": summary["resources"]["elapsed_seconds"],
            "peak_process_rss_mb": summary["resources"]["peak_process_rss_mb"]}
        for k, value in result["score_contributions"].items():
            row[k + "_contribution"] = value
            row[k + "_delta_baseline10"] = value - baseline["score_contributions"][k]
            row[k + "_delta_full34"] = value - full["score_contributions"][k]
        rows.append(row)
        frame = pd.read_csv(directory / "primary_2023/monthly_metrics.csv")
        frame.insert(0, "candidate", name)
        frame.insert(0, "experiment_id", f"G{i:02d}")
        monthly.append(frame)
    table = pd.DataFrame(rows)
    months = pd.concat(monthly, ignore_index=True)
    for ref, name in (("baseline10", "baseline10"), ("full34", "full34")):
        ref_frame = months[months.candidate.eq(name)].set_index("month")
        for k in ("score", "ic", "annual_excess", "turnover", "ic_contribution", "excess_contribution", "stability_contribution"):
            months[f"{k}_minus_{ref}"] = months[k] - months.month.map(ref_frame[k])
    table.to_csv(OUTPUT / "comparison.csv", index=False, float_format="%.17g")
    months.to_csv(OUTPUT / "monthly_comparison.csv", index=False, float_format="%.17g")
    groups = []
    for group in "ABCDEF":
        add = table[table.candidate.eq("10+" + group)].iloc[0]
        drop = table[table.candidate.eq("34-" + group)].iloc[0]
        add_month = months[months.candidate.eq("10+" + group)]
        drop_month = months[months.candidate.eq("34-" + group)]
        row = {"group": group, "add_score_delta": add.score_minus_baseline10,
               "presence_in_full_score_delta": -drop.score_minus_full34,
               "add_positive_months": int(add_month.score_minus_baseline10.gt(0).sum()),
               "presence_in_full_positive_months": int(drop_month.score_minus_full34.lt(0).sum())}
        for k in ("ic", "excess", "stability"):
            row[f"add_{k}_contribution_delta"] = add[f"{k}_delta_baseline10"]
            row[f"presence_in_full_{k}_contribution_delta"] = -drop[f"{k}_delta_full34"]
        for prefix, frame, sign, ref in (("add", add_month, 1, "baseline10"), ("presence_in_full", drop_month, -1, "full34")):
            deltas = sign * frame[f"score_minus_{ref}"]
            row[prefix + "_monthly_min"] = float(deltas.min())
            row[prefix + "_monthly_max"] = float(deltas.max())
            row[prefix + "_monthly_median"] = float(deltas.median())
            row[prefix + "_excess_positive_months"] = int((sign * frame[f"annual_excess_minus_{ref}"]).gt(0).sum())
        groups.append(row)
    pd.DataFrame(groups).to_csv(OUTPUT / "group_effects.csv", index=False, float_format="%.17g")


def membership_comparisons(entries):
    # Label availability is diagnostic only; no model/selection operation here.
    panel = load_raw_baseline_panel(RAW_DATA_PATH)
    p = panel[panel.trade_date.between(20230103, 20231229)]
    invalid = {int(date): set(g.loc[g.is_price_valid.eq(0), "ts_code"].astype(str)) for date, g in p.groupby("trade_date", observed=True)}
    missing = {int(date): set(g.loc[g.y_ret_1d.isna(), "ts_code"].astype(str)) for date, g in p.groupby("trade_date", observed=True)}
    all_keys = p[["ts_code", "trade_date"]].reset_index(drop=True)
    del p, panel
    sets = {}
    for name, (directory, _, _) in entries.items():
        pred_keys = pd.read_parquet(directory / "primary_2023/predictions.parquet", columns=["ts_code", "trade_date"])
        pd.testing.assert_frame_equal(pred_keys, all_keys, check_dtype=False, check_categorical=False)
        frame = pd.read_csv(directory / "primary_2023/daily_top_sets.csv")
        sets[name] = {int(row.trade_date): set(row.top_codes.split(",")) for row in frame.itertuples()}
    rows = []
    for name in ORDER:
        for reference in ("baseline10", "full34"):
            changed, changed_invalid, changed_missing, invalid_diff_days, missing_diff_days = 0, 0, 0, 0, 0
            for date, current_set in sets[name].items():
                ref_set = sets[reference][date]
                diff = current_set ^ ref_set
                changed += len(diff)
                changed_invalid += len(diff & invalid[date])
                changed_missing += len(diff & missing[date])
                invalid_diff_days += int(current_set & invalid[date] != ref_set & invalid[date])
                missing_diff_days += int(current_set & missing[date] != ref_set & missing[date])
            rows.append({"candidate": name, "reference": reference,
                         "changed_membership_stock_dates_symmetric_difference": changed,
                         "changed_invalid_price_stock_dates": changed_invalid,
                         "changed_missing_label_stock_dates": changed_missing,
                         "changed_invalid_fraction": changed_invalid / changed if changed else 0.,
                         "changed_missing_fraction": changed_missing / changed if changed else 0.,
                         "days_invalid_price_subset_differed": invalid_diff_days,
                         "days_missing_label_subset_differed": missing_diff_days})
    pd.DataFrame(rows).to_csv(OUTPUT / "top_membership_comparison.csv", index=False, float_format="%.17g")


def main():
    if not __debug__:
        raise RuntimeError("verification requires Python without -O")
    started = time.perf_counter()
    OUTPUT.mkdir(exist_ok=False)
    state = {"status": "running", "stage": "step2_groups_2023", "serial": True, "completed": []}
    write_json(OUTPUT / "status.json", state)
    try:
        config = read_json(CONFIG)
        acceptance = read_json(ROOT / "artifacts/features34_step1/acceptance.json")
        assert acceptance["accepted"] and acceptance["final_tests"]["passed"] == 78
        check_frozen(read_json(ROOT / config["frozen_manifest"]))
        current = provenance(ROOT, RAW_DATA_PATH)
        current["reference_summary_sha256"] = sha256_file(ROOT / config["reference_summary"])
        assert current["git"]["branch"] == "ivor-work"
        reference_summary = read_json(ROOT / config["reference_summary"])
        assert current["data"]["sha256"] == reference_summary["data"]["sha256"]
        reference = next(s for s in reference_summary["splits"] if s["split_name"] == "primary_2023")
        write_json(OUTPUT / "preflight.json", current)
        snapshot = OUTPUT / "executed_sources"
        snapshot.mkdir()
        for relative in current["source_sha256"]:
            dest = snapshot / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, dest)
        entries, reuse = {}, []
        for accepted in acceptance["runs"]:
            if accepted["split"] != "primary_2023":
                continue
            name = accepted["candidate"]
            directory = ROOT / accepted["directory"]
            summary, proof = verify_reuse(directory, name, config, current, reference, accepted)
            entries[name] = (directory, summary, "reused_step1")
            reuse.append(proof)
        assert set(entries) == {"baseline10", "full34"}
        write_json(OUTPUT / "reuse_verification.json", {"qualified": True, "runs": reuse})
        for i, name in enumerate(ORDER, 1):
            if name not in entries:
                experiment_id = "features34_step2_" + name.replace("+", "plus").replace("-", "minus")
                command = [sys.executable, "-B", "scripts/run_features34.py", "--config", str(CONFIG),
                           "--candidate", name, "--split", "primary_2023", "--experiment-id", experiment_id]
                print(f"[{i}/14] Running {name} serially", flush=True)
                log = OUTPUT / f"G{i:02d}_run.txt"
                with log.open("x", encoding="utf-8") as handle:
                    result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                write_json(OUTPUT / f"G{i:02d}_command.json", {"command": command, "exit_code": result.returncode})
                if result.returncode:
                    raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {log}")
                directories = sorted((ROOT / "artifacts/experiments" / experiment_id).iterdir())
                directory = directories[-1]
                summary = verify_run(directory, name, config, current, reference)
                assert read_json(directory / "provenance.json")["source_sha256"] == current["source_sha256"]
                entries[name] = (directory, summary, "new_run")
            print(f"[{i}/14] Verified {name}: score={entries[name][1]['splits'][0]['metrics']['final_score']:.12f}", flush=True)
            state["completed"].append(name)
            write_json(OUTPUT / "status.json", state)
        comparison_tables(entries)
        membership_comparisons(entries)
        check_frozen(read_json(ROOT / config["frozen_manifest"]))
        final_source = provenance(ROOT, RAW_DATA_PATH)
        assert final_source["source_sha256"] == current["source_sha256"] and final_source["data"] == current["data"]
        write_json(OUTPUT / "summary.json", {"accepted": True, "stage": "step2_groups_2023", "runs": [
            {"candidate": name, "directory": str(entries[name][0].relative_to(ROOT)).replace("\\", "/"),
             "result_mode": entries[name][2], "summary_sha256": sha256_file(entries[name][0] / "summary.json"),
             "prediction_sha256": entries[name][1]["splits"][0]["prediction_sha256"]} for name in ORDER],
            "serial": True, "reused": 2, "new_runs": 12, "frozen_files_unchanged": True,
            "source_data_unchanged_during_matrix": True, "2024_runs": 0,
            "final_retention_decisions": False, "elapsed_seconds": time.perf_counter() - started,
            "tables_sha256": {p.name: sha256_file(p) for p in OUTPUT.glob("*.csv")}})
        state.update(status="success", elapsed_seconds=time.perf_counter() - started)
        write_json(OUTPUT / "status.json", state)
    except BaseException as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.perf_counter() - started)
        write_json(OUTPUT / "status.json", state)
        raise


if __name__ == "__main__":
    main()
