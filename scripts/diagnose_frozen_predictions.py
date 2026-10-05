"""Read-only decomposition of indexed frozen predictions; never fit or predict."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.baseline_panel import load_raw_baseline_panel
from src.data.data_contract import assert_unique_keys
from src.metrics.official import score_official
from src.validation.experiment import prediction_hash, sha256_file
from src.features.baseline_v1 import FEATURE_COLUMNS as BASE

FREEZE = ROOT / "artifacts/features34_step5/FROZEN_CANDIDATES.json"
VALIDATION = ROOT / "artifacts/frozen_models_validation/20261005T062821736270Z_2d9c336c"
RAW = ROOT / "赛题五/赛题五数据/训练集.csv"
TEST = ROOT / "赛题五/赛题五数据/测试集_X.csv"
IDS = ["baseline10", "S4R_lean31_minus4", "S4R_full34_minus3"]
OWN = ["scripts/diagnose_frozen_predictions.py", "tests/test_frozen_prediction_diagnostics.py"]
REPORT = ROOT / "docs/model_optimization/STEP1_REPORT.md"
HANDOFF = ROOT / "docs/model_optimization/STEP1_HANDOFF.json"
METRICS = ["ic_mean", "annual_excess", "mean_turnover", "final_score",
           "ic_contribution", "excess_contribution", "stability_contribution"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def csv(frame, path):
    frame.to_csv(path, index=False, float_format="%.17g")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def hashes_check(mapping, base=ROOT):
    for name, expected in mapping.items():
        require(sha256_file(base / name) == expected, f"source/artifact hash mismatch: {base / name}")


def official_evaluator():
    spec = importlib.util.spec_from_file_location("unchanged_official", ROOT / "赛题五/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def metric_difference(a, b):
    require(set(a) == set(b) and bool(a), "metric names differ")
    require(np.isfinite(list(a.values())).all() and np.isfinite(list(b.values())).all(), "nonfinite scoring result")
    return {k: float(a[k]-b[k]) for k in a}


def contributions(frame):
    result = frame.copy()
    result["ic_contribution"] = .4 * result.ic_mean
    result["excess_contribution"] = .3 * result.annual_excess
    result["stability_contribution"] = .3 * (1-result.mean_turnover)
    result["final_score"] = result[["ic_contribution", "excess_contribution", "stability_contribution"]].sum(axis=1, min_count=3)
    return result


def daily_metrics(details):
    frames = []
    for name, column in [("daily_ic", "ic"), ("daily_excess", "excess"), ("daily_turnover", "turnover")]:
        frames.append(details[name].set_index("trade_date")[[column]])
    result = pd.concat(frames, axis=1).sort_index().reset_index()
    result = result.rename(columns={"ic": "ic_mean", "turnover": "mean_turnover"})
    result["annual_excess"] = result.excess * 252
    return contributions(result)


def aggregate_daily(frame, column):
    # Monthly turnover retains the preceding date outside the month. Each metric
    # has its own valid-day denominator; never average the monthly final scores.
    result = frame.groupby(column, sort=True).agg(
        ic_mean=("ic_mean", "mean"), annual_excess=("annual_excess", "mean"),
        mean_turnover=("mean_turnover", "mean"), ic_days=("ic_mean", "count"),
        excess_days=("excess", "count"), turnover_days=("mean_turnover", "count"))
    return contributions(result.reset_index())


def comparisons(frame, keys):
    rows = []
    for after, before in [(IDS[1], IDS[0]), (IDS[2], IDS[0]), (IDS[1], IDS[2])]:
        a = frame[frame.candidate.eq(after)].set_index(keys)
        b = frame[frame.candidate.eq(before)].set_index(keys)
        require(a.index.equals(b.index), "comparison keys differ")
        delta = a[METRICS]-b[METRICS]
        delta.columns = [c+"_delta" for c in METRICS]
        delta = delta.reset_index()
        delta["after"], delta["before"] = after, before
        rows.append(delta)
    return pd.concat(rows, ignore_index=True)


def concentration(values):
    v = np.asarray(values, dtype=float)
    require(len(v)>0 and np.isfinite(v).all(), "invalid excess series")
    ordered = np.sort(v)[::-1]
    positive = np.maximum(v, 0).sum()
    total = v.sum()
    result = dict(days=len(v), positive_days=int((v>0).sum()), negative_days=int((v<0).sum()),
                  arithmetic_excess_sum=float(total), annual_excess=float(v.mean()*252),
                  best_day=float(v.max()), worst_day=float(v.min()))
    for k in [1, 5, 10, max(1, int(np.ceil(len(v)*.05)))]:
        prefix = f"best_{k}_days"
        result[prefix+"_sum"] = float(ordered[:k].sum())
        result[prefix+"_positive_mass_fraction"] = float(np.maximum(ordered[:k],0).sum()/positive) if positive>0 else None
        result[prefix+"_net_sum_fraction"] = float(ordered[:k].sum()/total) if total!=0 else None
        result[prefix+"_removed_annual_excess"] = float(ordered[k:].mean()*252) if len(v)>k else None
    # Removing dates is descriptive concentration arithmetic, not a new score.
    return result


def top_diagnostics(joined, details=None):
    """Return/turnover universes stay separate; price-only filter uses today's X."""
    membership, transitions, quality, valid_turns = [], [], [], []
    lookups = {}
    if details is not None:
        for kind, key in [("return", "daily_return_top_sets"), ("turnover", "daily_top_sets")]:
            lookups[kind] = {int(r.trade_date): set(r.top_codes.split(",")) for r in details[key].itertuples()}
    previous = {k: None for k in ["return", "turnover", "price_valid"]}
    sets = {k: {} for k in previous}
    for date, group in joined.groupby("trade_date", sort=True):
        indexed = group.set_index("ts_code")
        universes = {"turnover": group[group.flag_limit_up.eq(0)],
                     "price_valid": group[group.flag_limit_up.eq(0) & group.is_price_valid.eq(1)]}
        if "y_ret_1d" in group:
            universes["return"] = group[group.flag_limit_up.eq(0) & group.y_ret_1d.notna()]
        for kind, eligible in universes.items():
            if len(eligible)<100:
                previous[kind] = None
                continue
            top = eligible.sort_values("pred", ascending=False).iloc[:len(eligible)//10]
            current = set(top.ts_code)
            if kind in lookups:
                require(current == lookups[kind][int(date)], f"official {kind} Top mismatch on {date}")
            sets[kind][int(date)] = current
            missing = int(top.y_ret_1d.isna().sum()) if "y_ret_1d" in top else None
            quality.append(dict(trade_date=int(date), top_type=kind, universe_count=len(eligible), top_count=len(top),
                missing_label_count=missing, invalid_price_count=int(top.is_price_valid.eq(0).sum())))
            membership.append(dict(trade_date=int(date), top_type=kind, top_codes=",".join(sorted(current))))
            if previous[kind] is not None:
                prev_date, prev_codes, prev_rows = previous[kind]
                entered, exited = current-prev_codes, prev_codes-current
                row = dict(trade_date=int(date), previous_trade_date=int(prev_date), top_type=kind,
                    intersection=len(current & prev_codes), union=len(current | prev_codes),
                    entered_count=len(entered), exited_count=len(exited),
                    entered_codes=",".join(sorted(entered)), exited_codes=",".join(sorted(exited)),
                    turnover=1-len(current & prev_codes)/len(current | prev_codes))
                for prefix, codes, data in [("entered", entered, indexed), ("exited", exited, prev_rows)]:
                    selection = data.loc[sorted(codes)]
                    row[prefix+"_invalid_price_count"] = int(selection.is_price_valid.eq(0).sum())
                    row[prefix+"_missing_label_count"] = int(selection.y_ret_1d.isna().sum()) if "y_ret_1d" in selection else None
                transitions.append(row)
                if kind == "price_valid":
                    valid_turns.append(dict(trade_date=int(date), price_valid_turnover=row["turnover"]))
            previous[kind] = (date, current, indexed)
    q = pd.DataFrame(quality)
    q["invalid_price_fraction"] = q.invalid_price_count/q.top_count
    q["missing_label_fraction"] = q.missing_label_count/q.top_count
    return q, pd.DataFrame(membership), pd.DataFrame(transitions), pd.DataFrame(valid_turns), sets


def aggregate_top(frame, period):
    result = frame.groupby(["candidate", period, "top_type"]).agg(
        top_count=("top_count", "sum"), missing_label_count=("missing_label_count", lambda s: s.sum(min_count=1)),
        invalid_price_count=("invalid_price_count", "sum"), observations=("trade_date", "size")).reset_index()
    result["missing_label_fraction"] = result.missing_label_count/result.top_count
    result["invalid_price_fraction"] = result.invalid_price_count/result.top_count
    return result


def cross_predictions(a, b, quality, sa, sb, after=IDS[1], before=IDS[2]):
    require(a[["ts_code","trade_date"]].equals(b[["ts_code","trade_date"]]), "correlation keys differ")
    joined = a.merge(b, on=["ts_code","trade_date"], suffixes=("_a","_b"), validate="one_to_one")
    joined = joined.merge(quality, on=["ts_code","trade_date"], validate="one_to_one")
    rows = []
    for date, group in joined.groupby("trade_date",sort=True):
        row = dict(trade_date=int(date), after=after, before=before)
        scopes = {"all": group, "price_valid":group[group.is_price_valid.eq(1)]}
        if "y_ret_1d" in group:
            scopes["label_valid_diagnostic"] = group[group.y_ret_1d.notna()]
        for scope, sub in scopes.items():
            row[scope+"_n"] = len(sub)
            row[scope+"_spearman"] = float(spearmanr(sub.pred_a,sub.pred_b)[0])
            row[scope+"_pearson"] = float(sub.pred_a.corr(sub.pred_b))
        for kind in ["return","turnover","price_valid"]:
            if int(date) not in sa[kind] or int(date) not in sb[kind]:
                continue
            aa,bb = sa[kind][int(date)],sb[kind][int(date)]
            changed = aa ^ bb
            sub = group[group.ts_code.isin(changed)]
            row[kind+"_jaccard"] = len(aa & bb)/len(aa | bb)
            row[kind+"_intersection_fraction"] = len(aa & bb)/min(len(aa),len(bb))
            row[kind+"_symmetric_difference"] = len(changed)
            row[kind+"_changed_codes"] = ",".join(sorted(changed))
            row[kind+"_changed_invalid_price"] = int(sub.is_price_valid.eq(0).sum())
            row[kind+"_changed_missing_label"] = int(sub.y_ret_1d.isna().sum()) if "y_ret_1d" in sub else None
        rows.append(row)
    return pd.DataFrame(rows)


def records():
    frozen = read(FREEZE)
    rows = list(frozen["development_runs"])
    for rec in read(ROOT/"artifacts/features34_step5/run_index.json")["runs"]:
        rows.append(dict(candidate=rec["item"]["candidate"],split=rec["item"]["split"],
            directory=rec["directory"],summary_sha256=rec["summary_sha256"]))
    d = ROOT/frozen["baseline_directory"]
    rows.append(dict(candidate="baseline10",split="oos_2024",directory=frozen["baseline_directory"],summary_sha256=sha256_file(d/"summary.json")))
    require(len(rows)==12 and len({(r['candidate'],r['split']) for r in rows})==12,"incomplete frozen matrix")
    return sorted(rows,key=lambda r:(r["split"],IDS.index(r["candidate"])))


def prepare(output):
    require(not REPORT.exists() and not HANDOFF.exists(),"existing report/handoff must not be overwritten")
    output.mkdir(parents=True,exist_ok=False)
    require(subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()=="ivor-work","wrong branch")
    names = subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard"],cwd=ROOT).decode("utf-8").splitlines()
    # git -z avoids quoted non-ASCII filenames and preserves spaces.
    names = subprocess.check_output(["git","ls-files","-z","--cached","--others","--exclude-standard"],cwd=ROOT).decode("utf-8").split("\0")
    snapshot = {n:sha256_file(ROOT/n) for n in names if n and (ROOT/n).is_file() and not n.startswith(output.relative_to(ROOT).as_posix()+"/")}
    for n in [".git/config",".git/hooks/pre-push",".git/info/exclude"]:
        if (ROOT/n).is_file(): snapshot[n]=sha256_file(ROOT/n)
    write(output/"preserved.json",snapshot)
    write(output/"registration.json",dict(stage="frozen_prediction_score_decomposition",started_at=datetime.now(timezone.utc).isoformat(),
        branch="ivor-work",freeze_sha256=sha256_file(FREEZE),new_training_runs=0,prediction_transformations=0,
        inputs=records(),test_validation_directory=str(VALIDATION),
        monthly_turnover="preceding valid day retained, including month boundary; resets at annual split boundary",
        concentration="best 1/5/10 and ceil(5% of days); no new score or rule after deleting days",
        correlations="all, current-price-valid, label-valid (diagnostic only); return/turnover/price-only Top separate",
        source_sha256={p:sha256_file(ROOT/p) for p in OWN},
        original_git_status=subprocess.check_output(["git","status","--porcelain"],cwd=ROOT).decode("utf-8")))
    for p in OWN+["赛题五/evaluate.py","src/metrics/official.py","src/data/baseline_panel.py"]:
        dest=output/"executed_sources"/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dest)


def verify_sources(output):
    freeze=read(FREEZE);reg=read(output/"registration.json")
    require(sha256_file(FREEZE)==reg["freeze_sha256"],"freeze changed")
    hashes_check(freeze["source"]["source_sha256"])
    require(sha256_file(RAW)==freeze["source"]["data"]["sha256"],"raw data changed")
    hashes_check(read(ROOT/"docs/features34/FROZEN_REFERENCE.json"))
    for phase in ["features34_step5","features34_step6"]:
        acc=read(ROOT/"artifacts"/phase/"acceptance.json")
        require(acc["accepted"],f"{phase} not accepted")
        hashes_check(acc["evidence_sha256"],ROOT/"artifacts"/phase)
    acc=read(VALIDATION/"delivery_acceptance.json")
    require(acc["accepted"],"test adaptation not accepted")
    hashes_check(acc["all_final_output_sha256"],VALIDATION)
    meta=read(VALIDATION/"provenance.json")
    require(sha256_file(TEST)==meta["test_data"]["sha256"],"test X changed")
    # Validate original and repaired executed source snapshots without requiring
    # historical orchestration versions to equal later authorization metadata.
    hashes_check(meta["source_sha256"],VALIDATION/"executed_sources")
    repaired=read(VALIDATION/"repair_registration.json")["repaired_source_sha256"]
    hashes_check(repaired,VALIDATION/"repaired_sources")
    hashes_check({**meta["source_sha256"],**repaired})
    for rec in read(ROOT/"artifacts/features34_step6/run_index.json")["runs"]:
        orig=ROOT/rec["original_directory"];repeat=ROOT/rec["directory"];s=rec["item"]["split"]
        require(sha256_file(repeat/"summary.json")==rec["summary_sha256"],"repeat indexed summary changed")
        a=next(x for x in read(orig/"summary.json")["splits"] if x["split_name"]==s)
        b=read(repeat/"summary.json")["splits"][0]
        require(a["file_sha256"]==b["file_sha256"],"repeat files differ")
        hashes_check(b["file_sha256"],repeat/s)
    hashes_check(reg["source_sha256"],output/"executed_sources")
    own_sources = dict(reg["source_sha256"])
    repair_path = output/"repair_registration.json"
    if repair_path.exists():
        repair = read(repair_path)
        require(repair["original_registration_sha256"]==sha256_file(output/"registration.json"),"original registration changed")
        require(set(repair["repaired_source_sha256"])=={OWN[0]},"repair outside independent diagnostic entry")
        hashes_check(repair["repaired_source_sha256"],output/"repaired_sources")
        own_sources.update(repair["repaired_source_sha256"])
    hashes_check(own_sources)
    print("Frozen sources, indices, prior acceptances, raw data and repeated files verified.",flush=True)


def load_record(rec, panel, output):
    d=ROOT/rec["directory"];s=rec["split"];sm=read(d/"summary.json")
    require(sha256_file(d/"summary.json")==rec["summary_sha256"],"indexed summary changed")
    result=next(r for r in sm["splits"] if r["split_name"]==s)
    hashes_check(result["file_sha256"],d/s)
    require(sha256_file(d/"provenance.json")==sm["provenance_sha256"],"run provenance changed")
    if "config_sha256" in sm:require(sha256_file(d/"config.json")==sm["config_sha256"],"run config changed")
    cols=list(BASE) if rec["candidate"]=="baseline10" else next(c["features"] for c in read(FREEZE)["candidates"] if c["candidate"]==rec["candidate"])
    require(sm["features"]==cols and sm["model_params"]==read(FREEZE)["candidates"][0]["model_params"],"frozen columns/parameters differ")
    model=lgb.Booster(model_file=str(d/s/"models/lightgbm.txt"))
    require(model.feature_name()==cols and model.num_trees()==100,"actual model columns/tree budget differ")
    spec=read(ROOT/"configs/features34_step3.json")["splits"].get(s,read(ROOT/"configs/splits.yaml").get(s))
    require(result["dates"]==spec,"old split changed")
    raw=panel[panel.trade_date.between(spec["valid_start"],spec["valid_end"])].copy()
    raw["ts_code"]=raw.ts_code.astype(str)
    raw=raw.reset_index(drop=True)
    train=panel.trade_date.between(spec["train_start"],spec["train_end"]) & panel.is_trainable.eq(1)
    require(int(train.sum())==result["train_samples"] and len(raw)==result["valid_prediction_rows"],"split row counts differ")
    pq=pd.read_parquet(d/s/"predictions.parquet");pq["ts_code"]=pq.ts_code.astype(str)
    require(np.isfinite(pq.pred).all() and prediction_hash(pq.pred)==result["prediction_sha256"],"saved prediction changed")
    fixture=d/s/"evaluate_input"
    pred=pd.read_csv(fixture/"submission.csv");truth=pd.read_csv(fixture/"测试集_Y.csv");x=pd.read_csv(fixture/"测试集_X.csv")
    for frame in [pq,pred,truth,x]:
        pd.testing.assert_frame_equal(frame[["ts_code","trade_date"]],raw[["ts_code","trade_date"]],check_dtype=False,check_exact=True)
        assert_unique_keys(frame,name="diagnostic input")
    pd.testing.assert_series_equal(truth.y_ret_1d,raw.y_ret_1d,check_dtype=False,check_exact=True)
    pd.testing.assert_series_equal(x.flag_limit_up,raw.flag_limit_up,check_dtype=False,check_exact=True)
    require(np.allclose(pred.pred,pq.pred,rtol=0,atol=1e-15),"CSV is not the unchanged saved prediction")
    origin=read(d/"provenance.json")
    require(origin["data"]==read(FREEZE)["source"]["data"],"model data source differs")
    snapshot=ROOT/"artifacts"/("features34_step5" if s=="oos_2024" and rec["candidate"]!="baseline10" else "features34_step4_revision" if rec["candidate"]!="baseline10" else "features34_step3" if s.startswith("dev") else "features34_step1")/"executed_sources"
    source_rows={}
    for name,h in origin["source_sha256"].items():
        if (snapshot/name).exists() and sha256_file(snapshot/name)==h:
            source_rows[name]=dict(sha256=h,snapshot=str(snapshot/name))
        elif (ROOT/name).exists() and sha256_file(ROOT/name)==h:
            source_rows[name]=dict(sha256=h,snapshot=str(ROOT/name))
        else:
            flattened=ROOT/"artifacts/features34_step1/executed_sources"/Path(name).name
            if flattened.is_file() and sha256_file(flattened)==h:
                source_rows[name]=dict(sha256=h,snapshot=str(flattened),layout="historical flattened snapshot")
                continue
            matches=[p for p in (ROOT/"artifacts/baseline_v1_1",ROOT/"artifacts/features34_step1",ROOT/"artifacts/features34_step2",ROOT/"artifacts/features34_step3",ROOT/"artifacts/features34_step4",ROOT/"artifacts/features34_step4_revision",ROOT/"artifacts/features34_step5") if (p/"executed_sources"/name).is_file() and sha256_file(p/"executed_sources"/name)==h]
            if matches:
                source_rows[name]=dict(sha256=h,snapshot=str(matches[0]/"executed_sources"/name))
            else:
                # Some earliest baseline sources live in local Git rather than
                # later execution snapshots. Verify bytes against provenance.
                commits=subprocess.check_output(["git","log","--all","--format=%H","--",name],cwd=ROOT,text=True).splitlines()
                match=None
                for commit in commits:
                    blob=subprocess.run(["git","show",commit+":"+name],cwd=ROOT,capture_output=True)
                    if blob.returncode==0 and hashlib.sha256(blob.stdout).hexdigest()==h:
                        dest=output/"historical_sources"/h/name
                        if dest.exists():require(sha256_file(dest)==h,"historical blob changed")
                        else:
                            dest.parent.mkdir(parents=True,exist_ok=True)
                            with dest.open("xb") as handle:handle.write(blob.stdout)
                        match=dict(sha256=h,snapshot=str(dest),git_commit=commit)
                        break
                require(match is not None,f"cannot locate actual historical source: {name} {h}")
                source_rows[name]=match
    paths={"predictions":d/s/"predictions.parquet","score_prediction":fixture/"submission.csv", "score_truth":fixture/"测试集_Y.csv","score_x":fixture/"测试集_X.csv","model":d/s/"models/lightgbm.txt","summary":d/"summary.json","provenance":d/"provenance.json"}
    evidence={**rec,"paths":{k:str(v) for k,v in paths.items()},"sha256":{k:sha256_file(v) for k,v in paths.items()},
              "split_spec":spec,"features":cols,"executed_sources":source_rows,
              "csv_minus_parquet_max_abs":float(np.max(np.abs(pred.pred-pq.pred))),
              "csv_rounding_note":"Official script uses default pandas CSV parser; parquet unchanged. Only existing CSV scored.",
              "all_keys_raw_labels_flags_and_train_counts_verified":True}
    return pred,truth,x,raw,result,evidence


def run(output):
    verify_sources(output)
    panel=load_raw_baseline_panel(RAW)
    evaluator=official_evaluator()
    daily_all=[];quality_all=[];member_all=[];transition_all=[];corr_all=[];annual_all=[];concentrations=[];inputs=[];parities=[];cache={}
    for rec in read(output/"registration.json")["inputs"]:
        start=time.perf_counter();pred,truth,x,raw,result,evidence=load_record(rec,panel,output)
        local=score_official(pred,truth,x,return_details=True);details=local.pop("details")
        fixture=Path(evidence["paths"]["score_prediction"]).parent
        official={k:float(v) for k,v in evaluator(str(fixture/"submission.csv"),str(fixture)).items()}
        diff=metric_difference(local,official);prior_diff=metric_difference(local,result["metrics"])
        parity={"candidate":rec["candidate"],"split":rec["split"],"local_minus_official":diff,"local_minus_frozen":prior_diff,"official_metrics":official,
                "max_abs_official_difference":max(abs(v) for v in diff.values()),"max_abs_frozen_difference":max(abs(v) for v in prior_diff.values())}
        parities.append(parity);write(output/"official_parity.json",parities)
        require(parity["max_abs_official_difference"]<=1e-12 and parity["max_abs_frozen_difference"]<=1e-12,"scoring mismatch; frozen scheme not revised")
        year=int(str(raw.trade_date.min())[:4]);name=rec["candidate"]
        daily=daily_metrics(details);daily["month"]=daily.trade_date.astype(str).str[:6];daily["year"]=year;daily["candidate"]=name
        quality=raw[["ts_code","trade_date","flag_limit_up","is_price_valid"]].copy()
        joined=pred.merge(truth,on=["ts_code","trade_date"],validate="one_to_one").merge(quality,on=["ts_code","trade_date"],validate="one_to_one")
        q,mem,trans,valid,sets=top_diagnostics(joined,details)
        daily=daily.merge(valid,on="trade_date",how="left",validate="one_to_one")
        for frame in [q,mem,trans]:frame["candidate"]=name;frame["year"]=year;frame["month"]=frame.trade_date.astype(str).str[:6]
        # Independently verify the official turnover series against reconstructed sets.
        turn=trans[trans.top_type.eq("turnover")]
        np.testing.assert_allclose(turn.turnover.to_numpy(),details["daily_turnover"].turnover.to_numpy(),rtol=0,atol=0)
        daily_all.append(daily);quality_all.append(q);member_all.append(mem);transition_all.append(trans)
        annual_all.append(dict(candidate=name,year=year,**local,ic_contribution=.4*local["ic_mean"],excess_contribution=.3*local["annual_excess"],stability_contribution=.3*(1-local["mean_turnover"]),price_valid_turnover=float(valid.price_valid_turnover.mean())))
        for scope,frame in [(str(year),daily)]+list(daily.groupby("month")):
            concentrations.append(dict(candidate=name,period=str(scope),**concentration(frame.excess.dropna())))
        best=daily.nlargest(10,"excess")[["trade_date","excess"]].to_dict("records")
        evidence.update(year=year,best_10_excess_dates=best,elapsed_seconds=time.perf_counter()-start)
        inputs.append(evidence);cache[name]=(pred,joined[["ts_code","trade_date","is_price_valid","y_ret_1d"]],sets)
        if len(cache)==3:
            for after,before in [(IDS[1],IDS[0]),(IDS[2],IDS[0]),(IDS[1],IDS[2])]:
                a,qual,sa=cache[after];b,_,sb=cache[before]
                corr=cross_predictions(a,b,qual,sa,sb,after,before);corr["year"]=year;corr["month"]=corr.trade_date.astype(str).str[:6];corr_all.append(corr)
            cache={}
        print(f"{name} {year}: score={local['final_score']:.9f}, official diff={parity['max_abs_official_difference']:.3g}, {time.perf_counter()-start:.1f}s",flush=True)
    annual=pd.DataFrame(annual_all);daily=pd.concat(daily_all,ignore_index=True);q=pd.concat(quality_all,ignore_index=True)
    monthly=pd.concat([aggregate_daily(g,"month").assign(candidate=n,year=y) for (n,y),g in daily.groupby(["candidate","year"],sort=False)],ignore_index=True)
    monthly=monthly.merge(daily.groupby(["candidate","month"]).price_valid_turnover.mean().reset_index(),on=["candidate","month"],validate="one_to_one")
    corrs=pd.concat(corr_all,ignore_index=True)
    tables=dict(annual_metrics=annual,monthly_metrics=monthly,daily_metrics=daily,
        annual_comparison=comparisons(annual,["year"]),monthly_comparison=comparisons(monthly,["year","month"]),daily_comparison=comparisons(daily,["year","trade_date"]),
        daily_top_quality=q,annual_top_quality=aggregate_top(q,"year"),monthly_top_quality=aggregate_top(q,"month"),
        daily_top_members=pd.concat(member_all,ignore_index=True),daily_top_transitions=pd.concat(transition_all,ignore_index=True),
        daily_prediction_comparison=corrs,return_concentration=pd.DataFrame(concentrations))
    numeric=corrs.select_dtypes(include=np.number).columns.difference(["trade_date","year"])
    for period in ["year","month"]:
        tables[period+"_prediction_comparison"]=corrs.groupby(["after","before",period])[numeric].mean().reset_index()
    summaries=[]
    for (after,before,year),g in tables["monthly_comparison"].groupby(["after","before","year"]):
        worst=g.loc[g.final_score_delta.idxmin()]
        summaries.append(dict(after=after,before=before,year=int(year),positive_months=int(g.final_score_delta.gt(0).sum()),negative_months=int(g.final_score_delta.lt(0).sum()),zero_months=int(g.final_score_delta.eq(0).sum()),worst_month=worst.month,worst_delta=float(worst.final_score_delta)))
    tables["monthly_risk_summary"]=pd.DataFrame(summaries)
    tables["negative_months"]=tables["monthly_comparison"].query("final_score_delta < 0").sort_values("final_score_delta")
    # Date concentration of the incremental excess-return stream is also disclosed.
    delta_rows=[]
    for (after,before,year),g in tables["daily_comparison"].groupby(["after","before","year"]):
        delta_rows.append(dict(after=after,before=before,year=int(year),**concentration(g.annual_excess_delta.dropna()/252)))
    tables["incremental_return_concentration"]=pd.DataFrame(delta_rows)
    for name,frame in tables.items():csv(frame,output/(name+".csv"))
    write(output/"input_manifest.json",inputs)
    del panel
    test_diagnostics(output)
    write(output/"run_complete.json",dict(complete=True,training_runs=0,prediction_transformations=0,official_scored_runs=12,
        annual_rows=len(annual),monthly_rows=len(monthly),daily_rows=len(daily)))


def test_diagnostics(output):
    raw=pd.read_csv(TEST,usecols=["ts_code","trade_date","open","high","low","close","flag_limit_up"])
    price=raw[["open","high","low","close"]]
    raw["is_price_valid"]=(np.isfinite(price).all(axis=1)&price.gt(0).all(axis=1)&raw.high.ge(price.max(axis=1))&raw.low.le(price.min(axis=1))).astype("int8")
    quality=raw[["ts_code","trade_date","flag_limit_up","is_price_valid"]]
    rows=[];sources=[];cache={}
    index=read(VALIDATION/"run_index.json")
    for rec in index["runs"]:
        if rec["item"]["phase"]!="A":continue
        name=rec["item"]["candidate"];d=ROOT/rec["directory"]
        require(sha256_file(d/"summary.json")==rec["summary_sha256"] and sha256_file(d/"files.json")==rec["files_sha256"],"test indexed source changed")
        hashes_check(read(d/"files.json"),d)
        pred=pd.read_parquet(d/"predictions.parquet");pred["ts_code"]=pred.ts_code.astype(str)
        pd.testing.assert_frame_equal(pred[["ts_code","trade_date"]],raw[["ts_code","trade_date"]],check_dtype=False)
        sm=read(d/"summary.json");require(prediction_hash(pred.pred)==sm["prediction_sha256"] and np.isfinite(pred.pred).all(),"test prediction changed")
        model=lgb.Booster(model_file=str(d/"lightgbm.txt"));require(model.feature_name()==sm["columns"],"test model columns differ")
        dest=output/("test_"+name);dest.mkdir(exist_ok=False)
        joined=pred.merge(quality,on=["ts_code","trade_date"],validate="one_to_one")
        q,mem,trans,valid,sets=top_diagnostics(joined)
        for frame in [q,mem,trans,valid]:frame["candidate"]=name
        for label,frame in [("daily_top_quality",q),("daily_top_members",mem),("daily_top_transitions",trans),("daily_price_valid_turnover",valid)]:csv(frame,dest/(label+".csv"))
        turn=trans[trans.top_type.eq("turnover")].turnover.mean()
        invalid=q[q.top_type.eq("turnover")]
        require(abs(turn-sm["diagnostics"]["official_prediction_turnover"])<=1e-12 and abs(valid.price_valid_turnover.mean()-sm["diagnostics"]["price_valid_turnover"])<=1e-12,"test diagnostic mismatch")
        grouped=pred.assign(month=pred.trade_date.astype(str).str[:6]).groupby("month").pred
        dist=grouped.agg(["count","mean","std","min","max","median"])
        for quantile in [.01,.05,.95,.99]:dist[str(quantile)]=grouped.quantile(quantile)
        csv(dist.reset_index(),dest/"monthly_prediction_distribution.csv")
        q["month"]=q.trade_date.astype(str).str[:6];csv(aggregate_top(q,"month"),dest/"monthly_top_quality.csv")
        rows.append(dict(candidate=name,rows=len(pred),dates=pred.trade_date.nunique(),mean=float(pred.pred.mean()),std=float(pred.pred.std()),min=float(pred.pred.min()),max=float(pred.pred.max()),
            official_prediction_turnover=float(turn),price_valid_turnover=float(valid.price_valid_turnover.mean()),top_invalid_price_fraction=float(invalid.invalid_price_count.sum()/invalid.top_count.sum()),
            true_ic=None,true_annual_excess=None,true_complete_score=None))
        sources.append(dict(candidate=name,directory=str(d),paths={p.name:str(p) for p in [d/"predictions.parquet",d/"lightgbm.txt",d/"summary.json",d/"files.json"]},sha256=read(d/"files.json"),test_x_path=str(TEST),test_x_sha256=sha256_file(TEST)))
        cache[name]=(pred,sets)
    require(len(rows)==2,"test adaptation inputs missing")
    a,sa=cache[IDS[1]];b,sb=cache[IDS[2]]
    csv(cross_predictions(a,b,quality,sa,sb),output/"test_daily_prediction_comparison.csv")
    csv(pd.DataFrame(rows),output/"test_unlabeled_summary.csv");write(output/"test_input_manifest.json",sources)
    # No test Y, return Top, IC, actual excess return, or complete score exists here.
    print("Existing test adaptation reused; unlabeled distribution and X-quality diagnostics only.",flush=True)


def table(frame, columns):
    def cell(v):return f"{v:.9f}" if isinstance(v,(float,np.floating)) else str(v)
    return "\n".join(["|"+"|".join(columns)+"|","|"+"|".join(["---"]*len(columns))+"|",*["|"+"|".join(cell(v) for v in row)+"|" for row in frame[columns].itertuples(index=False,name=None)]])


def report(output):
    a=pd.read_csv(output/"annual_metrics.csv");delta=pd.read_csv(output/"annual_comparison.csv")
    monthly=pd.read_csv(output/"monthly_comparison.csv");risk=pd.read_csv(output/"monthly_risk_summary.csv")
    q=pd.read_csv(output/"annual_top_quality.csv");corr=pd.read_csv(output/"year_prediction_comparison.csv")
    concentration_table=pd.read_csv(output/"return_concentration.csv");increment=pd.read_csv(output/"incremental_return_concentration.csv")
    baseline=delta[delta.before.eq(IDS[0])]
    lines=["# 冻结27/31列预测的评分分解与风险诊断", "", "## 结论", "",
        "四年官方全年评分重新核对：27列主候选S4R_lean31_minus4和31列备选S4R_full34_minus3均高于同年十特征。所有年度的正增益来自IC和收益，低换手分项相对十特征均为负；不能把优势归因于降低官方换手。", "",
        "2024年9月是两个候选相对十特征最严重的月度退化，主要由收益分项造成。2023全年优势最小，2023年11月也存在共同退化。2021—2023已大量开发、2024已查看，全部为历史事后诊断，不称盲测，不证明未来收益或因果关系。", "",
        "官方换手Top含大量缺标签和价格无效成员，特别是2021年几乎全为无效样本；价格有效样本换手明显更高。官方分数复现有效，但官方低换手不能直接解释为可交易稳定性。", "",
        "是否值得开展后续实验：可考虑预先固定、有限预算的融合和基于过去预测的平滑实验，但当前只有互补空间及换手风险的诊断，没有融合或平滑有效的证据。本次没有生成融合/平滑预测或选择权重，任务验收后停止。", "",
        "## 1. 来源与计分口径", "",
        f"权威冻结：{FREEZE}。只用登记的S4R候选，不使用第3步旧lean27/lean31。完整输入路径、源码快照、切分、模型和预测哈希见input_manifest.json；第6步重复文件哈希逐项相同，适配产物封存哈希核验。唯一目录：{output}。", "",
        "解释器为C:\\fintechathon\\.venv\\Scripts\\python.exe。既有历史submission.csv/测试集_Y.csv/测试集_X.csv直接由未修改赛题五/evaluate.py读取。CSV默认解析与parquet可能有浮点末位差，逐项记录原样保留；不改预测、不重建标签。源码历史版本通过provenance及executed_sources核对，不能把后续入口授权扩展误报为模型变更。", "",
        "全年按各指标有效日均值独立计分：0.4×IC + 0.3×252×日均超额 + 0.3×(1−官方换手)。月度按月内日值计算，月初换手保留此前有效日，年度切分首日无换手；不平均月分替代全年。逐日annual_excess为日超额×252的诊断量，逐日综合代理值仅三项俱全时给出，不冒称官方单日可提交评分。每日收益为Top收益减同一有效市场均值，年化不等于实际资金曲线、复利收益或成本后收益。", "",
        "## 2. 全年评分及分项", "",table(a,["candidate","year","ic_mean","annual_excess","mean_turnover","final_score","price_valid_turnover"]),"",
        table(delta,["after","before","year","final_score_delta","ic_contribution_delta","excess_contribution_delta","stability_contribution_delta"]),"",
        "IC与收益正贡献均不足以证明每列必需；官方低换手的绝对分值较大，和相对十特征的分差来源是两件事。分项是评分公式的算术分解，不是缺失或特征影响的因果分解。", "",
        "## 3. 月度风险与2024年9月", "",table(risk,["after","before","year","positive_months","negative_months","worst_month","worst_delta"]),"",
        "以下为两候选相对十特征所有负月份，按退化大小排序；27与31完整比较另见monthly_comparison.csv：", "",
        table(monthly[monthly.before.eq(IDS[0]) & monthly.final_score_delta.lt(0)].sort_values("final_score_delta"),["after","month","final_score_delta","ic_contribution_delta","excess_contribution_delta","stability_contribution_delta"]),"",
        "9月与其他差月逐项比较：", ""]
    for name in IDS[1:]:
        row=monthly[monthly.after.eq(name)&monthly.before.eq(IDS[0])&monthly.month.eq(202409)].iloc[0]
        components=[row.ic_contribution_delta,row.excess_contribution_delta,row.stability_contribution_delta]
        lines.append(f"- {name}：202409总分差{row.final_score_delta:+.9f}；IC/收益/低换手分差分别{components[0]:+.9f}/{components[1]:+.9f}/{components[2]:+.9f}。相对十特征的年化超额差{row.annual_excess_delta:+.9f}，是月内日均差×252，不能当月累计损失解释。")
    sep=pd.read_csv(output/"daily_comparison.csv");sep=sep[sep.before.eq(IDS[0]) & sep.trade_date.between(20240901,20240930)]
    for name,g in sep.groupby("after"):
        worst=g.loc[g.annual_excess_delta.idxmin()]
        lines.append(f"- {name}：9月收益落后日{int(g.annual_excess_delta.lt(0).sum())}/{len(g)}，最差收益差日期{int(worst.trade_date)}，当日超额差{worst.annual_excess_delta/252:+.6%}。完整逐日分差见daily_comparison.csv，可核对是否由少数大幅日期驱动。")
    lines += ["", "不根据9月拟合筛选或预测规则，不将市场状态、成员变化和收益的同期相关解释为因果。", "",
        "## 4. 收益集中度", "",
        "best_5/10_days_positive_mass_fraction表示最大正超额日占全部正超额总量的份额；net_sum_fraction除以全年净和，负日抵消可使该比率超过100%，不能单独判定稳定性。removed_annual_excess仅是删去最好日期的描述性算术量，不重新计算IC/换手，不提供新方案分数。超额算术和不是复利资金曲线。", "",
        table(concentration_table[concentration_table.period.astype(str).str.len().eq(4)],["candidate","period","positive_days","negative_days","annual_excess","best_5_days_positive_mass_fraction","best_10_days_positive_mass_fraction","best_10_days_net_sum_fraction","best_10_days_removed_annual_excess"]),"",
        "相对对照的增量收益集中度：", "",table(increment,["after","before","year","annual_excess","best_10_days_positive_mass_fraction","best_10_days_net_sum_fraction","best_10_days_removed_annual_excess"]), "",
        "全部年度、月份的1/5/10及前5%日期份额在return_concentration.csv。每项年度最好10日的日期和超额保存在input_manifest.json。", "",
        "## 5. 两种官方Top与样本质量", "",
        "收益Top：flag_limit_up=0且标签非缺失，至少100行，按pred降序取floor(n/10)。换手Top：只要求flag_limit_up=0及至少100行，缺标签仍参与。IC包含有标签的涨停行。价格有效Top依据当日OHLC有限、正数和高低价一致性及非涨停构建，不用未来标签；它只是替代换手诊断，不能替换官方分数，也不证明可执行交易。", "",
        table(q[q.top_type.isin(["return","turnover"])],["candidate","year","top_type","top_count","missing_label_fraction","invalid_price_fraction"]),"",
        "占比按入选股票—日期次数加权。daily_top_members.csv保存三种Top完整成员；daily_top_transitions.csv保存逐日进入/退出名单、交并集、Jaccard变化及进入/退出时各自日期的缺标签/无效价格计数。月度和年度质量表分别留档。收益Top缺标签为0是规则结果，不代表预测解决了缺失问题。", "",
        "daily_prediction_comparison.csv保存候选对十特征及27对31的同日Top对称差、缺标签/价格无效变化计数；集合变化不能精确分离缺失排名对换手的因果贡献。", "",
        "## 6. 27/31互补空间", "",
        table(corr[corr.after.eq(IDS[1])&corr.before.eq(IDS[2])],["year","all_spearman","price_valid_spearman","label_valid_diagnostic_spearman","return_jaccard","turnover_jaccard","price_valid_jaccard"]),"",
        "以上为逐日相关和重合的算术平均，完整逐日、逐月及同十特征的比较均保存。全样本相关可能被大量无效价格行影响，价格有效相关提供另一视角。收益Top依赖标签可用性，只用于事后诊断，不作为可部署筛选条件。Jaccard低于1及某些月份相反的优势方向表明预测并非完全重复，但不能推导线性混合后分数会提高；排序、Top切点和换手是非线性的。", "",
        "固定实验是否值得：存在有限的研究空间。应以本诊断暴露的逐年/逐月退化和真实价格有效换手风险作为检查项；尚无融合收益证明，也未计算任何新预测。平滑可能改变收益Top和IC并引入滞后，官方低换手又受无效样本影响，不能只优化该分项。后续需要单独预先固定实验，本任务不进入该阶段。", "",
        "## 7. 无标签测试期", "",table(pd.read_csv(output/"test_unlabeled_summary.csv"),["candidate","rows","dates","mean","std","min","max","official_prediction_turnover","price_valid_turnover","top_invalid_price_fraction"]),"",
        "只复用20261005T062821736270Z_2d9c336c阶段A的两个适配产物，不重新调用模型推断或训练。逐月预测分布、Top质量和逐日相关已保存。真实测试IC、收益、完整综合分全部为null；没有读取或重建测试标签，也没有据测试分布选择27/31、权重或参数。2026年6月截至8日，是不完整月份。", "",
        "## 8. 验收与接续", "",
        "12项年度评分与未修改官方Python脚本及原冻结标量逐项核对；真实差异见official_parity.json。run/audit两次核验，验收检查原始标签和键、旧切分资格数、模型列/参数、所有历史文件和源码哈希、分项加总及月初换手连续性。没有重新训练或重新生成模型预测。", "",
        "执行命令、完整stdout/stderr、退出码、耗时、测试结果、环境版本及失败记录在本目录。acceptance.json与delivery_acceptance.json为验收和交付依据；STEP1_HANDOFF.json列出准确路径、哈希和停止边界。仅本地ivor-work提交代码、测试、报告、handoff及小型验收；大预测和成员明细留本地。", "",
        "未验证：真实测试表现、未来跨期稳定性、实盘费用/容量/交易可执行性、融合、平滑、调参。完成后停止。", ""]
    return "\n".join(lines)


def audit(output):
    verify_sources(output)
    preserved=read(output/"preserved.json")
    if (output/"repair_registration.json").exists():
        # Original own source is still preserved byte-for-byte in executed_sources.
        require(preserved[OWN[0]]==sha256_file(output/"executed_sources"/OWN[0]),"original own source not preserved")
        preserved.pop(OWN[0])
    hashes_check(preserved)
    require(subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()=="ivor-work","branch changed")
    panel=load_raw_baseline_panel(RAW);saved=pd.read_csv(output/"daily_metrics.csv",float_precision="round_trip")
    verified=[]
    for rec in read(output/"registration.json")["inputs"]:
        pred,truth,x,raw,result,evidence=load_record(rec,panel,output)
        actual=score_official(pred,truth,x,return_details=True);detail=actual.pop("details")
        official={k:float(v) for k,v in official_evaluator()(evidence["paths"]["score_prediction"],str(Path(evidence["paths"]["score_prediction"]).parent)).items()}
        diff=metric_difference(actual,official);require(max(abs(v) for v in diff.values())<=1e-12,"audit official mismatch")
        daily=daily_metrics(detail);existing=saved[saved.candidate.eq(rec["candidate"])&saved.year.eq(int(str(raw.trade_date.min())[:4]))].reset_index(drop=True)
        pd.testing.assert_frame_equal(daily,existing[daily.columns],check_dtype=False,check_exact=True)
        verified.append(dict(candidate=rec["candidate"],split=rec["split"],official_difference=diff,raw_keys_labels_split_verified=True,daily_recomputed=True))
        print(f"Audit {rec['candidate']} {rec['split']} passed.",flush=True)
    annual=pd.read_csv(output/"annual_metrics.csv",float_precision="round_trip")
    monthly=pd.read_csv(output/"monthly_metrics.csv",float_precision="round_trip")
    for (name,year),g in saved.groupby(["candidate","year"]):
        agg=aggregate_daily(g,"year").iloc[0];a=annual[annual.candidate.eq(name)&annual.year.eq(year)].iloc[0]
        require(np.allclose(agg[METRICS].astype(float),a[METRICS].astype(float),rtol=0,atol=1e-12),"annual decomposition mismatch")
        m=aggregate_daily(g,"month");disk=monthly[monthly.candidate.eq(name)&monthly.year.eq(year)].sort_values("month")
        require(np.allclose(m[METRICS],disk[METRICS],rtol=0,atol=1e-12),"monthly decomposition mismatch")
    for period in ["annual","monthly","daily"]:
        comp=pd.read_csv(output/(period+"_comparison.csv"),float_precision="round_trip")
        total=comp.ic_contribution_delta+comp.excess_contribution_delta+comp.stability_contribution_delta
        require(np.allclose(total,comp.final_score_delta,rtol=0,atol=1e-12,equal_nan=True),"delta contributions do not sum")
    require(len(annual)==12 and len(monthly)==144,"incomplete diagnostic table")
    write(output/"acceptance.json",dict(accepted=True,new_training_runs=0,prediction_transformations=0,official_verified_runs=verified,
        preserved_source_artifacts_untracked_files_and_push_protections=True,annual_and_monthly_and_daily_decomposition=True,
        test_scope="unlabeled only; no true IC/excess/complete score",tests=read(output/"tests_result.json")))


def subprocess_phase(output, phase, command):
    started=time.perf_counter();at=datetime.now(timezone.utc).isoformat()
    with (output/(phase+".log")).open("x",encoding="utf-8") as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    receipt=dict(command=command,cwd=str(ROOT),started_at=at,exit_code=result.returncode,elapsed_seconds=time.perf_counter()-started,log=str(output/(phase+".log")),log_sha256=sha256_file(output/(phase+".log")))
    write(output/(phase+"_result.json"),receipt)
    require(result.returncode==0,f"{phase} failed; preserved log: {receipt['log']}")


def execute(output):
    started=time.perf_counter();prepare(output)
    subprocess_phase(output,"tests",[sys.executable,"-X","utf8","-B","-m","pytest","tests/test_frozen_prediction_diagnostics.py","tests/test_official_score.py","-q","-p","no:cacheprovider"])
    subprocess_phase(output,"dependency",[sys.executable,"-X","utf8","-B","-m","pip","check"])
    write(output/"environment.json",dict(python=sys.version,executable=sys.executable,pandas=pd.__version__,numpy=np.__version__,lightgbm=lgb.__version__))
    for phase in ["run","audit"]:
        subprocess_phase(output,phase,[sys.executable,"-X","utf8","-B",str(Path(__file__).resolve()),phase,"--output",str(output)])
    finish(output,started)


def finish(output,started):
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    body=report(output)
    with REPORT.open("x",encoding="utf-8") as handle:handle.write(body)
    (output/"REPORT.md").write_text(body,encoding="utf-8")
    evidence={p.relative_to(output).as_posix():sha256_file(p) for p in output.rglob("*") if p.is_file()}
    delivery=dict(accepted=True,report_path=str(REPORT),report_sha256=sha256_file(REPORT),artifact_directory=str(output),
        acceptance_sha256=sha256_file(output/"acceptance.json"),all_output_sha256=evidence,
        elapsed_seconds=time.perf_counter()-started,new_training_runs=0,prediction_transformations=0,stop_after_step1=True)
    write(output/"delivery_acceptance.json",delivery)
    handoff=dict(stage="step1_complete_stop",accepted=True,report=str(REPORT),report_sha256=sha256_file(REPORT),
        artifact_directory=str(output),freeze_path=str(FREEZE),freeze_sha256=sha256_file(FREEZE),
        acceptance_path=str(output/"acceptance.json"),acceptance_sha256=sha256_file(output/"acceptance.json"),
        delivery_acceptance_path=str(output/"delivery_acceptance.json"),delivery_acceptance_sha256=sha256_file(output/"delivery_acceptance.json"),
        input_manifest_path=str(output/"input_manifest.json"),input_manifest_sha256=sha256_file(output/"input_manifest.json"),
        historical_inputs=read(output/"input_manifest.json"),test_inputs=read(output/"test_input_manifest.json"),
        commands={p.stem:read(p) for p in output.glob("*_result.json")},
        failures=[read(p) for p in output.glob("failure_*.json")],
        limitations=["2021-2023 heavily developed; 2024 seen; no blind test","official turnover includes missing labels and invalid prices","test has no true labels; no real IC/return/complete score","no fusion/smoothing/tuning performed"],
        next_action="Stop. A new authorized task and fixed experimental design are required before fusion/smoothing/tuning.")
    write(HANDOFF,handoff)
    require(sha256_file(REPORT)==handoff["report_sha256"],"report seal failed")
    hashes_check(evidence,output)
    print(f"Accepted. Report: {REPORT}; output: {output}; elapsed: {time.perf_counter()-started:.1f}s",flush=True)


def resume(output):
    started=time.perf_counter()
    require((output/"repair_registration.json").exists(),"repair must be registered without changing original registration")
    require(not REPORT.exists() and not HANDOFF.exists(),"existing report/handoff must not be overwritten")
    attempt=1
    while (output/f"repair_tests{attempt}.log").exists():attempt+=1
    subprocess_phase(output,f"repair_tests{attempt}",[sys.executable,"-X","utf8","-B","-m","pytest","tests/test_frozen_prediction_diagnostics.py","tests/test_official_score.py","-q","-p","no:cacheprovider"])
    for phase in ["run","audit"]:
        label=phase+f"_resume{attempt}"
        subprocess_phase(output,label,[sys.executable,"-X","utf8","-B",str(Path(__file__).resolve()),phase,"--output",str(output)])
    finish(output,started)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=["execute","run","audit","resume"]);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args();output=args.output.resolve()
    require(output.is_relative_to(ROOT/"artifacts/model_optimization/step1"),"output outside authorized directory")
    try:
        {"execute":execute,"run":run,"audit":audit,"resume":resume}[args.phase](output)
    except BaseException as exc:
        if output.exists():
            path=output/("failure_"+args.phase+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
            write(path,dict(phase=args.phase,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),accepted=False))
        raise


if __name__=="__main__":main()
