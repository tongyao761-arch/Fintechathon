"""Bounded, exclusive-create historical diagnostics of the frozen contracts."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
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

from scripts.diagnose_frozen_predictions import (
    aggregate_daily, aggregate_top, concentration, contributions, daily_metrics,
    metric_difference, official_evaluator, records, require, top_diagnostics,
)
from scripts.optimize_frozen27 import check_model, frame_hash, array_hash
from scripts.validate_prediction_transforms import read, write, csv, validated, verify_hashes
from scripts.run_lightgbm_baseline import MODEL_PARAMS, split_masks, environment_versions
from src.data.baseline_panel import load_raw_baseline_panel
from src.features.baseline_v1 import FEATURE_COLUMNS as BASE
from src.features.features34 import build_features34, FEATURE_COLUMNS, FEATURE_DEFINITIONS
from src.metrics.official import score_official
from src.validation.splits import TimeSplit
from src.validation.experiment import sha256_file, prediction_hash

CFG = ROOT / 'configs/frozen_overfitting_risk.json'
FREEZE = ROOT / 'artifacts/features34_step5/FROZEN_CANDIDATES.json'
RAW = ROOT / '赛题五/赛题五数据/训练集.csv'
IDS = ['baseline10', 'S4R_lean31_minus4', 'S4R_full34_minus3']
KEYS = ['ts_code', 'trade_date']
ITERATIONS = [20, 40, 60, 80, 100]
OWN = ['configs/frozen_overfitting_risk.json', 'scripts/evaluate_frozen_overfitting_risk.py',
       'tests/test_frozen_overfitting_risk.py', 'scripts/plot_frozen_overfitting_risk.py']
REQUIRED = ['AGENTS.md', 'AGENTS.override.md', 'docs/features34/RESULTS.md',
    'docs/features34/STEP4_REVISION_REPORT.md', 'docs/features34/STEP5_REPORT.md',
    'docs/features34/FROZEN_MODEL_VALIDATION_REPORT.md', 'docs/model_optimization/STEP1_REPORT.md',
    'docs/model_optimization/STEP1_HANDOFF.json', 'docs/model_optimization/STEP3_REPORT.md',
    'docs/model_optimization/STEP3_HANDOFF.json', 'artifacts/features34_step5/FROZEN_CANDIDATES.json',
    'artifacts/features34_step6/acceptance.json', 'artifacts/features34_step6/run_index.json',
    'configs/features34_step3.json', 'configs/splits.yaml', '赛题五/evaluate.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def half_spec(dates, year):
    dates = sorted(set(int(d) for d in dates))
    h2 = [d for d in dates if year * 10000 + 701 <= d <= year * 10000 + 1231]
    require(bool(h2), 'missing half-year')
    pos = dates.index(h2[0]); require(pos >= 2, 'missing purge history')
    return dict(train_start=20180102, train_end=dates[pos-2], purge_date=dates[pos-1],
                valid_start=h2[0], valid_end=h2[-1])


def fit_statistics(y, pred, dates):
    y, pred, dates = np.asarray(y, float), np.asarray(pred, float), np.asarray(dates)
    require(len(y) > 0 and np.isfinite(y).all() and np.isfinite(pred).all(), 'invalid fitting sample')
    mse, zero = float(np.mean((y-pred)**2)), float(np.mean(y**2))
    rows = []
    for date in np.unique(dates):
        m = dates == date; a, b = y[m], pred[m]
        ic = float(spearmanr(a, b)[0]) if len(a) >= 30 and np.unique(a).size > 1 and np.unique(b).size > 1 else np.nan
        rows.append(dict(trade_date=int(date), n=int(m.sum()), ic=ic,
                         mse=float(np.mean((a-b)**2)), zero_mse=float(np.mean(a**2))))
    daily = pd.DataFrame(rows)
    return dict(n=len(y), market_days=len(daily), mse=mse, zero_mse=zero,
        normalized_mse=mse/zero if np.isfinite(zero) and zero > 0 else None,
        ic_mean=float(daily.ic.mean()) if daily.ic.notna().any() else None,
        ic_days=int(daily.ic.notna().sum())), daily


def moving_indices(n, block=20, repetitions=2000, seed=20261006):
    require(n >= block and block > 0, 'too few dates for prescribed moving blocks')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n-block+1, size=(repetitions, (n+block-1)//block))
    return (starts[..., None] + np.arange(block)).reshape(repetitions, -1)[:, :n]


def paired_bootstrap(a, b, indices):
    require(a.trade_date.tolist() == b.trade_date.tolist(), 'bootstrap date pairing differs')
    # Metric means have their own valid-date denominators; joint rows preserve covariance.
    metrics = ['ic_mean', 'annual_excess', 'mean_turnover']
    deltas = []
    for metric in metrics:
        aa, bb = a[metric].to_numpy(float), b[metric].to_numpy(float)
        deltas.append(np.nanmean(aa[indices], axis=1)-np.nanmean(bb[indices], axis=1))
    d = np.stack(deltas, axis=1)
    samples = np.column_stack([d, .4*d[:, 0], .3*d[:, 1], -.3*d[:, 2],
                               .4*d[:, 0]+.3*d[:, 1]-.3*d[:, 2]])
    names = metrics + ['ic_contribution', 'excess_contribution', 'stability_contribution', 'final_score']
    require(np.isfinite(samples).all(), 'bootstrap replicate has invalid denominator')
    return names, samples


def legacy_equivalence(rec, panel, features):
    modules=[]
    for name,module_name in [('src/data/baseline_panel.py','src.data.overfit_legacy_panel'),
                             ('src/features/baseline_v1.py','src.features.overfit_legacy_features')]:
        entry=rec['executed_sources'][name]
        spec=importlib.util.spec_from_file_location(module_name,entry['snapshot'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
    old_panel=modules[0].load_raw_baseline_panel(RAW)
    pd.testing.assert_frame_equal(old_panel,panel,check_exact=True)
    old_features=modules[1].build_baseline_v1_features(old_panel)
    pd.testing.assert_frame_equal(old_features,features[list(BASE)],check_exact=True)
    del old_panel,old_features;gc.collect()
    return dict(candidate='baseline10',split='oos_2024',full_raw_panel_and_flags_equal=True,
                full_ten_float32_features_equal=True,rows=len(panel),
                reason='Historical date validation addition and post-cast overflow protection; no changed actual inputs. features34 did not exist in the old ten-feature model dependency list.',
                legacy_sources={n:rec['executed_sources'][n] for n in ['src/data/baseline_panel.py','src/features/baseline_v1.py']})


def history_sources(panel, features):
    h = read(ROOT/'docs/model_optimization/STEP1_HANDOFF.json')
    require(h['accepted'], 'STEP1 not accepted')
    for key, hk in [('report','report_sha256'), ('input_manifest_path','input_manifest_sha256'),
                    ('acceptance_path','acceptance_sha256')]:
        require(sha256_file(Path(h[key])) == h[hk], 'STEP1 seal '+key)
    require(read(Path(h['acceptance_path']))['accepted'], 'STEP1 source not accepted')
    recs = h['historical_inputs']
    authoritative = records()
    require(len(recs) == 12, 'expected 12 originals')
    for r in recs:
        index = next(v for v in authoritative if (v['candidate'],v['split']) == (r['candidate'],r['split']))
        require(r['directory'] == index['directory'] and r['summary_sha256'] == index['summary_sha256'], 'wrong historical candidate')
        for key, path in r['paths'].items():
            require(sha256_file(Path(path)) == r['sha256'][key], 'historical path hash '+path)
        for v in r['executed_sources'].values():
            require(sha256_file(Path(v['snapshot'])) == v['sha256'], 'historical executable changed')
        prior = read(r['paths']['provenance'])
        require(prior['data'] == read(FREEZE)['source']['data'], 'historical data source')
        for name in ['src/data/baseline_panel.py','src/features/features34.py','src/features/baseline_v1.py',
                     'src/metrics/official.py','scripts/run_lightgbm_baseline.py','configs/splits.yaml','赛题五/evaluate.py']:
            expected=prior['source_sha256'].get(name)
            if expected!=sha256_file(ROOT/name):
                require(r['candidate']=='baseline10' and r['split']=='oos_2024' and
                        name in ['src/data/baseline_panel.py','src/features/features34.py','src/features/baseline_v1.py'],
                        'current executable differs from original '+name)
                if name=='src/features/features34.py':require(expected is None,'unexplained old features34')
                else:require(expected==r['executed_sources'][name]['sha256'],'legacy mismatch')
        expected = prior['source_sha256'].get('configs/features34_step3.json')
        if expected is None:
            require(r['candidate']=='baseline10' and r['split'] in ['primary_2023','oos_2024'], 'unexplained missing historical config')
            expected=read(FREEZE)['source']['source_sha256']['configs/features34_step3.json']
        require(sha256_file(ROOT/'configs/features34_step3.json') == expected, 'historical split config changed')
        check_model(r['paths']['model'], r['features'], MODEL_PARAMS)
    require(read(ROOT/'artifacts/features34_step6/acceptance.json')['accepted'], 'STEP6 not accepted')
    for r in read(ROOT/'artifacts/features34_step6/run_index.json')['runs']:
        d=ROOT/r['directory']; require(sha256_file(d/'summary.json')==r['summary_sha256'], 'repeat summary')
        orig=read(ROOT/r['original_directory']/'summary.json')
        a=next(s for s in orig['splits'] if s['split_name']==r['item']['split'])
        b=read(d/'summary.json')['splits'][0]
        require(a['file_sha256']==b['file_sha256'], 'repeat original files differ')
        verify_hashes(b['file_sha256'],d/r['item']['split'])
    legacy=legacy_equivalence(next(r for r in recs if r['candidate']=='baseline10' and r['split']=='oos_2024'),panel,features)
    return recs,legacy


def evidence(panel, features, spec, columns, next_dates):
    train, valid = split_masks(panel, TimeSplit(name='diagnostic', **spec))
    require(next_dates.loc[train].notna().all() and next_dates.loc[train].lt(spec['valid_start']).all(), 'label crosses validation')
    y=panel.loc[train,'y_ret_1d'].astype('float32')
    require(np.isfinite(y).all(), 'nonfinite float32 training target')
    return dict(training_samples=int(train.sum()), prediction_rows=int(valid.sum()),
        train_mask_sha256=array_hash(train.to_numpy()), training_keys_sha256=frame_hash(panel.loc[train,KEYS]),
        training_labels_sha256=array_hash(y.to_numpy()), training_features_sha256=frame_hash(features.loc[train,columns]),
        valid_keys_sha256=frame_hash(panel.loc[valid,KEYS]), validation_features_sha256=frame_hash(features.loc[valid,columns]),
        next_label_date_max=int(next_dates.loc[train].max()), purge_trained_rows=int((train & panel.trade_date.eq(spec['purge_date'])).sum()))


def prepare(out, panel, features):
    require(not (out/'registration.json').exists(), 'registration exists; refuse overwrite')
    cfg=read(CFG)
    require(cfg['candidates']==IDS and cfg['years']==[2021,2022,2023,2024] and
            cfg['maximum_fit_calls_including_failures']==12 and cfg['iterations']==ITERATIONS and
            cfg['bootstrap']==dict(block_days=20,repetitions=2000,seed=20261006,confidence=.95), 'matrix changed')
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='ivor-work', 'branch')
    freeze=read(FREEZE); require(sha256_file(RAW)==freeze['source']['data']['sha256'], 'raw data changed')
    packages=dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))
    require(packages==freeze['source']['dependencies'],'frozen training environment differs')
    columns={'baseline10':list(BASE)}
    for c in freeze['candidates']:
        require(c['model_params']==MODEL_PARAMS, 'original params differ')
        require(c['feature_count']==len(c['features'])==(27 if c['candidate']==IDS[1] else 31), 'count')
        require(c['feature_definitions']=={k:FEATURE_DEFINITIONS[k] for k in c['features']}, 'formula')
        columns[c['candidate']]=c['features']
    recs,legacy=history_sources(panel,features)
    next_dates=panel.groupby('ts_code',observed=True,sort=False).trade_date.shift(-1)
    matrix=[]; originals=[]
    for year in cfg['years']:
        spec=half_spec(panel.trade_date.unique(),year)
        for name in IDS:
            ev=evidence(panel,features,spec,columns[name],next_dates)
            matrix.append(dict(id=f'{name}_{year}_H2',candidate=name,year=year,split_spec=spec,
                               columns=columns[name],parameters=MODEL_PARAMS,**ev))
            r=next(v for v in recs if v['candidate']==name and int(v['split'][-4:])==year)
            require(r['features']==columns[name], 'original column order differs')
            ev0=evidence(panel,features,r['split_spec'],columns[name],next_dates)
            sm=next(v for v in read(r['paths']['summary'])['splits'] if v['split_name']==r['split'])
            require(sm['dates']==r['split_spec'] and sm['train_samples']==ev0['training_samples'], 'original split qualification')
            originals.append(dict(candidate=name,year=year,**ev0))
    # Guard all pre-existing repo files, including ignored model artifacts, independently of manifests.
    gitpaths=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode('utf8').split('\0')
    paths={ROOT/n for n in gitpaths if n} | {p for p in (ROOT/'artifacts').rglob('*') if p.is_file()}
    paths |= {RAW,ROOT/'.git/config',ROOT/'.git/hooks/pre-push',ROOT/'.git/info/exclude'}
    preserved={p.relative_to(ROOT).as_posix():sha256_file(p) for p in sorted(paths) if p.is_file() and
        not p.is_relative_to(out) and p.relative_to(ROOT).as_posix() not in OWN}
    write(out/'preserved.json',preserved)
    deps=set(REQUIRED+OWN) | {p.relative_to(ROOT).as_posix() for p in ROOT.glob('src/**/*.py')} | {
        p.relative_to(ROOT).as_posix() for p in ROOT.glob('scripts/*.py')} | {
        p.relative_to(ROOT).as_posix() for p in ROOT.glob('configs/*') if p.is_file()}
    sources={n:sha256_file(ROOT/n) for n in sorted(deps)}
    for n in sources:
        dest=out/'executed_sources'/n; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/n,dest)
    reg=dict(registered_at=now(),config=cfg,matrix=matrix,originals=originals,inputs=recs,columns=columns,
        legacy_source_equivalence=legacy,
        calendar=sorted(int(d) for d in panel.trade_date.unique()),parameters=MODEL_PARAMS,
        source_sha256=sources,raw_sha256=sha256_file(RAW),preserved_sha256=sha256_file(out/'preserved.json'),
        history_feature_sha256={c:frame_hash(features[[c]]) for c in features},environment=environment_versions(),
        interpreter=sys.executable,branch='ivor-work',parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        historical_metadata_exception='baseline10 primary_2023 and oos_2024 predate step3 config; step3 config checked against frozen authoritative hash, original oos_2024 dates checked against unchanged configs/splits.yaml and actual samples/model',
        metrics=['MSE','MSE/zero_MSE','daily_Rank_IC','official_scores','component_contributions','Top_quality','concentration'],
        periods='Annual fixed and stitched annual; separately reset official H1/H2; monthly from continuous annual daily sequence',
        predictions='Full validation keys; float32 X, native NaNs, float64 predictions; original parquet models must reload exactly. Scoring uses default CSV parse, reports rounding.')
    write(out/'registration.json',reg); write(out/'registration_sha256.json',dict(sha256=sha256_file(out/'registration.json')))
    print('Registration sealed; 12 fits; '+str(out),flush=True)
    return reg


def verify_registration(out):
    reg=read(out/'registration.json')
    require(sha256_file(out/'registration.json')==read(out/'registration_sha256.json')['sha256'], 'registration changed')
    current=dict(reg['source_sha256'])
    for repair_path in sorted(out.glob('repair_registration*.json')):
        repair=read(repair_path)
        require(repair['original_registration_sha256']==sha256_file(out/'registration.json'),'repair registration origin')
        require(set(repair['repaired_source_sha256'])=={'scripts/evaluate_frozen_overfitting_risk.py'},'repair scope')
        verify_hashes(repair['repaired_source_sha256'],out/repair.get('source_directory','repaired_sources'))
        current.update(repair['repaired_source_sha256'])
    verify_hashes(current); verify_hashes(reg['source_sha256'],out/'executed_sources')
    require(sha256_file(RAW)==reg['raw_sha256'] and sha256_file(out/'preserved.json')==reg['preserved_sha256'], 'source data/protection manifest changed')
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='ivor-work','branch changed')
    return reg


def fitting(out, model, panel, features, columns, spec, year, origin):
    market=sorted(int(d) for d in panel.trade_date.unique() if spec['train_start']<=d<=spec['train_end'])[-126:]
    train,_=split_masks(panel,TimeSplit(name='diagnostic',**spec))
    masks={'train':train & panel.trade_date.isin(market),
           'valid':panel.trade_date.between(spec['valid_start'],spec['valid_end']) &
               panel.is_price_valid.eq(1) & np.isfinite(panel.y_ret_1d)}
    rows=[]; daily=[]
    for scope,mask in masks.items():
        x=features.loc[mask,columns]; y=panel.loc[mask,'y_ret_1d']; dates=panel.loc[mask,'trade_date']
        full=model.predict(x)
        require(np.array_equal(full,model.predict(x,num_iteration=100)), 'saved model prefix 100 differs')
        for iteration in ITERATIONS:
            pred=full if iteration==100 else model.predict(x,num_iteration=iteration)
            stats,d=fit_statistics(y,pred,dates)
            meta=dict(candidate=origin['candidate'],year=year,origin=origin['origin'],scope=scope,
                      iteration=iteration,keys_sha256=frame_hash(panel.loc[mask,KEYS]),
                      sample_start=int(dates.min()),sample_end=int(dates.max()))
            rows.append(dict(**meta,**stats)); daily.append(d.assign(**{k:v for k,v in meta.items() if k!='keys_sha256'}))
    return rows,daily


def save_predictions(pred, dest):
    dest.mkdir(parents=True,exist_ok=True)
    validated(pred);pred.to_parquet(dest/'predictions.parquet',index=False)
    csv(pred,dest/'evaluation_predictions.csv')
    roundtrip=pd.read_csv(dest/'evaluation_predictions.csv',float_precision='round_trip')
    np.testing.assert_array_equal(roundtrip.pred,pred.pred)
    return float(np.max(np.abs(pd.read_csv(dest/'evaluation_predictions.csv').pred-pred.pred)))


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    require(not (out/'registration.json').exists(), 'existing registration; refuse rerunning training')
    print('Loading raw historical panel and building frozen X on all rows',flush=True)
    panel=load_raw_baseline_panel(RAW);features=build_features34(panel,FEATURE_COLUMNS)
    require(features.index.equals(panel.index) and all(v==np.dtype('float32') for v in features.dtypes), 'float32 aligned features')
    require(not np.isinf(features.to_numpy()).any(), 'infinite feature')
    reg=prepare(out,panel,features)
    original_predictions={}; fit_rows=[]; fit_daily=[]; verified=[]; run_index=[]
    for rec in reg['inputs']:
        name=rec['candidate'];year=int(rec['split'][-4:]);spec=rec['split_spec'];cols=reg['columns'][name]
        print(f'Verifying original {name} {year} and 20/40/60/80/100 prefixes',flush=True)
        model=check_model(rec['paths']['model'],cols,reg['parameters'])
        mask=panel.trade_date.between(spec['valid_start'],spec['valid_end'])
        pq=pd.read_parquet(rec['paths']['predictions']);pq['ts_code']=pq.ts_code.astype(str)
        validated(pq);raw=panel.loc[mask].reset_index(drop=True);raw['ts_code']=raw.ts_code.astype(str)
        pd.testing.assert_frame_equal(pq[KEYS],raw[KEYS],check_dtype=False,check_exact=True)
        np.testing.assert_array_equal(model.predict(features.loc[mask,cols]),pq.pred)
        pred=pd.read_csv(rec['paths']['score_prediction']);truth=pd.read_csv(rec['paths']['score_truth']);x=pd.read_csv(rec['paths']['score_x'])
        for fr in [pred,truth,x]:pd.testing.assert_frame_equal(fr[KEYS],raw[KEYS],check_dtype=False,check_exact=True)
        pd.testing.assert_series_equal(truth.y_ret_1d,raw.y_ret_1d,check_dtype=False,check_exact=True)
        pd.testing.assert_series_equal(x.flag_limit_up,raw.flag_limit_up,check_dtype=False,check_exact=True)
        sm=next(v for v in read(rec['paths']['summary'])['splits'] if v['split_name']==rec['split'])
        verify_hashes(sm['file_sha256'],ROOT/rec['directory']/rec['split'])
        require(prediction_hash(pq.pred)==sm['prediction_sha256'], 'historical prediction seal')
        rounded=float(np.max(np.abs(pred.pred-pq.pred)));require(rounded<=reg['config']['csv_prediction_tolerance'],'historical CSV changed')
        result=official_evaluator()(rec['paths']['score_prediction'],str(Path(rec['paths']['score_truth']).parent))
        diff=max(abs(v) for v in metric_difference(result,sm['metrics']).values())
        require(diff<=reg['config']['official_parity_tolerance'],'historical official differs')
        verified.append(dict(candidate=name,year=year,actual_X_reload_equal=True,prefix_100_equal=True,
                             official_max_difference=diff,csv_rounding=rounded))
        original_predictions[name,year]=pq
        diag_spec=dict(spec);diag_spec['valid_end']=max(d for d in reg['calendar'] if year*10000<d<year*10000+701)
        r,d=fitting(out,model,panel,features,cols,diag_spec,year,dict(candidate=name,origin='annual_start'))
        fit_rows.extend(r);fit_daily.extend(d)
    write(out/'original_verification.json',verified)
    updated={};attempts=0
    for item in reg['matrix']:
        require(attempts<12,'fit budget exhausted');name=item['candidate'];year=item['year'];cols=item['columns'];spec=item['split_spec']
        d=out/'models'/item['id'];d.mkdir(parents=True,exist_ok=False)
        train,valid=split_masks(panel,TimeSplit(name=item['id'],**spec))
        require(int(train.sum())==item['training_samples'],'registered samples changed')
        write(d/'attempt_started.json',dict(fit_call=attempts+1,started_at=now(),item=item));attempts+=1
        print(f'Fit {attempts}/12 {item["id"]} samples={int(train.sum())}',flush=True);started=time.perf_counter()
        try:
            model=lgb.LGBMRegressor(**item['parameters'])
            model.fit(features.loc[train,cols],panel.loc[train,'y_ret_1d'].astype('float32'),feature_name=cols)
            require(all(model.get_params()[k]==v for k,v in item['parameters'].items()),'actual parameters changed')
            values=model.predict(features.loc[valid,cols]);model.booster_.save_model(str(d/'lightgbm.txt'))
            reloaded=check_model(d/'lightgbm.txt',cols,item['parameters'])
            np.testing.assert_array_equal(values,reloaded.predict(features.loc[valid,cols]))
            pred=panel.loc[valid,KEYS].reset_index(drop=True);pred['ts_code']=pred.ts_code.astype(str);pred['pred']=values
            rounding=save_predictions(pred,d)
            require(rounding<=reg['config']['csv_prediction_tolerance'],'new CSV rounding')
            write(d/'summary.json',dict(item=item,actual_params=model.get_params(),reload_equal=True,prefix_100_equal=True,
                csv_rounding=rounding,prediction_sha256=prediction_hash(values),elapsed_seconds=time.perf_counter()-started))
            updated[name,year]=pred
            r,dd=fitting(out,reloaded,panel,features,cols,spec,year,dict(candidate=name,origin='midyear'))
            fit_rows.extend(r);fit_daily.extend(dd)
            write(d/'attempt_success.json',dict(fit_call=attempts,elapsed_seconds=time.perf_counter()-started))
            write(d/'files.json',{p.name:sha256_file(p) for p in d.iterdir() if p.is_file()})
            run_index.append(dict(id=item['id'],directory=str(d),files_sha256=sha256_file(d/'files.json')))
            del model,reloaded;gc.collect()
        except BaseException as exc:
            write(d/'attempt_failure.json',dict(fit_call=attempts,error=str(exc),traceback=traceback.format_exc()))
            raise
    write(out/'run_index.json',dict(runs=run_index,fit_calls=attempts,originals=reg['inputs']))
    csv(pd.DataFrame(fit_rows),out/'fitting_metrics.csv');csv(pd.concat(fit_daily,ignore_index=True),out/'fitting_daily.csv')
    for year in reg['config']['years']:
        for name in IDS:
            pq=original_predictions[name,year]; h2=updated[name,year];start=int(h2.trade_date.min())
            joined=pd.concat([pq[pq.trade_date.lt(start)],h2],ignore_index=True).sort_values(KEYS).reset_index(drop=True)
            # Annual stocks panel order is restored, with exactly one model switch.
            pd.testing.assert_frame_equal(joined[KEYS],pq[KEYS],check_dtype=False,check_exact=True)
            save_predictions(joined,out/'schemes'/f'halfyear_update_{name}_{year}')
    write(out/'training_complete.json',dict(fit_calls=attempts,status='training_complete'))
    print('All 12 fits and 24 model fitting curves complete; scoring follows',flush=True)
    score_all(out,panel)


def score_dataset(out, pred, raw, name, year, scheme, period):
    dest=out/'scores'/f'{scheme}_{name}_{year}_{period}';dest.mkdir(parents=True,exist_ok=False)
    csv(pred,dest/'evaluation_predictions.csv')
    # Reuse original label tokens as read by the official CSV parser; default parsing audited.
    shutil.copy2(out/'truth_reference'/f'{year}_{period}.csv',dest/'测试集_Y.csv')
    csv(raw[KEYS+['flag_limit_up']],dest/'测试集_X.csv')
    p=pd.read_csv(dest/'evaluation_predictions.csv');y=pd.read_csv(dest/'测试集_Y.csv');x=pd.read_csv(dest/'测试集_X.csv')
    pd.testing.assert_series_equal(y.y_ret_1d,raw.y_ret_1d.reset_index(drop=True),check_dtype=False,check_exact=True)
    local=score_official(p,y,x,return_details=True);details=local.pop('details')
    official=official_evaluator()(str(dest/'evaluation_predictions.csv'),str(dest))
    diff=max(abs(v) for v in metric_difference(local,official).values())
    require(diff<=1e-12,'official/local mismatch')
    meta=dict(candidate=name,year=year,scheme=scheme,period=period)
    write(dest/'parity.json',dict(**meta,max_difference=diff,metrics=official,
        csv_rounding=float(np.max(np.abs(p.pred-pred.pred)))))
    daily=daily_metrics(details)
    # Annual daily sequence is continuous over July and month boundaries; halves reset separately.
    for key in ['daily_ic','daily_excess','daily_turnover','daily_top_sets','daily_return_top_sets']:
        csv(details[key],dest/(key+'.csv'))
    joined=p.merge(raw[KEYS+['is_price_valid']],on=KEYS,validate='one_to_one').merge(y,on=KEYS,validate='one_to_one').merge(x,on=KEYS,validate='one_to_one')
    quality,members,transitions,turns,_=top_diagnostics(joined,details)
    daily=daily.merge(turns,on='trade_date',how='left',validate='one_to_one')
    csv(quality,dest/'top_quality.csv');csv(members,dest/'top_members.csv');csv(transitions,dest/'top_transitions.csv')
    csv(turns,dest/'price_valid_turnover.csv')
    summary=contributions(pd.DataFrame([dict(**meta,**official,price_valid_turnover=float(turns.price_valid_turnover.mean()))]))
    return summary,daily.assign(**meta),quality.assign(**meta)


def score_all(out,panel=None):
    reg=verify_registration(out)
    if panel is None:panel=load_raw_baseline_panel(RAW)
    summaries=[];days=[];qualities=[];concentrations=[]
    for year in reg['config']['years']:
        rec0=next(r for r in reg['inputs'] if int(r['split'][-4:])==year)
        raw=panel.loc[panel.trade_date.between(rec0['split_spec']['valid_start'],rec0['split_spec']['valid_end'])].reset_index(drop=True)
        raw['ts_code']=raw.ts_code.astype(str)
        # Preserve exact original labels as parsed by official evaluator, not a new label serialization.
        truth=pd.read_csv(rec0['paths']['score_truth']);raw['y_ret_1d']=truth.y_ret_1d
        reference=out/'truth_reference';reference.mkdir(exist_ok=True)
        tokens=pd.read_csv(rec0['paths']['score_truth'],dtype=str,keep_default_na=False)
        shutil.copy2(rec0['paths']['score_truth'],reference/f'{year}_annual.csv')
        for half in ['H1','H2']:
            m=tokens.trade_date.astype(int)%10000<701
            csv(tokens.loc[m if half=='H1' else ~m],reference/f'{year}_{half}.csv')
        for scheme in reg['config']['schemes']:
            for name in IDS:
                print(f'Scoring {scheme} {name} {year} annual/H1/H2',flush=True)
                rec=next(r for r in reg['inputs'] if r['candidate']==name and int(r['split'][-4:])==year)
                pq=pd.read_parquet(rec['paths']['predictions']) if scheme=='annual_fixed' else pd.read_parquet(out/'schemes'/f'{scheme}_{name}_{year}'/'predictions.parquet')
                pq['ts_code']=pq.ts_code.astype(str)
                for period in ['annual','H1','H2']:
                    mask=np.ones(len(raw),bool) if period=='annual' else ((raw.trade_date%10000<701).to_numpy() if period=='H1' else (raw.trade_date%10000>=701).to_numpy())
                    s,d,q=score_dataset(out,pq.loc[mask].reset_index(drop=True),raw.loc[mask].reset_index(drop=True),name,year,scheme,period)
                    if scheme=='annual_fixed' and period=='annual':
                        historical=next(v for v in read(rec['paths']['summary'])['splits'] if v['split_name']==rec['split'])['metrics']
                        require(max(abs(float(s.iloc[0][k])-v) for k,v in historical.items())<=1e-12,'rescored original differs from original CSV')
                    summaries.append(s);days.append(d);qualities.append(q)
                    concentrations.append(dict(candidate=name,year=year,scheme=scheme,period=period,**concentration(d.excess.dropna())))
    annual_half=pd.concat(summaries,ignore_index=True);daily=pd.concat(days,ignore_index=True);quality=pd.concat(qualities,ignore_index=True)
    csv(annual_half,out/'period_metrics.csv');csv(daily,out/'daily_metrics.csv');csv(quality,out/'daily_top_quality.csv')
    csv(pd.DataFrame(concentrations),out/'return_concentration.csv')
    monthly=[];topmonthly=[]
    for (scheme,name,year),d in daily[daily.period.eq('annual')].groupby(['scheme','candidate','year']):
        d=d.assign(month=d.trade_date//100)
        m=aggregate_daily(d,'month').assign(candidate=name,year=year,scheme=scheme)
        m=m.merge(d.groupby('month').price_valid_turnover.mean(),on='month',validate='one_to_one');monthly.append(m)
        q=quality[(quality.period=='annual')&(quality.scheme==scheme)&(quality.candidate==name)&(quality.year==year)].assign(month=lambda f:f.trade_date//100)
        topmonthly.append(aggregate_top(q,'month').assign(scheme=scheme,year=year))
    monthly=pd.concat(monthly,ignore_index=True)
    csv(monthly,out/'monthly_metrics.csv');csv(pd.concat(topmonthly,ignore_index=True),out/'monthly_top_quality.csv')
    topannual=[]
    for (scheme,period),q in quality.groupby(['scheme','period']):
        topannual.append(aggregate_top(q,'year').assign(scheme=scheme,period=period))
    csv(pd.concat(topannual,ignore_index=True),out/'period_top_quality.csv')
    compare_all(out,annual_half,monthly,daily,reg)
    selection_audit(out)
    write(out/'scoring_complete.json',dict(status='complete',official_comparisons=len(annual_half)))


def difference_row(a,b,keys):
    metrics=['ic_mean','annual_excess','mean_turnover','final_score','ic_contribution','excess_contribution','stability_contribution','price_valid_turnover']
    row={k:a[k] for k in keys}
    row.update(after=a['candidate'],before=b['candidate'])
    for m in metrics:row[m+'_delta']=float(a[m]-b[m])
    row['component_direction_conflict']=bool(any(row[m+'_delta']<0 for m in ['ic_contribution','excess_contribution','stability_contribution']) and any(row[m+'_delta']>0 for m in ['ic_contribution','excess_contribution','stability_contribution']))
    return row


def compare_all(out,periods,months,daily,reg):
    prows=[];mrows=[];intervals=[];increment=[]
    for frame,keys,rows in [(periods,['scheme','year','period'],prows),(months,['scheme','year','month'],mrows)]:
        for vals,g in frame.groupby(keys):
            for after,before in reg['config']['comparisons']:
                a=g[g.candidate.eq(after)].iloc[0];b=g[g.candidate.eq(before)].iloc[0]
                rows.append(difference_row(a,b,keys))
    csv(pd.DataFrame(prows),out/'period_comparison.csv');csv(pd.DataFrame(mrows),out/'monthly_comparison.csv')
    negatives=pd.DataFrame(mrows);csv(negatives[negatives.final_score_delta<0],out/'negative_months.csv')
    update_rows=[]
    for (year,name,period),g in periods.groupby(['year','candidate','period']):
        if period=='H1':continue
        a=g[g.scheme.eq('halfyear_update')].iloc[0];b=g[g.scheme.eq('annual_fixed')].iloc[0]
        row=difference_row(a,b,['year','period']);row['candidate']=name;row['comparison']='update_minus_fixed';update_rows.append(row)
    csv(pd.DataFrame(update_rows),out/'update_comparison.csv')
    bd=out/'bootstrap';bd.mkdir(exist_ok=False)
    tasks=[]
    for (scheme,year,period),g in daily.groupby(['scheme','year','period']):
        for after,before in reg['config']['comparisons']:
            tasks.append((dict(scheme=scheme,year=int(year),period=period,after=after,before=before),
                          g[g.candidate.eq(after)],g[g.candidate.eq(before)]))
    for (year,name,period),g in daily.groupby(['year','candidate','period']):
        if period!='H1':tasks.append((dict(scheme='update_minus_fixed',year=int(year),period=period,after=name,before=name),
            g[g.scheme.eq('halfyear_update')],g[g.scheme.eq('annual_fixed')]))
    for i,(meta,a,b) in enumerate(tasks):
        a=a.sort_values('trade_date').reset_index(drop=True);b=b.sort_values('trade_date').reset_index(drop=True)
        idx=moving_indices(len(a));names,samples=paired_bootstrap(a,b,idx)
        np.savez_compressed(bd/f'pair_{i:03d}.npz',dates=a.trade_date.to_numpy(),indices=idx,samples=samples,
            a=a[['ic_mean','annual_excess','mean_turnover']].to_numpy(),b=b[['ic_mean','annual_excess','mean_turnover']].to_numpy())
        for j,m in enumerate(names):
            intervals.append(dict(**meta,metric=m,delta=float(a[m].mean()-b[m].mean()) if m!='final_score' else float(.4*(a.ic_mean.mean()-b.ic_mean.mean())+.3*(a.annual_excess.mean()-b.annual_excess.mean())-.3*(a.mean_turnover.mean()-b.mean_turnover.mean())),
                lower=float(np.quantile(samples[:,j],.025)),upper=float(np.quantile(samples[:,j],.975)),
                a_valid_days=int(a[m].notna().sum()),b_valid_days=int(b[m].notna().sum()),replicates=2000,
                archive=f'bootstrap/pair_{i:03d}.npz'))
        if meta['scheme']!='update_minus_fixed':
            z=a.excess-b.excess
            increment.append(dict(**meta,**concentration(z.dropna())))
    csv(pd.DataFrame(intervals),out/'paired_intervals.csv');csv(pd.DataFrame(increment),out/'incremental_return_concentration.csv')
    # Positive month contributions, separating mass concentration from cancellation in net sums.
    pm=[]
    for key,g in pd.DataFrame(mrows).groupby(['scheme','after','before','year']):
        v=g.excess_contribution_delta.to_numpy();pos=np.maximum(v,0);order=np.argsort(v)[::-1]
        pm.append(dict(scheme=key[0],after=key[1],before=key[2],year=int(key[3]),
            best_month=int(g.iloc[order[0]].month),largest_positive_mass_fraction=float(pos.max()/pos.sum()) if pos.sum()>0 else None,
            best_3_positive_mass_fraction=float(pos[order[:3]].sum()/pos.sum()) if pos.sum()>0 else None,
            negative_months=int((g.final_score_delta<0).sum()),worst_score_month=int(g.loc[g.final_score_delta.idxmin(),'month']),
            worst_month_delta=float(g.final_score_delta.min())))
    csv(pd.DataFrame(pm),out/'monthly_concentration.csv')


def selection_audit(out):
    # Indexed histories are checked, never rerun. Logical comparisons are correlated.
    audit=[];index_checks=[]
    for stage,new,reuse,logical,years,use in [
        ('features34_step1',3,0,3,'2023;2024','baseline migration and full34 framework; 2024 already visible'),
        ('features34_step2',12,2,14,'2023','14 prespecified feature groups; candidate exploration'),
        ('features34_step3',12,3,15,'2021;2022;2023','five candidates across development years; choose full34/lean31 contexts'),
        ('features34_step4',135,9,135,'2021;2022;2023','45 conditional single deletions across 3 years; screen2023 then cross-years'),
        ('features34_step4_revision',6,144,6,'2021;2022;2023','post-observation rule change; two joint combinations; rank by2023'),
        ('features34_step5',2,7,2,'2024','freeze before these candidate results; restricted review, not blind'),
        ('features34_step6',4,2,4,'2023;2024','four exact repeats only; no new selection')]:
        folder=ROOT/'artifacts'/stage;accepted=read(folder/'acceptance.json');require(accepted['accepted'],'history stage not accepted')
        path=folder/'run_index.json'
        rows=read(path)['runs'] if path.exists() else []
        if stage=='features34_step2':rows=read(folder/'summary.json')['runs']
        if stage=='features34_step1':
            rows=accepted['runs']
        for row in rows:
            p=ROOT/row['directory']/'summary.json';require(p.exists(),'missing history summary')
            expected=row.get('summary_sha256');actual=sha256_file(p)
            if expected:require(actual==expected,'historical index mismatch')
            d=read(p)
            # Verify all model/prediction file hashes in each saved split where available.
            for split in d.get('splits',[]):verify_hashes(split['file_sha256'],p.parent/split['split_name'])
            index_checks.append(dict(stage=stage,path=str(p),sha256=actual,indexed_sha256=expected))
        registration=next((folder/n for n in ['registration.json','screen_registration.json','preregistration.json'] if (folder/n).exists()),None)
        rd=read(registration) if registration else {}
        stamp=next((rd[k] for k in ['registered_at','started_at','created_at'] if k in rd),None)
        audit.append(dict(stage=stage,new_fit_calls=new,reuse_references=reuse,logical_model_comparisons=logical,
            observed_years=years,decision_use=use,registration_time=stamp,
            registration_path=str(registration) if registration else None,acceptance_sha256=sha256_file(folder/'acceptance.json'),
            note='reuse references are overlapping source counts, not independent observations'))
    csv(pd.DataFrame(audit),out/'selection_process_audit.csv');write(out/'selection_index_checks.json',index_checks)
    old=read(ROOT/'artifacts/features34_step4/preregistration.json');rev=read(ROOT/'artifacts/features34_step4_revision/registration.json')
    write(out/'selection_rules.json',dict(original=old['config']['rules'],revision=rev['config']['rules'],
        observed_prior_results=rev['observed_prior_results'],freeze_time=read(FREEZE)['frozen_at'],
        complete_selection_bias_quantifiable=False,
        reason='No new independent labeled holdout; development reused and rules revised after seeing results; external/manual discarded paths and unlogged comparisons cannot be guaranteed complete. Fixed-prediction bootstrap does not model selection.'))
    shutil.copy2(ROOT/'artifacts/features34_step4_revision/single_decisions.csv',out/'historical_single_decisions.csv')
    # Parent marginal effects remain separate from baseline10 effects.
    shutil.copy2(ROOT/'artifacts/features34_step4_revision/candidate_results.csv',out/'historical_parent_marginal.csv')
    verify_sensitivity(out)


def verify_sensitivity(out):
    records_s=[]
    validation=ROOT/'artifacts/frozen_models_validation/20261005T062821736270Z_2d9c336c'
    require(read(validation/'acceptance.json')['accepted'],'sensitivity history not accepted')
    for r in read(validation/'run_index.json')['runs']:
        item=r['item']
        if item['phase']!='B':continue
        d=ROOT/r['directory'];require(sha256_file(d/'summary.json')==r['summary_sha256'],'B indexed summary')
        require(sha256_file(d/'files.json')==r['files_sha256'],'B manifest')
        verify_hashes(read(d/'files.json'),d)
        model=check_model(d/'lightgbm.txt',read(d/'config.json')['features'],item['parameters'])
        expected=dict(MODEL_PARAMS);expected.update(item['change']);require(expected==item['parameters'],'B independent params')
        s=read(d/'summary.json')
        records_s.append(dict(candidate=item['candidate'],experiment=item['probe'],year=int(item['split'][-4:]),
            final_score=s['metrics']['final_score'],directory=str(d),model_sha256=sha256_file(d/'lightgbm.txt'),
            scope='verified historical saved parameters, indices and all sealed files; no retraining'))
    h=read(ROOT/'docs/model_optimization/STEP3_HANDOFF.json');require(h['accepted'],'M history incomplete')
    for key,hk in [('report','report_sha256'),('acceptance_path','acceptance_sha256'),('registration_path','registration_sha256')]:
        require(sha256_file(Path(h[key]))==h[hk],'M seal')
    d0=Path(h['artifact_directory']);reg=read(d0/'registration.json')
    for r in read(d0/'run_index.json')['runs']:
        d=Path(r['directory']);require(sha256_file(d/'files.json')==r['files_sha256'],'M manifest');verify_hashes(read(d/'files.json'),d)
        s=read(d/'summary.json');check_model(d/'lightgbm.txt',reg['columns'],s['item']['parameters'])
    old=pd.read_csv(d0/'annual_metrics.csv');shutil.copy2(d0/'annual_comparison.csv',out/'historical_M_comparison.csv')
    shutil.copy2(validation/'annual_comparison.csv',out/'historical_B_comparison.csv')
    csv(pd.DataFrame(records_s),out/'historical_sensitivity_audit.csv')
    write(out/'sensitivity_verification.json',dict(B_models=12,M_new_models=8,M_reuse=1,
        all_indexed_files_and_actual_parameters_verified=True,refit_calls=0,
        scope='Reference accepted original reload/official-score audits; do not repeat historical experiments'))


def audit(out):
    reg=verify_registration(out)
    verify_hashes(read(out/'preserved.json'))
    attempts=list((out/'models').glob('*/attempt_started.json'));require(len(attempts)==12,'budget/matrix incomplete')
    require(not list((out/'models').glob('*/attempt_failure.json')),'fit failure remains')
    panel=load_raw_baseline_panel(RAW);features=build_features34(panel,FEATURE_COLUMNS)
    require({c:frame_hash(features[[c]]) for c in features}==reg['history_feature_sha256'],'independent rebuilt frozen X differs')
    next_dates=panel.groupby('ts_code',observed=True,sort=False).trade_date.shift(-1)
    for item in reg['matrix']:
        actual=evidence(panel,features,item['split_spec'],item['columns'],next_dates)
        require(actual=={k:item[k] for k in actual},'independent training boundary/input differs')
        d=out/'models'/item['id'];model=check_model(d/'lightgbm.txt',item['columns'],item['parameters'])
        valid=panel.trade_date.between(item['split_spec']['valid_start'],item['split_spec']['valid_end'])
        pred=pd.read_parquet(d/'predictions.parquet');validated(pred)
        raw=panel.loc[valid,KEYS].reset_index(drop=True);raw['ts_code']=raw.ts_code.astype(str)
        pd.testing.assert_frame_equal(pred[KEYS],raw,check_dtype=False,check_exact=True)
        np.testing.assert_array_equal(model.predict(features.loc[valid,item['columns']]),pred.pred)
        require(prediction_hash(pred.pred)==read(d/'summary.json')['prediction_sha256'],'new prediction seal')
        print('Independent audit '+item['id'],flush=True)
    fm=pd.read_csv(out/'fitting_metrics.csv',float_precision='round_trip')
    require(len(fm)==240,'24 model * 2 samples * 5 prefixes')
    require(fm.groupby(['year','origin','scope']).keys_sha256.nunique().eq(1).all(),'diagnostic qualification differs across candidates')
    signals=[]
    for (name,year,origin),g in fm.groupby(['candidate','year','origin']):
        a=g[g.scope.eq('train')].sort_values('iteration');b=g[g.scope.eq('valid')].sort_values('iteration')
        for k in range(1,5):
            signals.append(dict(candidate=name,year=int(year),origin=origin,from_iteration=int(a.iloc[k-1].iteration),to_iteration=int(a.iloc[k].iteration),
                train_mse_delta=float(a.iloc[k].mse-a.iloc[k-1].mse),valid_mse_delta=float(b.iloc[k].mse-b.iloc[k-1].mse),
                valid_ic_delta=float(b.iloc[k].ic_mean-b.iloc[k-1].ic_mean),
                error_divergence=bool(a.iloc[k].mse<a.iloc[k-1].mse and b.iloc[k].mse>b.iloc[k-1].mse),
                ic_divergence=bool(a.iloc[k].mse<a.iloc[k-1].mse and b.iloc[k].ic_mean<b.iloc[k-1].ic_mean),
                train_valid_normalized_mse_gap=float(b.iloc[k].normalized_mse-a.iloc[k].normalized_mse),
                train_valid_ic_gap=float(a.iloc[k].ic_mean-b.iloc[k].ic_mean)))
    signal_frame=pd.DataFrame(signals)
    if (out/'fitting_risk_signals.csv').exists():
        pd.testing.assert_frame_equal(pd.read_csv(out/'fitting_risk_signals.csv',float_precision='round_trip'),signal_frame,
                                      check_dtype=False,check_exact=True)
    else:csv(signal_frame,out/'fitting_risk_signals.csv')
    # Stitched boundary is checked against exact saved model outputs, not half-score averages.
    for year in reg['config']['years']:
        for name in IDS:
            rec=next(r for r in reg['inputs'] if r['candidate']==name and int(r['split'][-4:])==year)
            original=pd.read_parquet(rec['paths']['predictions']);h2=pd.read_parquet(out/'models'/f'{name}_{year}_H2'/'predictions.parquet')
            joined=pd.read_parquet(out/'schemes'/f'halfyear_update_{name}_{year}'/'predictions.parquet')
            original['ts_code']=original.ts_code.astype(str)
            pd.testing.assert_frame_equal(joined[KEYS],original[KEYS],check_dtype=False,check_exact=True)
            mask=joined.trade_date.ge(h2.trade_date.min())
            np.testing.assert_array_equal(joined.loc[mask,'pred'],h2.pred)
            np.testing.assert_array_equal(joined.loc[~mask,'pred'],original.loc[~mask,'pred'])
    daily=pd.read_csv(out/'daily_metrics.csv',float_precision='round_trip');monthly=pd.read_csv(out/'monthly_metrics.csv',float_precision='round_trip')
    for (scheme,name,year),g in daily[daily.period.eq('annual')].groupby(['scheme','candidate','year']):
        rebuilt=aggregate_daily(g.assign(month=g.trade_date//100),'month')
        saved=monthly[(monthly.scheme==scheme)&(monthly.candidate==name)&(monthly.year==year)].reset_index(drop=True)
        np.testing.assert_allclose(rebuilt[['ic_mean','annual_excess','mean_turnover','final_score']],saved[['ic_mean','annual_excess','mean_turnover','final_score']],rtol=0,atol=1e-12)
        # July annual turnover must retain June predecessor, while standalone H2 drops its first day.
        d=out/'scores'/f'{scheme}_{name}_{year}_annual'
        t=pd.read_csv(d/'daily_turnover.csv');july=t[t.trade_date%10000>=701].iloc[0]
        require(int(july.previous_trade_date)%10000<701,'July boundary not continuous')
        h=pd.read_csv(out/'scores'/f'{scheme}_{name}_{year}_H2'/'daily_turnover.csv')
        require(int(h.trade_date.min())>int(july.trade_date),'standalone half turnover did not reset')
    ci=pd.read_csv(out/'paired_intervals.csv',float_precision='round_trip')
    for path in sorted((out/'bootstrap').glob('*.npz')):
        z=np.load(path);idx=z['indices'];require(np.array_equal(idx,moving_indices(len(z['dates']))),'bootstrap seed/blocks changed')
        aa,bb=z['a'],z['b']
        # Match the registered per-metric reduction order for exact bit comparison.
        # A vectorized 3-column reduction otherwise changes last-bit summation order.
        d=np.column_stack([np.nanmean(aa[:,j][idx],axis=1)-np.nanmean(bb[:,j][idx],axis=1) for j in range(3)])
        sample=np.column_stack([d,.4*d[:,0],.3*d[:,1],-.3*d[:,2],.4*d[:,0]+.3*d[:,1]-.3*d[:,2]])
        np.testing.assert_array_equal(sample,z['samples'])
        group=ci[ci.archive.eq('bootstrap/'+path.name)]
        np.testing.assert_allclose(group.lower,np.quantile(sample,.025,axis=0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(group.upper,np.quantile(sample,.975,axis=0),rtol=0,atol=1e-12)
    parities=[read(p) for p in (out/'scores').glob('*/parity.json')]
    require(len(parities)==72 and all(p['max_difference']<=1e-12 for p in parities),'72 official comparisons')
    verify_hashes(read(out/'preserved.json'))
    write(out/'acceptance.json',dict(accepted=True,fit_calls_including_failures=12,maximum=12,originals_verified=12,
        official_evaluations=72,official_tolerance=1e-12,official_max_difference=max(v['max_difference'] for v in parities),
        csv_max_rounding=max(v['csv_rounding'] for v in parities),
        independent_full_panel_rebuild=True,registration_unchanged=True,same_diagnostic_samples=True,
        saved_actual_parameters_columns_reload_keys_verified=True,all_label_boundaries_verified=True,
        stitched_keys_and_July_boundary_verified=True,monthly_turnover_continuity_verified=True,
        bootstrap_joint_blocks_nan_rules_no_recomputed_turnover_verified=True,
        preserved_existing_files=len(read(out/'preserved.json')),previous_files_raw_data_git_protections_unchanged=True,
        test_X_used=False,frozen_replaced=False,selection_changed=False,submission_created=False))
    print('Independent acceptance passed',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['run','score_all','selection_audit','audit']);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output).resolve()
    require(out.is_relative_to(ROOT/'artifacts/overfitting_risk'),'output outside task tree')
    start=time.perf_counter()
    try:
        if args.phase=='selection_audit':verify_registration(out)
        globals()[args.phase](out)
        write(out/(args.phase+'_result.json'),dict(phase=args.phase,exit_code=0,started_at=now(),elapsed_seconds=time.perf_counter()-start))
    except BaseException as exc:
        if out.exists():write(out/('failure_'+args.phase+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),
            dict(phase=args.phase,error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,
                 fit_calls=len(list((out/'models').glob('*/attempt_started.json')))))
        raise


if __name__=='__main__':main()
