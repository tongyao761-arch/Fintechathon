"""Preregistered M1/M2/M3, eight fits, independent of frozen entrypoints."""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import io
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

from scripts.validate_prediction_transforms import (
    CONTROLS, SPLITS, KEYS, METRICS, gate, load_input, risk_summary,
    validated, verify_hashes, sha256_file, read, write, csv,
)
from scripts.diagnose_frozen_predictions import (
    require, aggregate_daily, aggregate_top, contributions, daily_metrics,
    top_diagnostics, metric_difference, official_evaluator, table,
)
from scripts.run_lightgbm_baseline import MODEL_PARAMS, split_masks, environment_versions
from scripts.run_features34_step3 import validate_dates
from src.data.baseline_panel import load_raw_baseline_panel
from src.features.features34 import build_features34, FEATURE_COLUMNS, FEATURE_DEFINITIONS
from src.validation.splits import TimeSplit
from src.validation.experiment import prediction_hash

CONFIG = ROOT / 'configs/model_optimization_step3.json'
FREEZE = ROOT / 'artifacts/features34_step5/FROZEN_CANDIDATES.json'
RAW = ROOT / '赛题五/赛题五数据/训练集.csv'
REPORT = ROOT / 'docs/model_optimization/STEP3_REPORT.md'
HANDOFF = ROOT / 'docs/model_optimization/STEP3_HANDOFF.json'
OWN = ['scripts/optimize_frozen27.py', 'configs/model_optimization_step3.json',
       'tests/test_model_optimization_step3.py']
EXPERIMENTS = ['M1', 'M2', 'M3']
CHANGES = {'M1': {'calendar_training_years': 3},
           'M2': {'n_estimators': 200, 'learning_rate': .025},
           'M3': {'min_child_samples': 500}}
FLAGS = {'annual_delta_vs27_below': -.005, 'monthly_delta_below': -.005,
         'severe_monthly_delta_below': -.02, 'small_mean_absolute_delta_below': .003}


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_config(c):
    require(c['stage'] == 'frozen27_fixed_small_budget_model_comparison', 'wrong stage')
    require(c['freeze'] == str(FREEZE.relative_to(ROOT)).replace('\\', '/') and
            c['candidate'] == CONTROLS[0], 'wrong frozen candidate')
    require(c['splits'] == SPLITS and c['controls'] == CONTROLS, 'split/control change')
    require(c['changes'] == CHANGES and c['risk_flags'] == FLAGS, 'matrix/risk change')
    require(c['budget'] == dict(logical_comparisons=9, new_full_training=8,
                               reuse=['M1_dev_2021']), 'budget change')
    require(c['scope'] == dict(evaluate_2024=False, official_test_fit_or_selection=False,
        early_stopping=False, search=False, replace_original=False, final_submission=False), 'scope change')


def parameters(original, experiment):
    require(original == MODEL_PARAMS, 'frozen original parameters inconsistent; stop')
    require(experiment in EXPERIMENTS, 'unknown independent experiment')
    p = original.copy()
    if experiment != 'M1':
        p.update(CHANGES[experiment])
    return p


def matrix(dates, frozen_params):
    cfg = read(ROOT / 'configs/features34_step3.json')
    splits = validate_dates(dates, cfg)
    dates = sorted(set(int(d) for d in dates))
    items = []
    for e in EXPERIMENTS:
        for s in SPLITS:
            spec = {k: v for k, v in vars(splits[s]).items() if k != 'name'}
            year = int(s[-4:])
            if e == 'M1':
                prior = [d for d in dates if year - 3 <= d // 10000 < year]
                require({d // 10000 for d in prior} == set(range(year - 3, year)), 'incomplete M1 years')
                spec['train_start'] = min(prior)
            items.append(dict(id=e+'_'+s, experiment=e, split=s, year=year,
                split_spec=spec, parameters=parameters(frozen_params, e),
                reuse=(e == 'M1' and year == 2021)))
    return items


def array_hash(values):
    a = np.asarray(values)
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True,
        categorize=True).to_numpy(dtype='<u8').tobytes()).hexdigest()


def context():
    panel = load_raw_baseline_panel(RAW)
    frozen = read(FREEZE)
    columns = tuple(c for c in FEATURE_COLUMNS if any(c in r['features'] for r in frozen['candidates']))
    features = build_features34(panel, columns)
    require(features.index.equals(panel.index) and len(features) == len(panel), 'feature row change')
    require(all(t == np.dtype('float32') for t in features.dtypes) and
            not np.isinf(features.to_numpy()).any(), 'feature precision/finite contract')
    return panel, features


def sample_evidence(panel, features, item, cols):
    split = TimeSplit(name=item['split'], **item['split_spec'])
    train, valid = split_masks(panel, split)
    y = panel.loc[train, 'y_ret_1d'].astype('float32')
    require(np.isfinite(y).all(), 'nonfinite labels after float32 conversion')
    # Same-stock next observed date verifies the boundary label cannot enter validation.
    next_dates = panel.groupby('ts_code', observed=True, sort=False).trade_date.shift(-1)
    require(next_dates.loc[train].notna().all() and
            next_dates.loc[train].lt(split.valid_start).all(), 'training label crosses validation boundary')
    require(not train[panel.trade_date.eq(split.purge_date)].any(), 'purge row trained')
    actual = dict(training_samples=int(train.sum()), prediction_rows=int(valid.sum()),
        training_min_date=int(panel.loc[train, 'trade_date'].min()),
        training_max_date=int(panel.loc[train, 'trade_date'].max()),
        training_year_counts={str(k): int(v) for k,v in (panel.loc[train, 'trade_date']//10000).value_counts().sort_index().items()},
        train_mask_sha256=array_hash(train.to_numpy()), valid_mask_sha256=array_hash(valid.to_numpy()),
        training_keys_sha256=frame_hash(panel.loc[train, KEYS]),
        training_features_sha256=frame_hash(features.loc[train, cols]),
        training_labels_sha256=array_hash(y.to_numpy()),
        validation_features_sha256=frame_hash(features.loc[valid, cols]),
        purge_rows=int(panel.trade_date.eq(split.purge_date).sum()),
        purge_trained_rows=0, next_label_date_max=int(next_dates.loc[train].max()))
    return train, valid, y, actual


def check_model(path, columns, params):
    model = lgb.Booster(model_file=str(path))
    require(model.feature_name() == list(columns) and model.num_trees() == params['n_estimators'], 'saved features/tree budget mismatch')
    # Saved LightGBM aliases are checked for every frozen constructor parameter.
    aliases = {'n_estimators': 'num_iterations', 'random_state': 'seed',
        'n_jobs': 'num_threads', 'min_child_samples': 'min_data_in_leaf',
        'subsample': 'bagging_fraction', 'subsample_freq': 'bagging_freq',
        'colsample_bytree': 'feature_fraction', 'reg_alpha': 'lambda_l1',
        'reg_lambda': 'lambda_l2', 'boosting_type': 'boosting'}
    for k,v in params.items():
        got = model.params[aliases.get(k, k)]
        if k == 'n_jobs':
            require(int(got) > 0, 'invalid actual all-core thread allocation')
        elif isinstance(v, bool):
            require(got in [v, str(v).lower(), int(v)], 'saved boolean mismatch '+k)
        elif isinstance(v, (int, float)):
            require(float(got) == v, 'saved parameter mismatch '+k)
        else:
            require(got == v, 'saved parameter mismatch '+k)
    return model


def source_gate():
    h, inputs, n = gate()
    h2 = read(ROOT / 'docs/model_optimization/STEP2_HANDOFF.json')
    require(h2['accepted'] and h2['stage'] == 'step2_complete_stop', 'unresolved STEP2')
    for p,k in [('report','report_sha256'), ('registration_path','registration_sha256'),
                ('acceptance_path','acceptance_sha256'), ('delivery_acceptance_path','delivery_acceptance_sha256')]:
        require(sha256_file(h2[p]) == h2[k], 'STEP2 seal mismatch '+p)
    delivery = read(h2['delivery_acceptance_path'])
    require(delivery['accepted'] and read(h2['acceptance_path'])['accepted'], 'STEP2 not accepted')
    verify_hashes(delivery['output_sha256'], Path(h2['artifact_directory']))
    frozen = read(FREEZE)
    require(sha256_file(RAW) == frozen['source']['data']['sha256'], 'raw source mismatch')
    packages=dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions()))
    require(packages==frozen['source']['dependencies'],'frozen training environment mismatch')
    for r in frozen['candidates']:
        require(r['feature_count'] == len(r['features']) == (27 if r['candidate'] == CONTROLS[0] else 31), 'frozen count mismatch')
        require(r['feature_definitions'] == {c: FEATURE_DEFINITIONS[c] for c in r['features']}, 'formula mismatch')
        parameters(r['model_params'], 'M1')
    for r in read(ROOT / 'artifacts/features34_step5/run_index.json')['runs']:
        require(sha256_file(ROOT/r['directory']/'summary.json') == r['summary_sha256'], 'step5 index mismatch')
    # Actual executable dependencies must match historical model provenance.
    for rec in inputs:
        prior = read(rec['paths']['provenance'])
        require(prior['data'] == frozen['source']['data'], 'historical data mismatch')
        for name in ['src/data/baseline_panel.py', 'src/features/features34.py', 'src/features/baseline_v1.py',
            'src/metrics/official.py', 'scripts/run_lightgbm_baseline.py', 'configs/features34_step3.json',
            'configs/splits.yaml', '赛题五/evaluate.py']:
            if name not in prior['source_sha256']:
                require((rec['candidate'],rec['split'],name)==('baseline10','primary_2023','configs/features34_step3.json'),
                        'unexplained missing historical source '+name)
                expected=frozen['source']['source_sha256'][name]
            else:
                expected=prior['source_sha256'][name]
            require(sha256_file(ROOT/name) == expected, 'historical source mismatch '+name)
    return inputs, n + len(delivery['output_sha256'])


def prepare(output):
    require(not REPORT.exists() and not HANDOFF.exists(), 'existing STEP3 docs; refuse overwrite')
    c = read(CONFIG); validate_config(c)
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() == 'ivor-work', 'branch')
    inputs, checked = source_gate()
    panel = load_raw_baseline_panel(RAW)
    frozen = read(FREEZE)
    items = matrix(panel.trade_date.unique(), frozen['candidates'][0]['model_params'])
    for rec in inputs:
        load_input(rec, panel)
        check_model(rec['paths']['model'], rec['features'], MODEL_PARAMS)
    paths = subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=ROOT).decode('utf-8').split('\0')
    paths += ['.git/config', '.git/hooks/pre-push']
    preserved = {p: sha256_file(ROOT/p) for p in paths if p and p not in OWN and
                 (ROOT/p).is_file() and not (ROOT/p).resolve().is_relative_to(output)}
    write(output/'preserved.json', preserved)
    dependencies = [*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('tests/*.py'),
        *ROOT.glob('configs/*'), *ROOT.glob('requirements*'), FREEZE, ROOT/'artifacts/features34_step5/run_index.json',
        ROOT/'赛题五/evaluate.py', ROOT/'赛题五/完整赛题说明_本地.md', ROOT/'AGENTS.md', ROOT/'AGENTS.override.md',
        ROOT/'docs/features34/RESULTS.md', ROOT/'docs/features34/FROZEN_MODEL_VALIDATION_REPORT.md',
        *ROOT.glob('docs/model_optimization/STEP[12]*')]
    sources = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in dependencies if p.is_file()}
    for p in sources:
        dest = output/'executed_sources'/p; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/p, dest)
    reg = dict(registered_at=now(), config=c, matrix=items, inputs=inputs,
        columns=frozen['candidates'][0]['features'], source_sha256=sources, raw_sha256=sha256_file(RAW),
        preserved_sha256=sha256_file(output/'preserved.json'), source_hashes_verified=checked,
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        environment=environment_versions(), interpreter=sys.executable, qualification=c['contract']['training'])
    reg['historical_metadata_exception']='baseline10/primary_2023 predates configs/features34_step3.json; current config checked against authoritative frozen source hash; saved actual split, sample counts, keys, labels and model reload checked separately.'
    write(output/'registration.json',reg)
    write(output/'registration_sha256.json',dict(sha256=sha256_file(output/'registration.json')))
    write(output/'preflight.json',dict(accepted=True, unresolved_issues=[], verified_hashes=checked,
        original_models_saved_parameters_verified=9, raw_panel_start=int(panel.trade_date.min())))
    print('Registered '+str(output),flush=True)


def stable(output):
    reg = read(output/'registration.json')
    require(sha256_file(output/'registration.json') == read(output/'registration_sha256.json')['sha256'], 'registration changed')
    validate_config(reg['config'])
    verify_hashes(reg['source_sha256']); verify_hashes(reg['source_sha256'],output/'executed_sources')
    require(sha256_file(RAW) == reg['raw_sha256'] and sha256_file(output/'preserved.json') == reg['preserved_sha256'], 'data/preservation changed')
    verify_hashes(read(output/'preserved.json'))
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() == 'ivor-work','branch changed')
    return reg


def verify_original(rec, panel, features):
    pq, pred, truth, x, raw, sm = load_input(rec,panel)
    mask = panel.trade_date.between(rec['split_spec']['valid_start'],rec['split_spec']['valid_end'])
    model = check_model(rec['paths']['model'],rec['features'],MODEL_PARAMS)
    np.testing.assert_array_equal(model.predict(features.loc[mask,rec['features']]),pq.pred)
    result = official_evaluator()(rec['paths']['score_prediction'],str(Path(rec['paths']['score_truth']).parent))
    diff = metric_difference(result,sm['metrics'])
    require(max(abs(v) for v in diff.values()) <= 1e-12,'original official score changed')
    return dict(candidate=rec['candidate'],split=rec['split'],actual_X_reload_equal=True,
                official_max_difference=max(abs(v) for v in diff.values()))


def run(output):
    reg = stable(output)
    panel,features = context()
    require(matrix(panel.trade_date.unique(),MODEL_PARAMS) == reg['matrix'],'matrix/date changed')
    reuse = [verify_original(r,panel,features) for r in reg['inputs']]
    write(output/'reuse_verification.json',reuse)
    write(output/'history_feature_hashes.json',{c:frame_hash(features[[c]]) for c in features})
    # Future truncation verifies earlier frozen features independently on actual X.
    prefix = panel.trade_date.lt(20240000)
    reconstructed = build_features34(panel.loc[prefix],tuple(features.columns))
    pd.testing.assert_frame_equal(reconstructed,features.loc[prefix],check_exact=True)
    del reconstructed; gc.collect()
    for item in reg['matrix']:
        if item['reuse']:
            continue
        ident = item['id']; dest = output/ident
        require(not dest.exists(),'completed/partial fit exists; refuse implicit retraining '+ident)
        train,valid,y,evidence = sample_evidence(panel,features,item,reg['columns'])
        dest.mkdir()
        write(dest/'before_fit.json',dict(item=item,features=reg['columns'],**evidence))
        print('Training '+ident+' '+str(evidence['training_samples'])+' samples',flush=True)
        start=time.perf_counter()
        write(dest/'attempt_started.json',dict(attempt=1,started_at=now(),parameters=item['parameters']))
        try:
            model=lgb.LGBMRegressor(**item['parameters'])
            model.fit(features.loc[train,reg['columns']],y,feature_name=reg['columns'])
            require(all(model.get_params()[k] == v for k,v in item['parameters'].items()),'actual constructor changed')
            values=model.predict(features.loc[valid,reg['columns']])
            model.booster_.save_model(str(dest/'lightgbm.txt'))
            reloaded=check_model(dest/'lightgbm.txt',reg['columns'],item['parameters'])
            np.testing.assert_array_equal(reloaded.predict(features.loc[valid,reg['columns']]),values)
            pred=panel.loc[valid,KEYS].reset_index(drop=True); pred['ts_code']=pred.ts_code.astype(str);pred['pred']=values
            validated(pred);pred.to_parquet(dest/'predictions.parquet',index=False)
            csv(pred,dest/'evaluation_predictions.csv')
            write(dest/'summary.json',dict(item=item,columns=reg['columns'],actual_parameters=model.get_params(),
                actual_booster_parameters=model.booster_.params,model_reload_equal=True,finite_complete_predictions=True,
                prediction_sha256=prediction_hash(values),**evidence,elapsed_seconds=time.perf_counter()-start))
            write(dest/'attempt_result.json',dict(attempt=1,status='success',exit_code=0,elapsed_seconds=time.perf_counter()-start))
            write(dest/'files.json',{p.name:sha256_file(p) for p in dest.iterdir() if p.is_file()})
            print('Completed '+ident,flush=True)
            del model,reloaded,pred,y;gc.collect()
        except BaseException as exc:
            write(dest/'attempt_failure.json',dict(attempt=1,status='failed',error=str(exc),traceback=traceback.format_exc(),
                elapsed_seconds=time.perf_counter()-start,parameters=item['parameters']))
            raise
    index=[dict(id=i['id'],directory=str(output/i['id']),files_sha256=sha256_file(output/i['id']/'files.json'))
           for i in reg['matrix'] if not i['reuse']]
    write(output/'run_index.json',dict(runs=index,new_full_training=8,reused=['M1_dev_2021']))
    summarize(output,panel)


def score_one(output,reg,name,split,panel):
    rec=next(r for r in reg['inputs'] if r['candidate'] == (name if name in CONTROLS else CONTROLS[0]) and r['split']==split)
    _,old,truth,x,raw,sm=load_input(rec,panel)
    path=Path(rec['paths']['score_prediction']) if name in CONTROLS or (name=='M1' and split=='dev_2021') else output/(name+'_'+split)/'evaluation_predictions.csv'
    pred=pd.read_csv(path);validated(pred)
    pd.testing.assert_frame_equal(pred[KEYS],old[KEYS],check_dtype=False,check_exact=True)
    from src.metrics.official import score_official
    result=score_official(pred,truth,x,return_details=True);details=result.pop('details')
    official=official_evaluator()(str(path),str(Path(rec['paths']['score_truth']).parent))
    diff=metric_difference(result,official);require(max(abs(v) for v in diff.values())<=1e-12,'official parity')
    if name in CONTROLS or (name=='M1' and split=='dev_2021'):
        require(max(abs(v) for v in metric_difference(result,sm['metrics']).values())<=1e-12,'reuse score changed')
    joined=pred.merge(truth,on=KEYS,validate='one_to_one').merge(raw[KEYS+['flag_limit_up','is_price_valid']],on=KEYS,validate='one_to_one')
    q,mem,trans,pv,_=top_diagnostics(joined,details)
    day=daily_metrics(details).merge(pv,on='trade_date',how='left',validate='one_to_one')
    for frame in [q,mem,trans,day]:
        frame['candidate']=name;frame['year']=int(split[-4:]);frame['month']=frame.trade_date//100
    annual=dict(candidate=name,year=int(split[-4:]),split=split,**result,price_valid_turnover=float(pv.price_valid_turnover.mean()))
    return annual,day,q,mem,trans,dict(candidate=name,split=split,official_difference=diff,score_path=str(path))


def compare(frame,keys):
    rows=[]
    for name in EXPERIMENTS:
        a=frame[frame.candidate.eq(name)].set_index(keys).sort_index()
        for control in CONTROLS:
            b=frame[frame.candidate.eq(control)].set_index(keys).sort_index()
            require(a.index.equals(b.index),'comparison periods')
            d=a[METRICS]-b[METRICS];d.columns=[c+'_delta' for c in METRICS]
            rows.append(d.reset_index().assign(after=name,before=control))
    return pd.concat(rows,ignore_index=True)


def summarize(output,panel):
    reg=read(output/'registration.json');results=[]
    for s in SPLITS:
        for n in [*CONTROLS,*EXPERIMENTS]:
            results.append(score_one(output,reg,n,s,panel))
            print('Scored '+n+' '+s,flush=True)
    a=contributions(pd.DataFrame([r[0] for r in results]));d=pd.concat([r[1] for r in results],ignore_index=True)
    m=pd.concat([aggregate_daily(g,'month').assign(candidate=n,year=y) for (n,y),g in d.groupby(['candidate','year'],sort=False)],ignore_index=True)
    m=m.merge(d.groupby(['candidate','month']).price_valid_turnover.mean().reset_index(),on=['candidate','month'],validate='one_to_one')
    q=pd.concat([r[2] for r in results],ignore_index=True)
    ad,md=compare(a,['year']),compare(m,['year','month'])
    ad['annual_risk_vs27']=ad.before.eq(CONTROLS[0])&ad.final_score_delta.lt(-.005)
    md['risk_below_minus005']=md.final_score_delta.lt(-.005);md['severe_below_minus02']=md.final_score_delta.lt(-.02)
    tables=dict(annual_metrics=a,monthly_metrics=m,daily_metrics=d,annual_comparison=ad,monthly_comparison=md,
        risk_summary=risk_summary(ad,md),negative_months=md[md.final_score_delta.lt(0)],daily_top_quality=q,
        annual_top_quality=aggregate_top(q,'year'),monthly_top_quality=aggregate_top(q,'month'),
        daily_top_members=pd.concat([r[3] for r in results],ignore_index=True),
        daily_top_transitions=pd.concat([r[4] for r in results],ignore_index=True))
    for name,frame in tables.items():csv(frame,output/(name+'.csv'))
    write(output/'official_parity.json',[r[5] for r in results])


def equal_disk(frame,path):
    actual=pd.read_csv(path,float_precision='round_trip',keep_default_na=False)
    expected=pd.read_csv(io.StringIO(frame.to_csv(index=False,float_format='%.17g')),float_precision='round_trip',keep_default_na=False)
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=True)


def audit(output):
    reg=stable(output);panel,features=context()
    require(matrix(panel.trade_date.unique(),MODEL_PARAMS)==reg['matrix'],'matrix mismatch')
    require({c:frame_hash(features[[c]]) for c in features}==read(output/'history_feature_hashes.json'),'full historical features changed')
    originals=[verify_original(r,panel,features) for r in reg['inputs']]
    idx=read(output/'run_index.json')['runs'];require(len(idx)==8,'eight fit budget')
    verified=[]
    for item in reg['matrix']:
        train,valid,y,e=sample_evidence(panel,features,item,reg['columns'])
        if item['reuse']:
            rec=next(r for r in reg['inputs'] if r['candidate']==CONTROLS[0] and r['split']==item['split'])
            sm=read(rec['paths']['summary'])['splits'][0]
            require(e['training_samples']==sm['train_samples'] and item['split_spec']==rec['split_spec'],'M1 reuse mismatch')
            verified.append(dict(id=item['id'],reused=True,training_samples=e['training_samples']))
            continue
        dest=output/item['id'];ir=next(r for r in idx if r['id']==item['id'])
        require(sha256_file(dest/'files.json')==ir['files_sha256'],'model index changed')
        verify_hashes(read(dest/'files.json'),dest)
        before=read(dest/'before_fit.json');sm=read(dest/'summary.json')
        for k,v in e.items():require(before[k]==v and sm[k]==v,'actual sample/input mismatch '+k)
        require(before['item']==item and sm['item']==item and sm['columns']==reg['columns'],'actual configuration mismatch')
        require(all(sm['actual_parameters'][k]==v for k,v in item['parameters'].items()),'constructor parameter mismatch')
        model=check_model(dest/'lightgbm.txt',reg['columns'],item['parameters'])
        pq=pd.read_parquet(dest/'predictions.parquet');validated(pq)
        keys=panel.loc[valid,KEYS].reset_index(drop=True);keys.ts_code=keys.ts_code.astype(str)
        pd.testing.assert_frame_equal(pq[KEYS],keys,check_exact=True)
        np.testing.assert_array_equal(model.predict(features.loc[valid,reg['columns']]),pq.pred)
        parsed=pd.read_csv(dest/'evaluation_predictions.csv',float_precision='round_trip')
        np.testing.assert_array_equal(parsed.pred,pq.pred)
        require(prediction_hash(pq.pred)==sm['prediction_sha256'],'prediction digest')
        verified.append(dict(id=item['id'],reused=False,training_samples=e['training_samples'],
            model_reload_equal=True,actual_parameters_verified=True,purge_and_label_boundary_verified=True,all_keys_finite=True))
        print('Audited model '+item['id'],flush=True)
    a=pd.read_csv(output/'annual_metrics.csv',float_precision='round_trip')
    require(len(a)==18,'18 annual rows')
    d=pd.read_csv(output/'daily_metrics.csv',float_precision='round_trip')
    m=pd.read_csv(output/'monthly_metrics.csv',float_precision='round_trip');require(len(m)==216,'216 monthly rows')
    for s in SPLITS:
        for n in [*CONTROLS,*EXPERIMENTS]:
            annual,day,q,mem,trans,parity=score_one(output,reg,n,s,panel)
            g=a[a.candidate.eq(n)&a.year.eq(int(s[-4:]))].iloc[0]
            for k,v in annual.items():
                if isinstance(v,float):require(abs(g[k]-v)<=1e-12,'annual report mismatch '+k)
            for f,name in [(day,'daily_metrics'),(q,'daily_top_quality'),(mem,'daily_top_members'),(trans,'daily_top_transitions')]:
                disk=pd.read_csv(output/(name+'.csv'),float_precision='round_trip',keep_default_na=False)
                old=disk[disk.candidate.eq(n)&disk.year.eq(int(s[-4:]))].reset_index(drop=True)
                normalized=pd.read_csv(io.StringIO(f.to_csv(index=False,float_format='%.17g')),float_precision='round_trip',keep_default_na=False)
                pd.testing.assert_frame_equal(old,normalized,check_dtype=False,check_exact=True)
            ag=aggregate_daily(day,'month')
            old=m[m.candidate.eq(n)&m.year.eq(int(s[-4:]))].reset_index(drop=True)
            pd.testing.assert_frame_equal(old[ag.columns],ag,check_dtype=False,check_exact=True)
            print('Audited score and Top '+n+' '+s,flush=True)
    ad,md=compare(a,['year']),compare(m,['year','month'])
    ad['annual_risk_vs27']=ad.before.eq(CONTROLS[0])&ad.final_score_delta.lt(-.005)
    md['risk_below_minus005']=md.final_score_delta.lt(-.005);md['severe_below_minus02']=md.final_score_delta.lt(-.02)
    equal_disk(ad,output/'annual_comparison.csv');equal_disk(md,output/'monthly_comparison.csv')
    equal_disk(risk_summary(ad,md),output/'risk_summary.csv')
    equal_disk(md[md.final_score_delta.lt(0)],output/'negative_months.csv')
    quality=pd.read_csv(output/'daily_top_quality.csv',float_precision='round_trip')
    equal_disk(aggregate_top(quality,'year'),output/'annual_top_quality.csv')
    equal_disk(aggregate_top(quality,'month'),output/'monthly_top_quality.csv')
    require(len(list(output.glob('M*/attempt_started.json')))==8 and not list(output.glob('M*/attempt_failure.json')),'fit attempt budget')
    require(read(output/'tests_result.json')['exit_code']==0 and read(output/'dependency_result.json')['exit_code']==0,'checks failed')
    stable(output)
    write(output/'acceptance.json',dict(accepted=True,new_full_training=8,logical_comparisons=9,
        verified_runs=verified,originals=originals,history_features_exact=True,official_score_and_full_monthly_top_rebuilt=True,
        source_data_previous_files_push_protections_unchanged=True,official_max_tolerance=1e-12,
        evaluate_2024=False,official_test_fit_or_selection=False,frozen_model_replaced=False))


def process(output,phase,command):
    start=time.perf_counter();at=now()
    with (output/(phase+'.log')).open('x',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    write(output/(phase+'_result.json'),dict(command=command,cwd=str(ROOT),started_at=at,
        exit_code=result.returncode,elapsed_seconds=time.perf_counter()-start,
        log=str(output/(phase+'.log')),log_sha256=sha256_file(output/(phase+'.log'))))
    require(result.returncode==0,phase+' failed; log retained')


def execute(output):
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();at=now()
    with (output/'prepare.log').open('x',encoding='utf-8') as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            prepare(output)
    write(output/'prepare_result.json',dict(command=[sys.executable,*sys.argv],cwd=str(ROOT),
        phase='prepare within execute',started_at=at,exit_code=0,elapsed_seconds=time.perf_counter()-start,
        log=str(output/'prepare.log'),log_sha256=sha256_file(output/'prepare.log')))
    print('Registered '+str(output),flush=True)
    process(output,'tests',[sys.executable,'-X','utf8','-B','-m','pytest','tests/test_model_optimization_step3.py',
        'tests/test_features34.py','tests/test_splits.py','tests/test_official_score.py',
        'tests/test_frozen_prediction_diagnostics.py','-q','-p','no:cacheprovider'])
    process(output,'dependency',[sys.executable,'-X','utf8','-B','-m','pip','check'])
    for phase in ['run','audit']:
        process(output,phase,[sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),phase,'--output',str(output)])


def finalize(output):
    require(read(output/'acceptance.json')['accepted'],'unaccepted');reg=stable(output)
    narrative=read(output/'interpretation.json')
    require(len(narrative['recommended_candidates'])<=1,'too many followup candidates')
    a=pd.read_csv(output/'annual_metrics.csv',float_precision='round_trip')
    ad=pd.read_csv(output/'annual_comparison.csv',float_precision='round_trip')
    md=pd.read_csv(output/'monthly_comparison.csv',float_precision='round_trip')
    risk=pd.read_csv(output/'risk_summary.csv',float_precision='round_trip',keep_default_na=False)
    q=pd.read_csv(output/'annual_top_quality.csv',float_precision='round_trip')
    runstats=[read(output/i['id']/'summary.json') for i in reg['matrix'] if not i['reuse']]
    lines=['# 冻结27列方案的小预算模型优化比较','',narrative['conclusion'],'',
        *[narrative[e] for e in EXPERIMENTS], '', '## 固定矩阵与来源','',
        f'唯一目录：{output}。事前登记SHA-256：{sha256_file(output/"registration.json")}。来源门核对{reg["source_hashes_verified"]}项哈希，9个原年度对照核验模型重载、完整键、标签、参数和官方评分后复用。权威名单仅为features34_step5/FROZEN_CANDIDATES.json内S4R候选。', '',
        'M1只改变训练起点，验证年前3个完整日历年；M1/2021与原方案相同复用。M2只将100/0.05改为200/0.025；树数与学习率乘积相同不代表模型等价。M3只将min_child_samples从100改为500。各项从原27列方案独立出发，不叠加。9项逻辑比较，严格8次新全量训练。', '',
        table(pd.DataFrame([dict(id=i['id'],**i['split_spec'],reuse=i['reuse'],n_estimators=i['parameters']['n_estimators'],learning_rate=i['parameters']['learning_rate'],min_child_samples=i['parameters']['min_child_samples']) for i in reg['matrix']]),
              ['id','train_start','train_end','purge_date','valid_start','valid_end','reuse','n_estimators','learning_rate','min_child_samples']), '',
        '完整2018年起的有序历史X先按冻结公式计算float32特征，再筛训练窗口；保留历史窗口与截面排名中间列。名单、顺序、标签、资格、NaN处理和其他参数不变。使用原始模型输出，未应用第二步融合或平滑。', '',
        '## 年度指标及三个对照','',table(a,['candidate','year','final_score','ic_mean','annual_excess','mean_turnover','price_valid_turnover']), '',
        table(ad,['after','before','year','final_score_delta','ic_mean_delta','annual_excess_delta','mean_turnover_delta','price_valid_turnover_delta','ic_contribution_delta','excess_contribution_delta','stability_contribution_delta']), '',
        '## 跨年及完整月度分差','',
        '沿用STEP2描述性风险标记：年度相对原27分差<−0.005；所有对照月度分差<−0.005、<−0.02；三年平均分差绝对值<0.003。门槛没有随结果改变，不是显著性或自动采用规则。方向冲突为年度同时有正负分差，任何负年另列。', '',
        table(risk,list(risk.columns)), '',
        '月度从全年逐日指标聚合，月初换手保留上月前一有效日；年度首日不计换手。全年各指标按自身有效日均值评分，不平均月分替代全年。月度年化超额为日均×252，不是月累计资金收益。', '',
        table(md.pivot(index=['after','month'],columns='before',values='final_score_delta').reset_index(),['after','month',*CONTROLS]), '',
        '全部退化月份见negative_months.csv；明显退化月份如下：','',
        table(md[md.final_score_delta.lt(-.005)],['after','before','month','final_score_delta','ic_contribution_delta','excess_contribution_delta','stability_contribution_delta','severe_below_minus02']), '',
        '## 缺标签与无效价格Top','',
        '收益Top排除涨停及缺标签；官方换手Top只排除涨停，缺标签及无效价格行可入选。价格有效换手只按当日OHLC有限、正数、高低价一致性和非涨停过滤，不按未来标签，只作诊断。质量占比按入选股票—日期次数加权。完整月度、逐日及成员转换留档。', '',
        table(q,['candidate','year','top_type','top_count','missing_label_fraction','invalid_price_fraction']), '',narrative['risk'],'',
        '## 训练与验收','',
        table(pd.DataFrame([dict(id=r['item']['id'],training_samples=r['training_samples'],min_date=r['training_min_date'],max_date=r['training_max_date'],prediction_rows=r['prediction_rows'],elapsed_seconds=r['elapsed_seconds']) for r in runstats]),['id','training_samples','min_date','max_date','prediction_rows','elapsed_seconds']), '',
        '独立进程从原始数据重建完整历史特征，核对每列哈希、实际训练mask/键/float32输入/标签哈希、训练年度数量、purge及同股票标签下一日期；逐项解析保存模型全部冻结参数与别名、树数、27列顺序。全部保存模型重载预测逐值一致、键覆盖完整、无重键、预测有限，17位CSV的round_trip预测逐值等于Parquet。历史2024以后截断重算开发特征逐值一致，该检查不拟合或评分2024。', '',
        '18项年度评分（9原对照、9逻辑比较）与未修改赛题五/evaluate.py一致≤1e-12。独立重建全年逐日/月度指标、三类Top成员/转换、质量表与分差，8次fit及9项矩阵预算核对通过。测试与pip check通过，完整命令、日志、退出码、耗时、失败记录、源码快照和模型/预测/数据哈希留档。原始数据、旧产物、既有未提交内容及.git/config/pre-push保护核验不变。', '',
        '## 推荐、限制与停止','',narrative['recommendation'],'',
        '2021—2023已大量开发和筛选，本轮任何改善均不证明未来有效。仅描述历史固定实验差异，不做显著性声明；官方换手包含缺标签和无效价格Top，不能直接解释为可交易稳定性。未验证成本、容量和未来收益。最多一个后续候选需要新的明确授权和固定验证设计。', '',
        '未替换原模型，未组合参数或第二步处理；未进行2024复核、测试集新模型训练或选择、网格/自动调参、早停、种子搜索及最终submission。完成验收后仅本地提交本任务代码、配置、测试、小型报告，停止等待下一项明确授权。', '',
        f'解释器：{sys.executable}。完整执行命令见execute_result.json及各*_result.json；登记、模型索引、验收和交付哈希见registration.json/run_index.json/acceptance.json/delivery_acceptance.json。','']
    with REPORT.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    write(output/'delivery_acceptance.json',dict(accepted=True,report_sha256=sha256_file(REPORT),
        acceptance_sha256=sha256_file(output/'acceptance.json'),stop_after_step3=True,
        output_sha256={p.relative_to(output).as_posix():sha256_file(p) for p in output.rglob('*') if p.is_file()}))
    write(HANDOFF,dict(stage='step3_complete_stop',accepted=True,artifact_directory=str(output),report=str(REPORT),report_sha256=sha256_file(REPORT),
        registration_path=str(output/'registration.json'),registration_sha256=sha256_file(output/'registration.json'),
        acceptance_path=str(output/'acceptance.json'),acceptance_sha256=sha256_file(output/'acceptance.json'),
        delivery_acceptance_path=str(output/'delivery_acceptance.json'),delivery_acceptance_sha256=sha256_file(output/'delivery_acceptance.json'),
        matrix=reg['matrix'],rules=reg['config'],sources=reg['inputs'],source_sha256=reg['source_sha256'],raw_sha256=reg['raw_sha256'],
        actual_runs=runstats,run_index=read(output/'run_index.json'),annual_metrics=a.to_dict('records'),
        annual_comparison=ad.to_dict('records'),monthly_comparison=md.to_dict('records'),comparison_summary=risk.to_dict('records'),
        recommendation=narrative,commands={p.stem:read(p) for p in output.glob('*_result.json')},
        failures=[read(p) for p in output.rglob('*failure*.json')],acceptance=read(output/'acceptance.json'),
        new_full_training=8,logical_comparisons=9,evaluated_2024=0,official_test_fit_or_selection=False,
        replaced_original=False,final_submission=False,next_action='Stop; wait for explicit next authorization.'))
    print('STEP3 accepted and sealed; stop.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['execute','run','audit','finalize'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(ROOT/'artifacts/model_optimization/step3'),'output outside authorized root')
    start=time.perf_counter();at=now()
    try:
        globals()[args.phase](out)
        if args.phase=='execute':
            write(out/'execute_result.json',dict(command=[sys.executable,*sys.argv],cwd=str(ROOT),started_at=at,
                exit_code=0,elapsed_seconds=time.perf_counter()-start))
    except BaseException as exc:
        if out.exists():
            write(out/('failure_'+args.phase+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),
                dict(phase=args.phase,accepted=False,error=str(exc),traceback=traceback.format_exc()))
            if args.phase=='execute':
                write(out/'execute_result.json',dict(command=[sys.executable,*sys.argv],cwd=str(ROOT),started_at=at,
                    exit_code=1,elapsed_seconds=time.perf_counter()-start))
        raise


if __name__=='__main__':main()
