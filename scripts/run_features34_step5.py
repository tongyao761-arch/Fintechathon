"""Freeze two development-selected candidates, then review 2024 once; no search."""
from __future__ import annotations

import argparse
import contextlib
import gc
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts import run_features34_step4 as old
from scripts import run_features34_step4_revision as revision
from scripts.run_features34 import run_split, check_frozen, feature_statistics
from scripts.run_features34_step3 import read_json, check_contract
from scripts.run_lightgbm_baseline import MODEL_PARAMS, RAW_DATA_PATH, EXPERIMENT_ROOT, PeakMemoryMonitor, split_masks, environment_versions
from src.data.baseline_panel import load_raw_baseline_panel, extract_truth_files
from src.features.baseline_v1 import FEATURE_COLUMNS as BASE_COLUMNS
from src.features.features34 import build_features34, select_features, FEATURE_DEFINITIONS
from src.validation.experiment import experiment_run, provenance, prediction_hash, sha256_file, write_json
from src.validation.splits import get_split

CONFIG = ROOT / 'configs/features34_step5.json'
OUTPUT = ROOT / 'artifacts/features34_step5'
IDS = ['S4R_lean31_minus4', 'S4R_full34_minus3']


def validate_config(cfg):
    expected = dict(stage='step5_limited_2024_review',
        candidate_source='artifacts/features34_step4_revision/frozen_candidates.json',
        development_config='configs/features34_step3.json', candidates=IDS, split='oos_2024',
        new_training_runs=2,
        baseline_directory='artifacts/experiments/baseline_v1_1/20260927T114614962794Z_2f5b6764',
        review_scope='2024 previously viewed; restricted review set, not untouched blind test',
        interpretation='all four annual score deltas positive supports next-stage consideration; any negative annual delta is cross-year degradation; no further search',
        bootstrap=dict(block_trading_days=20, repetitions=2000, seed=20261004, confidence=.95))
    if cfg != expected:
        raise ValueError('only the two frozen candidates and fixed oos_2024 review are authorized')


def candidates(cfg):
    validate_config(cfg)
    frozen = read_json(ROOT / cfg['candidate_source'])
    if not frozen['accepted'] or frozen['used_2024']:
        raise AssertionError('development candidates not accepted before 2024')
    records = frozen['candidates']
    if [r['candidate'] for r in records] != IDS:
        raise AssertionError('development selection changed')
    prior = read_json(revision.CONFIG)
    for r in records:
        version = next(v for v in revision.VERSIONS if v['candidate'] == r['candidate'])
        cols = select_features(groups=prior['contexts'][version['context']], exclude=version['exclude'])
        if r['features'] != list(cols) or r['exclude'] != version['exclude'] or r['context'] != version['context']:
            raise AssertionError('candidate formula/selection source mismatch')
    return records


def verify_baseline(cfg, meta):
    d = ROOT / cfg['baseline_directory']
    reference = read_json(ROOT / 'artifacts/baseline_v1_1/summary.json')
    sm = read_json(d / 'summary.json')
    if sm != reference or read_json(d / 'status.json')['status'] != 'success':
        raise AssertionError('frozen baseline directory/reference mismatch')
    if sm['model_params'] != MODEL_PARAMS or sm['environment'] != environment_versions():
        raise AssertionError('baseline model/environment changed')
    prior = read_json(d / 'provenance.json')
    if prior['data'] != meta['data'] or prior['dependencies'] != meta['dependencies']:
        raise AssertionError('baseline data/dependency contract changed')
    if sha256_file(d / 'provenance.json') != sm['provenance_sha256']:
        raise AssertionError('baseline provenance changed')
    result = next(x for x in sm['splits'] if x['split_name'] == cfg['split'])
    for name, expected in result['file_sha256'].items():
        if sha256_file(d / cfg['split'] / name) != expected:
            raise AssertionError(f'baseline artifact changed: {name}')
    pred = pd.read_parquet(d / cfg['split'] / 'predictions.parquet')
    if pred.duplicated(['ts_code', 'trade_date']).any() or not np.isfinite(pred.pred).all() or prediction_hash(pred.pred) != result['prediction_sha256']:
        raise AssertionError('baseline prediction validation failed')
    if lgb.Booster(model_file=str(d / cfg['split'] / 'models/lightgbm.txt')).feature_name() != list(BASE_COLUMNS):
        raise AssertionError('baseline model input changed')
    return d, dict(features=list(BASE_COLUMNS), splits=[result], **{k: sm[k] for k in ('model_params', 'environment')})


def development_evidence(cfg, meta):
    accepted = read_json(revision.OUTPUT / 'acceptance.json')
    if not accepted['accepted'] or accepted['used_2024']:
        raise AssertionError('step4 revision not accepted')
    for name, expected in accepted['evidence_sha256'].items():
        if sha256_file(revision.OUTPUT / name) != expected:
            raise AssertionError(f'previous evidence changed: {name}')
    prior = read_json(revision.OUTPUT / 'registration.json')['source']
    if prior['data'] != meta['data'] or prior['dependencies'] != meta['dependencies']:
        raise AssertionError('development data/dependencies changed')
    for name, expected in prior['source_sha256'].items():
        if sha256_file(ROOT / name) != expected or sha256_file(revision.OUTPUT / 'executed_sources' / name) != expected:
            raise AssertionError(f'development execution source changed: {name}')
    refs, _ = old.references(read_json(old.CONFIG), meta)
    entries = {('baseline10', s): refs[('baseline10', s)] for s in ('dev_2021', 'dev_2022', 'primary_2023')}
    for rec in read_json(revision.OUTPUT / 'run_index.json')['runs']:
        name, split = rec['item']['version'], rec['item']['split']
        if name not in IDS:
            continue
        d = ROOT / rec['directory']
        if sha256_file(d / 'summary.json') != rec['summary_sha256']:
            raise AssertionError('development run summary changed')
        cols = next(r['features'] for r in candidates(cfg) if r['candidate'] == name)
        sm = old.verify_artifact(d, split, tuple(cols))
        entries[(name, split)] = (d, sm)
    if len(entries) != 9:
        raise AssertionError('missing development comparisons')
    return entries


def preserve():
    snap = read_json(OUTPUT / 'preflight.json')
    for collection in ('protections', 'existing_untracked', 'previous_files'):
        for name, expected in snap[collection].items():
            p = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha256_file(p) != expected:
                raise AssertionError(f'preserved file changed: {name}')


def prepare():
    cfg = read_json(CONFIG); validate_config(cfg)
    if OUTPUT.exists():
        raise FileExistsError('step5 evidence directory already exists; use registered run/audit, never overwrite freeze')
    manifest, _, meta = check_contract(read_json(ROOT / cfg['development_config']))
    records = candidates(cfg)
    # No candidate 2024 outcomes are read before the immutable registration.
    entries = development_evidence(cfg, meta)
    baseline, _ = verify_baseline(cfg, meta)
    snapshot = old.protection_snapshot()
    # Files created by this step are recorded by source/command hashes, not as preexisting user files.
    own = {'configs/features34_step5.json', 'scripts/run_features34_step5.py',
           'scripts/audit_features34_step5.py', 'tests/test_features34_step5.py'}
    snapshot['existing_untracked'] = {p:h for p,h in snapshot['existing_untracked'].items()
        if p not in own and not p.startswith(('artifacts/features34_step5/', 'artifacts/features34_step5_checks/'))}
    previous = [p for folder in ('artifacts/features34_step1', 'artifacts/features34_step2',
        'artifacts/features34_step3', 'artifacts/features34_step4', 'artifacts/features34_step4_revision', 'docs/features34')
        for p in (ROOT / folder).rglob('*') if p.is_file() and p.name not in ('PLAN.md', 'PROGRESS.md')]
    snapshot['previous_files'] = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in previous}
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT / 'preflight.json', snapshot)
    write_json(OUTPUT / 'failures.json', dict(failures=[]))
    for r in records:
        r['feature_definitions'] = {c: FEATURE_DEFINITIONS[c] for c in r['features']}
        r['model_params'] = MODEL_PARAMS
        r['selection_reason'] = 'step4 revision accepted score-first ranking on 2023; three development years above baseline10 and original parent; monthly and statistical risks retained'
    freeze = dict(stage=cfg['stage'], frozen_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        candidate_2024_results_seen=False, candidates=records, config=cfg, config_sha256=sha256_file(CONFIG),
        candidate_source_sha256=sha256_file(ROOT / cfg['candidate_source']),
        development_acceptance_sha256=sha256_file(revision.OUTPUT / 'acceptance.json'),
        source=meta, split=get_split(cfg['split']).__dict__, baseline_directory=baseline.relative_to(ROOT).as_posix(),
        development_runs=[dict(candidate=n, split=s, directory=d.relative_to(ROOT).as_posix(), summary_sha256=sha256_file(d/'summary.json')) for (n,s),(d,_) in entries.items()],
        calculation_contract='full sorted historical X panel before masks; canonical float32 inputs, complete windows and NaN rules unchanged; rank intermediates retained; no labels/eligibility in features',
        prior_risks='exploratory step4 revision; negative months, return concentration and block intervals crossing zero; no individual necessity inferred from joint candidates')
    write_json(OUTPUT / 'FROZEN_CANDIDATES.json', freeze)
    write_json(OUTPUT / 'registration.json', dict(freeze_sha256=sha256_file(OUTPUT/'FROZEN_CANDIDATES.json'), registered_at=freeze['frozen_at'], candidate_2024_results_seen=False))
    for name in meta['source_sha256']:
        dest = OUTPUT / 'executed_sources' / name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/name, dest)
    write_json(OUTPUT / 'run_index.json', dict(runs=[]))
    write_json(OUTPUT / 'status.json', dict(status='registered', planned_new_runs=2, candidate_2024_results_seen=False))
    check_frozen(manifest); preserve()
    print('Frozen before candidate 2024 results: S4R_lean31_minus4 (27), S4R_full34_minus3 (31).', flush=True)


def registration_check(cfg, meta):
    validate_config(cfg)
    reg = read_json(OUTPUT / 'registration.json'); frozen = read_json(OUTPUT / 'FROZEN_CANDIDATES.json')
    if reg['candidate_2024_results_seen'] or reg['freeze_sha256'] != sha256_file(OUTPUT/'FROZEN_CANDIDATES.json'):
        raise AssertionError('immutable prereview freeze changed')
    if frozen['config'] != cfg or frozen['config_sha256'] != sha256_file(CONFIG) or frozen['candidate_source_sha256'] != sha256_file(ROOT/cfg['candidate_source']):
        raise AssertionError('candidate selection/config changed after freeze')
    for k in ('source_sha256', 'data', 'dependencies'):
        if frozen['source'][k] != meta[k]:
            raise AssertionError(f'registered {k} changed')
    for name, expected in meta['source_sha256'].items():
        if sha256_file(OUTPUT/'executed_sources'/name) != expected:
            raise AssertionError('source snapshot changed')
    preserve()
    return frozen


def run():
    began = time.perf_counter(); cfg = read_json(CONFIG)
    manifest, reference, meta = check_contract(read_json(ROOT/cfg['development_config']))
    frozen = registration_check(cfg, meta); baseline, bsm = verify_baseline(cfg, meta)
    split = get_split(cfg['split']); s = split.name
    write_json(OUTPUT/'status.json', dict(status='running', planned_new_runs=2))
    with PeakMemoryMonitor() as memory:
        panel = load_raw_baseline_panel(RAW_DATA_PATH)
        dates = sorted(panel.trade_date.unique()); prior = [int(d) for d in dates if d < split.valid_start]
        if prior[-2:] != [split.train_end, split.purge_date]:
            raise AssertionError('2024 purge must be last raw trading day before validation')
        full = build_features34(panel)
        stats = feature_statistics(full)
        if stats.infinite.any() or not stats.finite.gt(0).all():
            raise AssertionError('nonfinite/empty full historical feature')
        valid = split_masks(panel, split)[1]
        exact = panel.loc[valid, ['ts_code','trade_date']].reset_index(drop=True)
        pd.testing.assert_frame_equal(pd.read_parquet(baseline/s/'predictions.parquet')[['ts_code','trade_date']], exact, check_dtype=False, check_categorical=False)
        for r in frozen['candidates']:
            item = dict(id='S5_'+r['candidate']+'_2024', candidate=r['candidate'], split=s)
            cols = tuple(r['features']); started = time.perf_counter(); output = None
            index = read_json(OUTPUT/'run_index.json')['runs']
            done = next((x for x in index if x['item'] == item), None)
            if done:
                if sha256_file(ROOT/done['directory']/'summary.json') != done['summary_sha256']:
                    raise AssertionError('completed run changed')
                old.verify_artifact(ROOT/done['directory'], s, cols, exact); continue
            try:
                with experiment_run(EXPERIMENT_ROOT, item['id']) as output:
                    write_json(output/'config.json', dict(requested=cfg, item=item, features=list(cols), model_params=MODEL_PARAMS, freeze_sha256=sha256_file(OUTPUT/'FROZEN_CANDIDATES.json')))
                    write_json(output/'provenance.json', meta)
                    splitdir = output/s; fixture = splitdir/'evaluate_input'; fixture.mkdir(parents=True, exist_ok=False)
                    extract_truth_files(RAW_DATA_PATH, {fixture/'测试集_Y.csv': (split.valid_start, split.valid_end)})
                    if sha256_file(fixture/'测试集_Y.csv') != sha256_file(baseline/s/'evaluate_input/测试集_Y.csv'):
                        raise AssertionError('raw-token scoring truth differs from baseline')
                    with (output/'execution.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        result = run_split(panel, full.loc[:,cols], s, output_dir=splitdir, reference=bsm['splits'][0], columns=cols)
                    pd.testing.assert_frame_equal(pd.read_parquet(splitdir/'predictions.parquet')[['ts_code','trade_date']], exact, check_dtype=False, check_categorical=False)
                    check_frozen(manifest)
                    write_json(splitdir/'summary.json', result)
                    write_json(output/'summary.json', dict(stage=cfg['stage'], candidate=r['candidate'], item=item,
                        features=list(cols), model_params=MODEL_PARAMS, environment=environment_versions(), panel_rows=len(panel),
                        feature_index_preserved=True, prediction_keys_match_raw_panel=True, frozen_files_unchanged=True, splits=[result],
                        provenance_sha256=sha256_file(output/'provenance.json'), config_sha256=sha256_file(output/'config.json'),
                        freeze_sha256=sha256_file(OUTPUT/'FROZEN_CANDIDATES.json'),
                        resources=dict(elapsed_seconds=time.perf_counter()-started, peak_process_rss_mb=memory.peak_rss_bytes/1024**2)))
                state = read_json(output/'status.json'); state['elapsed_seconds'] = time.perf_counter()-started; write_json(output/'status.json',state)
                old.verify_artifact(output, s, cols, exact)
                index.append(dict(item=item, directory=output.relative_to(ROOT).as_posix(), summary_sha256=sha256_file(output/'summary.json')))
                write_json(OUTPUT/'run_index.json',dict(runs=index))
                print(f'Completed fixed review {r["candidate"]}: {result["metrics"]["final_score"]:.12f}', flush=True)
                gc.collect()
            except BaseException as exc:
                if output is not None:
                    state=read_json(output/'status.json'); state['elapsed_seconds']=time.perf_counter()-started; write_json(output/'status.json',state)
                failure('model', exc, directory=None if output is None else str(output), item=item, elapsed_seconds=time.perf_counter()-started)
                raise
    registration_check(cfg, provenance(ROOT, RAW_DATA_PATH)); check_frozen(manifest)
    write_json(OUTPUT/'status.json',dict(status='runs_complete_audit_pending', planned_new_runs=2, elapsed_seconds=time.perf_counter()-began, peak_process_rss_mb=memory.peak_rss_bytes/1024**2))


def entries(cfg, meta):
    result = development_evidence(cfg, meta)
    result[('baseline10','oos_2024')] = verify_baseline(cfg,meta)
    runs=read_json(OUTPUT/'run_index.json')['runs']
    if len(runs)!=2 or {r['item']['candidate'] for r in runs} != set(IDS):
        raise AssertionError('exact two-candidate review incomplete')
    for rec in runs:
        d=ROOT/rec['directory']; n=rec['item']['candidate']
        cols=tuple(next(r['features'] for r in candidates(cfg) if r['candidate']==n))
        if rec['item'] != dict(id='S5_'+n+'_2024',candidate=n,split='oos_2024') or sha256_file(d/'summary.json') != rec['summary_sha256']:
            raise AssertionError('review run index changed')
        result[(n,'oos_2024')] = (d,old.verify_artifact(d,'oos_2024',cols))
    return result


def comparison_tables(all_entries, cfg):
    annual=[]; monthly=[]; missing=[]; intervals=[]
    for (name,s),(d,sm) in all_entries.items():
        r=sm['splits'][0]; base=all_entries[('baseline10',s)][1]['splits'][0]
        for k in ('dates','train_samples','valid_prediction_rows','purge_rows','split_train_rows'):
            if r[k]!=base[k]: raise AssertionError('same-year eligibility changed')
        row=dict(candidate=name,split=s,year=int(s[-4:]),feature_count=len(sm['features']),directory=d.relative_to(ROOT).as_posix(),
            train_samples=r['train_samples'],valid_prediction_rows=r['valid_prediction_rows'],prediction_coverage=r['prediction_coverage'],
            official_max_abs_difference=r['official_comparison']['max_abs_difference'])
        for k,v in r['metrics'].items(): row.update({k:v,k+'_minus_baseline10':v-base['metrics'][k]})
        for k,v in r['score_contributions'].items(): row.update({k+'_contribution':v,k+'_contribution_minus_baseline10':v-base['score_contributions'][k]})
        diag=r['diagnostics']; top=diag['top_groups']['turnover']
        row.update(price_valid_turnover=diag['price_valid_only_turnover'],price_valid_turnover_minus_baseline10=diag['price_valid_only_turnover']-base['diagnostics']['price_valid_only_turnover'],
            top_missing_label=top['missing_label_fraction'],top_invalid_price=top['invalid_price_fraction'],top_baseline_all_missing=top['all_features_missing_fraction'],
            top_candidate_all_missing=diag.get('candidate_features_top_groups',diag['top_groups'])['turnover']['all_features_missing_fraction'])
        annual.append(row)
        frame=pd.read_csv(d/s/'monthly_metrics.csv',float_precision='round_trip')
        bm=pd.read_csv(all_entries[('baseline10',s)][0]/s/'monthly_metrics.csv',float_precision='round_trip')
        if len(frame)!=12 or frame.month.tolist()!=bm.month.tolist(): raise AssertionError('monthly alignment')
        for k in ('ic','annual_excess','turnover','score','ic_contribution','excess_contribution','stability_contribution'):
            frame[k+'_minus_baseline10']=frame[k]-bm[k]
        frame['candidate']=name; frame['split']=s; frame['year']=int(s[-4:]); monthly.append(frame)
        daily=pd.read_csv(d/s/'daily_missing_diagnostics.csv',float_precision='round_trip')
        candidate_path=d/s/'daily_candidate_missing_diagnostics.csv'
        cd=pd.read_csv(candidate_path,float_precision='round_trip') if candidate_path.exists() else daily
        pv=pd.read_csv(d/s/'daily_price_valid_turnover.csv',float_precision='round_trip'); pv['month']=pv.trade_date//100
        daily['month']=daily.trade_date//100; cd=cd.assign(month=cd.trade_date//100)
        for (month,kind),g in daily.groupby(['month','top_type']):
            m=dict(candidate=name,split=s,year=int(s[-4:]),month=int(month),top_type=kind,top_count=int(g.top_count.sum()),price_valid_turnover=float(pv[pv.month==month].turnover.mean()))
            for key in ('missing_label','invalid_price','all_features_missing'):
                m[key+'_count']=int(g[key+'_count'].sum()); m[key+'_fraction']=m[key+'_count']/m['top_count']
            cg=cd[(cd.month==month)&(cd.top_type==kind)]
            m['candidate_all_missing_count']=int(cg.all_features_missing_count.sum()); m['candidate_all_missing_fraction']=m['candidate_all_missing_count']/m['top_count']; missing.append(m)
        if name!='baseline10':
            intervals.append(dict(candidate=name,split=s,year=int(s[-4:]),**old.paired_blocks(old.daily(d,s),old.daily(all_entries[('baseline10',s)][0],s),cfg['bootstrap'])))
    return {'annual_comparison':pd.DataFrame(annual).sort_values(['year','candidate']).reset_index(drop=True),
            'monthly_comparison':pd.concat(monthly,ignore_index=True).sort_values(['month','candidate']).reset_index(drop=True),
            'monthly_missing_diagnostics':pd.DataFrame(missing).sort_values(['month','candidate','top_type']).reset_index(drop=True),
            'paired_intervals':pd.DataFrame(intervals).sort_values(['year','candidate']).reset_index(drop=True)}


def membership_table(all_entries, panel):
    rows=[]
    for s in ('dev_2021','dev_2022','primary_2023','oos_2024'):
        bd=all_entries[('baseline10',s)][0]
        def sets(d):
            return {int(r.trade_date):set(r.top_codes.split(',')) for r in pd.read_csv(d/s/'daily_top_sets.csv').itertuples()}
        before=sets(bd); p=panel[panel.trade_date.isin(before)]
        invalid={int(t):set(g.loc[g.is_price_valid.eq(0),'ts_code'].astype(str)) for t,g in p.groupby('trade_date',observed=True)}
        absent={int(t):set(g.loc[g.y_ret_1d.isna(),'ts_code'].astype(str)) for t,g in p.groupby('trade_date',observed=True)}
        for name in IDS:
            after=sets(all_entries[(name,s)][0]); changes=bad=miss=bad_days=miss_days=0
            if set(before)!=set(after): raise AssertionError('top dates changed')
            for t in before:
                diff=before[t]^after[t]; changes+=len(diff); bad+=len(diff&invalid[t]); miss+=len(diff&absent[t])
                bad_days+=bool((before[t]&invalid[t])!=(after[t]&invalid[t])); miss_days+=bool((before[t]&absent[t])!=(after[t]&absent[t]))
            rows.append(dict(candidate=name,year=int(s[-4:]),split=s,changed_members=changes,changed_invalid_price=bad,changed_missing_label=miss,
                invalid_price_fraction_of_changes=bad/changes if changes else 0,missing_label_fraction_of_changes=miss/changes if changes else 0,
                invalid_subset_changed_days=bad_days,missing_subset_changed_days=miss_days))
    return pd.DataFrame(rows).sort_values(['year','candidate']).reset_index(drop=True)


def conclusion_tables(tables):
    rows=[]
    for name in IDS:
        g=tables['annual_comparison']; g=g[g.candidate==name].sort_values('year')
        m=tables['monthly_comparison']; m=m[(m.candidate==name)&(m.year==2024)]
        r=g[g.year==2024].iloc[0]; delta=g.final_score_minus_baseline10
        positive=bool((delta>0).all())
        rows.append(dict(candidate=name,all_four_years_above_baseline10=positive,negative_score_years=';'.join(str(x) for x in g.loc[delta<0,'year']),
            delta_score_2024=float(r.final_score_minus_baseline10),worst_annual_delta=float(delta.min()),
            delta_2024_minus_delta_2023=float(r.final_score_minus_baseline10-g[g.year==2023].final_score_minus_baseline10.iloc[0]),
            positive_months_2024=int((m.score_minus_baseline10>0).sum()),negative_months_2024=int((m.score_minus_baseline10<0).sum()),
            worst_month_2024=int(m.loc[m.score_minus_baseline10.idxmin(),'month']),worst_month_delta_2024=float(m.score_minus_baseline10.min()),
            next_stage='有证据支持进入下一阶段，保留月度及统计风险' if positive else ('不支持替代baseline：存在负年度分差' if (delta<0).any() else '证据不足'),
            evidence_limit='restricted previously viewed 2024; exploratory development selection; not independent blind test or final replacement proof'))
    return pd.DataFrame(rows)


def failure(phase, exc, **extra):
    if OUTPUT.exists():
        path=OUTPUT/'failures.json'; records=read_json(path) if path.exists() else dict(failures=[])
        records['failures'].append(dict(phase=phase,error_type=type(exc).__name__,error=str(exc),at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),**extra))
        write_json(path,records)
        write_json(OUTPUT/'status.json',dict(status='failed',phase=phase,error_type=type(exc).__name__,error=str(exc)))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('phase',choices=['prepare','run']); args=parser.parse_args(argv)
    try:
        (prepare if args.phase=='prepare' else run)()
    except BaseException as exc:
        failure(args.phase,exc); raise


if __name__=='__main__':
    main()
