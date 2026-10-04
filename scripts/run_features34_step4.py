"""Bounded conditional single deletions; immutable phases, three development years only."""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
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
from scripts.run_features34 import run_split, check_frozen, feature_statistics
from scripts.run_features34_step3 import read_json, check_contract, validate_dates, SPLITS
from scripts.run_lightgbm_baseline import MODEL_PARAMS, RAW_DATA_PATH, EXPERIMENT_ROOT, PeakMemoryMonitor, split_masks, environment_versions
from src.data.baseline_panel import load_raw_baseline_panel
from src.features.features34 import FEATURE_COLUMNS, NEW_COLUMNS, GROUPS, select_features, build_features34
from src.validation.experiment import experiment_run, provenance, sha256_file, prediction_hash, write_json

CONFIG = ROOT / 'configs/features34_step4.json'
OUTPUT = ROOT / 'artifacts/features34_step4'
PREREG = ROOT / 'docs/features34/STEP4_PREREGISTRATION.md'


def validate_config(cfg):
    if cfg['stage'] != 'step4_single_deletion_development_only' or cfg['used_2024']:
        raise ValueError('step4 development only; 2024 prohibited')
    if cfg['contexts'] != {'full34': list('ABCDEF'), 'lean31': list('BCDEF')}:
        raise ValueError('only fixed full34/lean31 backgrounds')
    if set(cfg) != {'stage', 'contexts', 'step3_config', 'rules', 'budgets', 'used_2024'}:
        raise ValueError('unsupported config fields; no tuning')
    if cfg['budgets'] != dict(screen_2023=45, cross_year_max=90, joint_versions_max=2, addback_versions_max_total=2):
        raise ValueError('experiment budget changed')


def selection(cfg, item):
    validate_config(cfg)
    if item['split'] not in SPLITS or item['context'] not in cfg['contexts']:
        raise ValueError('unauthorized year or background')
    exclusions = item['exclude']
    if item['kind'] == 'single' and len(exclusions) != 1:
        raise ValueError('single deletion must remove exactly one added feature')
    if item['kind'] not in ('single', 'joint', 'addback'):
        raise ValueError('unsupported experiment kind')
    return select_features(groups=cfg['contexts'][item['context']], exclude=exclusions)


def matrix(cfg):
    return [dict(id=f'S4_{ctx}_{c}_2023', context=ctx, feature=c,
                 group=next(g for g, cs in GROUPS.items() if c in cs),
                 split='primary_2023', kind='single', exclude=[c])
            for ctx, groups in cfg['contexts'].items()
            for c in NEW_COLUMNS if c in select_features(groups=groups)]


def verify_artifact(directory, split, columns, exact_keys=None):
    summary = read_json(directory / 'summary.json')
    if read_json(directory / 'status.json')['status'] != 'success':
        raise AssertionError('failed prior run cannot be reused')
    result = summary['splits'][0]
    if result['split_name'] != split or summary['features'] != list(columns):
        raise AssertionError('split/model columns mismatch')
    if summary['model_params'] != MODEL_PARAMS or summary['environment'] != environment_versions():
        raise AssertionError('model/environment mismatch')
    if sha256_file(directory / 'provenance.json') != summary['provenance_sha256'] or sha256_file(directory / 'config.json') != summary['config_sha256']:
        raise AssertionError('metadata mismatch')
    for relative, expected in result['file_sha256'].items():
        if sha256_file(directory / split / relative) != expected:
            raise AssertionError(f'changed artifact: {relative}')
    pred = pd.read_parquet(directory / split / 'predictions.parquet')
    if pred.duplicated(['ts_code', 'trade_date']).any() or not np.isfinite(pred.pred).all() or prediction_hash(pred.pred) != result['prediction_sha256']:
        raise AssertionError('prediction validity failure')
    if exact_keys is not None:
        pd.testing.assert_frame_equal(pred[['ts_code', 'trade_date']], exact_keys,
                                      check_dtype=False, check_categorical=False)
    if result['prediction_coverage'] != 1 or not result['model_reload_predictions_equal']:
        raise AssertionError('coverage/reload mismatch')
    if lgb.Booster(model_file=str(directory / split / 'models/lightgbm.txt')).feature_name() != list(columns):
        raise AssertionError('saved model columns mismatch')
    if not np.isfinite(list(result['metrics'].values())).all() or result['official_comparison']['max_abs_difference'] > 1e-12:
        raise AssertionError('official parity failed')
    return summary


def references(cfg, meta):
    step3 = ROOT / 'artifacts/features34_step3'
    accepted = read_json(step3 / 'acceptance.json')
    if not accepted['accepted'] or not read_json(step3 / 'summary.json')['accepted']:
        raise AssertionError('step3 not accepted')
    for relative, expected in accepted['evidence_sha256'].items():
        if sha256_file(step3 / relative) != expected:
            raise AssertionError(f'step3 evidence changed: {relative}')
    prereg = read_json(step3 / 'preregistration.json')
    for relative, expected in prereg['source']['source_sha256'].items():
        if sha256_file(ROOT / relative) != expected or sha256_file(step3 / 'executed_sources' / relative) != expected:
            raise AssertionError(f'step3 executed source changed: {relative}')
    records, refs = [], {}
    for rec in read_json(step3 / 'run_index.json')['runs']:
        if rec['candidate'] not in ('baseline10', 'full34', 'lean31'):
            continue
        directory = ROOT / rec['directory']
        if sha256_file(directory / 'summary.json') != rec['summary_sha256']:
            raise AssertionError('step3 summary hash changed')
        cols = select_features(groups=[] if rec['candidate'] == 'baseline10' else cfg['contexts'][rec['candidate']])
        summary = verify_artifact(directory, rec['split'], cols)
        prior = read_json(directory / 'provenance.json')
        if prior['data'] != meta['data'] or prior['dependencies'] != meta['dependencies']:
            raise AssertionError('reuse data/dependency mismatch')
        # Older stages legitimately have fewer source files; every actually executed frozen dependency is checked above.
        refs[(rec['candidate'], rec['split'])] = (directory, summary)
        records.append({**rec, 'verified_files': len(summary['splits'][0]['file_sha256'])})
    if len(refs) != 9:
        raise AssertionError('missing three-year reference')
    return refs, records


def daily(directory, split):
    frames = []
    for filename, value in [('daily_ic', 'ic'), ('daily_excess', 'excess'), ('daily_turnover', 'turnover')]:
        frame = pd.read_csv(directory / split / f'{filename}.csv', float_precision='round_trip')
        frames.append(frame.set_index('trade_date')[[value]])
    return pd.concat(frames, axis=1).sort_index()


def paired_blocks(after, before, rule):
    if not after.index.equals(before.index):
        raise AssertionError('daily dates must align exactly')
    a, b = after.to_numpy(), before.to_numpy()
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise AssertionError('metric eligibility changed')
    n = len(a); length = rule['block_trading_days']
    if n < length:
        raise ValueError('insufficient dates for block length')
    rng = np.random.default_rng(rule['seed'])
    draws = []
    # A block first date has no resampled predecessor, so its turnover is omitted.
    for _ in range(rule['repetitions']):
        aa, bb = [], []
        remaining = n
        while remaining:
            take = min(length, remaining)
            start = int(rng.integers(0, n-length+1))
            xa, xb = a[start:start+take].copy(), b[start:start+take].copy()
            xa[0, 2] = np.nan; xb[0, 2] = np.nan
            aa.append(xa); bb.append(xb); remaining -= take
        delta = np.nanmean(np.concatenate(aa), axis=0) - np.nanmean(np.concatenate(bb), axis=0)
        draws.append(float(delta @ np.array([.4, .3*252, -.3])))
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(block_ci_lower=float(lo), block_ci_upper=float(hi), block_ci_spans_zero=bool(lo <= 0 <= hi),
                block_length=length, block_repetitions=rule['repetitions'])


def compare(item, directory, summary, refs, cfg):
    split = item['split']; result = summary['splits'][0]
    refdir, refsummary = refs[(item['context'], split)]
    before = refsummary['splits'][0]
    base = refs[('baseline10', split)][1]['splits'][0]
    for k in ('train_samples', 'valid_prediction_rows', 'purge_rows', 'split_train_rows', 'dates'):
        if result[k] != before[k] or result[k] != base[k]:
            raise AssertionError('sample eligibility/split mismatch')
    row = dict(experiment_id=item['id'], context=item['context'], feature=item.get('feature', ';'.join(item['exclude'])),
               group=item.get('group', 'joint'), kind=item['kind'], year=int(split[-4:]), split=split,
               directory=directory.relative_to(ROOT).as_posix(), before_directory=refdir.relative_to(ROOT).as_posix(),
               feature_count=len(summary['features']), train_samples=result['train_samples'], valid_prediction_rows=result['valid_prediction_rows'])
    for k, v in result['metrics'].items():
        row.update({f'before_{k}': before['metrics'][k], f'after_{k}': v,
                    f'delta_{k}': v-before['metrics'][k], f'after_{k}_minus_baseline10': v-base['metrics'][k]})
    for prefix, source, key in [('top_missing_label', 'top_groups', 'missing_label_fraction'),
                               ('top_invalid_price', 'top_groups', 'invalid_price_fraction'),
                               ('top_baseline_all_missing', 'top_groups', 'all_features_missing_fraction'),
                               ('top_candidate_all_missing', 'candidate_features_top_groups', 'all_features_missing_fraction')]:
        for tag, r in [('before', before), ('after', result)]:
            row[f'{tag}_{prefix}'] = r['diagnostics'][source]['turnover'][key]
        row[f'delta_{prefix}'] = row[f'after_{prefix}']-row[f'before_{prefix}']
    for tag, r in [('before', before), ('after', result)]:
        row[f'{tag}_price_valid_turnover'] = r['diagnostics']['price_valid_only_turnover']
    row['delta_price_valid_turnover'] = row['after_price_valid_turnover']-row['before_price_valid_turnover']
    monthly = pd.read_csv(directory / split / 'monthly_metrics.csv', float_precision='round_trip')
    bm = pd.read_csv(refdir / split / 'monthly_metrics.csv', float_precision='round_trip')
    basem = pd.read_csv(refs[('baseline10', split)][0] / split / 'monthly_metrics.csv', float_precision='round_trip')
    if len(monthly) != 12 or monthly.month.tolist() != bm.month.tolist() or monthly.month.tolist() != basem.month.tolist():
        raise AssertionError('monthly alignment failed')
    for k in ('ic', 'annual_excess', 'turnover', 'score'):
        monthly[f'before_{k}'] = bm[k]
        monthly[f'delta_{k}'] = monthly[k]-bm[k]
        monthly[f'{k}_minus_baseline10'] = monthly[k]-basem[k]
    for k in ('experiment_id', 'context', 'feature', 'year', 'kind'):
        monthly[k] = row[k]
    positive = monthly.delta_annual_excess.clip(lower=0)
    scorepositive = monthly.delta_score.clip(lower=0)
    row.update(positive_months=int((monthly.delta_score > 0).sum()), negative_months=int((monthly.delta_score < 0).sum()),
               near_months=int((monthly.delta_score.abs() <= cfg['rules']['near_score']).sum()),
               worst_month=int(monthly.loc[monthly.delta_score.idxmin(), 'month']), worst_month_delta=float(monthly.delta_score.min()),
               median_month_delta=float(monthly.delta_score.median()),
               positive_excess_months=int((monthly.delta_annual_excess > 0).sum()),
               largest_positive_excess_month_share=None if positive.sum() == 0 else float(positive.max()/positive.sum()),
               largest_positive_score_month_share=None if scorepositive.sum() == 0 else float(scorepositive.max()/scorepositive.sum()))
    row.update(paired_blocks(daily(directory, split), daily(refdir, split), cfg['rules']['bootstrap']))
    row['guards_pass'] = guards(row, cfg['rules'])
    row['near_effect'] = abs(row['delta_final_score']) <= cfg['rules']['near_score']
    row['clear_year_degradation'] = row['delta_final_score'] < -cfg['rules']['clear_year_degradation']
    row['preliminary_conclusion'] = 'possible deletion' if row['delta_final_score'] > cfg['rules']['near_score'] else ('possible retention' if row['delta_final_score'] < -cfg['rules']['near_score'] else 'near/uncertain')
    row['evidence_limit'] = 'conditional fixed background; 2023 alone cannot decide; bootstrap fixed predictions only'
    return row, monthly


def guards(row, r):
    return bool(row['delta_ic_mean'] >= -r['ic_degradation_guard'] and
        row['delta_annual_excess'] >= -r['annual_excess_degradation_guard'] and
        row['delta_mean_turnover'] <= r['turnover_increase_guard'] and
        row['delta_price_valid_turnover'] <= r['price_valid_turnover_increase_guard'] and
        all(row['delta_'+k] <= r['missing_fraction_increase_guard'] for k in
            ['top_missing_label', 'top_invalid_price', 'top_baseline_all_missing', 'top_candidate_all_missing']) and
        row['positive_months'] >= r['monthly_positive_required'] and
        row['worst_month_delta'] >= r['monthly_worst_floor'] and
        row['largest_positive_excess_month_share'] is not None and
        row['largest_positive_excess_month_share'] <= r['largest_positive_excess_month_share_max'])


def classify(frame, r):
    if set(frame.year) != {2021, 2022, 2023} or len(frame) != 3:
        return '不确定'
    if (frame.delta_final_score >= r['near_score']).all() and (frame.block_ci_lower > 0).all() and frame.guards_pass.all():
        return '删除'
    if (frame.delta_final_score <= -r['near_score']).all() and (frame.block_ci_upper < 0).all():
        return '保留'
    return '不确定'


def protection_snapshot():
    paths = [ROOT / '.git/hooks/pre-push']
    gitdir = Path(subprocess.check_output(['git', 'rev-parse', '--absolute-git-dir'], cwd=ROOT, text=True).strip())
    paths = [gitdir / 'hooks/pre-push', gitdir / 'config']
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    return dict(protections={str(p): sha256_file(p) for p in paths if p.exists()},
                existing_untracked={p: sha256_file(ROOT/p) for p in untracked if p and (ROOT/p).is_file()})


def execute(items, phase):
    started = time.perf_counter(); cfg = read_json(CONFIG); validate_config(cfg)
    for item in items:
        selection(cfg,item)
    step3cfg = read_json(ROOT / cfg['step3_config'])
    manifest, ref, meta = check_contract(step3cfg)
    refs, records = references(cfg, meta)
    if not OUTPUT.exists():
        OUTPUT.mkdir(parents=True, exist_ok=False)
        write_json(OUTPUT/'preflight.json', protection_snapshot())
        write_json(OUTPUT/'preregistration.json', dict(config=cfg, config_sha256=sha256_file(CONFIG),
            rules_sha256=sha256_file(PREREG), source=meta, reference_runs=records,
            created_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat()))
        write_json(OUTPUT/'screen_matrix.json', dict(direction='after minus before', experiments=matrix(cfg)))
        write_json(OUTPUT/'run_index.json', dict(runs=[]))
        write_json(OUTPUT/'failures.json', dict(failures=[]))
        for relative in meta['source_sha256']:
            target=OUTPUT/'executed_sources'/relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative,target)
        shutil.copy2(PREREG, OUTPUT/'STEP4_PREREGISTRATION.md')
    prereg=read_json(OUTPUT/'preregistration.json')
    if prereg['config_sha256'] != sha256_file(CONFIG) or prereg['rules_sha256'] != sha256_file(PREREG) or prereg['source']['source_sha256'] != meta['source_sha256']:
        raise AssertionError('predefined config/rules/source changed; refuse mixed execution')
    phase_path=OUTPUT/f'{phase}_status.json'
    plan_path=OUTPUT/f'{phase}_plan.json'
    plan_hash=sha256_file(plan_path) if plan_path.exists() else None
    registration_path=OUTPUT/f'{phase}_registration.json'
    if not registration_path.exists():
        write_json(registration_path,dict(experiments=items,plan_sha256=plan_hash,
            registered_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat()))
    elif read_json(registration_path)['experiments'] != items or read_json(registration_path)['plan_sha256'] != plan_hash:
        raise AssertionError('phase matrix/plan changed after registration')
    write_json(phase_path, dict(status='running', planned=len(items)))
    try:
        with PeakMemoryMonitor() as memory:
            panel=load_raw_baseline_panel(RAW_DATA_PATH)
            splits=validate_dates(panel.trade_date.unique(),step3cfg)
            print(f'{phase}: computing full historical 34 columns once before any masks',flush=True)
            full=build_features34(panel)
            stats=feature_statistics(full)
            if stats.infinite.any() or not stats.finite.gt(0).all(): raise AssertionError('feature invalid')
            exact={s:panel.loc[split_masks(panel,sp)[1],['ts_code','trade_date']].reset_index(drop=True) for s,sp in splits.items()}
            for (name,s),(d,sm) in refs.items():
                verify_artifact(d,s,tuple(sm['features']),exact[s])
            for item in items:
                columns=selection(cfg,item); split=item['split']; sp=splits[split]
                index=read_json(OUTPUT/'run_index.json')['runs']
                old=next((x for x in index if x['item']['id']==item['id']),None)
                if old is not None:
                    if old['item'] != item: raise AssertionError('existing experiment definition differs')
                    verify_artifact(ROOT/old['directory'],split,columns,exact[split]); continue
                began=time.perf_counter(); output=None
                try:
                    with experiment_run(EXPERIMENT_ROOT,item['id']) as output:
                        write_json(output/'config.json',dict(requested=cfg,item=item,features=list(columns),model_params=MODEL_PARAMS))
                        write_json(output/'provenance.json',meta)
                        features=full.loc[:,columns]
                        splitdir=output/split; fixture=splitdir/'evaluate_input'; fixture.mkdir(parents=True,exist_ok=False)
                        # The immutable original-token truth used in accepted step3 is copied byte-for-byte.
                        truth=refs[('baseline10',split)][0]/split/'evaluate_input/测试集_Y.csv'
                        shutil.copy2(truth,fixture/'测试集_Y.csv')
                        frozen=next((s for s in ref['splits'] if s['split_name']==split),None)
                        with (output/'execution.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            result=run_split(panel,features,split,output_dir=splitdir,reference=frozen,columns=columns,research_split=sp)
                        pred=pd.read_parquet(splitdir/'predictions.parquet')
                        pd.testing.assert_frame_equal(pred[['ts_code','trade_date']],exact[split],check_dtype=False,check_categorical=False)
                        if sha256_file(fixture/'测试集_Y.csv') != sha256_file(truth): raise AssertionError('truth changed')
                        check_frozen(manifest)
                        write_json(splitdir/'summary.json',result)
                        summary=dict(stage=cfg['stage'],candidate=item['id'],item=item,features=list(columns),
                            model_params=MODEL_PARAMS,environment=environment_versions(),panel_rows=len(panel),
                            feature_index_preserved=True,prediction_keys_match_raw_panel=True,frozen_files_unchanged=True,
                            splits=[result],provenance_sha256=sha256_file(output/'provenance.json'),config_sha256=sha256_file(output/'config.json'),
                            resources=dict(elapsed_seconds=time.perf_counter()-began,peak_process_rss_mb=memory.peak_rss_bytes/1024**2))
                        write_json(output/'summary.json',summary)
                    state=read_json(output/'status.json');state['elapsed_seconds']=time.perf_counter()-began;write_json(output/'status.json',state)
                    verify_artifact(output,split,columns,exact[split])
                    index.append(dict(item=item,directory=output.relative_to(ROOT).as_posix(),summary_sha256=sha256_file(output/'summary.json')))
                    write_json(OUTPUT/'run_index.json',dict(runs=index))
                    print(f'Accepted {item["id"]}: {result["metrics"]["final_score"]:.12f} ({time.perf_counter()-began:.1f}s)',flush=True)
                    del features,pred;gc.collect()
                except BaseException as exc:
                    failures=read_json(OUTPUT/'failures.json');failures['failures'].append(dict(item=item,error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.perf_counter()-began,directory=None if output is None else str(output)))
                    write_json(OUTPUT/'failures.json',failures)
                    if output is not None:
                        state=read_json(output/'status.json');state['elapsed_seconds']=time.perf_counter()-began;write_json(output/'status.json',state)
                    raise
            check_frozen(manifest)
            now=provenance(ROOT,RAW_DATA_PATH)
            if now['source_sha256'] != meta['source_sha256'] or now['data'] != meta['data']:
                raise AssertionError('source/data changed during execution')
            if plan_path.exists() and sha256_file(plan_path) != plan_hash:
                raise AssertionError('phase plan changed during execution')
        write_json(phase_path,dict(status='success',planned=len(items),elapsed_seconds=time.perf_counter()-started))
        aggregate(cfg,refs)
    except BaseException as exc:
        write_json(phase_path,dict(status='failed',error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.perf_counter()-started));raise


def aggregate(cfg,refs):
    rows,months=[],[]
    for rec in read_json(OUTPUT/'run_index.json')['runs']:
        d=ROOT/rec['directory'];item=rec['item']
        if sha256_file(d/'summary.json') != rec['summary_sha256']: raise AssertionError('indexed summary changed')
        sm=verify_artifact(d,item['split'],selection(cfg,item))
        row,month=compare(item,d,sm,refs,cfg);rows.append(row);months.append(month)
    annual=pd.DataFrame(rows)
    annual.to_csv(OUTPUT/'deletion_results.csv',index=False,float_format='%.17g')
    pd.concat(months,ignore_index=True).to_csv(OUTPUT/'monthly_deletion_results.csv',index=False,float_format='%.17g')
    decisions=[]
    for (ctx,c),frame in annual[annual.kind=='single'].groupby(['context','feature']):
        decision=classify(frame,cfg['rules'])
        rec=dict(context=ctx,feature=c,group=frame.group.iloc[0],decision=decision,
                 currently_in_original_input=True,temporarily_keep=decision!='删除',
                 direct_years=';'.join(str(y) for y in sorted(frame.year)),
                 experiment_ids=';'.join(frame.sort_values('year').experiment_id),
                 reason='conditional three-year evidence meets fixed rule' if decision!='不确定' else 'incomplete/near/conflicting/block CI or guard risk')
        for _,r in frame.iterrows(): rec[f'delta_{int(r.year)}']=r.delta_final_score
        decisions.append(rec)
    # A is absent from lean31; group deletion evidence does not prove each A column dispensable.
    for c in GROUPS['A']:
        decisions.append(dict(context='lean31',feature=c,group='A',decision='不确定',currently_in_original_input=False,
                              temporarily_keep=False,direct_years='',experiment_ids='',reason='absent by fixed starting candidate; no individual necessity conclusion'))
    pd.DataFrame(decisions).to_csv(OUTPUT/'feature_decisions.csv',index=False,float_format='%.17g')
    return annual,pd.DataFrame(decisions)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['screen','cross','joint','aggregate'])
    args=parser.parse_args();cfg=read_json(CONFIG)
    if args.phase=='screen': execute(matrix(cfg),'screen')
    elif args.phase in ('cross','joint'):
        plan=read_json(OUTPUT/f'{args.phase}_plan.json')
        items=plan['experiments']
        if args.phase=='cross' and (len(items)>90 or any(x['kind']!='single' or x['split']=='primary_2023' for x in items)): raise ValueError('cross budget/scope')
        if args.phase=='joint' and (len({x['version'] for x in items})>2 or any(x['kind']!='joint' for x in items)): raise ValueError('joint budget/scope')
        execute(items,args.phase)
    else:
        _,_,meta=check_contract(read_json(ROOT/cfg['step3_config']));refs,_=references(cfg,meta);aggregate(cfg,refs)


if __name__=='__main__': main()
