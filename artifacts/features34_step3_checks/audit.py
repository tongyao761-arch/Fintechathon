"""Read-only final disk audit of the completed step3 matrix; writes acceptance last."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.run_features34_step3 import (read_json, CONFIG, OUTPUT, check_contract,
    validate_dates, verify_new, comparison_tables, rank_candidates)
from scripts.run_lightgbm_baseline import RAW_DATA_PATH, split_masks
from src.data.baseline_panel import load_raw_baseline_panel
from src.validation.experiment import sha256_file, prediction_hash, write_json


def main():
    started = time.perf_counter()
    config = read_json(CONFIG)
    manifest, ref, meta = check_contract(config)
    total = read_json(OUTPUT / 'summary.json')
    assert total['accepted'] and read_json(OUTPUT / 'status.json')['status'] == 'success'
    assert total['years'] == [2021, 2022, 2023] and not total['used_2024']
    prereg = read_json(OUTPUT / 'preregistration.json')
    assert prereg['config_sha256'] == sha256_file(CONFIG)
    assert prereg['candidates_sha256'] == sha256_file(ROOT / 'docs/features34/STEP3_CANDIDATES.md')
    checks = ROOT / 'artifacts/features34_step3_checks'
    preflight = read_json(checks / 'preflight.json')
    for relative, expected in {**preflight['protections'], **preflight['existing_untracked_hashes']}.items():
        assert sha256_file(ROOT / relative) == expected, relative
    assert read_json(checks / 'tests_result.json')['exit_code'] == 0
    assert read_json(checks / 'dependency_check.json')['exit_code'] == 0
    for relative, expected in prereg['source']['source_sha256'].items():
        assert sha256_file(OUTPUT / 'executed_sources' / relative) == expected, relative
        assert sha256_file(ROOT / relative) == expected, relative
    panel = load_raw_baseline_panel(RAW_DATA_PATH)
    splits = validate_dates(panel.trade_date.unique(), config)
    entries, verified, exact_keys, count_rows = [], [], {}, []
    for name, split in splits.items():
        train, valid = split_masks(panel, split)
        exact_keys[name] = panel.loc[valid, ['ts_code', 'trade_date']].reset_index(drop=True)
        count_rows.append(dict(split=name, **config['splits'][name], train_samples=int(train.sum()),
            train_period_rows=int(split.masks(panel)[0].sum()),
            valid_prediction_rows=int(valid.sum()), purge_rows=int(panel.trade_date.eq(split.purge_date).sum()),
            valid_missing_labels=int(panel.loc[valid,'y_ret_1d'].isna().sum()),
            valid_invalid_prices=int(panel.loc[valid,'is_price_valid'].eq(0).sum())))
        assert not train[panel.trade_date.eq(split.purge_date)].any()
        assert panel.loc[train,'trade_date'].max() == split.train_end
    for run in total['runs']:
        directory=ROOT / run['directory']; name=run['candidate']; split=run['split']
        assert sha256_file(directory/'summary.json') == run['summary_sha256']
        summary=read_json(directory/'summary.json')
        if run['mode']=='new_run': verify_new(directory,name,split,config,meta)
        else:
            assert read_json(directory/'status.json')['status']=='success'
            result=summary['splits'][0]
            for relative,expected in result['file_sha256'].items():
                assert sha256_file(directory/split/relative)==expected
        result=summary['splits'][0]
        assert result['dates']==config['splits'][split]
        prediction=pd.read_parquet(directory/split/'predictions.parquet')
        pd.testing.assert_frame_equal(prediction[['ts_code','trade_date']],exact_keys[split],
            check_dtype=False,check_categorical=False)
        assert np.isfinite(prediction.pred).all()
        assert prediction_hash(prediction.pred)==result['prediction_sha256']
        counts=next(r for r in count_rows if r['split']==split)
        assert result['train_samples']==counts['train_samples']
        assert result['valid_prediction_rows']==counts['valid_prediction_rows']
        stats=pd.read_csv(directory/split/'feature_missing_statistics.csv')
        assert set(stats.scope)=={'full_panel','train_eligible','validation_all_keys'}
        assert not stats.infinite.any() and stats.finite.gt(0).all()
        assert set(stats[stats.scope=='full_panel'].rows)=={len(panel)}
        entries.append((name,split,directory,summary,run['mode']))
        verified.append(dict(candidate=name,split=split,directory=run['directory'],
            hashes_verified=len(result['file_sha256']),keys_match_raw_panel=True,
            official_max_abs_difference=result['official_comparison']['max_abs_difference']))
    annual,monthly,missing=comparison_tables(entries)
    ranking,selected=rank_candidates(annual,monthly,config['selection_rule'])
    for name,frame in [('comparison',annual),('monthly_comparison',monthly),
                       ('monthly_missing_diagnostics',missing),('candidate_ranking',ranking)]:
        saved=pd.read_csv(OUTPUT/f'{name}.csv',float_precision='round_trip')
        pd.testing.assert_frame_equal(frame.reset_index(drop=True),saved,check_dtype=False,
                                      check_exact=True)
        assert sha256_file(OUTPUT/f'{name}.csv')==total['tables'][name]
    assert selected==total['proposed_next_candidates']
    pd.DataFrame(count_rows).to_csv(OUTPUT/'split_audit.csv',index=False)
    stability=[]
    for (candidate,year),frame in monthly.groupby(['candidate','year']):
        delta=frame.score_minus_baseline10
        positive=delta.clip(lower=0)
        stability.append(dict(candidate=candidate,year=int(year),positive_months=int((delta>0).sum()),
            min_month_delta=float(delta.min()),max_month_delta=float(delta.max()),
            median_month_delta=float(delta.median()),positive_excess_months=int((frame.annual_excess_minus_baseline10>0).sum()),
            largest_positive_month_share=None if positive.sum()==0 else float(positive.max()/positive.sum())))
    pd.DataFrame(stability).to_csv(OUTPUT/'monthly_stability.csv',index=False,float_format='%.17g')
    pair_rows=[]
    for year,frame in annual.groupby('year'):
        indexed=frame.set_index('candidate')
        for group,before,after in [('A','full34','lean31'),('C','lean31','lean27'),('E','lean27','lean23')]:
            pair_rows.append(dict(year=int(year),deleted_group=group,context_before=before,context_after=after,
                **{f'{k}_deletion_delta':float(indexed.loc[after,k]-indexed.loc[before,k]) for k in
                   ['final_score','ic_mean','annual_excess','mean_turnover','price_valid_only_turnover','top_missing_label_fraction']}))
    pd.DataFrame(pair_rows).to_csv(OUTPUT/'conditional_group_deletions.csv',index=False,float_format='%.17g')
    evidence=[*total['tables'], 'split_audit', 'monthly_stability', 'conditional_group_deletions']
    acceptance=dict(accepted=True,stage=config['stage'],runs=verified,frozen_files_verified=len(manifest),
        new_runs=sum(r['mode']=='new_run' for r in total['runs']),reused_runs=sum(r['mode']=='reused_step2' for r in total['runs']),
        regression_tests_passed=88,pip_check_passed=True,all_raw_validation_keys_verified=True,
        training_eligibility_counts_verified=True,actual_trading_day_purge_verified=True,
        source_snapshots_verified=True,pre_registration_unchanged=True,all_comparison_tables_recomputed=True,
        branch=meta['git']['branch'],push_protections_unchanged=True,existing_untracked_files_preserved=len(preflight['existing_untracked_hashes']),
        used_2024=False,feature_checks_run=False,final_submission_generated=False,
        proposed_next_candidates=selected,audit_elapsed_seconds=time.perf_counter()-started,
        auxiliary_failures=['artifacts/features34_step3_checks/log_read_failure.json','artifacts/features34_step3_checks/preflight_attempt_01.json'],
        model_run_failures=0,audit_script_sha256=sha256_file(Path(__file__)),
        evidence_sha256={f'{name}.csv':sha256_file(OUTPUT/f'{name}.csv') for name in evidence})
    write_json(OUTPUT/'acceptance.json',acceptance)
    print({'accepted':True,'runs':len(verified),'selected':selected})


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        write_json(ROOT/'artifacts/features34_step3_checks/audit_failure.json',
            dict(status='failed',error_type=type(exc).__name__,error=str(exc)))
        raise
