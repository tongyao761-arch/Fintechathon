"""Independent disk acceptance; no model retraining and no 2024 scoring."""
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts import run_features34_step4 as runner
from scripts.run_features34 import check_frozen
from scripts.run_features34_step3 import check_contract,validate_dates,read_json
from scripts.run_lightgbm_baseline import RAW_DATA_PATH,split_masks
from src.data.baseline_panel import load_raw_baseline_panel
from src.validation.experiment import sha256_file,write_json


def main():
    started=time.perf_counter();out=runner.OUTPUT;cfg=read_json(runner.CONFIG)
    manifest,ref,meta=check_contract(read_json(ROOT/cfg['step3_config']))
    refs,records=runner.references(cfg,meta)
    prereg=read_json(out/'preregistration.json')
    assert prereg['config_sha256']==sha256_file(runner.CONFIG)
    assert prereg['rules_sha256']==sha256_file(runner.PREREG)
    correction=read_json(out/'audit_source_correction.json') if (out/'audit_source_correction.json').exists() else None
    assert set(prereg['source']['source_sha256'])==set(meta['source_sha256'])
    for relative,expected in prereg['source']['source_sha256'].items():
        assert sha256_file(out/'executed_sources'/relative)==expected,relative
        if relative=='scripts/audit_features34_step4.py' and correction:
            assert expected==correction['executed_sha256']
            assert meta['source_sha256'][relative]==correction['corrected_sha256']
            assert correction['training_runner_changed'] is False
        else:assert meta['source_sha256'][relative]==expected,relative
    preflight=read_json(out/'preflight.json')
    for p,expected in preflight['protections'].items():assert sha256_file(Path(p))==expected,p
    for p,expected in preflight['existing_untracked'].items():
        if p=='scripts/audit_features34_step4.py' and correction:
            assert expected==correction['executed_sha256']
            assert sha256_file(ROOT/p)==correction['corrected_sha256']
        else:assert sha256_file(ROOT/p)==expected,p
    panel=load_raw_baseline_panel(RAW_DATA_PATH)
    splits=validate_dates(panel.trade_date.unique(),read_json(ROOT/cfg['step3_config']))
    exact={};counts={}
    for s,sp in splits.items():
        train,valid=split_masks(panel,sp)
        exact[s]=panel.loc[valid,['ts_code','trade_date']].reset_index(drop=True)
        counts[s]=dict(train_samples=int(train.sum()),valid_prediction_rows=int(valid.sum()))
        assert not train[panel.trade_date.eq(sp.purge_date)].any()
    runs=read_json(out/'run_index.json')['runs'];verified=[]
    for (name,s),(d,sm) in refs.items():runner.verify_artifact(d,s,tuple(sm['features']),exact[s])
    expected=runner.matrix(cfg)+read_json(out/'cross_plan.json')['experiments']+read_json(out/'joint_plan.json')['experiments']
    assert len({i['id'] for i in expected})==len(expected)==len(runs)
    assert {i['id'] for i in expected}=={r['item']['id'] for r in runs}
    for rec in runs:
        item=rec['item'];s=item['split'];d=ROOT/rec['directory']
        assert item==next(x for x in expected if x['id']==item['id'])
        assert sha256_file(d/'summary.json')==rec['summary_sha256']
        sm=runner.verify_artifact(d,s,runner.selection(cfg,item),exact[s]);result=sm['splits'][0]
        assert result['dates']==read_json(ROOT/cfg['step3_config'])['splits'][s]
        for k,v in counts[s].items():assert result[k]==v
        saved_source=read_json(d/'provenance.json')
        for key in ['source_sha256','dependencies','data']:
            assert saved_source[key]==prereg['source'][key],key
        # Git status records phase-specific evidence files and changed docs; it is not executable provenance.
        for key in ['branch','commit']:
            assert saved_source['git'][key]==prereg['source']['git'][key],key
        assert sha256_file(d/s/'evaluate_input/测试集_Y.csv')==sha256_file(refs[('baseline10',s)][0]/s/'evaluate_input/测试集_Y.csv')
        stats=pd.read_csv(d/s/'feature_missing_statistics.csv')
        assert not stats.infinite.any() and stats.finite.gt(0).all()
        assert set(stats.scope)=={'full_panel','train_eligible','validation_all_keys'}
        assert set(stats[stats.scope=='full_panel'].rows)=={len(panel)}
        # Independently reconstruct yearly score from preserved day-level official details.
        daily=runner.daily(d,s)
        actual=.4*daily.ic.mean()+.3*252*daily.excess.mean()+.3*(1-daily.turnover.mean())
        assert abs(actual-result['metrics']['final_score'])<=1e-12
        verified.append(dict(id=item['id'],directory=rec['directory'],hashes_verified=len(result['file_sha256']),raw_keys_equal=True,
            official_max_abs_difference=result['official_comparison']['max_abs_difference']))
    old_results=pd.read_csv(out/'deletion_results.csv',float_precision='round_trip')
    old_decisions=pd.read_csv(out/'feature_decisions.csv',float_precision='round_trip')
    annual,decisions=runner.aggregate(cfg,refs)
    pd.testing.assert_frame_equal(annual,old_results,check_dtype=False,check_exact=True)
    # Compare decisions via CSV because absent-context records deliberately contain empty strings.
    pd.testing.assert_frame_equal(pd.read_csv(out/'feature_decisions.csv',float_precision='round_trip'),old_decisions,check_dtype=False,check_exact=True)
    assert len(decisions)==48 and decisions.groupby('context').size().eq(24).all()
    for phase in ['screen','cross']:
        assert read_json(out/f'{phase}_status.json')['status']=='success'
    for phase in ['screen','cross','joint']:
        registration=read_json(out/f'{phase}_registration.json') if (out/f'{phase}_registration.json').exists() else None
        if registration and registration['plan_sha256'] is not None:assert registration['plan_sha256']==sha256_file(out/f'{phase}_plan.json')
    for name in ['tests_result','dependency_check']:
        assert read_json(out/f'{name}.json')['exit_code']==0
    frozen=read_json(out/'frozen_candidates.json')
    assert len(frozen['candidates'])<=2 and not frozen['used_2024']
    assert len(read_json(out/'joint_plan.json')['versions'])<=2
    check_frozen(manifest)
    evidence={p.name:sha256_file(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md') and p.name not in ('acceptance.json','summary.json','status.json')}
    acceptance=dict(accepted=True,stage=cfg['stage'],runs=verified,reused_references=records,frozen_files_verified=len(manifest),
        all_raw_validation_keys_verified=True,training_eligibility_counts_verified=True,all_saved_hashes_verified=True,
        daily_official_scores_reconstructed=True,all_tables_and_decisions_recomputed=True,source_snapshots_verified=True,
        push_protections_unchanged=True,existing_untracked_files_preserved=len(preflight['existing_untracked']),
        used_2024=False,final_submission_generated=False,model_run_failures=len(read_json(out/'failures.json')['failures']),
        auxiliary_failures=read_json(out/'auxiliary_failures.json')['failures'] if (out/'auxiliary_failures.json').exists() else [],
        audit_source_correction=correction,
        elapsed_seconds=time.perf_counter()-started,evidence_sha256=evidence)
    write_json(out/'acceptance.json',acceptance)
    write_json(out/'summary.json',dict(accepted=True,stage=cfg['stage'],new_runs=len(runs),reused_references=9,
        screen_runs=45,cross_runs=len(read_json(out/'cross_plan.json')['experiments']),
        joint_versions=len(read_json(out/'joint_plan.json')['versions']),addback_versions=0,
        used_2024=False,final_submission_generated=False,frozen_candidates=frozen['candidates'],
        acceptance_sha256=sha256_file(out/'acceptance.json')))
    write_json(out/'status.json',dict(status='success',step4_complete=True))
    print(f'Accepted step4: {len(runs)} runs; 9 reference reuses; {len(decisions)} background decisions',flush=True)


if __name__=='__main__':main()
