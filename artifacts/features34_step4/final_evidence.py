"""Verify only new explanation output after the full model/disk audit passed."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from src.validation.experiment import sha256_file,write_json
from scripts.run_features34_step3 import read_json

p=ROOT/'artifacts/features34_step4';a=read_json(p/'acceptance.json')
assert a['accepted'] and len(a['runs'])==135
changed=[n for n,h in a['evidence_sha256'].items() if sha256_file(p/n)!=h]
assert set(changed)<= {'STEP4_REPORT.md'},changed
annual=pd.read_csv(p/'deletion_results.csv',float_precision='round_trip')
d=pd.read_csv(p/'decision_evidence_details.csv',keep_default_na=False)
decisions=pd.read_csv(p/'feature_decisions.csv',keep_default_na=False)
r=read_json(ROOT/'configs/features34_step4.json')['rules']
assert len(d)==48
for _,row in d.iterrows():
    fs=annual[(annual.context==row.context)&(annual.feature==row.feature)&(annual.kind=='single')].sort_values('year')
    expected=decisions[(decisions.context==row.context)&(decisions.feature==row.feature)].iloc[0]
    assert row.decision==expected.decision
    assert row.experiment_ids==';'.join(fs.experiment_id)
    yy=lambda mask:';'.join(str(int(y)) for y in fs.loc[mask,'year'])
    assert row.near_years==yy(fs.delta_final_score.abs()<=r['near_score'])
    assert row.block_ci_spans_zero_years==yy((fs.block_ci_lower<=0)&(fs.block_ci_upper>=0))
    assert row.deletion_guard_failure_years==yy(~fs.guards_pass)
    assert row.clear_degradation_years==yy(fs.delta_final_score < -r['clear_year_degradation'])
    assert bool(row.annual_direction_conflict)==bool((fs.delta_final_score>0).any() and (fs.delta_final_score<0).any())
note=dict(accepted=True,full_disk_audit_unchanged=True,numerical_tables_and_models_unchanged=True,
    added_explanation_rows_verified=48,changed_evidence=changed,
    unchanged_frozen_candidates=True,
    additions=['decision_evidence_details.csv','EVIDENCE_INDEX.md'],
    note='Only report explanation and evidence index extended after full audit; derived trigger years independently checked against accepted numerical results.')
write_json(p/'final_evidence_check.json',note)
for name in changed+note['additions']+['final_evidence_check.json']:
    a['evidence_sha256'][name]=sha256_file(p/name)
a['post_audit_explanation_check']=note
write_json(p/'acceptance.json',a)
s=read_json(p/'summary.json');s['acceptance_sha256']=sha256_file(p/'acceptance.json')
write_json(p/'summary.json',s)
print('Verified 48 explanation rows; audited numerical tables, frozen candidates, and all models unchanged')
