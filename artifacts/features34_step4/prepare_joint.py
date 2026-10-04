"""Freeze at most one joint deletion per fixed background; no opportunistic search."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.run_features34_step4 import OUTPUT,CONFIG,read_json,write_json,SPLITS

cfg=read_json(CONFIG)
assert read_json(OUTPUT/'cross_status.json')['status']=='success'
decisions=pd.read_csv(OUTPUT/'feature_decisions.csv')
assert len(decisions)==48
annual=pd.read_csv(OUTPUT/'deletion_results.csv')
assert len(annual)==135
versions=[];experiments=[];fallbacks=[]
for ctx in cfg['contexts']:
    selected=decisions[(decisions.context==ctx)&(decisions.decision=='删除')]
    removed=selected.feature.tolist()
    if not removed:
        fallbacks.append(dict(context=ctx,candidate=ctx,reason='没有满足预登记跨年直接证据及全部护栏的确定删除列，冻结原候选；不强行精简',exclude=[]))
        continue
    version=f'joint_{ctx}'
    reasons=[dict(feature=row.feature,decision=row.decision,experiments=row.experiment_ids,
                  reason='该背景三年单删综合分≥0.001，配对块下界>0，全部分项/月度/缺失护栏通过') for _,row in selected.iterrows()]
    versions.append(dict(version=version,context=ctx,exclude=removed,reasons=reasons))
    for split in SPLITS:
        experiments.append(dict(id=f'S4_{version}_{split[-4:]}',version=version,context=ctx,
                               split=split,kind='joint',exclude=removed))
assert len(versions)<=2
assert not (OUTPUT/'joint_plan.json').exists()
write_json(OUTPUT/'joint_plan.json',dict(frozen_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
    versions=versions,experiments=experiments,fallbacks=fallbacks,
    rule='每个背景最多一个联合版本；三年完整验证，任一年负删除分差或护栏不通过退回原候选；没有确定删除列可保留原候选',
    addback_budget_total=2,addback_versions_planned=0,
    addback_reason='尚无明确替代或交互依据支持额外回补，不按猜测使用额度'))
print('Frozen joint versions:',len(versions),'new runs:',len(experiments))
