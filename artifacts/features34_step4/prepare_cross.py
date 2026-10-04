"""Freeze cross-year list only after the complete 2023 screen is accepted."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.run_features34_step4 import OUTPUT,CONFIG,read_json,write_json,matrix

cfg=read_json(CONFIG)
assert read_json(OUTPUT/'screen_status.json')['status']=='success'
annual=pd.read_csv(OUTPUT/'deletion_results.csv',float_precision='round_trip')
assert len(annual)==45 and set(annual.year)=={2023}
items=[];listrows=[]
for item in matrix(cfg):
    row=annual.set_index('experiment_id').loc[item['id']]
    reasons=[]
    if item['group'] in ('A','C'):reasons.append('A/C组第3步跨年反转；单列直接检验不能被组级结果替代')
    if item['group'] in ('D','F'):reasons.append('D源与F排名条件性效应；独立控制保留中间源')
    if row.delta_final_score>cfg['rules']['near_score']:reasons.append('2023可能删除，需要跨年验证是否退化')
    elif row.delta_final_score<-cfg['rules']['near_score']:reasons.append('2023可能应保留，需要跨年直接证据才可确定')
    else:reasons.append('2023效应接近，补齐年份观察是否稳定或冲突')
    if row.block_ci_spans_zero:reasons.append('2023配对块区间跨零，仍不确定')
    if not row.guards_pass:reasons.append('2023删除分项/月度/缺失诊断至少一项护栏不通过')
    reason='；'.join(reasons)
    listrows.append(dict(context=item['context'],feature=item['feature'],group=item['group'],
        screen_id=item['id'],score_delta_2023=row.delta_final_score,reason=reason,extra_runs=2))
    for split in ('dev_2021','dev_2022'):
        items.append({**item,'id':f'S4_{item["context"]}_{item["feature"]}_{split[-4:]}','split':split,'reason':reason,'screen_id':item['id']})
plan=dict(frozen_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
    rationale='完整45项2023结果已核验；对可能删除、可能保留、接近/冲突列全部补齐固定背景的两年直接证据；不增加候选或组合。此预算覆盖全名单，避免把未验证留在输入当作确定保留。',
    screen_runs=45,cross_budget=90,experiments=items)
assert len(items)==90
assert not (OUTPUT/'cross_plan.json').exists()
write_json(OUTPUT/'cross_plan.json',plan)
pd.DataFrame(listrows).to_csv(OUTPUT/'cross_review_list.csv',index=False,float_format='%.17g')
print('Frozen 45 conditional feature reviews, 90 additional 2021/2022 runs; no 2024')
