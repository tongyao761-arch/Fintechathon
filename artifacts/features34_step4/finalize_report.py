"""Create conditional 24-feature report and freeze no more than two candidates."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.run_features34_step4 import OUTPUT,CONFIG,read_json,write_json,select_features,NEW_COLUMNS,GROUPS
from src.validation.experiment import sha256_file

cfg=read_json(CONFIG);r=cfg['rules']
annual=pd.read_csv(OUTPUT/'deletion_results.csv',float_precision='round_trip')
decisions=pd.read_csv(OUTPUT/'feature_decisions.csv',keep_default_na=False,float_precision='round_trip')
monthly=pd.read_csv(OUTPUT/'monthly_deletion_results.csv',float_precision='round_trip')
original=pd.read_csv(ROOT/'artifacts/features34_step3/comparison.csv',float_precision='round_trip')
om=pd.read_csv(ROOT/'artifacts/features34_step3/monthly_comparison.csv',float_precision='round_trip')
plan=read_json(OUTPUT/'joint_plan.json');runs=read_json(OUTPUT/'run_index.json')['runs']
assert len(annual)==135+len(plan['experiments'])
details=[]
for _,dr in decisions.iterrows():
    fs=annual[(annual.context==dr.context)&(annual.feature==dr.feature)&(annual.kind=='single')].sort_values('year')
    years=lambda mask:';'.join(str(int(y)) for y in fs.loc[mask,'year'])
    details.append(dict(context=dr.context,feature=dr.feature,decision=dr.decision,
        experiment_ids=';'.join(fs.experiment_id),direct_years=';'.join(str(int(y)) for y in fs.year),
        near_years=years(fs.delta_final_score.abs()<=r['near_score']),
        annual_direction_conflict=bool((fs.delta_final_score>0).any() and (fs.delta_final_score<0).any()),
        block_ci_spans_zero_years=years((fs.block_ci_lower<=0)&(fs.block_ci_upper>=0)),
        deletion_guard_failure_years=years(~fs.guards_pass),
        clear_degradation_years=years(fs.delta_final_score < -r['clear_year_degradation']),
        input_present_in_frozen_start=dr.currently_in_original_input,
        evidence_limit='固定背景原候选未含A，不代表单列已证实可删' if len(fs)==0 else
            ('三年直接负增量且区间上界<0；官方换手含缺标签行，保留只针对该评分口径' if dr.decision=='保留' else
             '接近/方向冲突/区间跨零/删除护栏未通过；有年度负增量也不能自动确定必需')))
pd.DataFrame(details).to_csv(OUTPUT/'decision_evidence_details.csv',index=False)
frozen=[];comparison=[];monthrows=[]
for ctx in cfg['contexts']:
    ver=next((v for v in plan['versions'] if v['context']==ctx),None)
    passed=False
    if ver:
        frame=annual[(annual.kind=='joint')&(annual.context==ctx)]
        assert set(frame.year)=={2021,2022,2023}
        passed=bool((frame.delta_final_score>=0).all() and frame.guards_pass.all() and not frame.clear_year_degradation.any())
    name=ver['version'] if passed else ctx
    removed=ver['exclude'] if passed else []
    candidate=dict(candidate=name,original_context=ctx,exclude=removed,
        features=list(select_features(groups=cfg['contexts'][ctx],exclude=removed)),
        joint_tested=ver is not None,joint_passed=passed,
        selection_reason='联合三年验证及预登记护栏通过' if passed else ('联合未通过，回到原候选' if ver else '无满足预登记的确定删除列，保留原候选'),
        individual_decisions_file='feature_decisions.csv',
        input_retention_does_not_establish_necessity=True)
    frozen.append(candidate)
    for year in (2021,2022,2023):
        base=original[(original.candidate=='baseline10')&(original.year==year)].iloc[0]
        parent=original[(original.candidate==ctx)&(original.year==year)].iloc[0]
        if passed:
            a=frame[frame.year==year].iloc[0]
            vals={k:a['after_'+k] for k in ['ic_mean','annual_excess','mean_turnover','final_score']}
            experiment_id=a.experiment_id;directory=a.directory
            mm=monthly[monthly.experiment_id==experiment_id].copy()
            delta=mm.score_minus_baseline10
            excess_delta=mm.annual_excess_minus_baseline10
            parent_delta=mm.delta_score
            ci=(a.block_ci_lower,a.block_ci_upper)
        else:
            vals={k:parent[k] for k in ['ic_mean','annual_excess','mean_turnover','final_score']}
            experiment_id=f'step3_{ctx}_{year}';directory=parent.directory
            mm=om[(om.candidate==ctx)&(om.year==year)].copy()
            delta=mm.score_minus_baseline10;excess_delta=mm.annual_excess_minus_baseline10
            parent_delta=pd.Series([0.]*12,index=mm.index);ci=(0.,0.)
        pos_excess=excess_delta.clip(lower=0)
        comparison.append(dict(candidate=name,original_context=ctx,year=year,experiment_id=experiment_id,directory=directory,
            **vals,score_minus_baseline10=vals['final_score']-base.final_score,
            score_minus_parent=vals['final_score']-parent.final_score,
            positive_months_vs_baseline=int((delta>0).sum()),worst_month_vs_baseline=float(delta.min()),
            positive_months_vs_parent=int((parent_delta>0).sum()),worst_month_vs_parent=float(parent_delta.min()),
            largest_positive_excess_month_share=None if pos_excess.sum()==0 else float(pos_excess.max()/pos_excess.sum()),
            parent_block_ci_lower=ci[0],parent_block_ci_upper=ci[1]))
    candrows=comparison[-3:]
    candidate['worst_year_delta_vs_parent']=min(x['score_minus_parent'] for x in candrows)
    candidate['worst_year_delta_vs_baseline10']=min(x['score_minus_baseline10'] for x in candrows)

existing_frozen=read_json(OUTPUT/'frozen_candidates.json') if (OUTPUT/'frozen_candidates.json').exists() else None
if existing_frozen:assert existing_frozen['candidates']==frozen
write_json(OUTPUT/'frozen_candidates.json',dict(frozen_at=existing_frozen['frozen_at'] if existing_frozen else pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
    candidates=frozen,used_2024=False,addback_versions=0,
    individual_evidence_scope='固定背景逐列直接证据；A在lean31中缺席不能解释为每列确定删除',
    next_step_authorized=False))
pd.DataFrame(comparison).to_csv(OUTPUT/'frozen_candidate_comparison.csv',index=False,float_format='%.17g')

lines=['# 第4步：逐特征删除诊断与跨年复核','',
    f'完成两个固定背景的45项2023单删及90项2021/2022直接复核；联合版本{len(plan["versions"])}个，回补0个。复用第3步同年十特征/full34/lean31九项参照，没有重做。未使用2024训练、评分或筛选，没有生成最终比赛预测。','',
    '判断方向均为删除后减删除前。确定结论仅适用于表中背景；暂留输入不代表证明必需。原十特征骨架保留。','',
    '## 运行前规则及不确定性','',
    '接近±0.001，单年明显退化<−0.005；不是沿用第3步0.003排序阈值。确定删除要求三个开发年分差均≥0.001、配对块95%下界均>0且全部分项/月度/缺失护栏通过；确定保留要求三年分差均≤−0.001且区间上界均<0。其他为不确定。完整规则在STEP4_PREREGISTRATION.md及配置，并于训练前保存哈希。','',
    '分项护栏：IC下降不超过0.002、年化超额下降不超过0.01、官方/价格有效换手增加不超过0.01、四种Top缺失占比增加不超过0.01。月度护栏：每年≥8/12正分差月份、最差月≥−0.03、正超额增量最大月占比≤50%。护栏是保守研究约定，不能解释为统计等效或实盘可交易证明。','',
    '所有删除比较使用对齐逐日IC、超额及官方换手的20交易日非循环连续块，2000次，固定种子20261004，95%百分位区间。每块首日换手剔除，不把重采样块之间的持仓拼接；内部保持原始相邻交易日换手。固定预测的配对重采样只解释条件性采样波动，不涵盖训练/跨年/选择不确定性，也不是多重检验后的显著性证明。','',
    '## 24列的背景条件结论','',
    '每个单元列出结论及2021/2022/2023删除分差。完整分项、两种参照差、月度风险、缺失占比与区间在deletion_results.csv；逐月分差在monthly_deletion_results.csv。实验ID精确模式为`S4_{背景}_{特征}_{年份}`，每项实际目录和摘要哈希在run_index.json。','',
    '|组|新增特征|full34：结论；三年删除分差|lean31：结论；三年删除分差|','|---|---|---|---|']
for c in NEW_COLUMNS:
    cells=[]
    for ctx in cfg['contexts']:
        dr=decisions[(decisions.context==ctx)&(decisions.feature==c)].iloc[0]
        fs=annual[(annual.context==ctx)&(annual.feature==c)&(annual.kind=='single')].sort_values('year')
        if len(fs)==3:
            cell=dr.decision+'；'+'/'.join(f'{v:+.6f}' for v in fs.delta_final_score)
        else:cell='不确定；固定起点已不含A，无单列删除直接证据'
        cells.append(cell)
    group=next(g for g,cs in GROUPS.items() if c in cs)
    lines.append(f'|{group}|{c}|{cells[0]}|{cells[1]}|')

lines+=['','## 暂留输入与证据限制','']
for ctx in cfg['contexts']:
    frame=decisions[decisions.context==ctx]
    counts=frame.decision.value_counts().to_dict()
    kept=frame[frame.decision=='保留'].feature.tolist();removed=frame[frame.decision=='删除'].feature.tolist()
    lines.append(f'- {ctx}：有直接证据保留{counts.get("保留",0)}列，确定删除{counts.get("删除",0)}列，不确定{counts.get("不确定",0)}列。保留名单：'+('、'.join(kept) or '无')+'；删除名单：'+('、'.join(removed) or '无')+'。')
lines+=['','每列的不确定触发年份、方向冲突、配对区间跨零年份、明显退化年份、护栏失败年份及实际实验ID，单独列于decision_evidence_details.csv。A/C组已有跨年组级冲突，当前保留每列各背景结论，不合并无条件结论。D源与F排名的单删分别测试，删除log_mean_amount_20d模型列时，排名源中间值仍计算；没有源/排名同时删除的直接实验，不能推断其替代性。','',
    'ret_1d_rank_pct虽在两个背景三年单删均降分，但full34的2021/2023、lean31的2022/2023配对区间跨零，所以仍为不确定并暂留输入。flag_limit_down删除的六项综合分差均明确为负，官方换手增加约0.36–0.63、Top缺标签占比降到约0.03%–0.07%；其主要作用来自当前官方含缺失行的换手集合，不能解释为收益预测能力同幅改善。','',
    '2023结果先完成并保存后，才冻结cross_review_list.csv及cross_plan.json。复核覆盖全部45个背景—列组合、补齐90项，以免把未复核列写成确定保留。名单列出逐项2023信号、A/C跨年争议、源/排名依赖和复核理由，没有引入其他起点或组合搜索。','',
    '## 联合删除与冻结候选','']
if not plan['versions']:
    lines.append('没有满足确定删除规则的列，因此未构造联合删除版本，冻结两个原候选。没有用“几个单删看似接近”推断联合删除安全；没有明确替代/交互依据，未使用回补额度。')
else:
    for v in plan['versions']:
        f=next(x for x in frozen if x['original_context']==v['context'])
        lines.append(f'- {v["version"]}删除'+ '、'.join(v['exclude'])+f'；三年联合验证'+('通过，冻结该组合。' if f['joint_passed'] else '未通过，冻结原候选。'))
lines+=['','|冻结候选|年份|IC|年化超额|官方换手|综合分|相对同年10|相对原候选|','|---|---:|---:|---:|---:|---:|---:|---:|']
for row in comparison:
    lines.append(f'|{row["candidate"]}|{row["year"]}|{row["ic_mean"]:.6f}|{row["annual_excess"]:.6f}|{row["mean_turnover"]:.6f}|{row["final_score"]:.6f}|{row["score_minus_baseline10"]:+.6f}|{row["score_minus_parent"]:+.6f}|')
lines+=['','三年逐年证据优先于平均值；相对10涨分不能证明删除操作有帮助。上表保留与原候选的直接差值。','',
    '|冻结候选|最差年度相对10|最差年度相对原候选|正月数/36（对10）|最差月差（对10）|','|---|---:|---:|---:|---:|']
for f in frozen:
    rows=[x for x in comparison if x['candidate']==f['candidate']]
    lines.append(f'|{f["candidate"]}|{f["worst_year_delta_vs_baseline10"]:+.6f}|{f["worst_year_delta_vs_parent"]:+.6f}|{sum(x["positive_months_vs_baseline"] for x in rows)}|{min(x["worst_month_vs_baseline"] for x in rows):+.6f}|')
lines+=['','## 缺失样本与月度风险','',
    '原候选2021换手Top缺标签/价格无效几乎100%，2022约84.6%–85.5%，2023约56.15%；低官方换手主要描述含大量缺失样本的集合，不能当作可交易股票稳定。完整四种缺失占比、删除前后差及价格有效换手逐项保存，日级细表仍在不可覆盖运行目录。涨跌停标记可使候选全缺失占比为0，这不代表价格有效。价格有效换手仅诊断，不替代官方评分。','',
    '冻结候选仍有负月份、年度方向及输入必要性不确定。各年正超额增量最大月占比和月度差在frozen_candidate_comparison.csv。月首换手保持与上月末的真实边界；月度分数不能平均替代全年官方分数。三年开发样本内选择也不证明未来表现。','',
    '## 证据与验收','',
    '- 预登记：configs/features34_step4.json、docs/features34/STEP4_PREREGISTRATION.md、preregistration.json、screen_matrix.json、各phase_registration.json。','- 复核预算与联合冻结：cross_plan.json、cross_review_list.csv、joint_plan.json；回补0，额度未使用。','- 完整结果与条件名单：deletion_results.csv、monthly_deletion_results.csv、feature_decisions.csv、frozen_candidates.json、frozen_candidate_comparison.csv。','- 完整实验：run_index.json；各目录模型、预测、原始评分输入、status/config/provenance、官方比较、逐日/月度/缺失诊断、特征缺失统计、资源和执行日志。','- 原参照复用、源码快照、保护与未跟踪文件：preregistration.json、executed_sources/、preflight.json。','- 测试/依赖及失败：tests_result.json、tests_result.log、dependency_check.json、dependency_check.log、failures.json、auxiliary_failures.json及各阶段status。','- 独立验收：scripts/audit_features34_step4.py、acceptance.json、summary.json；实际检查原始验证键、同年训练数、全部文件哈希、模型输入、有限预测、日级重建官方分数、汇总及背景名单重算。','',
    '首次独立审计因把阶段变化的Git工作区status也要求完全相同而失败，日志audit.log及命令记录保留，未生成成功摘要。仅修正审计器，严格比较实际执行源码、数据、依赖和Git分支/提交；保留执行时原审计源码快照，并用audit_source_correction.json登记原版/修正版精确哈希。训练、特征、评分、阈值和矩阵源码未改。模型运行失败0，辅助审计失败1。','',
    '复现入口：`python -B scripts/run_features34_step4.py screen/cross/joint`按阶段执行并核验已完成ID后复用，phase计划和规则拒绝变更。已完成的实验不是重跑待办。独立验收：`python -B scripts/audit_features34_step4.py`。','',
    '第4步完成后停止；2024少量复核和后续步骤未执行、未获本次授权。']
report='\n'.join(lines)+'\n'
(ROOT/'docs/features34/STEP4_REPORT.md').write_text(report,encoding='utf-8')
(OUTPUT/'STEP4_REPORT.md').write_text(report,encoding='utf-8')
print('Frozen:',[f['candidate'] for f in frozen])
