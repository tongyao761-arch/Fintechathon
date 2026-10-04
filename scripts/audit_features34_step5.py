"""Read actual models/predictions/official inputs and accept the fixed 2024 review."""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts import run_features34_step5 as runner
from scripts import run_features34_step4 as old
from scripts.run_features34_step3 import read_json, check_contract, validate_dates
from scripts.run_lightgbm_baseline import RAW_DATA_PATH, split_masks, score_saved_inputs, compare_metrics
from src.data.baseline_panel import load_raw_baseline_panel
from src.features.features34 import build_features34
from src.features.baseline_v1 import FEATURE_COLUMNS as BASE_COLUMNS
from src.metrics.diagnostics import baseline_diagnostics
from src.validation.experiment import sha256_file, write_json
from src.validation.splits import get_split


def markdown_table(frame, columns):
    def cell(x):
        if isinstance(x,(float,np.floating)): return f'{x:.9f}'
        return str(x).replace('|','/')
    return '\n'.join(['|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|',
        *['|'+'|'.join(cell(x) for x in row)+'|' for row in frame[columns].itertuples(index=False,name=None)]])


def report(tables, freeze):
    annual=tables['annual_comparison']; review=annual[annual.year==2024]
    conclusions=tables['candidate_conclusions']; lines=[
        '# 第5步：最终候选2024受限复核', '',
        '两个候选均在查看本次候选2024结果前固定；只新增两项训练，核验复用十特征和六项候选开发运行。2024此前已被查看，本步仅称受限使用的复核集，不称完全未接触的盲测。完成本步后停止，没有依2024搜索特征、改公式、调参或试新组合。', '',
        '## 结论', '']
    for row in conclusions.itertuples():
        lines += [f'- {row.candidate}：{row.next_stage}。2024相对十特征分差{row.delta_score_2024:+.9f}；四年最差分差{row.worst_annual_delta:+.9f}；负年度：{row.negative_score_years or "无"}。2024正月份{row.positive_months_2024}/12，最差月{row.worst_month_2024}为{row.worst_month_delta_2024:+.9f}。']
    lines += ['', '支持进入下一阶段仅指固定官方得分的四年证据，不等于已经证明最终替代baseline、每列必需、未来表现或实盘可交易稳定。第4步规则是看过开发单删后的探索性修订；2021—2023经过大量内部选择。', '',
        '## 事前冻结与执行口径', '',
        f'冻结时刻：{freeze["frozen_at"]}。精确清单、逐列公式、参数、选择理由、源码/配置来源及哈希保存在FROZEN_CANDIDATES.json；不可覆盖注册在registration.json。候选来源为第4步修订的已验收frozen_candidates.json，按2023官方分排序固定27列、31列顺序；本步不按2024重做特征选择。', '',
        '十特征骨架全部保留；沿用100棵LightGBM、学习率0.05、31叶、随机种子20260927及原参数，目标y_ret_1d。先在完整7,900,350行历史面板计算X特征，再投影列和筛训练资格；不使用标签/资格字段作为特征。排名源中间计算保留。2024训练20180102—20231228，purge 20231229；验证20240102—20241231，训练资格5,659,954行，全部1,125,300验证键保留。', '']
    for r in freeze['candidates']:
        lines += [f'- {r["candidate"]}（{len(r["features"])}列，背景{r["context"]}）：删除{", ".join(r["exclude"])}；lean31背景另外固定缺席A三列。' if r['context']=='lean31' else f'- {r["candidate"]}（{len(r["features"])}列，背景{r["context"]}）：删除{", ".join(r["exclude"])}。']
    lines += ['', '## 2024全部官方指标与分项贡献', '',
        markdown_table(review,['candidate','ic_mean','ic_std','icir','ic_positive_ratio','annual_excess','top1_annual_ret','mean_turnover','final_score','final_score_minus_baseline10']), '',
        markdown_table(review,['candidate','ic_contribution','excess_contribution','stability_contribution','ic_contribution_minus_baseline10','excess_contribution_minus_baseline10','stability_contribution_minus_baseline10']), '',
        '所有比率均用小数；分差是同年同切分候选减十特征。全年直接按官方评分计算，月度不平均替代全年。', '',
        '## 开发结果与2024合并', '',
        markdown_table(annual,['candidate','year','ic_mean','annual_excess','mean_turnover','final_score','final_score_minus_baseline10']), '',
        '跨年份退化区分：相对同年baseline转为负分差，才称不支持跨年替代；对照优势从2023到2024的缩小另行报告。年度绝对分数受到年份市场与样本构成变化影响，不能单凭绝对分降低判失败。', '',
        markdown_table(conclusions,['candidate','delta_2024_minus_delta_2023','negative_score_years','worst_annual_delta']), '',
        '## 2024月度表现', '',
        markdown_table(tables['monthly_comparison'].query('year==2024'),['candidate','month','ic','annual_excess','turnover','score','score_minus_baseline10']), '',
        '月度超额为当月日均×252，不是累计月收益；月首换手包含与前一个有效交易日比较。全部四年144行月度分项见monthly_comparison.csv；288行月度收益/换手Top计数、缺失占比及价格有效换手见monthly_missing_diagnostics.csv。', '',
        '## 缺失样本与价格有效换手', '',
        markdown_table(annual,['candidate','year','top_missing_label','top_invalid_price','top_baseline_all_missing','top_candidate_all_missing','price_valid_turnover','price_valid_turnover_minus_baseline10']), '',
        markdown_table(tables['top_membership_comparison'],['candidate','year','changed_members','changed_missing_label','missing_label_fraction_of_changes','changed_invalid_price','invalid_price_fraction_of_changes','missing_subset_changed_days','invalid_subset_changed_days']), '']
    for name in runner.IDS:
        r=review[review.candidate==name].iloc[0]
        member=tables['top_membership_comparison']; member=member[(member.candidate==name)&(member.year==2024)].iloc[0]
        lines += [f'{name}：2024 IC/收益/低换手的总分贡献变化分别{r.ic_contribution_minus_baseline10:+.9f}/{r.excess_contribution_minus_baseline10:+.9f}/{r.stability_contribution_minus_baseline10:+.9f}。Top成员变化中标签缺失占{member.missing_label_fraction_of_changes:.4%}、价格无效占{member.invalid_price_fraction_of_changes:.4%}。']
        if r.ic_contribution_minus_baseline10+r.excess_contribution_minus_baseline10 > 0 and r.stability_contribution_minus_baseline10 <= 0 and member.missing_label_fraction_of_changes < .01 and member.invalid_price_fraction_of_changes < .01:
            lines += ['证据不支持其涨分主要来自缺失样本排名：增益来自IC和收益，低换手贡献反而下降，缺失/价格无效成员变化占比很小。']
        else:
            lines += ['分项与Top成员变化不足以直接完成缺失排名的因果归因；保留不确定，不能只看缺失占比下结论。']
    lines += ['', '官方换手绝对水平仍受长期入选缺标签/价格无效行影响，尤其2021/2022；候选含涨跌停标记使全输入缺失占比可能为0，不等于价格有效。价格有效换手仅筛当日OHLC和涨停，不按未来标签筛选，不替代官方分数。Top集合对称差只描述入选变化，不是缺失贡献的精确因果分解；本步不运行降缺失排名策略。', '',
        '## 不确定性与逐列结论', '',
        markdown_table(tables['paired_intervals'],['candidate','year','block_ci_lower','block_ci_upper','block_ci_spans_zero']), '',
        '配对块区间沿用20交易日、2000次、种子20261004；块首换手剔除。它仅覆盖已固定预测的条件性日序列波动，不涵盖训练、选择、多重检验与真正未来分布风险，不能证明统计显著的最终替代。', '',
        '本步仅复核组合，不做2024单列删除；逐列背景结论原样沿用开发证据，不因2024重判。最终背景名单及实际模型输入状态见feature_decisions.csv：', '']
    decisions=tables['feature_decisions']
    for context in ('full34','lean31'):
        for decision in ('保留','删除','不确定'):
            names=decisions[(decisions.context==context)&(decisions.decision==decision)].feature.tolist()
            lines += [f'- {context}：{decision}{len(names)}列：{", ".join(names)}。']
    lines += ['', '十特征始终固定保留；候选移出列不自动等于确定删除，lean31缺席A三列仍不确定。背景相反或区间跨零的列无法统一成无条件名单。没有复核原full34/lean31的2024表现，因此本步不能证明联合删除相对原候选在2024仍改善，也不能验证候选2024排序的普遍性。', '',
        '## 留档与验收', '',
        '只新增2项模型运行；所有2024本地/官方标量差异≤1e-12，模型保存重载预测逐值一致，全部原始键、资格数和保存哈希核验。开发九项及冻结十特征2024核验复用，没有重训已完成实验。独立验收重新从2024保存CSV调用官方评分，复算月度/缺失诊断，从原始X面板重载模型并核对原预测。39个冻结文件、前四步产物、既有未跟踪文件和push保护保持哈希。', '',
        '准确命令：`python -B scripts/run_features34_step5.py prepare`，`python -B scripts/run_features34_step5.py run`，`python -B scripts/audit_features34_step5.py`；解释器为项目.venv。不可覆盖冻结；完成运行按run_index复用，不增加组合。失败保存在failures.json和commands，不生成成功摘要。', '',
        '证据：FROZEN_CANDIDATES.json、registration.json、run_index.json、annual_comparison.csv、monthly_comparison.csv、monthly_missing_diagnostics.csv、top_membership_comparison.csv、paired_intervals.csv、candidate_conclusions.csv、feature_decisions.csv、acceptance.json、summary.json、tests_result.json、dependency_check.json。完整模型/全部键预测/历史官方校验输入由run_index索引。评分用submission.csv仅为历史验证输入，没有生成最终比赛submission。', '',
        '本步只在本地ivor-work，无远程写入，完成后停止。']
    return '\n'.join(lines)+'\n'


def main():
    started=time.perf_counter(); cfg=read_json(runner.CONFIG)
    manifest,_,meta=check_contract(read_json(ROOT/cfg['development_config']))
    frozen=runner.registration_check(cfg,meta)
    state=read_json(runner.OUTPUT/'status.json')
    if state['status'] not in ('runs_complete_audit_pending','success'):
        raise AssertionError('review runs not complete')
    all_entries=runner.entries(cfg,meta)
    panel=load_raw_baseline_panel(RAW_DATA_PATH)
    splits=validate_dates(panel.trade_date.unique(),read_json(ROOT/cfg['development_config']))
    splits['oos_2024']=get_split('oos_2024')
    full=build_features34(panel)
    verified=[]
    for (name,s),(d,sm) in all_entries.items():
        r=sm['splits'][0]; split=splits[s]; train,valid=split_masks(panel,split)
        keys=panel.loc[valid,['ts_code','trade_date']].reset_index(drop=True)
        saved=pd.read_parquet(d/s/'predictions.parquet')
        pd.testing.assert_frame_equal(saved[['ts_code','trade_date']],keys,check_dtype=False,check_categorical=False)
        assert r['train_samples']==int(train.sum()) and r['valid_prediction_rows']==int(valid.sum())
        assert not train[panel.trade_date.eq(split.purge_date)].any()
        assert r['dates']=={k:getattr(split,k) for k in ('train_start','train_end','purge_date','valid_start','valid_end')}
        dd=old.daily(d,s); actual=.4*dd.ic.mean()+.3*252*dd.excess.mean()+.3*(1-dd.turnover.mean())
        assert abs(actual-r['metrics']['final_score'])<=1e-12
        item=dict(candidate=name,split=s,directory=d.relative_to(ROOT).as_posix(),summary_sha256=sha256_file(d/'summary.json'),all_keys_verified=True,train_samples=int(train.sum()),day_score_error=abs(actual-r['metrics']['final_score']))
        if s=='oos_2024':
            model=lgb.Booster(model_file=str(d/s/'models/lightgbm.txt'))
            assert model.feature_name()==sm['features']
            assert np.array_equal(model.predict(full.loc[valid,tuple(sm['features'])]),saved.pred.to_numpy())
            metrics,comparison,pred,truth=score_saved_inputs(d/s/'evaluate_input'); details=metrics.pop('details')
            compare_metrics(r['metrics'],metrics)
            rawtruth=panel.loc[valid,['ts_code','trade_date','y_ret_1d']].reset_index(drop=True)
            pd.testing.assert_frame_equal(truth,rawtruth,check_dtype=False,check_categorical=False,check_exact=True)
            quality=panel.loc[valid,['ts_code','trade_date','flag_limit_up','is_price_valid']].copy()
            quality['baseline_features_all_missing']=full.loc[valid,BASE_COLUMNS].isna().all(axis=1)
            diag,dt=baseline_diagnostics(pred,truth,quality,details)
            for k in ('top_groups','price_valid_only_turnover'): assert diag[k]==r['diagnostics'][k]
            for filename,frame in dt.items():
                normalized=pd.read_csv(io.StringIO(frame.to_csv(index=False,float_format='%.17g')),float_precision='round_trip')
                disk=pd.read_csv(d/s/(filename+'.csv'),float_precision='round_trip')
                pd.testing.assert_frame_equal(normalized,disk,check_exact=True,check_dtype=False)
            if name!='baseline10':
                prior=read_json(d/'provenance.json')
                for k in ('data','dependencies','source_sha256'): assert prior[k]==frozen['source'][k]
                assert sm['freeze_sha256']==sha256_file(runner.OUTPUT/'FROZEN_CANDIDATES.json')
                assert read_json(d/'config.json')['freeze_sha256']==sm['freeze_sha256']
                run_time=pd.to_datetime(read_json(d/'status.json')['run_id'].split('_')[0],format='%Y%m%dT%H%M%S%fZ',utc=True)
                assert pd.Timestamp(read_json(runner.OUTPUT/'registration.json')['registered_at']) < run_time
                cq=quality.copy(); cq['baseline_features_all_missing']=full.loc[valid,tuple(sm['features'])].isna().all(axis=1)
                cd,ct=baseline_diagnostics(pred,truth,cq,details)
                assert cd['top_groups']==r['diagnostics']['candidate_features_top_groups']
                disk=pd.read_csv(d/s/'daily_candidate_missing_diagnostics.csv',float_precision='round_trip')
                normalized=pd.read_csv(io.StringIO(ct['daily_missing_diagnostics'].to_csv(index=False,float_format='%.17g')),float_precision='round_trip')
                pd.testing.assert_frame_equal(normalized,disk,check_exact=True,check_dtype=False)
                stats=pd.read_csv(d/s/'feature_missing_statistics.csv',float_precision='round_trip')
                rebuilt=runner.feature_statistics(full.loc[:,tuple(sm['features'])],train_mask=train,valid_mask=valid)
                pd.testing.assert_frame_equal(stats,rebuilt,check_exact=True,check_dtype=False)
            item.update(reload_predictions_equal=True,official_rescore_max_abs_difference=comparison['max_abs_difference'],diagnostics_recomputed=True)
        verified.append(item)
    tables=runner.comparison_tables(all_entries,cfg)
    tables['top_membership_comparison']=runner.membership_table(all_entries,panel)
    tables['candidate_conclusions']=runner.conclusion_tables(tables)
    decisions=pd.read_csv(runner.revision.OUTPUT/'single_decisions.csv',float_precision='round_trip').fillna('')
    context_to_candidate={r['context']:r for r in frozen['candidates']}
    decisions['review_candidate']=decisions.context.map({k:v['candidate'] for k,v in context_to_candidate.items()})
    decisions['in_frozen_model_input']=[row.feature in context_to_candidate[row.context]['features'] for row in decisions.itertuples()]
    decisions['step5_evidence_scope']='development conditional single-column conclusions retained; 2024 joint review does not reclassify columns'
    tables['feature_decisions']=decisions
    assert len(tables['annual_comparison'])==12 and len(tables['monthly_comparison'])==144 and len(tables['monthly_missing_diagnostics'])==288
    assert len(decisions)==48 and len(tables['candidate_conclusions'])==2
    # Independent arithmetic and conclusion checks; total is never reconstructed from monthly averages.
    for row in tables['annual_comparison'].itertuples():
        assert abs(.4*row.ic_mean+.3*row.annual_excess+.3*(1-row.mean_turnover)-row.final_score)<=1e-12
        assert abs(row.ic_contribution_minus_baseline10+row.excess_contribution_minus_baseline10+row.stability_contribution_minus_baseline10-row.final_score_minus_baseline10)<=1e-12
    for row in tables['candidate_conclusions'].itertuples():
        deltas=[all_entries[(row.candidate,s)][1]['splits'][0]['metrics']['final_score']-all_entries[('baseline10',s)][1]['splits'][0]['metrics']['final_score'] for s in splits]
        assert row.all_four_years_above_baseline10==all(x>0 for x in deltas)
    for name in ('tests_result','dependency_check'):
        assert read_json(runner.OUTPUT/(name+'.json'))['exit_code']==0
    runner.registration_check(cfg,runner.provenance(ROOT,RAW_DATA_PATH)); runner.check_frozen(manifest)
    for name,table in tables.items(): table.to_csv(runner.OUTPUT/(name+'.csv'),index=False,float_format='%.17g')
    body=report(tables,frozen)
    (runner.OUTPUT/'REPORT.md').write_text(body,encoding='utf-8')
    (ROOT/'docs/features34/STEP5_REPORT.md').write_text(body,encoding='utf-8')
    evidence={p.name:sha256_file(p) for p in runner.OUTPUT.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md') and p.name not in ('acceptance.json','summary.json','status.json')}
    accepted=dict(accepted=True,stage=cfg['stage'],new_training_runs=2,verified_runs=verified,
        candidates_frozen_before_review=True,review_scope=cfg['review_scope'],frozen_files_verified=len(manifest),
        old_evidence_and_untracked_and_push_protections_unchanged=True,all_keys_and_eligibility_verified=True,
        official_2024_scores_and_diagnostics_recomputed=True,saved_model_predictions_recomputed=True,
        evidence_sha256=evidence,elapsed_seconds=time.perf_counter()-started,failures=read_json(runner.OUTPUT/'failures.json')['failures'])
    write_json(runner.OUTPUT/'acceptance.json',accepted)
    write_json(runner.OUTPUT/'summary.json',dict(accepted=True,stage=cfg['stage'],new_runs=2,reused_development_runs=9,reused_2024_baseline=1,
        candidates=tables['candidate_conclusions'].to_dict('records'),review_scope=cfg['review_scope'],final_submission_generated=False,
        acceptance_sha256=sha256_file(runner.OUTPUT/'acceptance.json')))
    state.update(status='success',accepted=True,audit_elapsed_seconds=time.perf_counter()-started); write_json(runner.OUTPUT/'status.json',state)
    print(tables['candidate_conclusions'].to_string(index=False),flush=True)
    print(f'Accepted fixed step5 review: {time.perf_counter()-started:.1f}s.',flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        runner.failure('audit',exc); raise
