"""Exploratory score-focused revision; two immutable joint versions, no 2024."""
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
import numpy as np
import pandas as pd

from scripts import run_features34_step4 as old
from scripts.run_features34 import run_split, check_frozen, feature_statistics
from scripts.run_features34_step3 import read_json, check_contract, validate_dates, SPLITS
from scripts.run_lightgbm_baseline import MODEL_PARAMS, RAW_DATA_PATH, EXPERIMENT_ROOT, PeakMemoryMonitor, split_masks, environment_versions
from src.data.baseline_panel import load_raw_baseline_panel
from src.features.features34 import build_features34, select_features
from src.validation.experiment import experiment_run, provenance, sha256_file, write_json

CONFIG = ROOT / 'configs/features34_step4_revision.json'
RULES = ROOT / 'docs/features34/STEP4_REVISION_RULES.md'
OUTPUT = ROOT / 'artifacts/features34_step4_revision'
VERSIONS = [
    dict(candidate='S4R_full34_minus3', context='full34', exclude=['ret_10d', 'limit_up_count_5d', 'volatility_20d_rank_pct']),
    dict(candidate='S4R_lean31_minus4', context='lean31', exclude=['price_position_20d', 'lower_shadow', 'ret_5d_rank_pct', 'volatility_20d_rank_pct']),
]


def validate_config(cfg):
    if cfg['stage'] != 'step4_score_revision_development_only' or cfg['used_2024'] or cfg['versions'] != VERSIONS:
        raise ValueError('revision scope or fixed versions changed')
    if cfg['contexts'] != old.read_json(old.CONFIG)['contexts']:
        raise ValueError('fixed starting candidates changed')
    if cfg['rules']['bootstrap'] != {k: old.read_json(old.CONFIG)['rules']['bootstrap'][k] for k in ('block_trading_days', 'repetitions', 'seed', 'confidence')}:
        raise ValueError('bootstrap configuration changed')
    if cfg['budgets'] != dict(new_joint_versions=2, new_training_runs=6, new_addback_versions=0, stage_joint_versions_max=2, stage_addback_versions_max=2):
        raise ValueError('budget changed')
    if set(cfg) != {'stage', 'revision_type', 'step3_config', 'old_config', 'contexts', 'rules', 'versions', 'budgets', 'used_2024'}:
        raise ValueError('unsupported config or tuning')


def matrix(cfg):
    validate_config(cfg)
    return [dict(id=f'{v["candidate"]}_{s[-4:]}', version=v['candidate'], context=v['context'],
                 exclude=v['exclude'], split=s, kind='joint') for v in cfg['versions'] for s in SPLITS]


def columns(cfg, item):
    if item not in matrix(cfg):
        raise ValueError('run outside fixed six experiments')
    return select_features(groups=cfg['contexts'][item['context']], exclude=item['exclude'])


def classify(frame, r):
    if len(frame) != 3 or set(frame.year) != {2021, 2022, 2023} or not np.isfinite(frame.delta_final_score).all():
        return '不确定'
    if (frame.delta_final_score < -r['near_score']).all():
        return '保留'
    if (frame.delta_final_score > r['near_score']).all():
        return '删除'
    return '不确定'


def probe(frame, r):
    return bool(len(frame) == 3 and set(frame.year) == {2021, 2022, 2023} and
                frame.delta_final_score.min() >= -r['near_score'] and
                frame.delta_final_score.mean() > r['near_score'] and
                frame.loc[frame.year == 2023, 'delta_final_score'].iloc[0] > r['near_score'])


def joint_gate(frame, r):
    return bool(len(frame) == 3 and set(frame.year) == {2021, 2022, 2023} and
                np.isfinite(frame[['after_final_score', 'delta_final_score', 'after_final_score_minus_baseline10']]).all().all() and
                frame.loc[frame.year == 2023, 'delta_final_score'].iloc[0] > r['near_score'] and
                frame.delta_final_score.mean() > 0 and frame.delta_final_score.min() >= -r['clear_year_degradation'] and
                (frame.after_final_score_minus_baseline10 > 0).all())


def order_candidates(records, tolerance):
    remaining = [r for r in records if r['eligible']]
    ordered = []
    while remaining:
        best = max(r['score_2023'] for r in remaining)
        tied = [r for r in remaining if best - r['score_2023'] <= tolerance]
        chosen = sorted(tied, key=lambda r: (-r['worst_delta_vs_baseline10'], -r['mean_score'], r['feature_count'], r['candidate']))[0]
        ordered.append(chosen); remaining = [r for r in remaining if r['candidate'] != chosen['candidate']]
    return ordered


def preserve_old():
    snap = read_json(OUTPUT / 'preflight.json')
    for collection in ('protections', 'existing_untracked', 'old_step4_files', 'old_rule_files'):
        for name, expected in snap[collection].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha256_file(path) != expected:
                raise AssertionError(f'preserved file changed: {name}')


def verify_old_evidence(meta, deep=False):
    accepted = read_json(old.OUTPUT / 'acceptance.json')
    if not accepted['accepted'] or not read_json(old.OUTPUT / 'summary.json')['accepted']:
        raise AssertionError('old stage not accepted')
    for name, expected in accepted['evidence_sha256'].items():
        if sha256_file(old.OUTPUT / name) != expected:
            raise AssertionError(f'old evidence changed: {name}')
    prior = read_json(old.OUTPUT / 'preregistration.json')
    correction = read_json(old.OUTPUT / 'audit_source_correction.json')
    for name, expected in prior['source']['source_sha256'].items():
        if sha256_file(old.OUTPUT / 'executed_sources' / name) != expected:
            raise AssertionError('executed old snapshot changed')
        current = correction['corrected_sha256'] if name == 'scripts/audit_features34_step4.py' else expected
        if sha256_file(ROOT / name) != current:
            raise AssertionError(f'old execution dependency changed: {name}')
    if prior['source']['data'] != meta['data'] or prior['source']['dependencies'] != meta['dependencies']:
        raise AssertionError('reuse data/dependency contract changed')
    if read_json(old.OUTPUT / 'joint_plan.json')['versions'] or read_json(old.OUTPUT / 'frozen_candidates.json')['addback_versions']:
        raise AssertionError('old joint/addback budget already used; revise scope before running')
    data = pd.read_csv(old.OUTPUT / 'deletion_results.csv', float_precision='round_trip')
    records = read_json(old.OUTPUT / 'run_index.json')['runs']
    if len(data) != 135 or len(records) != 135 or set(data.experiment_id) != {x['item']['id'] for x in records}:
        raise AssertionError('old single matrix incomplete')
    for rec in records:
        d = ROOT / rec['directory']; item = rec['item']
        if sha256_file(d / 'summary.json') != rec['summary_sha256']:
            raise AssertionError('reused summary changed')
        sm = old.verify_artifact(d, item['split'], old.selection(read_json(old.CONFIG), item)) if deep else read_json(d / 'summary.json')
        row = data[data.experiment_id == item['id']].iloc[0]
        if sm['splits'][0]['metrics']['final_score'] != row.after_final_score:
            raise AssertionError('old table/model result mismatch')
    return data, records


def prepare():
    cfg = read_json(CONFIG); validate_config(cfg)
    manifest, _, meta = check_contract(read_json(ROOT / cfg['step3_config']))
    preserve_old()
    if (OUTPUT / 'registration.json').exists():
        registration = read_json(OUTPUT / 'registration.json')
        if registration['config_sha256'] != sha256_file(CONFIG) or registration['rules_sha256'] != sha256_file(RULES) or registration['source']['source_sha256'] != meta['source_sha256']:
            raise AssertionError('registered revision changed')
        return
    data, reused = verify_old_evidence(meta, deep=True)
    refs, refrecords = old.references(read_json(old.CONFIG), meta)
    decisions, proposals = single_tables(data, cfg)
    selected = {(v['context'], f) for v in cfg['versions'] for f in v['exclude']}
    proposed = {(r.context, r.feature) for r in proposals[proposals.probe_eligible].itertuples()}
    if selected != proposed:
        raise AssertionError('frozen joint lists differ from probe rule')
    write_json(OUTPUT / 'registration.json', dict(config=cfg, config_sha256=sha256_file(CONFIG), rules_sha256=sha256_file(RULES),
        source=meta, original_step4_acceptance_sha256=sha256_file(old.OUTPUT / 'acceptance.json'),
        reused_single_runs=reused, reference_runs=refrecords, registered_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        observed_prior_results=True, new_training_started=False))
    for name in meta['source_sha256']:
        target = OUTPUT / 'executed_sources' / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, target)
    shutil.copy2(RULES, OUTPUT / 'STEP4_REVISION_RULES.md')
    write_json(OUTPUT / 'matrix.json', dict(direction='after minus original same-year candidate', experiments=matrix(cfg),
        joint_versions_before=0, joint_versions_planned=2, addback_versions_before=0, addback_versions_planned=0))
    write_json(OUTPUT / 'run_index.json', dict(runs=[]))
    write_json(OUTPUT / 'failures.json', dict(failures=[]))
    decisions.to_csv(OUTPUT / 'single_decisions.csv', index=False, float_format='%.17g')
    proposals.to_csv(OUTPUT / 'probe_evidence.csv', index=False, float_format='%.17g')
    check_frozen(manifest)
    print('Registered exploratory rule revision, 135 verified singles, 9 references, 2 fixed joint versions.', flush=True)


def single_tables(data, cfg):
    previous = pd.read_csv(old.OUTPUT / 'feature_decisions.csv').fillna('')
    decisions = []; proposals = []
    for prev in previous.to_dict('records'):
        g = data[(data.context == prev['context']) & (data.feature == prev['feature'])].sort_values('year')
        complete = len(g) == 3 and set(g.year) == {2021, 2022, 2023}
        decision = classify(g, cfg['rules'])
        near = g.loc[g.delta_final_score.abs() <= cfg['rules']['near_score'], 'year'].tolist()
        conflict = bool(len(g) and (g.delta_final_score > 0).any() and (g.delta_final_score < 0).any())
        record = dict(context=prev['context'], feature=prev['feature'], group=prev['group'], decision=decision,
            old_decision=prev['decision'], direct_years=';'.join(str(x) for x in g.year),
            experiment_ids=';'.join(g.experiment_id), near_years=';'.join(str(x) for x in near),
            annual_direction_conflict=conflict, block_ci_zero_years=';'.join(str(x) for x in g.loc[g.block_ci_spans_zero, 'year']),
            old_guards_failed_years=';'.join(str(x) for x in g.loc[~g.guards_pass, 'year']),
            statistical_uncertainty=bool(not complete or g.block_ci_spans_zero.any()),
            evidence_scope='conditional direct annual score evidence; not statistical significance or joint necessity',
            absent_from_original_input=not bool(prev['currently_in_original_input']))
        for _, row in g.iterrows():
            record[f'delta_{row.year}'] = row.delta_final_score
        decisions.append(record)
        if complete:
            proposals.append(dict(context=prev['context'], feature=prev['feature'], decision=decision,
                probe_eligible=probe(g, cfg['rules']), min_annual_delta=float(g.delta_final_score.min()),
                mean_annual_delta=float(g.delta_final_score.mean()), delta_2023=float(g[g.year == 2023].delta_final_score.iloc[0]),
                reason='fixed score rule; near rows remain uncertain even if joint exploration eligible',
                experiment_ids=record['experiment_ids']))
    return pd.DataFrame(decisions), pd.DataFrame(proposals)


def registration_check(cfg, meta):
    reg = read_json(OUTPUT / 'registration.json')
    if reg['config_sha256'] != sha256_file(CONFIG) or reg['rules_sha256'] != sha256_file(RULES) or reg['source']['source_sha256'] != meta['source_sha256'] or reg['source']['data'] != meta['data']:
        raise AssertionError('revision sources/config/data changed after joint registration')
    if read_json(OUTPUT / 'matrix.json')['experiments'] != matrix(cfg):
        raise AssertionError('joint matrix changed')
    preserve_old()
    return reg


def run():
    started = time.perf_counter(); cfg = read_json(CONFIG); validate_config(cfg)
    manifest, reference, meta = check_contract(read_json(ROOT / cfg['step3_config']))
    registration_check(cfg, meta)
    refs, _ = old.references(read_json(old.CONFIG), meta)
    write_json(OUTPUT / 'run_status.json', dict(status='running', planned=6))
    with PeakMemoryMonitor() as memory:
        panel = load_raw_baseline_panel(RAW_DATA_PATH)
        splits = validate_dates(panel.trade_date.unique(), read_json(ROOT / cfg['step3_config']))
        print('Computing full historical panel before masks; only 2021/2022/2023 validation.', flush=True)
        full = build_features34(panel)
        stats = feature_statistics(full)
        if stats.infinite.any() or not stats.finite.gt(0).all():
            raise AssertionError('invalid full panel features')
        exact = {s: panel.loc[split_masks(panel, sp)[1], ['ts_code', 'trade_date']].reset_index(drop=True) for s, sp in splits.items()}
        for item in matrix(cfg):
            cols = columns(cfg, item); s = item['split']; began = time.perf_counter(); output = None
            index = read_json(OUTPUT / 'run_index.json')['runs']
            completed = next((x for x in index if x['item']['id'] == item['id']), None)
            if completed:
                if completed['item'] != item or sha256_file(ROOT / completed['directory'] / 'summary.json') != completed['summary_sha256']:
                    raise AssertionError('completed revision run changed')
                old.verify_artifact(ROOT / completed['directory'], s, cols, exact[s]); continue
            try:
                with experiment_run(EXPERIMENT_ROOT, item['id']) as output:
                    write_json(output / 'config.json', dict(requested=cfg, item=item, features=list(cols), model_params=MODEL_PARAMS))
                    write_json(output / 'provenance.json', meta)
                    features = full.loc[:, cols]; splitdir = output / s; fixture = splitdir / 'evaluate_input'
                    fixture.mkdir(parents=True, exist_ok=False)
                    truth = refs[('baseline10', s)][0] / s / 'evaluate_input/测试集_Y.csv'
                    shutil.copy2(truth, fixture / '测试集_Y.csv')
                    frozen = next((x for x in reference['splits'] if x['split_name'] == s), None)
                    with (output / 'execution.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        result = run_split(panel, features, s, output_dir=splitdir, reference=frozen, columns=cols, research_split=splits[s])
                    pred = pd.read_parquet(splitdir / 'predictions.parquet')
                    pd.testing.assert_frame_equal(pred[['ts_code', 'trade_date']], exact[s], check_dtype=False, check_categorical=False)
                    if sha256_file(fixture / '测试集_Y.csv') != sha256_file(truth):
                        raise AssertionError('original-token scoring truth changed')
                    check_frozen(manifest)
                    write_json(splitdir / 'summary.json', result)
                    summary = dict(stage=cfg['stage'], candidate=item['version'], item=item, features=list(cols), model_params=MODEL_PARAMS,
                        environment=environment_versions(), panel_rows=len(panel), feature_index_preserved=True, prediction_keys_match_raw_panel=True,
                        frozen_files_unchanged=True, splits=[result], provenance_sha256=sha256_file(output / 'provenance.json'),
                        config_sha256=sha256_file(output / 'config.json'), resources=dict(elapsed_seconds=time.perf_counter()-began, peak_process_rss_mb=memory.peak_rss_bytes/1024**2))
                    write_json(output / 'summary.json', summary)
                old.verify_artifact(output, s, cols, exact[s])
                index.append(dict(item=item, directory=output.relative_to(ROOT).as_posix(), summary_sha256=sha256_file(output / 'summary.json')))
                write_json(OUTPUT / 'run_index.json', dict(runs=index))
                print(f'Accepted {item["id"]}: score={result["metrics"]["final_score"]:.12f}', flush=True)
                del features, pred; gc.collect()
            except BaseException as exc:
                failure = read_json(OUTPUT / 'failures.json'); failure['failures'].append(dict(phase='model', item=item,
                    error_type=type(exc).__name__, error=str(exc), directory=None if output is None else str(output), elapsed_seconds=time.perf_counter()-began))
                write_json(OUTPUT / 'failures.json', failure); raise
    registration_check(cfg, provenance(ROOT, RAW_DATA_PATH)); check_frozen(manifest)
    write_json(OUTPUT / 'run_status.json', dict(status='success', planned=6, elapsed_seconds=time.perf_counter()-started))
    aggregate()


def derive(cfg, refs):
    oldcfg = read_json(old.CONFIG)
    reused = pd.read_csv(old.OUTPUT / 'deletion_results.csv', float_precision='round_trip')
    decisions, probes = single_tables(reused, cfg)
    rows = []; months = []
    for rec in read_json(OUTPUT / 'run_index.json')['runs']:
        item = rec['item']; d = ROOT / rec['directory']; sm = read_json(d / 'summary.json')
        row, month = old.compare(item, d, sm, refs, oldcfg)
        row['candidate'] = item['version']; month['candidate'] = item['version']
        rows.append(row); months.append(month)
    joints = pd.DataFrame(rows)
    if len(joints) != 6:
        raise AssertionError('six completed joint runs required before aggregation')
    for ctx in cfg['contexts']:
        for s in SPLITS:
            d, sm = refs[(ctx, s)]
            item = dict(id=f'reused_{ctx}_{s[-4:]}', context=ctx, kind='reference', exclude=[], split=s)
            row, month = old.compare(item, d, sm, refs, oldcfg)
            row['candidate'] = ctx; month['candidate'] = ctx; rows.append(row); months.append(month)
    annual = pd.DataFrame(rows); monthly = pd.concat(months, ignore_index=True)
    candidates = []
    for candidate, g in annual.groupby('candidate'):
        original = candidate in cfg['contexts']; g = g.sort_values('year')
        candidates.append(dict(candidate=candidate, context=g.context.iloc[0], original=original,
            eligible=original or joint_gate(g, cfg['rules']), feature_count=int(g.feature_count.iloc[0]),
            score_2023=float(g[g.year == 2023].after_final_score.iloc[0]), mean_score=float(g.after_final_score.mean()),
            worst_delta_vs_baseline10=float(g.after_final_score_minus_baseline10.min()), worst_delta_vs_parent=float(g.delta_final_score.min()),
            mean_delta_vs_parent=float(g.delta_final_score.mean()), annual_conflict=bool((g.delta_final_score < -cfg['rules']['near_score']).any()),
            statistical_uncertainty=bool(not original and g.block_ci_spans_zero.any()),
            positive_months_vs_parent=int(g.positive_months.sum()), worst_month_vs_parent=float(g.worst_month_delta.min()),
            reason='original fallback' if original else ('score rule passed; diagnostics disclosed' if joint_gate(g, cfg['rules']) else 'score rule failed; not ranked')))
    ordered = order_candidates(candidates, cfg['rules']['score_tie']); chosen = ordered[:2]
    frozen = []
    for rank, rec in enumerate(chosen, 1):
        spec = next((x for x in cfg['versions'] if x['candidate'] == rec['candidate']), None)
        excludes = [] if spec is None else spec['exclude']
        cols = select_features(groups=cfg['contexts'][rec['context']], exclude=excludes)
        frozen.append({**rec, 'rank': rank, 'exclude': excludes, 'features': list(cols),
            'evidence_scope': 'development-only selection; individual decisions do not imply joint necessity'})
    for i, row in decisions.iterrows():
        applicable = [v for v in frozen if v['context'] == row.context]
        decisions.loc[i, 'selected_candidates_in_context'] = ';'.join(v['candidate'] for v in applicable)
        decisions.loc[i, 'present_in_selected_candidates'] = ';'.join(v['candidate'] for v in applicable if row.feature in v['features'])
        decisions.loc[i, 'removed_in_selected_candidates'] = ';'.join(v['candidate'] for v in applicable if row.feature not in v['features'])
    ranking = pd.DataFrame(candidates)
    ranking['rank'] = ranking.candidate.map({r['candidate']: i+1 for i, r in enumerate(ordered)})
    return dict(single_decisions=decisions, probe_evidence=probes, joint_results=joints, candidate_results=annual,
                monthly_candidate_results=monthly, candidate_ranking=ranking), frozen


def aggregate():
    cfg = read_json(CONFIG); _, _, meta = check_contract(read_json(ROOT / cfg['step3_config']))
    registration_check(cfg, meta); refs, _ = old.references(read_json(old.CONFIG), meta)
    tables, frozen = derive(cfg, refs)
    for name, data in tables.items():
        data.to_csv(OUTPUT / f'{name}.csv', index=False, float_format='%.17g')
    write_json(OUTPUT / 'frozen_candidates.json', dict(candidates=frozen, accepted=False, audit_pending=True,
        used_2024=False, joint_versions_total=2, addback_versions_total=0, observed_existing_results=True))
    print('Derived revised decisions and four-candidate ranking; independent audit still required.', flush=True)


def render_report(tables, frozen):
    annual = tables['candidate_results']; decisions = tables['single_decisions']; ranking = tables['candidate_ranking']
    text = ['# 第4步修订：官方得分导向删除与跨年复核', '',
        '本次修订已通过独立验收。新规则基于已见的135项开发单删结果，属于探索性修订；两个联合清单及执行配置在六项新训练前冻结。旧规则、旧名单、旧报告、旧实验及冻结基线保持原哈希。', '',
        '只复用135项单删和9项参照，新增两个联合版本×三个开发年，共6项训练。2023是内部开发验证，模型训练到2022并purge；2021/2022同样验证。没有使用2024训练、评分或筛选，无最终比赛submission。', '',
        '删除分差=删除后减同年对应原候选；官方综合分优先。跨年三项直接分差均<−0.001判保留，均>+0.001判删除，接近/冲突/缺证据不确定。区间与月度/分项/缺失诊断保留但不再单独否决总分。结论是条件性开发决策，不是统计显著、组合可加性或未来名次保证。', '',
        '## 24列的背景条件结论', '', '|背景|特征|修订结论|旧结论|2021/2022/2023删除分差|区间跨零年份|冻结输入中移除|', '|---|---|---|---|---|---|---|']
    for row in decisions.to_dict('records'):
        values = '/'.join(f'{row[f"delta_{y}"]:+.6f}' if pd.notna(row.get(f'delta_{y}')) else '缺席无直接证据' for y in (2021, 2022, 2023))
        text.append(f'|{row["context"]}|{row["feature"]}|{row["decision"]}|{row["old_decision"]}|{values}|{row["block_ci_zero_years"]}|{row["removed_in_selected_candidates"]}|')
    text += ['', '暂留输入不等于证明必需；移出联合候选不等于该列确定删除。lean31缺席的A三列仍无单列直接证据。实验ID、直接年份、原判断、近零/方向/区间/旧护栏触发在single_decisions.csv；完整135项分项与月度继续引用旧结果，未覆盖。', '',
        '## 四个候选的三年官方结果', '', '|候选|年|IC|年化超额|官方换手|综合分|对十特征|对原候选|正月份/12|最差月对原候选|', '|---|---|---|---|---|---|---|---|---|---|']
    for row in annual.sort_values(['candidate', 'year']).to_dict('records'):
        text.append(f'|{row["candidate"]}|{row["year"]}|{row["after_ic_mean"]:.6f}|{row["after_annual_excess"]:.6f}|{row["after_mean_turnover"]:.6f}|{row["after_final_score"]:.9f}|{row["after_final_score_minus_baseline10"]:+.9f}|{row["delta_final_score"]:+.9f}|{row["positive_months"]}|{row["worst_month_delta"]:+.6f}|')
    text += ['', '候选ID与旧同列数组合不同；联合收益不能把单删收益相加。联合准入要求2023分差>0.001、三年平均>0、任一年≥−0.005、每年优于十特征。小于−0.001的年度退化单独标冲突，不能称稳健组合；区间跨零继续标统计不确定。', '',
        '|候选|联合准入/原参照|2023分数|最差年度对原候选|排名|', '|---|---|---|---|---|']
    for row in ranking.to_dict('records'):
        text.append(f'|{row["candidate"]}|{row["reason"]}|{row["score_2023"]:.9f}|{row["worst_delta_vs_parent"]:+.9f}|{int(row["rank"]) if pd.notna(row["rank"]) else "不入选"}|')
    text += ['', '## 冻结候选与剩余风险', '']
    for rec in frozen:
        text.append(f'- 第{rec["rank"]}候选：{rec["candidate"]}（{rec["feature_count"]}列），原背景{rec["context"]}，删除：{", ".join(rec["exclude"]) or "无"}；2023分数{rec["score_2023"]:.9f}，最差年对十特征{rec["worst_delta_vs_baseline10"]:+.9f}，对原候选{rec["worst_delta_vs_parent"]:+.9f}。')
    text += ['', '年度与月度风险、统计区间、四项Top缺失占比、价格有效换手及收益集中保存在candidate_results.csv和monthly_candidate_results.csv。评分贡献分别为0.4×ΔIC、0.3×Δ年化超额、−0.3×Δ官方换手；含缺失样本的官方Top集合必须按官方口径解释，不能等同于可交易稳定性。', '',
        '达到本次两版联合额度后停止；回补0，不追加搜索。所有年份属于开发数据，2024最后复核尚未执行。', '',
        '## 证据与验收', '',
        '- registration.json、matrix.json、probe_evidence.csv：探索性修订、训练前规则/源码哈希、固定预算和每列依据。',
        '- run_index.json：六项不可覆盖完整模型、全部键预测、官方评分输入、逐日/月度/缺失诊断、特征缺失统计、执行日志及来源。',
        '- single_decisions.csv、joint_results.csv、candidate_results.csv、monthly_candidate_results.csv、candidate_ranking.csv、frozen_candidates.json：完整决策与两种参照差。',
        '- acceptance.json：独立逐日重建官方分数、重算区间/表/排名/48条决策、原始键/资格/模型重载/保存哈希检查、源快照和旧产物保护核查。',
        '- preflight.json、tests_result.json、dependency_check.json、failures.json及commands：保护、测试、依赖及失败/命令记录。失败不进入候选排名；已失败尝试保留。',
        '- 复现：scripts/run_features34_step4_revision.py prepare/run/aggregate；审计：scripts/audit_features34_step4_revision.py。已完成一致实验核验后复用，不重跑单删。', '']
    (ROOT / 'docs/features34/STEP4_REVISION_REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    (OUTPUT / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('phase', choices=['prepare', 'run', 'aggregate'])
    args = parser.parse_args()
    try:
        {'prepare': prepare, 'run': run, 'aggregate': aggregate}[args.phase]()
    except BaseException as exc:
        failure_path = OUTPUT / 'failures.json'
        failures = read_json(failure_path) if failure_path.exists() else dict(failures=[])
        failures['failures'].append(dict(phase=args.phase, error_type=type(exc).__name__, error=str(exc)))
        write_json(failure_path, failures)
        write_json(OUTPUT / f'{args.phase}_status.json', dict(status='failed', error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
