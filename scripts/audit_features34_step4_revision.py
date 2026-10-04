"""Independent disk acceptance for score revision; no retraining or old writes."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts import run_features34_step4_revision as runner
from scripts import run_features34_step4 as old
from scripts.run_features34_step3 import read_json, check_contract, validate_dates
from scripts.run_lightgbm_baseline import RAW_DATA_PATH, split_masks
from src.data.baseline_panel import load_raw_baseline_panel
from src.features.features34 import build_features34
from src.validation.experiment import sha256_file, write_json


def independent_label(frame, near):
    if len(frame) != 3 or set(frame.year) != {2021, 2022, 2023}:
        return '不确定'
    deltas = frame.delta_final_score.to_numpy()
    if not np.isfinite(deltas).all():
        return '不确定'
    return '保留' if max(deltas) < -near else ('删除' if min(deltas) > near else '不确定')


def main():
    began = time.perf_counter(); out = runner.OUTPUT; cfg = read_json(runner.CONFIG)
    manifest, _, meta = check_contract(read_json(ROOT / cfg['step3_config']))
    reg = runner.registration_check(cfg, meta)
    for name, expected in reg['source']['source_sha256'].items():
        assert sha256_file(out / 'executed_sources' / name) == expected, name
    singles, old_runs = runner.verify_old_evidence(meta, deep=False)
    refs, records = old.references(read_json(old.CONFIG), meta)
    assert read_json(out / 'run_status.json')['status'] == 'success'
    runs = read_json(out / 'run_index.json')['runs']
    expected = runner.matrix(cfg)
    assert len(runs) == len(expected) == 6
    assert {r['item']['id'] for r in runs} == {r['id'] for r in expected}
    panel = load_raw_baseline_panel(RAW_DATA_PATH)
    splits = validate_dates(panel.trade_date.unique(), read_json(ROOT / cfg['step3_config']))
    exact = {}; counts = {}
    for name, split in splits.items():
        train, valid = split_masks(panel, split)
        exact[name] = panel.loc[valid, ['ts_code', 'trade_date']].reset_index(drop=True)
        counts[name] = dict(train_samples=int(train.sum()), valid_prediction_rows=int(valid.sum()))
        assert not train[panel.trade_date.eq(split.purge_date)].any()
    # Reuse was fully hash/model verified before training; independently verify all retained keys and day scores now.
    reused_checked = []
    for rec in old_runs:
        d = ROOT / rec['directory']; s = rec['item']['split']; sm = read_json(d / 'summary.json'); result = sm['splits'][0]
        pred = pd.read_parquet(d / s / 'predictions.parquet', columns=['ts_code', 'trade_date'])
        pd.testing.assert_frame_equal(pred, exact[s], check_dtype=False, check_categorical=False)
        for k, v in counts[s].items():
            assert result[k] == v
        dd = old.daily(d, s)
        score = .4 * dd.ic.mean() + .3 * 252 * dd.excess.mean() + .3 * (1-dd.turnover.mean())
        assert abs(score-result['metrics']['final_score']) <= 1e-12
        assert set(str(x)[:4] for x in dd.index) == {s[-4:]}
        reused_checked.append(dict(id=rec['item']['id'], summary_sha256=rec['summary_sha256'], keys_verified=True, day_score_error=abs(score-result['metrics']['final_score'])))
    full = build_features34(panel)
    verified = []
    for rec in runs:
        item = rec['item']; s = item['split']; d = ROOT / rec['directory']; cols = runner.columns(cfg, item)
        assert item == next(i for i in expected if i['id'] == item['id'])
        assert sha256_file(d / 'summary.json') == rec['summary_sha256']
        sm = old.verify_artifact(d, s, cols, exact[s]); result = sm['splits'][0]
        assert result['dates'] == read_json(ROOT / cfg['step3_config'])['splits'][s]
        for k, v in counts[s].items():
            assert result[k] == v
        prior = read_json(d / 'provenance.json')
        for key in ('data', 'dependencies', 'source_sha256'):
            assert prior[key] == reg['source'][key]
        for key in ('branch', 'commit'):
            assert prior['git'][key] == reg['source']['git'][key]
        assert sha256_file(d / s / 'evaluate_input/测试集_Y.csv') == sha256_file(refs[('baseline10', s)][0] / s / 'evaluate_input/测试集_Y.csv')
        model = lgb.Booster(model_file=str(d / s / 'models/lightgbm.txt'))
        valid = split_masks(panel, splits[s])[1]
        prediction = model.predict(full.loc[valid, cols])
        saved = pd.read_parquet(d / s / 'predictions.parquet')
        assert np.array_equal(prediction, saved.pred.to_numpy())
        stats = pd.read_csv(d / s / 'feature_missing_statistics.csv')
        assert not stats.infinite.any() and stats.finite.gt(0).all()
        assert set(stats.scope) == {'full_panel', 'train_eligible', 'validation_all_keys'}
        assert set(stats[stats.scope == 'full_panel'].rows) == {len(panel)}
        dd = old.daily(d, s)
        actual = .4*dd.ic.mean()+.3*252*dd.excess.mean()+.3*(1-dd.turnover.mean())
        assert abs(actual-result['metrics']['final_score']) <= 1e-12
        verified.append(dict(id=item['id'], directory=rec['directory'], saved_hashes_verified=len(result['file_sha256']),
            all_keys_verified=True, reload_predictions_equal=True, day_score_error=abs(actual-result['metrics']['final_score']),
            official_max_abs_difference=result['official_comparison']['max_abs_difference']))
    tables, frozen = runner.derive(cfg, refs)
    for name, table in tables.items():
        # CSV roundtrip normalizes intentionally absent fields and rank NA, without touching saved tables.
        import io
        text = table.to_csv(index=False, float_format='%.17g')
        normalized = pd.read_csv(io.StringIO(text), float_precision='round_trip')
        saved = pd.read_csv(out / f'{name}.csv', float_precision='round_trip')
        pd.testing.assert_frame_equal(normalized, saved, check_exact=True, check_dtype=False)
    assert len(tables['single_decisions']) == 48
    for row in tables['single_decisions'].itertuples():
        g = singles[(singles.context == row.context) & (singles.feature == row.feature)]
        assert row.decision == independent_label(g, cfg['rules']['near_score'])
    # Check joint eligibility and selection independently of runner's gating/ranking functions.
    records_by_id = {r['candidate']: r for r in tables['candidate_ranking'].to_dict('records')}
    for name, frame in tables['candidate_results'].groupby('candidate'):
        r = records_by_id[name]; dd = frame.delta_final_score
        expected_gate = name in cfg['contexts'] or bool(frame[frame.year == 2023].delta_final_score.iloc[0] > .001 and dd.mean() > 0 and dd.min() >= -.005 and (frame.after_final_score_minus_baseline10 > 0).all())
        assert r['eligible'] == expected_gate
    remaining = [r for r in records_by_id.values() if r['eligible']]; independent_order = []
    while remaining:
        best = max(r['score_2023'] for r in remaining)
        tie = [r for r in remaining if best-r['score_2023'] <= 1e-12]
        winner = min(tie, key=lambda r: (-r['worst_delta_vs_baseline10'], -r['mean_score'], r['feature_count'], r['candidate']))
        independent_order.append(winner['candidate']); remaining.remove(winner)
    assert [r['candidate'] for r in frozen] == independent_order[:2]
    assert read_json(out / 'frozen_candidates.json')['candidates'] == frozen
    for check in ('tests_result', 'dependency_check'):
        assert read_json(out / f'{check}.json')['exit_code'] == 0
    runner.preserve_old(); old.check_frozen(manifest)
    failures = read_json(out / 'failures.json')['failures']
    assert not [f for f in failures if f['phase'] == 'model']
    runner.render_report(tables, frozen)
    write_json(out / 'frozen_candidates.json', dict(candidates=frozen, accepted=True, audit_pending=False,
        used_2024=False, joint_versions_total=2, addback_versions_total=0, observed_existing_results=True))
    evidence = {p.name: sha256_file(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json', '.csv', '.md') and p.name not in ('acceptance.json', 'summary.json')}
    accepted = dict(accepted=True, stage=cfg['stage'], exploratory_after_observed_results=True, used_2024=False,
        new_runs=verified, reused_single_runs=reused_checked, reused_references=records, frozen_files_verified=len(manifest),
        old_step4_and_rules_unchanged=True, sources_verified=True, all_validation_keys_verified=True,
        all_tables_and_labels_recomputed=True, official_scores_reconstructed=True, candidate_ranking_independently_verified=True,
        joint_versions_total=2, addback_versions_total=0, failures=failures, elapsed_seconds=time.perf_counter()-began, evidence_sha256=evidence)
    write_json(out / 'acceptance.json', accepted)
    write_json(out / 'summary.json', dict(accepted=True, stage=cfg['stage'], new_runs=6, reused_singles=135, reused_references=9,
        candidates=frozen, used_2024=False, final_submission_generated=False, acceptance_sha256=sha256_file(out / 'acceptance.json')))
    print(f'Independent revision acceptance passed: 6 joint runs, 135 reused singles, 48 decisions ({time.perf_counter()-began:.1f}s).', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        p = runner.OUTPUT / 'failures.json'; f = read_json(p) if p.exists() else dict(failures=[])
        f['failures'].append(dict(phase='audit', error_type=type(exc).__name__, error=str(exc)))
        write_json(p, f)
        raise
