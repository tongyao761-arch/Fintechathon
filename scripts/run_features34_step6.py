"""Accept the two already frozen versions; repeat 2023/2024 without selection."""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts import run_features34_step5 as prior
from scripts import run_features34_step4 as old
from scripts.run_features34 import run_split, check_frozen, feature_statistics
from scripts.run_features34_step3 import read_json, check_contract, validate_dates
from scripts.run_lightgbm_baseline import (
    RAW_DATA_PATH, EXPERIMENT_ROOT, MODEL_PARAMS, environment_versions,
    PeakMemoryMonitor, split_masks, score_saved_inputs, compare_metrics,
)
from src.data.baseline_panel import load_raw_baseline_panel, extract_truth_files
from src.features.features34 import build_features34, FEATURE_DEFINITIONS, NEW_COLUMNS
from src.features.baseline_v1 import FEATURE_COLUMNS as BASE_COLUMNS
from src.metrics.diagnostics import baseline_diagnostics
from src.validation.experiment import experiment_run, provenance, sha256_file, write_json
from src.validation.splits import get_split

CONFIG = ROOT / 'configs/features34_step6.json'
OUTPUT = ROOT / 'artifacts/features34_step6'
SPLITS = ['primary_2023', 'oos_2024']


def validate_config(cfg):
    expected = dict(stage='step6_final_acceptance_no_selection',
        freeze='artifacts/features34_step5/FROZEN_CANDIDATES.json',
        candidates=prior.IDS, splits=SPLITS, new_training_runs=4,
        purpose='repeat frozen versions only; no feature selection or tuning')
    if cfg != expected:
        raise ValueError('only four repeats of the two previously frozen versions are authorized')


def verify_evidence(directory):
    accepted = read_json(directory / 'acceptance.json')
    if not accepted['accepted']:
        raise AssertionError(f'previous acceptance missing: {directory}')
    for name, expected in accepted['evidence_sha256'].items():
        if sha256_file(directory / name) != expected:
            raise AssertionError(f'previous evidence changed: {directory / name}')
    return accepted


def contract():
    cfg = read_json(CONFIG); validate_config(cfg)
    manifest, reference, meta = check_contract(read_json(ROOT / 'configs/features34_step3.json'))
    freeze = read_json(ROOT / cfg['freeze'])
    reg = read_json(prior.OUTPUT / 'registration.json')
    if sha256_file(ROOT / cfg['freeze']) != reg['freeze_sha256']:
        raise AssertionError('step5 freeze changed')
    for folder in (old.OUTPUT, prior.revision.OUTPUT, prior.OUTPUT):
        verify_evidence(folder)
    # New acceptance files may be additive. Every file actually used in step5 stays identical.
    for name, expected in freeze['source']['source_sha256'].items():
        if sha256_file(ROOT / name) != expected or sha256_file(prior.OUTPUT / 'executed_sources' / name) != expected:
            raise AssertionError(f'previous executed source changed: {name}')
    for key in ('data', 'dependencies'):
        if meta[key] != freeze['source'][key]:
            raise AssertionError(f'frozen {key} changed')
    records = prior.candidates(read_json(prior.CONFIG))
    for a, b in zip(records, freeze['candidates'], strict=True):
        if a['features'] != b['features'] or a['candidate'] != b['candidate']:
            raise AssertionError('candidate list mismatch')
        if b['model_params'] != MODEL_PARAMS or b['feature_definitions'] != {c: FEATURE_DEFINITIONS[c] for c in b['features']}:
            raise AssertionError('formula or model contract changed')
    entries = prior.entries(read_json(prior.CONFIG), meta)
    fullrefs, _ = old.references(read_json(old.CONFIG), meta)
    entries.update({k: v for k, v in fullrefs.items() if k[0] == 'full34'})
    return cfg, manifest, reference, meta, freeze, entries


def preserve():
    snapshot = read_json(OUTPUT / 'preflight.json')
    for key in ('protections', 'existing_untracked', 'previous_files'):
        for name, expected in snapshot[key].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha256_file(path) != expected:
                raise AssertionError(f'preserved file changed: {name}')


def registration_check():
    cfg, manifest, reference, meta, freeze, entries = contract()
    reg = read_json(OUTPUT / 'registration.json')
    if reg['config_sha256'] != sha256_file(CONFIG) or reg['freeze_sha256'] != sha256_file(ROOT / cfg['freeze']):
        raise AssertionError('repeat registration changed')
    for key in ('source_sha256', 'data', 'dependencies'):
        if reg['source'][key] != meta[key]:
            raise AssertionError(f'repeat execution {key} changed')
    for name, expected in reg['source']['source_sha256'].items():
        if sha256_file(OUTPUT / 'executed_sources' / name) != expected:
            raise AssertionError('repeat source snapshot changed')
    preserve()
    return cfg, manifest, reference, meta, freeze, entries


def prepare():
    if OUTPUT.exists():
        raise FileExistsError('step6 directory exists; never overwrite registration')
    cfg, manifest, _, meta, freeze, entries = contract()
    snapshot = old.protection_snapshot()
    own = {'configs/features34_step6.json', 'scripts/run_features34_step6.py', 'tests/test_features34_step6.py'}
    snapshot['existing_untracked'] = {p: h for p, h in snapshot['existing_untracked'].items() if p not in own}
    paths = [p for folder in ('artifacts/features34_step1', 'artifacts/features34_step2',
        'artifacts/features34_step3', 'artifacts/features34_step4', 'artifacts/features34_step4_revision',
        'artifacts/features34_step5', 'docs/features34') for p in (ROOT / folder).rglob('*')
        if p.is_file() and p.name not in ('PLAN.md', 'PROGRESS.md', 'RESULTS.md')]
    snapshot['previous_files'] = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT / 'preflight.json', snapshot)
    write_json(OUTPUT / 'failures.json', dict(failures=[]))
    write_json(OUTPUT / 'run_index.json', dict(runs=[]))
    write_json(OUTPUT / 'registration.json', dict(config=cfg, config_sha256=sha256_file(CONFIG),
        freeze_sha256=sha256_file(ROOT / cfg['freeze']), source=meta,
        candidates=freeze['candidates'], registered_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
        reference_runs=[dict(candidate=n, split=s, directory=d.relative_to(ROOT).as_posix(),
            summary_sha256=sha256_file(d / 'summary.json')) for (n, s), (d, _) in entries.items()]))
    for name in meta['source_sha256']:
        dest = OUTPUT / 'executed_sources' / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, dest)
    write_json(OUTPUT / 'status.json', dict(status='registered', planned_new_runs=4))
    check_frozen(manifest); preserve()
    print('Registered four frozen repeats; no new candidates.', flush=True)


def compare_repeat(before, after):
    for key in ('dates', 'train_samples', 'valid_prediction_rows', 'prediction_coverage',
        'prediction_sha256', 'purge_rows', 'split_train_rows', 'metrics', 'diagnostics',
        'score_contributions', 'file_sha256'):
        if before[key] != after[key]:
            raise AssertionError(f'repeat differs from frozen original: {key}')


def run():
    started = time.perf_counter()
    cfg, manifest, _, meta, freeze, entries = registration_check()
    write_json(OUTPUT / 'status.json', dict(status='running', planned_new_runs=4))
    with PeakMemoryMonitor() as memory:
        panel = load_raw_baseline_panel(RAW_DATA_PATH)
        validate_dates(panel.trade_date.unique(), read_json(ROOT / 'configs/features34_step3.json'))
        full = build_features34(panel)
        stats = feature_statistics(full)
        if stats.infinite.any() or not stats.finite.gt(0).all():
            raise AssertionError('infinite or empty feature')
        for record in freeze['candidates']:
            name = record['candidate']; cols = tuple(record['features'])
            for s in SPLITS:
                item = dict(candidate=name, split=s)
                index = read_json(OUTPUT / 'run_index.json')['runs']
                done = next((r for r in index if r['item'] == item), None)
                if done:
                    sm = old.verify_artifact(ROOT / done['directory'], s, cols)
                    if sha256_file(ROOT / done['directory'] / 'summary.json') != done['summary_sha256']:
                        raise AssertionError('registered repeat changed')
                    compare_repeat(entries[(name, s)][1]['splits'][0], sm['splits'][0]); continue
                split = get_split(s); beforedir, beforesm = entries[(name, s)]
                base = entries[('baseline10', s)][1]['splits'][0]
                began = time.perf_counter()
                with experiment_run(EXPERIMENT_ROOT, f'S6_{name}_{s[-4:]}') as output:
                    print(f'Repeating {name} {s}: {output}', flush=True)
                    write_json(output / 'config.json', dict(requested=cfg, item=item, features=list(cols),
                        model_params=MODEL_PARAMS, freeze_sha256=sha256_file(ROOT / cfg['freeze'])))
                    write_json(output / 'provenance.json', meta)
                    splitdir = output / s; fixture = splitdir / 'evaluate_input'
                    fixture.mkdir(parents=True, exist_ok=False)
                    extract_truth_files(RAW_DATA_PATH, {fixture / '测试集_Y.csv': (split.valid_start, split.valid_end)})
                    if sha256_file(fixture / '测试集_Y.csv') != sha256_file(beforedir / s / 'evaluate_input/测试集_Y.csv'):
                        raise AssertionError('raw scoring truth changed')
                    with (output / 'execution.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        result = run_split(panel, full.loc[:, cols], s, output_dir=splitdir, reference=base, columns=cols)
                    compare_repeat(beforesm['splits'][0], result)
                    saved = pd.read_parquet(splitdir / 'predictions.parquet')
                    keys = panel.loc[split_masks(panel, split)[1], ['ts_code', 'trade_date']].reset_index(drop=True)
                    pd.testing.assert_frame_equal(saved[['ts_code', 'trade_date']], keys, check_dtype=False, check_categorical=False)
                    check_frozen(manifest)
                    write_json(splitdir / 'summary.json', result)
                    write_json(output / 'summary.json', dict(stage=cfg['stage'], candidate=name, features=list(cols),
                        model_params=MODEL_PARAMS, environment=environment_versions(), splits=[result],
                        provenance_sha256=sha256_file(output / 'provenance.json'), config_sha256=sha256_file(output / 'config.json'),
                        freeze_sha256=sha256_file(ROOT / cfg['freeze']), original_directory=beforedir.relative_to(ROOT).as_posix(),
                        repeat_matches_original=True, resources=dict(elapsed_seconds=time.perf_counter()-began,
                            peak_process_rss_mb=memory.peak_rss_bytes / 1024**2)))
                old.verify_artifact(output, s, cols, keys)
                index.append(dict(item=item, directory=output.relative_to(ROOT).as_posix(),
                    original_directory=beforedir.relative_to(ROOT).as_posix(), summary_sha256=sha256_file(output / 'summary.json')))
                write_json(OUTPUT / 'run_index.json', dict(runs=index))
                print(f'Repeat identical: {name} {s}, score={result["metrics"]["final_score"]:.12f}', flush=True)
                gc.collect()
    registration_check()
    write_json(OUTPUT / 'status.json', dict(status='runs_complete_audit_pending', new_training_runs=4,
        elapsed_seconds=time.perf_counter()-started, peak_process_rss_mb=memory.peak_rss_bytes / 1024**2))


def verify_decisions(entries):
    decisions = pd.read_csv(prior.OUTPUT / 'feature_decisions.csv', float_precision='round_trip').fillna('')
    runs = {r['item']['id']: r for r in read_json(old.OUTPUT / 'run_index.json')['runs']}
    if len(decisions) != 48 or any(set(g.feature) != set(NEW_COLUMNS) for _, g in decisions.groupby('context')):
        raise AssertionError('24-column background decisions incomplete')
    for row in decisions.itertuples():
        if row.absent_from_original_input:
            if row.decision != '不确定' or row.experiment_ids:
                raise AssertionError('absent feature cannot have direct evidence')
            continue
        deltas = []
        for experiment_id in row.experiment_ids.split(';'):
            rec = runs[experiment_id]; d = ROOT / rec['directory']; sm = read_json(d / 'summary.json')
            if sha256_file(d / 'summary.json') != rec['summary_sha256'] or read_json(d / 'status.json')['status'] != 'success':
                raise AssertionError('single deletion evidence changed')
            s = rec['item']['split']
            cols = old.selection(read_json(old.CONFIG), rec['item'])
            if sm['features'] != list(cols) or sm['model_params'] != MODEL_PARAMS:
                raise AssertionError('single deletion configuration mismatch')
            parent = read_json(ROOT / 'artifacts/features34_step3/run_index.json')['runs']
            p = next(r for r in parent if r['candidate'] == row.context and r['split'] == s)
            parent_sm = read_json(ROOT / p['directory'] / 'summary.json')
            if sha256_file(ROOT / p['directory'] / 'summary.json') != p['summary_sha256']:
                raise AssertionError('single deletion parent changed')
            delta = sm['splits'][0]['metrics']['final_score'] - parent_sm['splits'][0]['metrics']['final_score']
            if abs(delta - getattr(row, 'delta_' + s[-4:])) > 1e-12:
                raise AssertionError('single deletion delta mismatch')
            deltas.append(delta)
        if len(deltas) != 3:
            raise AssertionError('three development years required')
        expected = '保留' if all(x < -.001 for x in deltas) else ('删除' if all(x > .001 for x in deltas) else '不确定')
        if row.decision != expected:
            raise AssertionError('decision differs from score-first development rule')
    return decisions


def normalize_frame(frame):
    return pd.read_csv(io.StringIO(frame.to_csv(index=False, float_format='%.17g')), float_precision='round_trip')


def audit():
    started = time.perf_counter()
    cfg, manifest, _, meta, freeze, entries = registration_check()
    if read_json(OUTPUT / 'status.json')['status'] not in ('runs_complete_audit_pending', 'success'):
        raise AssertionError('all four repeats must finish first')
    runs = read_json(OUTPUT / 'run_index.json')['runs']
    if len(runs) != 4 or {(r['item']['candidate'], r['item']['split']) for r in runs} != {(n, s) for n in prior.IDS for s in SPLITS}:
        raise AssertionError('repeat matrix incomplete or extra')
    for rec in runs:
        d = ROOT / rec['directory']
        if sha256_file(d / 'summary.json') != rec['summary_sha256']:
            raise AssertionError('repeat index summary changed')
        saved_meta = read_json(d / 'provenance.json')
        for key in ('data', 'dependencies', 'source_sha256'):
            if saved_meta[key] != meta[key]:
                raise AssertionError(f'repeat provenance mismatch: {key}')
        saved_cfg = read_json(d / 'config.json')
        if (saved_cfg['requested'] != cfg or saved_cfg['item'] != rec['item'] or
                saved_cfg['freeze_sha256'] != sha256_file(ROOT / cfg['freeze'])):
            raise AssertionError('repeat configuration mismatch')
    for name in ('tests_result', 'dependency_check'):
        if read_json(OUTPUT / (name + '.json'))['exit_code'] != 0:
            raise AssertionError('regression or dependencies failed')
    # Reuse verified development evidence; no model is retrained by the audit.
    fullrefs, _ = old.references(read_json(old.CONFIG), meta)
    entries.update({k: v for k, v in fullrefs.items() if k[0] == 'lean31'})
    decisions = verify_decisions(entries)
    panel = load_raw_baseline_panel(RAW_DATA_PATH); full = build_features34(panel)
    repeated = []
    targets = [(r['item']['candidate'], r['item']['split'], ROOT / r['directory']) for r in runs]
    targets += [('baseline10', s, entries[('baseline10', s)][0]) for s in SPLITS]
    for name, s, d in targets:
        sm = old.verify_artifact(d, s, tuple(entries[(name, s)][1]['features'])) if name != 'baseline10' or s != 'oos_2024' else entries[(name, s)][1]
        r = sm['splits'][0]; split = get_split(s); train, valid = split_masks(panel, split)
        keys = panel.loc[valid, ['ts_code', 'trade_date']].reset_index(drop=True)
        saved = pd.read_parquet(d / s / 'predictions.parquet')
        pd.testing.assert_frame_equal(saved[['ts_code', 'trade_date']], keys, check_dtype=False, check_categorical=False)
        if r['train_samples'] != int(train.sum()) or r['valid_prediction_rows'] != int(valid.sum()) or train[panel.trade_date.eq(split.purge_date)].any():
            raise AssertionError('raw keys, purge, or eligibility mismatch')
        model = lgb.Booster(model_file=str(d / s / 'models/lightgbm.txt'))
        cols = tuple(sm['features'])
        if model.feature_name() != list(cols) or not np.array_equal(model.predict(full.loc[valid, cols]), saved.pred.to_numpy()):
            raise AssertionError('raw X saved-model reproduction failed')
        metrics, comparison, pred, truth = score_saved_inputs(d / s / 'evaluate_input'); details = metrics.pop('details')
        compare_metrics(r['metrics'], metrics)
        pd.testing.assert_frame_equal(truth, panel.loc[valid, ['ts_code', 'trade_date', 'y_ret_1d']].reset_index(drop=True),
            check_dtype=False, check_categorical=False, check_exact=True)
        quality = panel.loc[valid, ['ts_code', 'trade_date', 'flag_limit_up', 'is_price_valid']].copy()
        quality['baseline_features_all_missing'] = full.loc[valid, BASE_COLUMNS].isna().all(axis=1)
        diag, tables = baseline_diagnostics(pred, truth, quality, details)
        for key in ('top_groups', 'price_valid_only_turnover'):
            if diag[key] != r['diagnostics'][key]: raise AssertionError('diagnostics mismatch')
        for filename, frame in {**details, **tables}.items():
            pd.testing.assert_frame_equal(normalize_frame(frame), pd.read_csv(d / s / (filename + '.csv'), float_precision='round_trip'), check_exact=True, check_dtype=False)
        if name != 'baseline10':
            compare_repeat(entries[(name, s)][1]['splits'][0], r)
            if not np.array_equal(saved.pred.to_numpy(), pd.read_parquet(entries[(name, s)][0] / s / 'predictions.parquet').pred.to_numpy()):
                raise AssertionError('repeat prediction array changed')
            quality['baseline_features_all_missing'] = full.loc[valid, cols].isna().all(axis=1)
            cd, ct = baseline_diagnostics(pred, truth, quality, details)
            if cd['top_groups'] != r['diagnostics']['candidate_features_top_groups']: raise AssertionError('candidate missing diagnostics')
            pd.testing.assert_frame_equal(normalize_frame(ct['daily_missing_diagnostics']), pd.read_csv(d / s / 'daily_candidate_missing_diagnostics.csv', float_precision='round_trip'), check_exact=True, check_dtype=False)
            pd.testing.assert_frame_equal(feature_statistics(full.loc[:, cols], train_mask=train, valid_mask=valid),
                pd.read_csv(d / s / 'feature_missing_statistics.csv', float_precision='round_trip'), check_exact=True, check_dtype=False)
        repeated.append(dict(candidate=name, split=s, directory=d.relative_to(ROOT).as_posix(),
            prediction_sha256=r['prediction_sha256'], prediction_coverage=r['prediction_coverage'],
            train_samples=r['train_samples'], valid_prediction_rows=r['valid_prediction_rows'],
            official_rescore_max_abs_difference=comparison['max_abs_difference'], model_reload_equal=True,
            all_keys_equal=True, diagnostics_recomputed=True, repeat_equal=True if name != 'baseline10' else None))
    # Assemble historical comparisons from actual run summaries, without re-ranking candidates.
    annual = []; monthly = []; missing = []
    for (name, s), (d, sm) in entries.items():
        if name == 'lean31': continue
        table = prior.comparison_tables({('baseline10', s): entries[('baseline10', s)], (name, s): (d, sm)}, read_json(prior.CONFIG))
        annual.append(table['annual_comparison'].query('candidate == @name'))
        monthly.append(table['monthly_comparison'].query('candidate == @name'))
        missing.append(table['monthly_missing_diagnostics'].query('candidate == @name'))
    frames = dict(annual_comparison=pd.concat(annual, ignore_index=True).sort_values(['year', 'candidate']),
        monthly_comparison=pd.concat(monthly, ignore_index=True).sort_values(['month', 'candidate']),
        monthly_missing_diagnostics=pd.concat(missing, ignore_index=True).sort_values(['month', 'candidate', 'top_type']),
        feature_decisions=decisions, repeat_comparison=pd.DataFrame(repeated))
    if len(frames['annual_comparison']) != 15 or len(frames['monthly_comparison']) != 180:
        raise AssertionError('historical comparison matrix incomplete')
    registration_check(); check_frozen(manifest)
    for name, frame in frames.items(): frame.to_csv(OUTPUT / (name + '.csv'), index=False, float_format='%.17g')
    evidence = {name + '.csv': sha256_file(OUTPUT / (name + '.csv')) for name in frames}
    for name in ('registration', 'preflight', 'run_index', 'tests_result', 'dependency_check', 'failures'):
        evidence[name + '.json'] = sha256_file(OUTPUT / (name + '.json'))
    accepted = dict(accepted=True, stage=cfg['stage'], new_training_runs=4, no_selection=True,
        repeat_runs=4, baseline_readonly_checks=2, frozen_files_verified=len(manifest),
        previous_execution_sources_verified=len(freeze['source']['source_sha256']),
        all_keys_finite_predictions_official_scores_and_diagnostics_verified=True,
        repeats_match_original_predictions_metrics_diagnostics_and_all_split_files=True,
        single_decision_rows_verified=48, full34_2024='not_run_not_authorized',
        protections_and_previous_files_unchanged=True, verified_runs=repeated,
        evidence_sha256=evidence, failures=read_json(OUTPUT / 'failures.json')['failures'],
        elapsed_seconds=time.perf_counter()-started)
    write_json(OUTPUT / 'acceptance.json', accepted)
    write_json(OUTPUT / 'summary.json', dict(accepted=True, stage=cfg['stage'], repeat_runs=4,
        final_submission_generated=False, acceptance_sha256=sha256_file(OUTPUT / 'acceptance.json')))
    write_json(OUTPUT / 'status.json', dict(status='success', accepted=True, new_training_runs=4))
    print('Accepted: four frozen repeats identical; baseline read-only checks and 48 decisions verified.', flush=True)


def failure(phase, exc, elapsed):
    if OUTPUT.exists():
        p = OUTPUT / 'failures.json'; state = read_json(p) if p.exists() else dict(failures=[])
        state['failures'].append(dict(phase=phase, error_type=type(exc).__name__, error=str(exc), elapsed_seconds=elapsed,
            at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat()))
        write_json(p, state)
        write_json(OUTPUT / 'status.json', dict(status='failed', phase=phase, error=str(exc), elapsed_seconds=elapsed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run', 'audit'])
    args = parser.parse_args(argv); began = time.perf_counter()
    try:
        {'prepare': prepare, 'run': run, 'audit': audit}[args.phase]()
    except BaseException as exc:
        failure(args.phase, exc, time.perf_counter()-began); raise


if __name__ == '__main__': main()
