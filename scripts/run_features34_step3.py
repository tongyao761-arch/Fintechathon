"""Fixed lean candidates on three development years; no 2024 search."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_features34 import check_frozen, feature_statistics, run_split
from scripts.run_features34_step2 import verify_run
from scripts.run_lightgbm_baseline import MODEL_PARAMS, RAW_DATA_PATH, EXPERIMENT_ROOT, PeakMemoryMonitor, split_masks, environment_versions
from src.data.baseline_panel import load_raw_baseline_panel, extract_truth_files
from src.features.features34 import build_features34, select_features, FEATURE_DEFINITIONS
from src.validation.experiment import experiment_run, provenance, sha256_file, prediction_hash, write_json
from src.validation.splits import TimeSplit, get_split

CONFIG = ROOT / 'configs/features34_step3.json'
OUTPUT = ROOT / 'artifacts/features34_step3'
EXPECTED = {'baseline10': [], 'full34': list('ABCDEF'), 'lean31': list('BCDEF'),
            'lean27': list('BDEF'), 'lean23': list('BDF')}
SPLITS = ('dev_2021', 'dev_2022', 'primary_2023')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_config(config, candidate=None, split_name=None):
    if config['stage'] != 'step3_development_only':
        raise ValueError('unsupported stage')
    if config['experiments'] != {n: {'groups': g} for n, g in EXPECTED.items()}:
        raise ValueError('only the five preregistered group combinations are authorized')
    if set(config['splits']) != set(SPLITS):
        raise ValueError('only 2021/2022/2023 are authorized; 2024 is prohibited')
    if candidate is not None and candidate not in EXPECTED:
        raise ValueError('candidate outside fixed step3 matrix')
    if split_name is not None and split_name not in SPLITS:
        raise ValueError('split outside step3 authorization; 2024 is prohibited')
    if config['splits']['primary_2023'] != {
        k: getattr(get_split('primary_2023'), k) for k in config['splits']['primary_2023']
    }:
        raise ValueError('2023 must use the frozen split')
    if set(config) != {'stage', 'experiments', 'splits', 'selection_rule', 'reference_summary',
                       'frozen_manifest', 'boundary_source'}:
        raise ValueError('unsupported fields; model tuning is prohibited')
    if candidate is not None:
        return select_features(**config['experiments'][candidate])


def validate_dates(dates, config):
    dates = sorted(set(int(d) for d in dates))
    digest = hashlib.sha256(np.asarray(dates, dtype='<i4').tobytes()).hexdigest()
    if digest != config['boundary_source']['dates_sha256'] or len(dates) != config['boundary_source']['date_count']:
        raise AssertionError('raw trading dates differ from preregistered boundary source')
    for name, spec in config['splits'].items():
        year = int(name[-4:])
        valid = [d for d in dates if d // 10000 == year]
        prior = [d for d in dates if d < valid[0]]
        actual = dict(train_start=dates[0], train_end=prior[-2], purge_date=prior[-1],
                      valid_start=valid[0], valid_end=valid[-1])
        if spec != actual:
            raise AssertionError(f'{name}: must purge last observed trading day before validation year')
    return {name: TimeSplit(name=name, **spec) for name, spec in config['splits'].items()}


def check_contract(config):
    validate_config(config)
    manifest = read_json(ROOT / config['frozen_manifest'])
    check_frozen(manifest)
    reference = read_json(ROOT / config['reference_summary'])
    meta = provenance(ROOT, RAW_DATA_PATH)
    if meta['git']['branch'] != 'ivor-work':
        raise AssertionError('step3 requires local ivor-work')
    if meta['data']['sha256'] != reference['data']['sha256'] or MODEL_PARAMS != reference['model_params']:
        raise AssertionError('frozen data/model contract changed')
    if meta['data']['sha256'] != config['boundary_source']['sha256']:
        raise AssertionError('raw boundary source changed')
    return manifest, reference, meta


def assert_source_stable(meta):
    now = provenance(ROOT, RAW_DATA_PATH)
    if now['source_sha256'] != meta['source_sha256'] or now['data'] != meta['data']:
        raise AssertionError('source/config/raw data changed during execution')


def run_one(candidate, split_name):
    started = time.perf_counter()
    output = None
    try:
        with experiment_run(EXPERIMENT_ROOT, f'features34_step3_{candidate}_{split_name}') as output:
            print(f'Experiment directory: {output}', flush=True)
            config = read_json(CONFIG)
            columns = validate_config(config, candidate, split_name)
            manifest, reference, meta = check_contract(config)
            meta['candidate_config_sha256'] = sha256_file(CONFIG)
            meta['reference_summary_sha256'] = sha256_file(ROOT / config['reference_summary'])
            meta['frozen_manifest_sha256'] = sha256_file(ROOT / config['frozen_manifest'])
            write_json(output / 'provenance.json', meta)
            write_json(output / 'config.json', dict(requested=config, candidate=candidate, split=split_name,
                       features=list(columns), model_params=MODEL_PARAMS, purpose=config['stage']))
            with PeakMemoryMonitor() as memory:
                panel = load_raw_baseline_panel(RAW_DATA_PATH)
                split = validate_dates(panel.trade_date.unique(), config)[split_name]
                # Complete raw historical panel is intentionally retained, including all missing rows.
                features = build_features34(panel, columns)
                if tuple(features.columns) != columns or not features.index.equals(panel.index):
                    raise AssertionError('feature contract changed')
                stats = feature_statistics(features)
                if stats.infinite.any() or not stats.finite.gt(0).all():
                    raise AssertionError('non-finite feature contract')
                stats.to_csv(output / 'feature_missing_statistics.csv', index=False, float_format='%.17g')
                split_output = output / split_name
                extract_truth_files(RAW_DATA_PATH, {split_output / 'evaluate_input/测试集_Y.csv':
                                                     (split.valid_start, split.valid_end)})
                frozen = next((s for s in reference['splits'] if s['split_name'] == split_name), None)
                result = run_split(panel, features, split_name, output_dir=split_output,
                                   reference=frozen, columns=columns, research_split=split)
                train, valid = split_masks(panel, split)
                if result['train_samples'] != int(train.sum()) or result['valid_prediction_rows'] != int(valid.sum()):
                    raise AssertionError('training eligibility or validation coverage changed')
                saved = pd.read_parquet(split_output / 'predictions.parquet')
                pd.testing.assert_frame_equal(saved[['ts_code', 'trade_date']],
                    panel.loc[valid, ['ts_code', 'trade_date']].reset_index(drop=True),
                    check_dtype=False, check_categorical=False)
            check_frozen(manifest)
            assert_source_stable(meta)
            write_json(split_output / 'summary.json', result)
            summary = dict(stage=config['stage'], candidate=candidate, run_id=output.name,
                features=list(columns), feature_count=len(columns),
                feature_definitions={c: FEATURE_DEFINITIONS[c] for c in columns},
                model_params=MODEL_PARAMS, environment=environment_versions(), panel_rows=len(panel),
                feature_index_preserved=True, frozen_files_unchanged=True, splits=[result],
                reference_summary_sha256=meta['reference_summary_sha256'],
                provenance_sha256=sha256_file(output / 'provenance.json'),
                config_sha256=sha256_file(output / 'config.json'),
                resources=dict(elapsed_seconds=time.perf_counter()-started,
                    peak_process_rss_mb=memory.peak_rss_bytes / 1024**2),
                prediction_keys_match_raw_panel=True,
                baseline_comparison_note='2021/2022 have no frozen prediction reference; comparison is to same-year baseline10 in aggregate tables')
            write_json(output / 'summary.json', summary)
        print(json.dumps({'accepted': str(output), 'metrics': result['metrics']}), flush=True)
    finally:
        if output is not None:
            state = read_json(output / 'status.json')
            state['elapsed_seconds'] = time.perf_counter()-started
            write_json(output / 'status.json', state)


def verify_shared_extension():
    """Undo only the explicit research split/reference guards and compare entire AST."""
    previous = ROOT / 'artifacts/features34_step2/executed_sources/scripts/run_features34.py'
    current = ROOT / 'scripts/run_features34.py'
    old = previous.read_text(encoding='utf-8')
    expected = old.replace('from src.validation.splits import get_split\n',
        'from src.validation.splits import get_split\nfrom src.validation.splits import TimeSplit\n')
    replacements = [
        ('    reference: dict,', '    reference: dict | None,'),
        ('    columns: tuple[str, ...],\n', '    columns: tuple[str, ...],\n    research_split: TimeSplit | None = None,\n'),
        ('    split = get_split(split_name)\n', '    split = research_split if research_split is not None else get_split(split_name)\n    if split.name != split_name:\n        raise ValueError("research split name mismatch")\n'),
        ('    if columns == BASE_COLUMNS and prediction_sha256', '    if reference is not None and columns == BASE_COLUMNS and prediction_sha256'),
        ('    if int(train_mask.sum()) != reference["train_samples"] or int(valid_mask.sum()) != reference["valid_prediction_rows"]:',
         '    if reference is not None and (int(train_mask.sum()) != reference["train_samples"] or int(valid_mask.sum()) != reference["valid_prediction_rows"]):'),
        ('    if columns == BASE_COLUMNS:\n', '    if reference is not None and columns == BASE_COLUMNS:\n'),
        ('"original_prediction_hash_matches": prediction_sha256 ==', '"original_prediction_hash_matches": None if reference is None else prediction_sha256 =='),
        ('"metrics_minus_baseline_v1_1": {key:', '"metrics_minus_baseline_v1_1": None if reference is None else {key:'),
    ]
    for before, after in replacements:
        if expected.count(before) != 1:
            raise AssertionError(f'unexpected prior runner source: {before}')
        expected = expected.replace(before, after)
    if ast.dump(ast.parse(expected)) != ast.dump(ast.parse(current.read_text(encoding='utf-8'))):
        raise AssertionError('shared runner changed beyond explicit split and optional frozen reference guards')
    return dict(previous_sha256=sha256_file(previous), current_sha256=sha256_file(current),
                full_ast_equal_after_only_documented_extensions=True,
                unchanged_training_scoring_feature_diagnostic_logic=True)


def reuse_2023(config, meta, reference):
    previous = read_json(ROOT / 'artifacts/features34_step2/summary.json')
    acceptance = read_json(ROOT / 'artifacts/features34_step2/acceptance.json')
    if not previous['accepted'] or not acceptance['accepted']:
        raise AssertionError('step2 not accepted')
    for relative, expected in acceptance['evidence_sha256'].items():
        if sha256_file(ROOT / 'artifacts/features34_step2' / relative) != expected:
            raise AssertionError(f'step2 evidence changed: {relative}')
    shared = verify_shared_extension()
    snapshot = ROOT / 'artifacts/features34_step2/executed_sources'
    unchanged = {}
    for p in snapshot.rglob('*'):
        if not p.is_file():
            continue
        relative = p.relative_to(snapshot).as_posix()
        if relative == 'scripts/run_features34.py':
            continue
        if sha256_file(ROOT / relative) != sha256_file(p):
            raise AssertionError(f'prior executed source changed: {relative}')
        unchanged[relative] = sha256_file(p)
    entries, records = [], []
    old_config = read_json(ROOT / 'configs/features34_step2.json')
    old_ref = next(s for s in reference['splits'] if s['split_name'] == 'primary_2023')
    current = {**meta, 'reference_summary_sha256': sha256_file(ROOT / config['reference_summary'])}
    for name, old_name in [('baseline10', 'baseline10'), ('full34', 'full34'), ('lean31', '34-A')]:
        record = next(r for r in previous['runs'] if r['candidate'] == old_name)
        directory = ROOT / record['directory']
        if sha256_file(directory / 'summary.json') != record['summary_sha256']:
            raise AssertionError('prior summary changed')
        summary = verify_run(directory, old_name, old_config, current, old_ref)
        if summary['features'] != list(select_features(**config['experiments'][name])):
            raise AssertionError('reuse feature mismatch')
        entries.append((name, 'primary_2023', directory, summary, 'reused_step2'))
        records.append(dict(candidate=name, previous_candidate=old_name, directory=record['directory'],
                            summary_sha256=record['summary_sha256'], saved_files_verified=True))
    write_json(OUTPUT / 'reuse_verification.json', dict(shared_extension=shared, unchanged_sources=unchanged,
               runs=records, step2_acceptance_sha256=sha256_file(ROOT / 'artifacts/features34_step2/acceptance.json')))
    return entries


def verify_new(directory, candidate, split_name, config, meta):
    summary = read_json(directory / 'summary.json')
    saved_config = read_json(directory / 'config.json')
    saved_meta = read_json(directory / 'provenance.json')
    if read_json(directory / 'status.json')['status'] != 'success':
        raise AssertionError('run failed')
    if summary['features'] != list(select_features(**config['experiments'][candidate])) or saved_config['requested'] != config:
        raise AssertionError('saved selection/config mismatch')
    if summary['model_params'] != MODEL_PARAMS or summary['environment'] != environment_versions():
        raise AssertionError('model/environment mismatch')
    if saved_meta['source_sha256'] != meta['source_sha256'] or saved_meta['dependencies'] != meta['dependencies'] or saved_meta['data'] != meta['data']:
        raise AssertionError('source/dependency/data provenance mismatch')
    if sha256_file(directory / 'provenance.json') != summary['provenance_sha256'] or sha256_file(directory / 'config.json') != summary['config_sha256']:
        raise AssertionError('metadata hash mismatch')
    result = summary['splits'][0]
    if result['dates'] != config['splits'][split_name] or result != read_json(directory / split_name / 'summary.json'):
        raise AssertionError('result/split mismatch')
    for relative, expected in result['file_sha256'].items():
        if sha256_file(directory / split_name / relative) != expected:
            raise AssertionError(f'saved artifact changed: {relative}')
    pred = pd.read_parquet(directory / split_name / 'predictions.parquet')
    if pred.duplicated(['ts_code', 'trade_date']).any() or not np.isfinite(pred.pred).all() or prediction_hash(pred.pred) != result['prediction_sha256']:
        raise AssertionError('prediction hash/key/finiteness failure')
    if result['prediction_coverage'] != 1 or not result['model_reload_predictions_equal'] or not summary['prediction_keys_match_raw_panel']:
        raise AssertionError('coverage/reload failure')
    if not np.isfinite(list(result['metrics'].values())).all() or result['official_comparison']['max_abs_difference'] > 1e-12:
        raise AssertionError('official score parity failure')
    if lgb.Booster(model_file=str(directory / split_name / 'models/lightgbm.txt')).feature_name() != summary['features']:
        raise AssertionError('model feature mismatch')
    if len(pd.read_csv(directory / split_name / 'monthly_metrics.csv')) != 12:
        raise AssertionError('incomplete monthly results')
    return summary


def comparison_tables(entries):
    baseline = {s: (d, r) for n, s, d, r, mode in entries if n == 'baseline10'}
    rows, months, diagnostics = [], [], []
    for name, split_name, directory, summary, mode in entries:
        result = summary['splits'][0]
        base_dir, base_summary = baseline[split_name]
        base = base_summary['splits'][0]
        for key in ('train_samples', 'valid_prediction_rows', 'purge_rows', 'split_train_rows'):
            if result[key] != base[key]:
                raise AssertionError(f'{split_name} candidates differ in {key}')
        diag = result['diagnostics']
        row = dict(candidate=name, year=int(split_name[-4:]), split=split_name, mode=mode,
            directory=directory.relative_to(ROOT).as_posix(), features=';'.join(summary['features']),
            feature_count=len(summary['features']), train_samples=result['train_samples'],
            valid_prediction_rows=result['valid_prediction_rows'], prediction_coverage=result['prediction_coverage'],
            **result['metrics'], score_minus_baseline10=result['metrics']['final_score']-base['metrics']['final_score'],
            **{f'{k}_minus_baseline10': v-base['metrics'][k] for k, v in result['metrics'].items() if k != 'final_score'},
            top_missing_label_fraction=diag['top_groups']['turnover']['missing_label_fraction'],
            top_invalid_price_fraction=diag['top_groups']['turnover']['invalid_price_fraction'],
            top_baseline_all_missing_fraction=diag['top_groups']['turnover']['all_features_missing_fraction'],
            top_candidate_all_missing_fraction=diag['candidate_features_top_groups']['turnover']['all_features_missing_fraction'],
            price_valid_only_turnover=diag['price_valid_only_turnover'],
            official_max_abs_difference=result['official_comparison']['max_abs_difference'],
            prediction_sha256=result['prediction_sha256'],
            elapsed_seconds=summary['resources']['elapsed_seconds'],
            peak_process_rss_mb=summary['resources']['peak_process_rss_mb'],
            **{f'{k}_contribution': v for k, v in result['score_contributions'].items()})
        rows.append(row)
        monthly = pd.read_csv(directory / split_name / 'monthly_metrics.csv')
        base_monthly = pd.read_csv(base_dir / split_name / 'monthly_metrics.csv')
        if monthly.month.tolist() != base_monthly.month.tolist():
            raise AssertionError('monthly dates mismatch')
        for k in ('ic', 'annual_excess', 'turnover', 'score'):
            monthly[f'{k}_minus_baseline10'] = monthly[k]-base_monthly[k]
        monthly.insert(0, 'candidate', name)
        monthly.insert(1, 'year', int(split_name[-4:]))
        months.append(monthly)
        daily = pd.read_csv(directory / split_name / 'daily_missing_diagnostics.csv')
        candidate_daily = pd.read_csv(directory / split_name / 'daily_candidate_missing_diagnostics.csv')
        price = pd.read_csv(directory / split_name / 'daily_price_valid_turnover.csv')
        daily['month'] = daily.trade_date // 100
        candidate_daily['month'] = candidate_daily.trade_date // 100
        price['month'] = price.trade_date // 100
        for (month, kind), frame in daily.groupby(['month', 'top_type']):
            counts = frame[['top_count', 'missing_label_count', 'invalid_price_count', 'all_features_missing_count']].sum()
            selected = candidate_daily[(candidate_daily.month == month) & (candidate_daily.top_type == kind)]
            rec = dict(candidate=name, year=int(split_name[-4:]), month=int(month), top_type=kind,
                       **{k: int(v) for k, v in counts.items()})
            for k in ('missing_label', 'invalid_price', 'all_features_missing'):
                rec[k+'_fraction'] = int(counts[k+'_count']) / int(counts.top_count)
            rec['candidate_all_features_missing_fraction'] = int(selected.all_features_missing_count.sum()) / int(counts.top_count)
            rec['price_valid_only_turnover'] = float(price.loc[price.month == month, 'turnover'].mean())
            diagnostics.append(rec)
    return pd.DataFrame(rows).sort_values(['year', 'feature_count']), pd.concat(months, ignore_index=True), pd.DataFrame(diagnostics)


def rank_candidates(annual, monthly, rule):
    rows = []
    for name, frame in annual[annual.candidate != 'baseline10'].groupby('candidate'):
        mf = monthly[monthly.candidate == name]
        delta = frame.score_minus_baseline10
        rows.append(dict(candidate=name, feature_count=int(frame.feature_count.iloc[0]),
            mean_annual_delta=float(delta.mean()), worst_annual_delta=float(delta.min()),
            negative_years=int((delta < 0).sum()), positive_months=int((mf.score_minus_baseline10 > 0).sum()),
            months=len(mf), worst_month_delta=float(mf.score_minus_baseline10.min()),
            median_month_delta=float(mf.score_minus_baseline10.median())))
    ranked = sorted(rows, key=lambda r: (r['negative_years'] > 0, -r['mean_annual_delta'],
                    -r['worst_annual_delta'], -r['positive_months']))
    selected = []
    pool = ranked.copy()
    while pool and len(selected) < rule['max_next_candidates']:
        first = pool[0]
        near = [r for r in pool if (r['negative_years'] > 0) == (first['negative_years'] > 0)
            and abs(r['mean_annual_delta']-first['mean_annual_delta']) <= rule['near_mean_delta']
            and abs(r['worst_annual_delta']-first['worst_annual_delta']) <= rule['near_worst_delta']
            and abs(r['positive_months']-first['positive_months']) <= rule['near_positive_months']]
        chosen = min(near, key=lambda r: r['feature_count'])
        selected.append(chosen['candidate'])
        pool.remove(chosen)
    for r in rows:
        r['proposed_for_feature_checks'] = r['candidate'] in selected
    return pd.DataFrame(rows), selected


def run_matrix():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / 'status.json', dict(status='running', stage='step3_development_only'))
    try:
        config = read_json(CONFIG)
        manifest, reference, meta = check_contract(config)
        dates = pd.read_csv(RAW_DATA_PATH, usecols=['trade_date'], dtype='int32').trade_date.unique()
        validate_dates(dates, config)
        candidates_path = ROOT / 'docs/features34/STEP3_CANDIDATES.md'
        prereg = dict(config_sha256=sha256_file(CONFIG), candidates_sha256=sha256_file(candidates_path),
                      config=config, source=meta, started_at=pd.Timestamp.now(tz='Asia/Shanghai').isoformat())
        write_json(OUTPUT / 'preregistration.json', prereg)
        entries = reuse_2023(config, meta, reference)
        snapshot = OUTPUT / 'executed_sources'
        for relative in meta['source_sha256']:
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        shutil.copy2(candidates_path, OUTPUT / 'STEP3_CANDIDATES.md')
        write_json(OUTPUT / 'run_index.json', dict(runs=[]))
        for split_name in SPLITS:
            for candidate in EXPECTED:
                if any(n == candidate and s == split_name for n, s, *_ in entries):
                    continue
                label = f'{candidate}_{split_name}'
                command = [sys.executable, '-B', str(Path(__file__).resolve()), '--candidate', candidate, '--split', split_name]
                with (OUTPUT / f'{label}.txt').open('w', encoding='utf-8') as handle:
                    done = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                write_json(OUTPUT / f'{label}_command.json', dict(command=command, exit_code=done.returncode))
                if done.returncode:
                    raise RuntimeError(f'{label} failed; see retained log and per-run failed status')
                parent = EXPERIMENT_ROOT / f'features34_step3_{candidate}_{split_name}'
                runs = list(parent.iterdir())
                if len(runs) != 1:
                    raise AssertionError('ambiguous run directory; refuse silently selecting a prior run')
                directory = runs[0]
                summary = verify_new(directory, candidate, split_name, config, meta)
                entries.append((candidate, split_name, directory, summary, 'new_run'))
                write_json(OUTPUT / 'run_index.json', dict(runs=[dict(candidate=n, split=s,
                    directory=d.relative_to(ROOT).as_posix(), mode=mode,
                    summary_sha256=sha256_file(d / 'summary.json')) for n, s, d, r, mode in entries]))
                print(f'Accepted {label}: {summary["splits"][0]["metrics"]["final_score"]:.12f}', flush=True)
        annual, monthly, missing = comparison_tables(entries)
        if (len(annual), len(monthly), len(missing)) != (15, 180, 360):
            raise AssertionError('incomplete matrix/monthly/diagnostic tables')
        ranking, selected = rank_candidates(annual, monthly, config['selection_rule'])
        for name, frame in [('comparison', annual), ('monthly_comparison', monthly),
                            ('monthly_missing_diagnostics', missing), ('candidate_ranking', ranking)]:
            frame.to_csv(OUTPUT / f'{name}.csv', index=False, float_format='%.17g')
        check_frozen(manifest)
        assert_source_stable(meta)
        if sha256_file(candidates_path) != prereg['candidates_sha256']:
            raise AssertionError('preregistered candidate rationale changed during matrix')
        summary = dict(accepted=True, stage=config['stage'], runs=read_json(OUTPUT / 'run_index.json')['runs'],
            proposed_next_candidates=selected, selection_rule=config['selection_rule'],
            new_runs=12, reused_runs=3, years=[2021, 2022, 2023], used_2024=False,
            feature_checks_run=False, final_retention_list_created=False,
            frozen_files_unchanged=True, elapsed_seconds=time.perf_counter()-started,
            tables={name: sha256_file(OUTPUT / f'{name}.csv') for name in
                    ['comparison', 'monthly_comparison', 'monthly_missing_diagnostics', 'candidate_ranking']})
        write_json(OUTPUT / 'summary.json', summary)
        write_json(OUTPUT / 'status.json', dict(status='success', elapsed_seconds=time.perf_counter()-started))
        print(json.dumps({'accepted': str(OUTPUT), 'proposed': selected}), flush=True)
    except BaseException as exc:
        write_json(OUTPUT / 'status.json', dict(status='failed', error_type=type(exc).__name__,
                   error=str(exc), elapsed_seconds=time.perf_counter()-started))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate')
    parser.add_argument('--split')
    args = parser.parse_args(argv)
    if args.candidate is None and args.split is None:
        run_matrix()
    else:
        # Keep invalid requests inside experiment_run so rejection has an immutable failed record.
        run_one(args.candidate, args.split)


if __name__ == '__main__':
    main()
