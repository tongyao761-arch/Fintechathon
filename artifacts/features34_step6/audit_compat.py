"""Retry the failed audit without changing the registered training sources."""
from pathlib import Path
import contextlib
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import run_features34_step6 as runner
from src.validation.experiment import sha256_file, write_json

ORIGINAL_TABLES = runner.prior.comparison_tables
REFERENCES = {}


def complete_tables(entries, cfg):
    """Give the legacy interval builder a real frozen comparator for baseline rows."""
    expanded = dict(entries)
    if expanded and all(name == 'baseline10' for name, _ in expanded):
        for _, split in entries:
            key = (runner.prior.IDS[0], split)
            if key not in REFERENCES:
                raise AssertionError('missing already frozen comparator; never synthesize a candidate')
            expanded[key] = REFERENCES[key]
    return ORIGINAL_TABLES(expanded, cfg)


def main():
    global REFERENCES
    REFERENCES = runner.registration_check()[-1]
    state = runner.read_json(runner.OUTPUT / 'status.json')
    if state['status'] == 'failed':
        failures = runner.read_json(runner.OUTPUT / 'failures.json')['failures']
        if not any(r['phase'] == 'audit' and r['error_type'] == 'KeyError' and r['error'] == "'year'" for r in failures):
            raise AssertionError('only the recorded baseline-only table failure may be retried here')
        write_json(runner.OUTPUT / 'audit_attempt01_status.json', state)
        write_json(runner.OUTPUT / 'status.json', dict(status='runs_complete_audit_pending', new_training_runs=4,
            retry_reason='legacy paired interval table empty for baseline-only aggregation'))
    checks = runner.read_json(runner.OUTPUT / 'compat_tests_result.json')
    if checks['exit_code'] != 0:
        raise AssertionError('compatibility regression checks failed')
    with contextlib.ExitStack() as stack:
        from unittest.mock import patch
        stack.enter_context(patch.object(runner.prior, 'comparison_tables', complete_tables))
        runner.main(['audit'])
    accepted = runner.read_json(runner.OUTPUT / 'acceptance.json')
    for name in ('audit_compat.py', 'test_audit_compat.py', 'compat_tests_result.json', 'audit_attempt01_status.json'):
        accepted['evidence_sha256'][name] = sha256_file(runner.OUTPUT / name)
    accepted['audit_retry_did_not_retrain_models'] = True
    accepted['additional_compatibility_tests'] = 2
    write_json(runner.OUTPUT / 'acceptance.json', accepted)
    summary = runner.read_json(runner.OUTPUT / 'summary.json')
    summary['acceptance_sha256'] = sha256_file(runner.OUTPUT / 'acceptance.json')
    write_json(runner.OUTPUT / 'summary.json', summary)


if __name__ == '__main__': main()
