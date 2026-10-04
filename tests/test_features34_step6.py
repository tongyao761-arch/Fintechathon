"""Final repeat authorization, strict reproducibility, and failure recording."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_features34_step6 as runner
from src.validation.experiment import write_json


class Step6Tests(unittest.TestCase):
    def test_reject_new_candidate_year_budget_and_tuning(self):
        cfg = runner.read_json(runner.CONFIG); runner.validate_config(cfg)
        for key, value in [('candidates', runner.prior.IDS + ['full34']), ('splits', ['dev_2022']),
                           ('new_training_runs', 5), ('model_params', {'num_leaves': 63})]:
            changed = copy.deepcopy(cfg); changed[key] = value
            with self.assertRaises(ValueError): runner.validate_config(changed)

    def test_repeat_rejects_changed_predictions_scores_and_missing_diagnostics(self):
        keys = ('dates', 'train_samples', 'valid_prediction_rows', 'prediction_coverage', 'prediction_sha256',
                'purge_rows', 'split_train_rows', 'metrics', 'diagnostics', 'score_contributions', 'file_sha256')
        original = {key: {'value': 1} for key in keys}
        runner.compare_repeat(original, copy.deepcopy(original))
        for key in keys:
            changed = copy.deepcopy(original); changed[key] = {'value': 2}
            with self.assertRaises(AssertionError): runner.compare_repeat(original, changed)

    def test_failed_run_records_failure_without_success_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); write_json(out / 'failures.json', dict(failures=[]))
            with (patch.object(runner, 'OUTPUT', out),
                  patch.object(runner, 'registration_check', side_effect=ValueError('bad freeze')),
                  patch.object(runner, 'load_raw_baseline_panel') as loader):
                with self.assertRaises(ValueError): runner.main(['run'])
                loader.assert_not_called()
                self.assertEqual(runner.read_json(out / 'status.json')['status'], 'failed')
                self.assertEqual(len(runner.read_json(out / 'failures.json')['failures']), 1)
                self.assertFalse((out / 'summary.json').exists())

    def test_prepare_refuses_to_overwrite_registration(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, 'OUTPUT', Path(tmp)):
            with self.assertRaises(FileExistsError): runner.prepare()


if __name__ == '__main__': unittest.main()
