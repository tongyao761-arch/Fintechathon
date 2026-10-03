"""Development-year purge, fixed matrix, reuse, selection, and failure contracts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import run_features34 as shared
from scripts import run_features34_step3 as runner
from src.data.baseline_panel import load_raw_baseline_panel, extract_truth_files
from src.features.baseline_v1 import FEATURE_COLUMNS as BASE_COLUMNS
from src.features.features34 import build_features34
from src.validation.splits import TimeSplit
from tests.test_features34 import panel


class TestStep3(unittest.TestCase):
    def setUp(self):
        self.config = runner.read_json(runner.CONFIG)

    def test_fixed_matrix_and_2024_rejected(self):
        for name, count in [('baseline10', 10), ('full34', 34), ('lean31', 31), ('lean27', 27), ('lean23', 23)]:
            for split in runner.SPLITS:
                selected = runner.validate_config(self.config, name, split)
                self.assertEqual(len(selected), count)
                self.assertEqual(selected[:10], BASE_COLUMNS)
        for name, split in [('lean23', 'oos_2024'), ('10+F', 'primary_2023')]:
            with self.assertRaises(ValueError):
                runner.validate_config(self.config, name, split)
        for edit in ('feature', 'model', 'year'):
            cfg = copy.deepcopy(self.config)
            if edit == 'feature': cfg['experiments']['lean23']['include'] = ['ret_2d']
            if edit == 'model': cfg['model_params'] = {'num_leaves': 99}
            if edit == 'year': cfg['splits']['oos_2024'] = {}
            with self.assertRaises(ValueError): runner.validate_config(cfg)

    def test_raw_date_boundary_and_purge(self):
        # Synthetic calendar deliberately has gaps: infer previous observed day, not calendar subtraction.
        dates = [20180102, 20201229, 20201230, 20201231, 20210104, 20211230,
                 20211231, 20220104, 20221229, 20221230, 20230103, 20231229]
        cfg = copy.deepcopy(self.config)
        cfg['boundary_source']['date_count'] = len(dates)
        cfg['boundary_source']['dates_sha256'] = runner.hashlib.sha256(np.asarray(dates, dtype='<i4').tobytes()).hexdigest()
        splits = runner.validate_dates(dates, cfg)
        for name in runner.SPLITS:
            s = splits[name]
            frame = pd.DataFrame({'trade_date': dates, 'is_trainable': 1, 'y_ret_1d': .01})
            frame.loc[frame.trade_date == s.train_end, 'is_trainable'] = 0
            train, valid = runner.split_masks(frame, s)
            self.assertFalse(train[frame.trade_date == s.purge_date].any())
            self.assertFalse(train[frame.trade_date == s.train_end].any())
            self.assertTrue((frame.loc[train, 'trade_date'] < s.valid_start).all())
            self.assertTrue(valid[frame.trade_date == s.valid_start].all())
        cfg['splits']['dev_2022']['train_end'] = 20211231
        with self.assertRaises(AssertionError): runner.validate_dates(dates, cfg)

    def test_shared_extension_only_documented_changes(self):
        self.assertTrue(runner.verify_shared_extension()['full_ast_equal_after_only_documented_extensions'])

    def test_failure_rejected_before_data_and_no_success_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner, 'EXPERIMENT_ROOT', Path(tmp)), patch.object(runner, 'load_raw_baseline_panel') as load:
                with self.assertRaises(ValueError): runner.run_one('lean23', 'oos_2024')
                load.assert_not_called()
            output = next(Path(tmp).glob('*/*'))
            self.assertEqual(runner.read_json(output / 'status.json')['status'], 'failed')
            self.assertIn('elapsed_seconds', runner.read_json(output / 'status.json'))
            self.assertFalse((output / 'summary.json').exists())

    def test_three_year_conflict_and_simplicity_tie(self):
        annual, monthly = [], []
        for name, count, deltas in [('full34', 34, [.05,.05,.05]), ('lean23', 23, [.049,.049,.049]),
                                    ('lean31',31,[-.01,.12,.13])]:
            for year, delta in zip((2021,2022,2023), deltas):
                annual.append(dict(candidate=name, feature_count=count, score_minus_baseline10=delta))
                for month in range(12): monthly.append(dict(candidate=name, score_minus_baseline10=delta))
        ranking, chosen = runner.rank_candidates(pd.DataFrame(annual),pd.DataFrame(monthly),self.config['selection_rule'])
        self.assertEqual(chosen, ['lean23','full34'])
        self.assertEqual(int(ranking.set_index('candidate').loc['lean31','negative_years']),1)

    def test_new_baseline_split_all_missing_rows_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'raw.csv'
            raw=panel(stocks=120,days=70).drop(columns=['is_price_valid','is_trainable'])
            dates=sorted(raw.trade_date.unique())
            missing=raw.ts_code.eq('000119.SZ') & raw.trade_date.ge(dates[66])
            raw.loc[missing,['open','high','low','close','y_ret_1d']]=np.nan
            raw.to_csv(source,index=False,float_format='%.17g')
            p=load_raw_baseline_panel(source); features=build_features34(p,BASE_COLUMNS)
            split=TimeSplit('development_fixture',dates[0],dates[64],dates[65],dates[66],dates[-1])
            output=root/'result'
            extract_truth_files(source,{output/'evaluate_input/测试集_Y.csv':(split.valid_start,split.valid_end)})
            with patch.object(shared,'get_split',side_effect=AssertionError('frozen split loader must not be used')):
                result=shared.run_split(p,features,split.name,output_dir=output,
                    reference=None,columns=BASE_COLUMNS,research_split=split)
            saved=pd.read_parquet(output/'predictions.parquet')
            self.assertEqual(len(saved),480)
            self.assertTrue(np.isfinite(saved.pred).all())
            self.assertEqual(result['prediction_coverage'],1.)
            self.assertTrue(result['model_reload_predictions_equal'])
            self.assertIsNone(result['original_prediction_hash_matches'])
            self.assertEqual(result['official_comparison']['max_abs_difference'],0)


if __name__ == '__main__': unittest.main()
