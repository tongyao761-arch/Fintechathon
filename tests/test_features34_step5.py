"""Authorization, immutable prereview registration, and failed-run boundaries."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import run_features34_step5 as runner
from src.validation.experiment import write_json


class Step5Tests(unittest.TestCase):
    def test_reject_extra_candidate_split_tuning_and_budget(self):
        cfg=runner.read_json(runner.CONFIG)
        runner.validate_config(cfg)
        for key,value in [('candidates',runner.IDS+['full34']),('split','primary_2023'),('new_training_runs',3),('model_params',{'num_leaves':63})]:
            changed=copy.deepcopy(cfg); changed[key]=value
            with self.assertRaises(ValueError): runner.validate_config(changed)

    def test_candidates_are_prior_frozen_canonical_inputs(self):
        records=runner.candidates(runner.read_json(runner.CONFIG))
        self.assertEqual([len(r['features']) for r in records],[27,31])
        for record in records:
            self.assertEqual(record['features'][:10],list(runner.BASE_COLUMNS))
            self.assertNotIn('is_trainable',record['features'])

    def test_failure_before_data_load_records_failed_without_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); cfg=runner.read_json(runner.CONFIG); cfg['split']='dev_2021'
            config=out/'bad.json'; write_json(config,cfg)
            with patch.object(runner,'OUTPUT',out),patch.object(runner,'CONFIG',config),patch.object(runner,'check_contract',return_value=({}, {}, {})),patch.object(runner,'load_raw_baseline_panel') as loader:
                with self.assertRaises(ValueError): runner.main(['run'])
                loader.assert_not_called()
                self.assertEqual(runner.read_json(out/'status.json')['status'],'failed')
                self.assertEqual(len(runner.read_json(out/'failures.json')['failures']),1)
                self.assertFalse((out/'summary.json').exists())

    def test_negative_year_is_not_cancelled_by_other_years(self):
        annual=[]; monthly=[]
        for name in runner.IDS:
            for year,delta in [(2021,.15),(2022,.12),(2023,.07),(2024,-.01)]:
                annual.append(dict(candidate=name,year=year,final_score_minus_baseline10=delta))
            for month in range(1,13): monthly.append(dict(candidate=name,year=2024,month=202400+month,score_minus_baseline10=-.01))
        table=runner.conclusion_tables(dict(annual_comparison=pd.DataFrame(annual),monthly_comparison=pd.DataFrame(monthly)))
        self.assertFalse(table.all_four_years_above_baseline10.any())
        self.assertTrue(table.next_stage.str.startswith('不支持').all())
        self.assertTrue(table.negative_score_years.eq('2024').all())


if __name__=='__main__': unittest.main()
