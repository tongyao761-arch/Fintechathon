"""Meaningful scope, conditional evidence, and paired block boundary checks."""
import copy
import unittest

import numpy as np
import pandas as pd

from scripts import run_features34_step4 as runner
from src.features.features34 import build_features34, GROUPS
from tests.test_features34 import panel


class TestStep4(unittest.TestCase):
    def setUp(self):
        self.cfg=runner.read_json(runner.CONFIG)

    def test_independent_45_deletions_and_scope(self):
        items=runner.matrix(self.cfg)
        self.assertEqual(len(items),45)
        self.assertEqual(sum(x['context']=='full34' for x in items),24)
        self.assertEqual(sum(x['context']=='lean31' for x in items),21)
        for item in items:
            columns=runner.selection(self.cfg,item)
            self.assertEqual(len(columns),33 if item['context']=='full34' else 30)
            self.assertNotIn(item['feature'],columns)
            self.assertEqual(columns[:10],runner.FEATURE_COLUMNS[:10])
        for change in [{'split':'oos_2024'}, {'exclude':['ret_1d']}, {'exclude':['ret_2d','ret_10d']}, {'context':'lean27'}]:
            with self.assertRaises(ValueError):runner.selection(self.cfg,{**items[0],**change})
        cfg=copy.deepcopy(self.cfg);cfg['model_params']={}
        with self.assertRaises(ValueError):runner.validate_config(cfg)

    def test_full_panel_projection_preserves_rank_source(self):
        raw=panel(stocks=3,days=70)
        full=build_features34(raw)
        for item in runner.matrix(self.cfg):
            columns=runner.selection(self.cfg,item)
            pd.testing.assert_frame_equal(full.loc[:,columns],build_features34(raw,columns),check_exact=True)
        item=next(x for x in runner.matrix(self.cfg) if x['feature']=='log_mean_amount_20d')
        self.assertIn('log_mean_amount_20d_rank_pct',runner.selection(self.cfg,item))

    def test_cross_year_conflict_and_incomplete_never_decided(self):
        r=self.cfg['rules']
        frame=pd.DataFrame(dict(year=[2021,2022,2023],delta_final_score=[.01,.01,.01],
            block_ci_lower=[.005]*3,block_ci_upper=[.02]*3,guards_pass=[True]*3))
        self.assertEqual(runner.classify(frame,r),'删除')
        self.assertEqual(runner.classify(frame.iloc[1:],r),'不确定')
        frame.loc[0,'delta_final_score']=-.02
        self.assertEqual(runner.classify(frame,r),'不确定')
        frame['delta_final_score']=-.01;frame['block_ci_upper']=-.005
        self.assertEqual(runner.classify(frame,r),'保留')
        frame.loc[0,'block_ci_upper']=.001
        self.assertEqual(runner.classify(frame,r),'不确定')

    def test_paired_blocks_exclude_incoming_and_seam_turnover(self):
        idx=pd.Index(range(40),name='trade_date')
        before=pd.DataFrame(dict(ic=np.zeros(40),excess=np.zeros(40),turnover=np.zeros(40)),index=idx)
        after=before.copy();after['turnover']=.2;after.loc[0,'turnover']=999
        rule={**self.cfg['rules']['bootstrap'],'block_trading_days':40,'repetitions':20}
        result=runner.paired_blocks(after,before,rule)
        self.assertAlmostEqual(result['block_ci_lower'],-.06)
        self.assertAlmostEqual(result['block_ci_upper'],-.06)
        with self.assertRaises(AssertionError):runner.paired_blocks(after.iloc[::-1],before,rule)
        after.loc[3,'ic']=np.nan
        with self.assertRaises(AssertionError):runner.paired_blocks(after,before,rule)


if __name__=='__main__':unittest.main()
