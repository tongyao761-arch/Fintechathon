"""Score decisions, joint risk limits, fixed budgets and source/rank controls."""
import copy
import unittest
import numpy as np
import pandas as pd
from scripts import run_features34_step4_revision as runner
from src.features.features34 import build_features34
from tests.test_features34 import panel


class TestRevision(unittest.TestCase):
    def setUp(self):
        self.cfg = runner.read_json(runner.CONFIG)
        self.rules = self.cfg['rules']

    def frame(self, values):
        return pd.DataFrame(dict(year=[2021, 2022, 2023], delta_final_score=values,
            after_final_score=[.3, .31, .32], after_final_score_minus_baseline10=[.1, .11, .12],
            block_ci_lower=[-.02]*3, block_ci_upper=[.02]*3, guards_pass=[False]*3))

    def test_score_labels_not_ci_or_component_veto(self):
        self.assertEqual(runner.classify(self.frame([.002, .003, .004]), self.rules), '删除')
        self.assertEqual(runner.classify(self.frame([-.002, -.003, -.004]), self.rules), '保留')
        self.assertEqual(runner.classify(self.frame([-.002, .003, .004]), self.rules), '不确定')
        self.assertEqual(runner.classify(self.frame([.001, .003, .004]), self.rules), '不确定')
        self.assertEqual(runner.classify(self.frame([-.001, -.003, -.004]), self.rules), '不确定')
        self.assertEqual(runner.classify(self.frame([.003, .004, .005]).iloc[1:], self.rules), '不确定')
        self.assertEqual(runner.classify(self.frame([np.nan, .003, .004]), self.rules), '不确定')

    def test_joint_gain_not_compensate_clear_bad_year(self):
        self.assertFalse(runner.joint_gate(self.frame([-.006, .02, .02]), self.rules))
        self.assertTrue(runner.joint_gate(self.frame([-.004, .02, .02]), self.rules))
        self.assertTrue(runner.joint_gate(self.frame([-.005, .02, .02]), self.rules))
        self.assertFalse(runner.joint_gate(self.frame([.01, .01, .001]), self.rules))
        f = self.frame([.01, .01, .02]); f.loc[0, 'after_final_score_minus_baseline10'] = 0
        self.assertFalse(runner.joint_gate(f, self.rules))

    def test_component_tradeoff_official_formula(self):
        f = self.frame([.004, .004, .004]); f['delta_ic_mean'] = -.003
        # Worsened IC legitimately offset by a larger return contribution in official total.
        improvement = .4*(-.003)+.3*.02-.3*.001
        self.assertGreater(improvement, .001)
        f['delta_final_score'] = improvement
        self.assertTrue(runner.joint_gate(f, self.rules))

    def test_near_probe_does_not_establish_single_deletion(self):
        f = self.frame([.006, -.0007, .018])
        self.assertTrue(runner.probe(f, self.rules))
        self.assertEqual(runner.classify(f, self.rules), '不确定')
        self.assertFalse(runner.probe(self.frame([-.0011, .005, .006]), self.rules))

    def test_fixed_matrix_projection_and_prohibited_year(self):
        items = runner.matrix(self.cfg); self.assertEqual(len(items), 6)
        full = build_features34(panel(stocks=3, days=70))
        for item in items:
            cols = runner.columns(self.cfg, item)
            self.assertEqual(len(cols), 31 if item['context'] == 'full34' else 27)
            self.assertEqual(cols[:10], runner.old.FEATURE_COLUMNS[:10])
            pd.testing.assert_frame_equal(full.loc[:, cols], build_features34(panel(stocks=3, days=70), cols), check_exact=True)
        for changes in ({'split': 'oos_2024'}, {'exclude': ['ret_1d']}, {'version': 'unexpected'}):
            with self.assertRaises(ValueError): runner.columns(self.cfg, {**items[0], **changes})
        cfg = copy.deepcopy(self.cfg); cfg['versions'][0]['exclude'].append('body_ratio')
        with self.assertRaises(ValueError): runner.matrix(cfg)

    def test_score_priority_fewer_features_only_tie(self):
        records = [dict(candidate='many', eligible=True, score_2023=.31, worst_delta_vs_baseline10=.01, mean_score=.3, feature_count=34),
                   dict(candidate='few', eligible=True, score_2023=.309, worst_delta_vs_baseline10=.02, mean_score=.4, feature_count=20),
                   dict(candidate='failed', eligible=False, score_2023=.9, worst_delta_vs_baseline10=-.1, mean_score=.5, feature_count=10)]
        self.assertEqual([r['candidate'] for r in runner.order_candidates(records, 1e-12)], ['many', 'few'])
        records[0]['score_2023'] = .309
        self.assertEqual(runner.order_candidates(records, 1e-12)[0]['candidate'], 'few')


if __name__ == '__main__':
    unittest.main()
