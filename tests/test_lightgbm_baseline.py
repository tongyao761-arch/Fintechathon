import unittest

import numpy as np
import pandas as pd

from scripts.run_lightgbm_baseline import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    MODEL_PARAMS,
    split_masks,
)
from src.validation.splits import TimeSplit


class TestLightGBMBaseline(unittest.TestCase):
    def setUp(self):
        self.split = TimeSplit(
            name="test",
            train_start=20230101,
            train_end=20230102,
            purge_date=20230103,
            valid_start=20230104,
            valid_end=20230105,
        )

    def test_feature_contract_is_exact_and_excludes_keys_and_labels(self):
        self.assertEqual(
            FEATURE_COLUMNS,
            [
                "open",
                "high",
                "low",
                "close",
                "vol",
                "amount",
                "flag_limit_up",
                "flag_limit_down",
            ],
        )
        self.assertTrue(
            set(["ts_code", "trade_date", "y_ret_1d", "is_price_valid", "is_trainable", "row_id"])
            <= set(EXCLUDED_COLUMNS)
        )

    def test_train_filter_does_not_filter_validation_rows(self):
        panel = pd.DataFrame(
            {
                "trade_date": [20230101, 20230102, 20230103, 20230104, 20230105],
                "is_trainable": [1, 0, 1, 0, 0],
                "y_ret_1d": [0.01, 0.02, 0.03, np.nan, 0.05],
            }
        )
        train, valid = split_masks(panel, self.split)
        self.assertEqual(train.tolist(), [True, False, False, False, False])
        self.assertEqual(valid.tolist(), [False, False, False, True, True])

    def test_nonfinite_training_label_is_rejected(self):
        panel = pd.DataFrame(
            {
                "trade_date": [20230101, 20230104],
                "is_trainable": [1, 0],
                "y_ret_1d": [np.inf, 0.01],
            }
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            split_masks(panel, self.split)

    def test_model_configuration_is_fixed_and_deterministic(self):
        self.assertTrue(MODEL_PARAMS["deterministic"])
        self.assertEqual(MODEL_PARAMS["subsample"], 1.0)
        self.assertEqual(MODEL_PARAMS["colsample_bytree"], 1.0)
        self.assertIsInstance(MODEL_PARAMS["random_state"], int)


if __name__ == "__main__":
    unittest.main()
