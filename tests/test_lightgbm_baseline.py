import unittest

import numpy as np
import pandas as pd

from scripts.run_lightgbm_baseline import (
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS as MODEL_FEATURE_COLUMNS,
    MODEL_PARAMS,
    _compare_with_official_evaluator,
    split_masks,
)
from src.data.data_contract import DataContractError
from src.features.baseline_v1 import FEATURE_COLUMNS, build_baseline_v1_features
from src.metrics.official import score_official
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
            (
                "ret_1d",
                "ret_5d",
                "ret_20d",
                "gap_1d",
                "intraday_ret",
                "high_low_range",
                "volatility_5d",
                "volatility_20d",
                "volume_ratio_5d",
                "amount_ratio_5d",
            ),
        )
        self.assertEqual(MODEL_FEATURE_COLUMNS, FEATURE_COLUMNS)
        self.assertTrue(
            set(
                [
                    "ts_code",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                    "amount",
                    "flag_limit_up",
                    "flag_limit_down",
                    "y_ret_1d",
                    "is_price_valid",
                    "is_trainable",
                    "row_id",
                ]
            )
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

    def test_official_csv_round_trip_matches_local_score(self):
        dates = np.repeat([20240102, 20240103, 20240104], 100)
        rank = np.tile(np.arange(100), 3)
        pred = pd.DataFrame(
            {
                "ts_code": [f"{i % 100:06d}.SZ" for i in range(300)],
                "trade_date": dates,
                "pred": rank.astype(np.float64) / 97.0,
            }
        )
        truth = pred[["ts_code", "trade_date"]].copy()
        truth["y_ret_1d"] = (rank / 10000.0).astype(np.float32).astype(np.float64)
        x = pred[["ts_code", "trade_date"]].copy()
        x["flag_limit_up"] = 0
        local = score_official(pred, truth, x)
        comparison = _compare_with_official_evaluator(pred, truth, x, local)
        self.assertLessEqual(comparison["max_abs_difference"], 1e-12)


class TestBaselineV1Features(unittest.TestCase):
    @staticmethod
    def make_panel(stocks=("A",), rows_per_stock=25):
        rows = []
        for stock_number, stock in enumerate(stocks, start=1):
            scale = float(stock_number * 100)
            for offset in range(rows_per_stock):
                close = scale + offset + 1.0
                rows.append(
                    {
                        "ts_code": stock,
                        "trade_date": 20230000 + offset + 1,
                        "open": close - 0.5,
                        "high": close + 1.0,
                        "low": close - 1.0,
                        "close": close,
                        "vol": float((offset + 1) * stock_number * 10),
                        "amount": float((offset + 1) * stock_number * 1000),
                    }
                )
        return pd.DataFrame(rows)

    def test_exact_price_return_range_and_volatility_values(self):
        panel = self.make_panel()
        original = panel.copy(deep=True)
        features = build_baseline_v1_features(panel)
        pd.testing.assert_frame_equal(panel, original)
        row = 20
        close = panel["close"]
        expected_ret_1d = close.iloc[row] / close.iloc[row - 1] - 1.0
        expected_returns_5 = [
            close.iloc[position] / close.iloc[position - 1] - 1.0
            for position in range(row - 4, row + 1)
        ]
        expected_returns_20 = [
            close.iloc[position] / close.iloc[position - 1] - 1.0
            for position in range(1, row + 1)
        ]
        self.assertAlmostEqual(features.loc[row, "ret_1d"], expected_ret_1d)
        self.assertAlmostEqual(
            features.loc[row, "ret_5d"],
            close.iloc[row] / close.iloc[row - 5] - 1.0,
        )
        self.assertAlmostEqual(
            features.loc[row, "ret_20d"],
            close.iloc[row] / close.iloc[row - 20] - 1.0,
        )
        self.assertAlmostEqual(
            features.loc[row, "gap_1d"],
            panel.loc[row, "open"] / close.iloc[row - 1] - 1.0,
        )
        self.assertAlmostEqual(
            features.loc[row, "intraday_ret"],
            close.iloc[row] / panel.loc[row, "open"] - 1.0,
        )
        self.assertAlmostEqual(
            features.loc[row, "high_low_range"],
            panel.loc[row, "high"] / panel.loc[row, "low"] - 1.0,
        )
        self.assertAlmostEqual(
            features.loc[row, "volatility_5d"],
            np.std(expected_returns_5, ddof=1),
        )
        self.assertAlmostEqual(
            features.loc[row, "volatility_20d"],
            np.std(expected_returns_20, ddof=1),
        )

    def test_volume_and_amount_ratios_exclude_current_row(self):
        panel = self.make_panel(rows_per_stock=7)
        features = build_baseline_v1_features(panel)
        row = 5
        self.assertAlmostEqual(
            features.loc[row, "volume_ratio_5d"],
            panel.loc[row, "vol"] / panel.loc[: row - 1, "vol"].mean(),
        )
        self.assertAlmostEqual(
            features.loc[row, "amount_ratio_5d"],
            panel.loc[row, "amount"] / panel.loc[: row - 1, "amount"].mean(),
        )
        self.assertTrue(pd.isna(features.loc[row - 1, "volume_ratio_5d"]))

    def test_stock_boundary_does_not_share_history(self):
        panel = self.make_panel(stocks=("A", "B"), rows_per_stock=21)
        features = build_baseline_v1_features(panel)
        first_b = panel.index[panel["ts_code"].eq("B")][0]
        self.assertTrue(features.loc[first_b].isna()[["ret_1d", "ret_5d", "ret_20d"]].all())
        self.assertTrue(pd.isna(features.loc[first_b, "volatility_5d"]))
        self.assertTrue(pd.isna(features.loc[first_b, "volume_ratio_5d"]))

    def test_missing_history_and_zero_denominators_become_nan_not_infinity(self):
        panel = self.make_panel(rows_per_stock=7)
        panel.loc[0:5, ["vol", "amount"]] = 0.0
        panel.loc[3, "open"] = 0.0
        panel.loc[4, "low"] = 0.0
        panel.loc[4, "close"] = 0.0
        features = build_baseline_v1_features(panel)
        self.assertTrue(pd.isna(features.loc[3, "intraday_ret"]))
        self.assertTrue(pd.isna(features.loc[4, "high_low_range"]))
        self.assertTrue(pd.isna(features.loc[5, "ret_1d"]))
        self.assertTrue(pd.isna(features.loc[5, "gap_1d"]))
        self.assertTrue(pd.isna(features.loc[5, "volume_ratio_5d"]))
        self.assertTrue(pd.isna(features.loc[5, "amount_ratio_5d"]))
        self.assertFalse(np.isinf(features.to_numpy()).any())

    def test_future_changes_do_not_change_past_features(self):
        panel = self.make_panel(rows_per_stock=25)
        before = build_baseline_v1_features(panel)
        changed = panel.copy()
        future = changed.index > 15
        changed.loc[future, ["open", "high", "low", "close", "vol", "amount"]] *= 100.0
        after = build_baseline_v1_features(changed)
        pd.testing.assert_frame_equal(before.loc[:15], after.loc[:15])

    def test_rejects_interleaved_unsorted_rows(self):
        panel = self.make_panel(stocks=("A", "B"), rows_per_stock=3)
        interleaved = panel.sort_values("trade_date", kind="stable").reset_index(drop=True)
        with self.assertRaisesRegex(DataContractError, "sorted"):
            build_baseline_v1_features(interleaved)

    def test_rejects_duplicate_keys_and_missing_columns(self):
        panel = self.make_panel(rows_per_stock=3)
        duplicated = pd.concat([panel, panel.iloc[[0]]], ignore_index=True).sort_values(
            ["ts_code", "trade_date"], kind="stable"
        ).reset_index(drop=True)
        with self.assertRaisesRegex(DataContractError, "duplicate"):
            build_baseline_v1_features(duplicated)
        with self.assertRaisesRegex(DataContractError, "missing required columns"):
            build_baseline_v1_features(panel.drop(columns="amount"))


if __name__ == "__main__":
    unittest.main()
