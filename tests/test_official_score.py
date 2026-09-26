import unittest

import numpy as np
import pandas as pd

from src.data.data_contract import DataContractError
from src.metrics.official import score_official


def sample_frames():
    dates = np.repeat([20240102, 20240103, 20240104], 100)
    rank = np.tile(np.arange(100), 3)
    pred = pd.DataFrame({"ts_code": [f"{i % 100:06d}.SZ" for i in range(300)], "trade_date": dates, "pred": rank.astype(float)})
    truth = pred[["ts_code", "trade_date"]].copy()
    truth["y_ret_1d"] = rank / 10000.0
    x = pred[["ts_code", "trade_date"]].copy()
    x["flag_limit_up"] = 0
    return pred, truth, x


class TestOfficialScore(unittest.TestCase):
    def test_perfect_rank_has_expected_ic_and_zero_turnover(self):
        pred, truth, x = sample_frames()
        result = score_official(pred, truth, x)
        self.assertAlmostEqual(result["ic_mean"], 1.0, places=12)
        self.assertAlmostEqual(result["mean_turnover"], 0.0, places=12)
        self.assertGreater(result["annual_excess"], 0.0)

    def test_rejects_missing_prediction_key(self):
        pred, truth, x = sample_frames()
        with self.assertRaises(DataContractError):
            score_official(pred.iloc[1:], truth, x)

    def test_rejects_nonfinite_prediction(self):
        pred, truth, x = sample_frames()
        pred.loc[0, "pred"] = np.inf
        with self.assertRaises(DataContractError):
            score_official(pred, truth, x)


if __name__ == "__main__":
    unittest.main()
