"""Meaningful label-free bridge and fixed-budget authorization checks."""
import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.validate_frozen_models import CONFIG, read, validate_config, parameters
from scripts.run_lightgbm_baseline import MODEL_PARAMS
from src.data.baseline_panel import load_raw_baseline_panel
from src.data.data_contract import DataContractError, X_COLUMNS
from src.data.frozen_test_adapter import adapt_test_x, bridge_x, training_boundary
from src.features.features34 import build_features34


def fixture():
    rows = []
    for stock in ("A", "B"):
        for i, date in enumerate(pd.bdate_range("2024-09-02", periods=100)):
            price = (10 if stock == "A" else 30) + i*.13
            rows.append(dict(ts_code=stock, trade_date=int(date.strftime("%Y%m%d")),
                open=price, high=price+1, low=price-1, close=price+.2,
                vol=100+i, amount=1000+i*20, flag_limit_up=0, flag_limit_down=i%7==0,
                y_ret_1d=.001))
    return pd.DataFrame(rows)


class ValidationTests(unittest.TestCase):
    def test_adapter_matches_original_float64_validity_before_float32(self):
        frame = fixture()
        frame.loc[0, "open"] = np.nan
        frame.loc[1, "high"] = frame.loc[1, "close"]-1e-8
        with TemporaryDirectory() as temp:
            p = Path(temp)/"train.csv"; frame.to_csv(p, index=False)
            history = load_raw_baseline_panel(p)
            test = adapt_test_x(pd.read_csv(p).drop(columns="y_ret_1d"))
        pd.testing.assert_frame_equal(history.loc[:, [*X_COLUMNS,"is_price_valid"]].reset_index(drop=True),
            test.loc[:, [*X_COLUMNS,"is_price_valid"]], check_dtype=False, check_categorical=False, check_exact=True)
        self.assertFalse("is_trainable" in test or "y_ret_1d" in test)

    def test_missing_prices_not_filled_and_stock_windows_do_not_cross(self):
        raw = fixture().drop(columns="y_ret_1d")
        raw.loc[10, ["open", "high", "low", "close", "vol", "amount"]] = np.nan
        x = adapt_test_x(raw).sort_values(["ts_code","trade_date"]).reset_index(drop=True)
        f = build_features34(x)
        self.assertTrue(np.isnan(f.loc[10, "intraday_ret"]))
        self.assertTrue(np.isnan(f.loc[11, "ret_1d"]))
        self.assertTrue(np.isnan(f.loc[100, "ret_1d"]))
        self.assertTrue(np.isnan(f.loc[159, "ret_60d"]))
        self.assertTrue(np.isfinite(f.loc[160, "ret_60d"]))
        self.assertTrue(np.isnan(f.loc[10, "ret_1d_rank_pct"]))

    def test_label_free_bridge_preserves_order_ids_and_history(self):
        frame = fixture()
        first = frame.groupby("ts_code", sort=False).head(1).assign(trade_date=20180102)
        past = pd.concat([first, frame[frame.trade_date.le(20241231)]]).sort_values(["ts_code", "trade_date"])
        future = frame[frame.trade_date.gt(20241231)].drop(columns="y_ret_1d").iloc[::-1].reset_index(drop=True)
        with TemporaryDirectory() as temp:
            p = Path(temp)/"train.csv"; past.to_csv(p, index=False)
            history = load_raw_baseline_panel(p)
        test = adapt_test_x(future); joined = bridge_x(history, test)
        f = build_features34(joined)
        old = build_features34(history)
        idx = joined[joined.history_row_id.ge(0)].sort_values("history_row_id").index
        pd.testing.assert_frame_equal(f.loc[idx].reset_index(drop=True), old, check_exact=True)
        out = joined[joined.test_row_id.ge(0)].sort_values("test_row_id")
        self.assertEqual(list(out.trade_date), list(future.trade_date))
        boundary = training_boundary(history, test)
        self.assertEqual(boundary["purge_date"], 20241231)
        self.assertEqual(boundary["train_end"], 20241230)

    def test_frozen_projections_equal_direct_build(self):
        freeze = read(CONFIG.parent.parent/"artifacts/features34_step5/FROZEN_CANDIDATES.json")
        x = adapt_test_x(fixture().drop(columns="y_ret_1d"))
        full = build_features34(x)
        for rec in freeze["candidates"]:
            direct = build_features34(x, tuple(rec["features"]))
            pd.testing.assert_frame_equal(full.loc[:, rec["features"]], direct, check_exact=True)

    def test_future_perturbation_and_truncation(self):
        raw = fixture().drop(columns="y_ret_1d"); x = adapt_test_x(raw)
        f = build_features34(x); cutoff = sorted(x.trade_date.unique())[70]
        early = x.trade_date.le(cutoff)
        truncated = build_features34(x.loc[early])
        changed = x.copy(); changed.loc[~early, "close"] = np.nan
        changed.loc[~early, "is_price_valid"] = 0
        changed.loc[~early, "flag_limit_down"] = 1
        modified = build_features34(changed)
        pd.testing.assert_frame_equal(f.loc[early], truncated, check_exact=True)
        pd.testing.assert_frame_equal(f.loc[early], modified.loc[early], check_exact=True)

    def test_reject_bad_keys_dates_labels_and_overlap(self):
        raw = fixture().drop(columns="y_ret_1d")
        for bad in (pd.concat([raw,raw.iloc[:1]]), raw.assign(trade_date=20250230),
                    raw.assign(y_ret_1d=np.nan), raw.assign(is_trainable=1)):
            with self.assertRaises(DataContractError): adapt_test_x(bad)
        x = adapt_test_x(raw)
        with self.assertRaises(DataContractError): bridge_x(x, x)

    def test_exact_independent_changes_and_budget(self):
        cfg = read(CONFIG); validate_config(cfg)
        for key, value in (("splits", ["oos_2024"]), ("candidates", ["lean27"]),
                           ("changes", {"B1":{"num_leaves":15,"reg_lambda":10.}}),
                           ("new_training_runs", {"A":2,"B":13})):
            bad = copy.deepcopy(cfg); bad[key]=value
            with self.assertRaises(ValueError): validate_config(bad)
        for probe, key, value in (("B1","num_leaves",15),("B2","reg_lambda",10.)):
            params = parameters(MODEL_PARAMS, probe)
            self.assertEqual([k for k in params if params[k] != MODEL_PARAMS[k]], [key])
            self.assertEqual(params[key], value)
        self.assertEqual(parameters(MODEL_PARAMS), MODEL_PARAMS)
        with self.assertRaises(ValueError): parameters(MODEL_PARAMS, "grid")


if __name__ == "__main__": unittest.main()
