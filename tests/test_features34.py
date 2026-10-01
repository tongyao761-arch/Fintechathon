import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts import run_features34 as runner
from src.data.baseline_panel import extract_truth_files, load_raw_baseline_panel
from src.data.data_contract import DataContractError
from src.features.baseline_v1 import FEATURE_COLUMNS as BASE_COLUMNS, build_baseline_v1_features
from src.features.features34 import FEATURE_COLUMNS, GROUPS, RANK_SOURCES, build_features34, select_features
from src.validation.splits import TimeSplit


def panel(stocks=3, days=75):
    rows = []
    dates = pd.bdate_range("2023-01-02", periods=days).strftime("%Y%m%d").astype(int)
    for stock in range(stocks):
        for t, date in enumerate(dates):
            close = 10 + stock * 3 + t * .2 + t*t*.003
            rows.append([f"{stock:06d}.SZ", date, close-.3, close+.8, close-.7,
                         close, 100+t, 1000+10*t+stock*50, int(t%7==0), int(t%9==0),
                         .03*np.sin(stock*.15+t*.2), 1, 1])
    frame = pd.DataFrame(rows, columns=["ts_code", "trade_date", "open", "high", "low", "close",
                                      "vol", "amount", "flag_limit_up", "flag_limit_down", "y_ret_1d",
                                      "is_price_valid", "is_trainable"])
    # Noncontiguous, reversed index labels: features must align by labels, not positions.
    frame.index = np.arange(len(frame))[::-1] * 3 + 7
    return frame


class TestFeatures34(unittest.TestCase):
    def test_all_24_formulas_against_independent_numpy_values(self):
        p = panel(); f = build_features34(p); b = build_baseline_v1_features(p)
        pd.testing.assert_frame_equal(f.loc[:, BASE_COLUMNS], b, check_exact=True)
        self.assertEqual(len(FEATURE_COLUMNS), 34)
        self.assertEqual(len(set(FEATURE_COLUMNS)), 34)
        self.assertEqual(tuple(f.columns), FEATURE_COLUMNS)
        expected = {}
        g = p[p.ts_code.eq("000000.SZ")]; last = g.iloc[-1]; idx = g.index[-1]
        c = g.close.to_numpy(); a = g.amount.to_numpy()
        for n in (2, 10, 60):
            expected[f"ret_{n}d"] = c[-1]/c[-1-n]-1
        expected.update(close_location=(last.close-last.low)/(last.high-last.low),
                        upper_shadow=(last.high-max(last.open,last.close))/last.close,
                        lower_shadow=(min(last.open,last.close)-last.low)/last.close,
                        body_ratio=(last.close-last.open)/(last.high-last.low))
        for n in (20,60):
            expected[f"bias_{n}d"] = c[-1]/np.mean(c[-n:])-1
            expected[f"price_position_{n}d"] = (c[-1]-np.min(g.low.to_numpy()[-n:]))/(np.max(g.high.to_numpy()[-n:])-np.min(g.low.to_numpy()[-n:]))
        expected["amount_ratio_20d"] = a[-1]/np.mean(a[-21:-1])
        expected["log_mean_amount_20d"] = np.log1p(np.mean(a[-20:]))
        expected["volatility_60d"] = np.std(b.loc[g.index, "ret_1d"].to_numpy()[-60:], ddof=1)
        expected["flag_limit_up"] = last.flag_limit_up
        expected["flag_limit_down"] = last.flag_limit_down
        expected["limit_up_count_5d"] = np.sum(g.flag_limit_up.to_numpy()[-5:])
        expected["limit_down_count_5d"] = np.sum(g.flag_limit_down.to_numpy()[-5:])
        for name,value in expected.items():
            with self.subTest(feature=name):
                np.testing.assert_allclose(f.loc[idx,name],value,rtol=2e-6,atol=1e-8)
        same_day = p.trade_date.eq(last.trade_date)
        for source in RANK_SOURCES:
            values = f.loc[same_day,source].to_numpy()
            for row,value in zip(f.index[same_day],values):
                average_position = np.sum(values<value)+(np.sum(values==value)+1)/2
                self.assertAlmostEqual(f.loc[row,source+"_rank_pct"],average_position/len(values),places=6)

    def test_stock_boundary_and_index_alignment(self):
        p = panel(); f = build_features34(p)
        for _,g in p.groupby("ts_code"):
            isolated = build_features34(g)
            pd.testing.assert_frame_equal(f.loc[g.index, FEATURE_COLUMNS[:-6]],isolated.loc[:,FEATURE_COLUMNS[:-6]])
            self.assertTrue(f.loc[g.index[0], ["ret_2d","ret_60d","volatility_60d","amount_ratio_20d"]].isna().all())
        self.assertTrue(f.index.equals(p.index))

    def test_history_warmup_and_current_vs_prior_amount(self):
        p=panel(stocks=1); f=build_features34(p)
        for name,n in (("ret_2d",2),("ret_10d",10),("ret_60d",60),("volatility_60d",60),("amount_ratio_20d",20)):
            self.assertTrue(f[name].iloc[:n].isna().all())
            self.assertTrue(np.isfinite(f[name].iloc[n]))
        for name,n in (("bias_20d",20),("bias_60d",60),("log_mean_amount_20d",20),("limit_up_count_5d",5)):
            self.assertTrue(f[name].iloc[:n-1].isna().all())
            self.assertTrue(np.isfinite(f[name].iloc[n-1]))
        changed=p.copy(); changed.loc[p.index[20],"amount"]*=100
        altered=build_features34(changed)
        self.assertAlmostEqual(altered.amount_ratio_20d.iloc[20]/f.amount_ratio_20d.iloc[20],100,places=4)

    def test_full_finite_windows_and_missing_shadow_input(self):
        p=panel(stocks=1); p.loc[p.index[65], ["amount","flag_limit_up","low","open"]]=np.nan
        f=build_features34(p)
        self.assertTrue(f.amount_ratio_20d.iloc[66:].isna().all())
        self.assertTrue(f.log_mean_amount_20d.iloc[65:].isna().all())
        self.assertTrue(f.price_position_20d.iloc[65:].isna().all())
        self.assertTrue(f.limit_up_count_5d.iloc[65:70].isna().all())
        self.assertTrue(np.isfinite(f.limit_up_count_5d.iloc[70]))
        self.assertTrue(f.loc[p.index[65],["upper_shadow","lower_shadow","body_ratio"]].isna().all())
        p=panel(stocks=1); p.loc[p.index[30],"close"]=np.inf
        f=build_features34(p)
        self.assertTrue(f.bias_20d.iloc[30:50].isna().all())
        self.assertFalse(np.isinf(f.to_numpy()).any())

    def test_zero_denominators_and_nonfinite_results_are_nan(self):
        p=panel(stocks=1); p[["open","high","low","close","amount"]]=0.
        f=build_features34(p)
        names=(*GROUPS["A"],*GROUPS["B"],*GROUPS["C"],"amount_ratio_20d")
        self.assertTrue(f.loc[:,names].isna().all().all())
        p.amount=-1.; f=build_features34(p)
        self.assertTrue(f.log_mean_amount_20d.isna().all())

    def test_future_x_never_changes_past_features(self):
        p=panel(); before=build_features34(p); changed=p.copy()
        cutoff=sorted(p.trade_date.unique())[65]
        future=p.trade_date.gt(cutoff)
        changed.loc[future,["open","high","low","close","vol","amount"]]*=3
        changed.loc[future,["flag_limit_up","flag_limit_down"]]=1
        after=build_features34(changed)
        pd.testing.assert_frame_equal(before.loc[~future],after.loc[~future],check_exact=True)

    def test_label_and_training_flags_do_not_enter_features(self):
        p=panel(); before=build_features34(p); p.y_ret_1d=np.nan; p.is_trainable=0
        pd.testing.assert_frame_equal(before,build_features34(p),check_exact=True)
        pd.testing.assert_frame_equal(before,build_features34(p.drop(columns=["y_ret_1d","is_trainable"])),check_exact=True)

    def test_same_date_rank_ties_pool_and_other_date_independence(self):
        p=panel(stocks=4); last=p.trade_date.max()
        for stock in range(4):
            positions=p.index[p.ts_code.eq(f"{stock:06d}.SZ")]
            p.loc[positions[-2],"close"]=10.
            p.loc[positions[-1],"close"]=[11.,11.,12.,100.][stock]
        p.loc[p.ts_code.eq("000003.SZ") & p.trade_date.eq(last),"is_price_valid"]=0
        f=build_features34(p); mask=p.trade_date.eq(last)
        np.testing.assert_allclose(f.loc[mask,"ret_1d_rank_pct"].to_numpy(),[.5,.5,1.,np.nan],equal_nan=True)
        changed=p.copy(); changed.loc[p.trade_date.eq(last),"close"]*=10
        after=build_features34(changed)
        pd.testing.assert_frame_equal(f.loc[~mask],after.loc[~mask],check_exact=True)
        # Changing a different day's cross-section does not enter a purely same-day source rank.
        changed=p.copy(); changed.loc[p.trade_date.eq(sorted(p.trade_date.unique())[0]),"amount"]*=1e6
        after=build_features34(changed)
        pd.testing.assert_series_equal(f.loc[mask,"log_mean_amount_20d_rank_pct"],after.loc[mask,"log_mean_amount_20d_rank_pct"])
        p.loc[p.ts_code.eq("000002.SZ") & p.trade_date.eq(last),"close"]=np.nan
        f=build_features34(p)
        np.testing.assert_allclose(f.loc[mask,"ret_1d_rank_pct"],[.75,.75,np.nan,np.nan],equal_nan=True)

    def test_rank_source_independent_selection(self):
        p=panel(); full=build_features34(p)
        columns=select_features(include=["log_mean_amount_20d_rank_pct"])
        result=build_features34(p,columns)
        self.assertNotIn("log_mean_amount_20d",result)
        pd.testing.assert_frame_equal(result,full.loc[:,columns])
        columns=select_features(groups=["D","F"],exclude=["log_mean_amount_20d"])
        pd.testing.assert_frame_equal(build_features34(p,columns),full.loc[:,columns])

    def test_invalid_selection_and_panel_rejected(self):
        for spec in ({"groups":["G"]},{"include":["is_trainable"]},{"exclude":["ret_1d"]},
                     {"groups":["A","A"]},{"exclude":["ret_2d"]}):
            with self.assertRaises(ValueError): select_features(**spec)
        p=panel()
        for broken in (p.iloc[::-1],p.set_axis([0]*len(p))):
            with self.assertRaises(DataContractError): build_features34(broken)
        with self.assertRaises(ValueError): build_features34(p, ["y_ret_1d"])

    def test_frozen_ten_only_exact(self):
        p=panel()
        pd.testing.assert_frame_equal(build_features34(p,BASE_COLUMNS),build_baseline_v1_features(p),check_exact=True)


class TestCandidateRunner(unittest.TestCase):
    def test_stage_authorization_and_2024_rejection(self):
        config=json.loads((runner.ROOT/"configs/features34.json").read_text())
        self.assertEqual(runner.resolve_selection(config,"baseline10","oos_2024"),BASE_COLUMNS)
        self.assertEqual(runner.resolve_selection(config,"full34","primary_2023"),FEATURE_COLUMNS)
        for name,split in (("full34","oos_2024"),("10+A","primary_2023"),("34-F","primary_2023")):
            with self.assertRaises(ValueError): runner.resolve_selection(config,name,split)
        config["allowed_runs"]["full34"].append("oos_2024")
        with self.assertRaises(ValueError): runner.resolve_selection(config,"full34","oos_2024")

    def test_failed_run_records_status_without_success_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner,"EXPERIMENT_ROOT",Path(directory)), patch.object(runner,"load_raw_baseline_panel") as load:
                with self.assertRaises(ValueError):
                    runner.main(["--candidate","full34","--split","oos_2024"])
                load.assert_not_called()
            output=next(Path(directory).glob("*/*"))
            state=json.loads((output/"status.json").read_text())
            self.assertEqual(state["status"],"failed")
            self.assertIn("elapsed_seconds",state)
            self.assertFalse((output/"summary.json").exists())

    def test_small_full34_end_to_end_all_validation_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"raw.csv"
            raw=panel(stocks=120,days=70).drop(columns=["is_price_valid","is_trainable"])
            dates=sorted(raw.trade_date.unique())
            # Missing validation prices/labels remain prediction keys.
            missing=raw.ts_code.eq("000119.SZ") & raw.trade_date.ge(dates[66])
            raw.loc[missing,["open","high","low","close","y_ret_1d"]]=np.nan
            raw.to_csv(source,index=False,float_format="%.17g")
            p=load_raw_baseline_panel(source); f=build_features34(p)
            split=TimeSplit("test",dates[0],dates[64],dates[65],dates[66],dates[-1])
            train,valid=runner.split_masks(p,split)
            # Full34 need not reproduce a baseline hash, but masks/counts must match.
            reference={"prediction_sha256":"baseline-reference", "train_samples":int(train.sum()),
                       "valid_prediction_rows":int(valid.sum()),
                       "metrics":json.loads((runner.ROOT/"artifacts/baseline_v1_1/summary.json").read_text(encoding="utf-8"))["splits"][0]["metrics"]}
            output=root/"result"
            extract_truth_files(source,{output/"evaluate_input/测试集_Y.csv":(split.valid_start,split.valid_end)})
            with patch.object(runner,"get_split",return_value=split):
                result=runner.run_split(p,f,"test",output_dir=output,reference=reference,columns=FEATURE_COLUMNS)
            saved=pd.read_parquet(output/"predictions.parquet")
            self.assertEqual(len(saved),480)
            self.assertTrue(np.isfinite(saved.pred).all())
            self.assertEqual(result["prediction_coverage"],1.)
            self.assertTrue(result["model_reload_predictions_equal"])
            self.assertLessEqual(result["official_comparison"]["max_abs_difference"],1e-12)
            self.assertTrue((output/"monthly_metrics.csv").exists())
            self.assertTrue((output/"daily_candidate_missing_diagnostics.csv").exists())
            self.assertTrue((output/"feature_missing_statistics.csv").exists())
            self.assertEqual(str(pd.read_csv(output/"evaluate_input/测试集_Y.csv").y_ret_1d.dtype),"float64")


if __name__ == "__main__":
    unittest.main()
