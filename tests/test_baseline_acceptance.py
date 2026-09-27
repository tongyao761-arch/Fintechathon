import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts import run_lightgbm_baseline as runner
from src.data.baseline_panel import load_raw_baseline_panel, compare_clean_input, extract_truth_files
from src.data.data_contract import DataContractError
from src.features.baseline_v1 import FEATURE_COLUMNS, build_baseline_v1_features
from src.metrics.diagnostics import baseline_diagnostics
from src.metrics.official import score_official
from src.validation.experiment import experiment_run, prediction_hash, write_json
from src.validation.splits import TimeSplit


def frames(days=4, stocks=120):
    rank = np.tile(np.arange(stocks), days)
    pred = pd.DataFrame({"ts_code": [f"{i:06d}.SZ" for i in rank],
                         "trade_date": np.repeat(np.arange(20240102, 20240102 + days), stocks),
                         "pred": rank.astype(float)})
    truth = pred[["ts_code", "trade_date"]].copy()
    truth["y_ret_1d"] = rank / 10000.0
    x = pred[["ts_code", "trade_date"]].copy()
    x["flag_limit_up"] = 0
    return pred, truth, x


class TestScoringFailurePaths(unittest.TestCase):
    def test_constant_predictions_in_labeled_subset_fail(self):
        pred, truth, x = frames()
        day = pred.trade_date.eq(20240102)
        pred.loc[day, "pred"] = 1.0
        pred.loc[0, "pred"] = 2.0
        truth.loc[0, "y_ret_1d"] = np.nan
        with self.assertRaisesRegex(DataContractError, "constant"):
            score_official(pred, truth, x)

    def test_constant_labels_fail(self):
        pred, truth, x = frames()
        truth.loc[truth.trade_date.eq(20240102), "y_ret_1d"] = 0.0
        with self.assertRaisesRegex(DataContractError, "constant"):
            score_official(pred, truth, x)

    def test_infinite_label_fails_even_if_limit_up(self):
        pred, truth, x = frames()
        for value in (np.inf, -np.inf):
            truth.loc[0, "y_ret_1d"] = value
            x.loc[0, "flag_limit_up"] = 1
            with self.assertRaisesRegex(DataContractError, "non-finite"):
                score_official(pred, truth, x)

    def test_single_valid_ic_day_fails(self):
        pred, truth, x = frames()
        truth.loc[truth.trade_date.ne(20240102), "y_ret_1d"] = np.nan
        with self.assertRaisesRegex(DataContractError, "not enough"):
            score_official(pred, truth, x)

    def test_no_turnover_pair_fails(self):
        pred, truth, x = frames(days=3, stocks=100)
        x.loc[100, "flag_limit_up"] = 1
        with self.assertRaisesRegex(DataContractError, "not enough"):
            score_official(pred, truth, x)

    def test_skipped_day_breaks_turnover_and_matches_official(self):
        pred, truth, x = frames(days=4, stocks=100)
        pred.loc[pred.trade_date.ge(20240104), "pred"] *= -1
        x.loc[100, "flag_limit_up"] = 1
        result = score_official(pred, truth, x, return_details=True)
        self.assertEqual(result["mean_turnover"], 0.0)
        self.assertEqual(result["details"]["daily_turnover"].previous_trade_date.tolist(), [20240104])
        result.pop("details")
        self.assertLessEqual(runner._compare_with_official_evaluator(pred, truth, x, result)["max_abs_difference"], 1e-12)

    def test_short_first_or_last_date_matches_official(self):
        for row in (0, 399):
            pred, truth, x = frames(days=4, stocks=100)
            x.loc[row, "flag_limit_up"] = 1
            result = score_official(pred, truth, x)
            self.assertLessEqual(runner._compare_with_official_evaluator(pred, truth, x, result)["max_abs_difference"], 1e-12)

    def test_nonfinite_official_or_local_metrics_never_pass(self):
        for value in (np.nan, np.inf, -np.inf):
            for local, official in (({"ic": value}, {"ic": 0.1}), ({"ic": 0.1}, {"ic": value})):
                with self.assertRaisesRegex(AssertionError, "non-finite"):
                    runner.compare_metrics(local, official)
        with self.assertRaises(AssertionError):
            runner.compare_metrics({}, {})

    def test_actual_official_nan_is_rejected(self):
        pred, truth, x = frames()
        local = score_official(pred, truth, x)
        official = {**local, "final_score": np.nan}
        with patch.object(runner, "_load_official_evaluator", return_value=lambda *args: official):
            with self.assertRaisesRegex(AssertionError, "official metrics contain non-finite"):
                runner._compare_with_official_evaluator(pred, truth, x, local)


class TestBaselineDiagnostics(unittest.TestCase):
    def test_hand_calculated_counts_and_no_mutation(self):
        pred, truth, x = frames(days=3)
        rank = np.tile(np.arange(120), 3)
        truth.loc[rank >= 118, "y_ret_1d"] = np.nan
        quality = x.copy()
        quality["is_price_valid"] = (rank < 118).astype(int)
        quality["baseline_features_all_missing"] = rank >= 117
        before = pred.copy(deep=True)
        scored = score_official(pred, truth, x, return_details=True)
        summary, tables = baseline_diagnostics(pred, truth, quality, scored["details"])
        top = summary["top_groups"]["turnover"]
        self.assertEqual(top["top_count"], 36)
        self.assertEqual(top["missing_label_count"], 6)
        self.assertEqual(top["invalid_price_count"], 6)
        self.assertEqual(top["all_features_missing_count"], 9)
        self.assertEqual(summary["top_groups"]["return"]["missing_label_count"], 0)
        self.assertEqual(summary["top_groups"]["return"]["all_features_missing_count"], 3)
        self.assertEqual(summary["price_valid_only_turnover"], 0)
        self.assertEqual(len(tables["daily_missing_diagnostics"]), 6)
        pd.testing.assert_frame_equal(pred, before)
        self.assertEqual({k: v for k, v in scored.items() if k != "details"}, score_official(pred, truth, x))


class TestDataAndArtifacts(unittest.TestCase):
    def test_raw_loader_builds_current_row_flags_without_filling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            raw = pd.DataFrame({"ts_code": ["A"] * 5, "trade_date": range(20240101, 20240106),
                                "open": [10., 10, 0, 10, 10], "high": [12., 9, 12, 12, 12],
                                "low": [9., 9, 9, 9, 9], "close": [11., 11, 11, np.nan, 11],
                                "vol": 100., "amount": 1000., "flag_limit_up": 0,
                                "flag_limit_down": 0, "y_ret_1d": [0.12345678901234567, .1, .1, .1, np.nan]})
            raw.to_csv(path, index=False)
            panel = load_raw_baseline_panel(path)
            self.assertEqual(panel.is_price_valid.tolist(), [1, 0, 0, 0, 1])
            self.assertEqual(panel.is_trainable.tolist(), [1, 0, 0, 0, 0])
            self.assertEqual(str(panel.close.dtype), "float32")
            self.assertEqual(str(panel.y_ret_1d.dtype), "float64")
            self.assertTrue(pd.isna(panel.close.iloc[3]))
            clean = panel.copy(); clean["y_ret_1d"] = clean.y_ret_1d.astype("float32")
            self.assertEqual(compare_clean_input(panel, clean)["status"], "passed")
            clean.loc[0, "is_trainable"] = 0
            with self.assertRaises(AssertionError):
                compare_clean_input(panel, clean)

    def test_original_label_tokens_and_shared_csv_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / "raw.csv"; fixture = root / "fixture"
            pred, _, x = frames(days=3, stocks=100)
            pred["pred"] = np.floor(pred.pred / 2) + np.where(pred.index % 2, 1e-15, 0)
            tokens = []
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["ts_code", "trade_date", "extra", "y_ret_1d"])
                for row in pred.itertuples(index=False):
                    rank = int(row.ts_code[:6])
                    token = "2.2351742234860698e-08" if rank < 2 else format(rank / 10000., ".18g")
                    tokens.append(token)
                    writer.writerow([row.ts_code, row.trade_date, "unused", token])
            destination = fixture / "测试集_Y.csv"
            counts = extract_truth_files(source, {destination: (20240102, 20240104)})
            self.assertEqual(counts[str(destination)], 300)
            with destination.open(encoding="utf-8", newline="") as handle:
                self.assertEqual([r["y_ret_1d"] for r in csv.DictReader(handle)], tokens)
            pd.testing.assert_series_equal(pd.read_csv(source).y_ret_1d, pd.read_csv(destination).y_ret_1d)
            pred.to_csv(fixture / "submission.csv", index=False, float_format="%.17g")
            x.to_csv(fixture / "测试集_X.csv", index=False)
            _, comparison, _, _ = runner.score_saved_inputs(fixture)
            self.assertLessEqual(comparison["max_abs_difference"], 1e-12)
            with self.assertRaises(FileExistsError):
                extract_truth_files(source, {destination: (20240102, 20240104)})

    def test_run_collision_failure_and_missing_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with experiment_run(root, "test", "ok") as output:
                write_json(output / "summary.json", {"accepted": True})
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "success")
            with self.assertRaises(FileExistsError):
                with experiment_run(root, "test", "ok"):
                    pass
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with experiment_run(root, "test", "failed") as failed:
                    raise RuntimeError("intentional failure")
            self.assertEqual(json.loads((failed / "status.json").read_text())["status"], "failed")
            self.assertFalse((failed / "summary.json").exists())
            with self.assertRaisesRegex(RuntimeError, "without an accepted summary"):
                with experiment_run(root, "test", "missing"):
                    pass
            with self.assertRaises(ValueError):
                with experiment_run(root, "../escape"):
                    pass

    def test_nonfinite_summary_cannot_mark_run_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                with experiment_run(Path(directory), "test", "nan") as output:
                    write_json(output / "summary.json", {"score": np.nan})
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")
            self.assertFalse((output / "summary.json").exists())

    def test_default_is_2023_and_full_validation_requires_explicit_all(self):
        self.assertEqual(runner.parse_args([]).split, "primary_2023")
        self.assertEqual(runner.parse_args(["--split", "all"]).split, "all")

    def test_small_end_to_end_model_reload_and_key_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw_path = root / "raw.csv"
            rows = []
            for i in range(100):
                for day in range(1, 26):
                    close = 10 + i * .2 + day * .1
                    rows.append([f"{i:06d}.SZ", 20230100 + day, close - .1, close + .2, close - .2,
                                 close, 100 + i + day, 1000 + i + day, 0, 0, .03 * np.sin(i * .15 + day * .2)])
            pd.DataFrame(rows, columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol",
                                        "amount", "flag_limit_up", "flag_limit_down", "y_ret_1d"]).to_csv(raw_path, index=False)
            panel = load_raw_baseline_panel(raw_path); features = build_baseline_v1_features(panel)
            split = TimeSplit("test", 20230101, 20230120, 20230121, 20230122, 20230125)
            train, valid = runner.split_masks(panel, split)
            model = lgb.LGBMRegressor(**runner.MODEL_PARAMS)
            model.fit(features.loc[train, FEATURE_COLUMNS], panel.loc[train, "y_ret_1d"].astype("float32"), feature_name=FEATURE_COLUMNS)
            prediction = model.predict(features.loc[valid, FEATURE_COLUMNS])
            pred = panel.loc[valid, ["ts_code", "trade_date"]].copy(); pred["pred"] = prediction
            truth = panel.loc[valid, ["ts_code", "trade_date", "y_ret_1d"]]
            x = panel.loc[valid, ["ts_code", "trade_date", "flag_limit_up"]]
            reference = {"prediction_sha256": prediction_hash(prediction), "train_samples": int(train.sum()),
                         "valid_prediction_rows": int(valid.sum()), "metrics": score_official(pred, truth, x)}
            output = root / "result"
            extract_truth_files(raw_path, {output / "evaluate_input" / "测试集_Y.csv": (20230122, 20230125)})
            with patch.object(runner, "get_split", return_value=split):
                result = runner.run_split(panel, features, "test", output_dir=output, reference=reference)
            self.assertTrue(result["model_reload_predictions_equal"])
            self.assertEqual(result["valid_prediction_rows"], 400)
            saved = pd.read_parquet(output / "predictions.parquet")
            pd.testing.assert_frame_equal(saved, pred.reset_index(drop=True))
            self.assertTrue((output / "daily_return_top_sets.csv").exists())
            self.assertTrue((output / "monthly_metrics.csv").exists())
            self.assertLessEqual(result["official_comparison"]["max_abs_difference"], 1e-12)


if __name__ == "__main__":
    unittest.main()
