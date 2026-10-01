"""Independent formulas, adversarial boundaries, and real training-path checks."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts import audit_baseline as audit
from scripts import run_lightgbm_baseline as runner
from src.data.baseline_panel import extract_truth_files, load_raw_baseline_panel
from src.data.data_contract import DataContractError
from src.features.baseline_v1 import FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, build_baseline_v1_features
from src.metrics.official import score_official
from src.validation.experiment import experiment_run, prediction_hash, sha256_file, write_json
from src.validation.splits import TimeSplit
from src.validation.submission import export_submission, validate_submission


def panel_fixture(stocks=4, days=45, dtype="float32"):
    dates = pd.bdate_range("2022-01-03", periods=days).strftime("%Y%m%d").astype(int)
    rows = []
    for i in range(stocks):
        for j, date in enumerate(dates):
            close = 10 + i * .21 + j * .12 + np.sin(i * .31 + j * .2) * .25
            rows.append([f"{i:06d}.SZ", int(date), close - .1, close + .3, close - .3,
                         close, 100 + i + j * 2, 1000 + i * 8 + j * 5, 0, 0,
                         .03 * np.sin(i * .15 + j * .2)])
    frame = pd.DataFrame(rows, columns=[*audit.KEYS, *RAW_FEATURE_COLUMNS,
                                       "flag_limit_up", "flag_limit_down", "y_ret_1d"])
    frame[list(RAW_FEATURE_COLUMNS)] = frame[list(RAW_FEATURE_COLUMNS)].astype(dtype)
    frame["is_price_valid"] = 1
    frame["is_trainable"] = 1
    return frame


def score_frames(sizes=(120, 120, 120, 120), ties=False):
    rows = []
    for day, n in enumerate(sizes):
        for i in range(n):
            rows.append([f"{i:06d}.SZ", 20240102 + day,
                         float(i // 30 if ties else i),
                         (i if day % 2 == 0 else n - 1 - i) / 10000., 0])
    frame = pd.DataFrame(rows, columns=[*audit.KEYS, "pred", "y_ret_1d", "flag_limit_up"])
    return frame[[*audit.KEYS, "pred"]], frame[[*audit.KEYS, "y_ret_1d"]], frame[[*audit.KEYS, "flag_limit_up"]]


def independent_score(pred, truth, x):
    """Hand-style rank correlations and set economics; no scipy/scorer helpers."""
    labels = dict(zip(map(tuple, truth[audit.KEYS].to_numpy()), truth.y_ret_1d))
    limits = dict(zip(map(tuple, x[audit.KEYS].to_numpy()), x.flag_limit_up))
    days = {}
    for row in pred.itertuples(index=False):
        key = (row.ts_code, row.trade_date)
        days.setdefault(row.trade_date, []).append((row.ts_code, row.pred, labels[key], limits[key]))
    correlations, top_returns, excesses, turnover = [], [], [], []
    previous = None
    for date in sorted(days):
        rows = days[date]
        labeled = [r for r in rows if np.isfinite(r[2])]
        if len(labeled) >= 30:
            ranks = []
            for col in (1, 2):
                values = [r[col] for r in labeled]
                ranks.append(np.array([sum(v < a for v in values) + (sum(v == a for v in values) + 1) / 2 for a in values]))
            a, b = [r - sum(r) / len(r) for r in ranks]
            correlations.append(float(sum(a * b) / np.sqrt(sum(a * a) * sum(b * b))))
        eligible = sorted([r for r in labeled if r[3] == 0], key=lambda r: r[1], reverse=True)
        if len(eligible) >= 100:
            count = len(eligible) // 10
            top = sum(r[2] for r in eligible[:count]) / count
            market = sum(r[2] for r in eligible) / len(eligible)
            top_returns.append(top)
            excesses.append(top - market)
        eligible = sorted([r for r in rows if r[3] == 0], key=lambda r: r[1], reverse=True)
        if len(eligible) < 100:
            previous = None
            continue
        current = {r[0] for r in eligible[:len(eligible) // 10]}
        if previous is not None:
            turnover.append(1 - len(previous & current) / len(previous | current))
        previous = current
    mean = sum(correlations) / len(correlations)
    std = (sum((r - mean) ** 2 for r in correlations) / (len(correlations) - 1)) ** .5
    annual = sum(excesses) / len(excesses) * 252
    turn = sum(turnover) / len(turnover)
    return {"ic_mean": mean, "ic_std": std, "icir": mean / std if std > 0 else 0.,
            "ic_positive_ratio": sum(r > 0 for r in correlations) / len(correlations),
            "annual_excess": annual, "top1_annual_ret": sum(top_returns) / len(top_returns) * 252,
            "mean_turnover": turn, "final_score": mean * .4 + annual * .3 + (1 - turn) * .3}


class TestIndependentFeatures(unittest.TestCase):
    def test_positive_stock_price_rescaling_preserves_ratio_features(self):
        frame = panel_fixture()
        original = build_baseline_v1_features(frame)
        for i, code in enumerate(frame.ts_code.unique()):
            frame.loc[frame.ts_code.eq(code), ["open", "high", "low", "close"]] *= 2 ** (i + 1)
        np.testing.assert_allclose(build_baseline_v1_features(frame), original, rtol=1e-6, atol=1e-7, equal_nan=True)

    def test_all_formulas_multiple_lengths_and_precision(self):
        for dtype in ("float32", "float64"):
            frame = panel_fixture(dtype=dtype)
            frame = frame.drop(frame.index[frame.ts_code.eq("000001.SZ")][-11:])
            frame.index = np.arange(len(frame)) * 17 + 9
            frame.ts_code = pd.Categorical(frame.ts_code, categories=["unused", *sorted(frame.ts_code.unique())])
            original = frame.copy(deep=True)
            result = audit.compare_feature_oracle(frame)
            self.assertTrue(result["nan_positions_equal"])
            pd.testing.assert_frame_equal(frame, original)
            full = build_baseline_v1_features(frame)
            for code in frame.ts_code.drop_duplicates():
                subset = frame.ts_code.eq(code)
                pd.testing.assert_frame_equal(full.loc[subset], build_baseline_v1_features(frame.loc[subset]))

    def test_missing_zero_and_all_missing_histories(self):
        frame = panel_fixture()
        frame.loc[3, "close"] = np.nan
        frame.loc[8, "open"] = 0
        frame.loc[9, "low"] = 0
        frame.loc[20:25, ["vol", "amount"]] = 0
        frame.loc[frame.ts_code.eq("000003.SZ"), list(RAW_FEATURE_COLUMNS)] = np.nan
        audit.compare_feature_oracle(frame)
        self.assertFalse(np.isinf(build_baseline_v1_features(frame)).any().any())

    def test_cast_overflow_is_nan_after_conversion(self):
        frame = panel_fixture(stocks=1, days=2, dtype="float64")
        frame.loc[0, ["open", "low", "close"]] = 1e-40
        frame.loc[0, "high"] = 1
        frame.loc[1, list(RAW_FEATURE_COLUMNS)] = 1
        features = build_baseline_v1_features(frame)
        self.assertFalse(np.isinf(features).any().any())
        self.assertTrue(pd.isna(features.loc[1, "ret_1d"]))
        self.assertTrue(pd.isna(features.loc[0, "high_low_range"]))

    def test_nonfinite_inputs_and_large_finite_values(self):
        frame = panel_fixture(dtype="float64")
        frame.loc[1, "close"] = np.inf
        frame.loc[3, "vol"] = -np.inf
        frame.loc[4, "amount"] = 1e300
        frame.loc[6, "open"] = 1e-300
        result = build_baseline_v1_features(frame)
        self.assertFalse(np.isinf(result).any().any())
        self.assertTrue(all(str(dtype) == "float32" for dtype in result.dtypes))

    def test_truncation_future_perturbation_and_stock_isolation(self):
        frame = panel_fixture()
        dates = sorted(frame.trade_date.unique())
        evidence = audit.assert_prefix_invariance(frame, [dates[5], dates[20], dates[21]])
        self.assertTrue(evidence["future_perturbation_equal"])

    def test_rejects_duplicate_index_keys_unsorted_missing_fields(self):
        frame = panel_fixture()
        invalid = frame.copy(); invalid.index = [0] * len(invalid)
        duplicate_key = pd.concat([frame, frame.iloc[[0]]]).sort_values(audit.KEYS).reset_index(drop=True)
        for bad in (invalid, duplicate_key, frame.iloc[::-1], frame.drop(columns="vol")):
            with self.assertRaises(DataContractError):
                build_baseline_v1_features(bad)

    def test_history_to_test_continuation_and_observation_windows(self):
        frame = panel_fixture()
        dates = sorted(frame.trade_date.unique())
        history = frame.loc[frame.trade_date.le(dates[25])]
        future = frame.loc[frame.trade_date.gt(dates[25])]
        joined = pd.concat([history, future]).sort_values(audit.KEYS)
        combined = build_baseline_v1_features(joined)
        pd.testing.assert_frame_equal(combined, build_baseline_v1_features(frame))
        separate = build_baseline_v1_features(future)
        first = future.groupby("ts_code", observed=True).head(1).index
        self.assertTrue(combined.loc[first, "ret_20d"].notna().all())
        self.assertTrue(separate.loc[first, "ret_20d"].isna().all())
        # Missing calendar observations do not silently change row-based windows.
        gapped = frame.drop(frame.index[::7])
        audit.compare_feature_oracle(gapped)


class TestRawAndTraining(unittest.TestCase):
    def test_independent_labels_and_terminal_target_price(self):
        frame = panel_fixture(stocks=2, days=8, dtype="float64")
        dates = sorted(frame.trade_date.unique())
        targets = []
        for _, group in frame.groupby("ts_code", sort=False):
            prices = group.close.to_numpy()
            frame.loc[group.index[:-1], "y_ret_1d"] = prices[1:] / prices[:-1] - 1
            frame.loc[group.index[-1], "y_ret_1d"] = .1
            targets.append([group.ts_code.iloc[0], 20220201, prices[-1] * 1.1])
        next_rows = pd.DataFrame(targets, columns=[*audit.KEYS, "close"])
        split = TimeSplit("fixture", dates[0], dates[3], dates[4], dates[5], dates[-1])
        with patch.object(runner, "SPLIT_NAMES", ["fixture"]), patch.object(audit, "get_split", return_value=split):
            result = audit.label_check(frame, next_rows)
            self.assertEqual(result["checked"], len(frame))
            self.assertEqual(result["terminal_cross_file_checked"], 2)
            self.assertEqual(result["nonmissing_uncheckable"], 0)
            wrong = frame.copy(); wrong.loc[0, "y_ret_1d"] += .001
            with self.assertRaisesRegex(AssertionError, "mismatches"):
                audit.label_check(wrong, next_rows)
            missing = frame.copy(); missing.loc[2, "close"] = np.nan
            self.assertGreater(audit.label_check(missing, next_rows)["nonmissing_uncheckable"], 0)
        unsafe = TimeSplit("fixture", dates[0], dates[4], 20500101, dates[5], dates[-1])
        with patch.object(runner, "SPLIT_NAMES", ["fixture"]), patch.object(audit, "get_split", return_value=unsafe):
            with self.assertRaisesRegex(AssertionError, "reaches validation"):
                audit.label_check(frame, next_rows)

    def test_invalid_calendar_dates_fail_and_raw_rows_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.csv"
            frame = panel_fixture(stocks=1)
            frame.loc[1, "trade_date"] = 20220199
            frame = frame.sort_values(audit.KEYS)
            frame.to_csv(source, index=False)
            with self.assertRaisesRegex(DataContractError, "calendar"):
                load_raw_baseline_panel(source)
            frame = panel_fixture(stocks=1)
            frame.loc[1, list(RAW_FEATURE_COLUMNS)] = np.nan
            frame.to_csv(source, index=False)
            loaded = load_raw_baseline_panel(source)
            self.assertEqual(len(loaded), len(frame))
            self.assertTrue(loaded.loc[1, list(RAW_FEATURE_COLUMNS)].isna().all())

    def test_raw_audit_rejects_sign_flags_infinite_and_ohlc_errors(self):
        for column, value in (("vol", -1), ("amount", -1), ("close", np.inf), ("flag_limit_up", 2), ("high", .01)):
            frame = panel_fixture()
            frame.loc[0, column] = value
            with self.assertRaises(AssertionError):
                audit.validate_raw(frame, "fixture")

    def test_actual_fit_inputs_and_prediction_invariance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = panel_fixture(stocks=110)
            # Exercise the real raw-CSV parsing contract, rather than comparing
            # in-memory synthetic floats with independently parsed CSV tokens.
            initial_path = root / "initial.csv"
            initial.to_csv(initial_path, index=False, float_format="%.17g")
            initial = load_raw_baseline_panel(initial_path)
            dates = sorted(initial.trade_date.unique())
            split = TimeSplit("fixture", dates[0], dates[24], dates[25], dates[26], dates[-1])
            train, valid = runner.split_masks(initial, split)
            features = build_baseline_v1_features(initial)
            model = lgb.LGBMRegressor(**runner.MODEL_PARAMS)
            model.fit(features.loc[train, FEATURE_COLUMNS], initial.loc[train, "y_ret_1d"].astype("float32"), feature_name=FEATURE_COLUMNS)
            values = model.predict(features.loc[valid, FEATURE_COLUMNS])
            pred = initial.loc[valid, audit.KEYS].copy(); pred["pred"] = values
            reference = {"prediction_sha256": prediction_hash(values), "train_samples": int(train.sum()),
                         "valid_prediction_rows": int(valid.sum()),
                         "metrics": score_official(pred, initial.loc[valid, [*audit.KEYS, "y_ret_1d"]], initial.loc[valid, [*audit.KEYS, "flag_limit_up"]])}
            captured = []
            original_fit = lgb.LGBMRegressor.fit
            def capture(instance, x, y, **kwargs):
                captured.append((x.copy(), y.copy(), kwargs.copy()))
                return original_fit(instance, x, y, **kwargs)
            for variant in ("base", "perturbed"):
                frame = initial.copy(deep=True)
                if variant == "perturbed":
                    frame.loc[valid, "y_ret_1d"] *= -3
                    frame.loc[valid, ["is_trainable", "is_price_valid"]] = 0
                    frame.loc[frame.trade_date.eq(split.purge_date), "y_ret_1d"] = .7
                    frame.loc[frame.index[valid][0], "y_ret_1d"] = np.nan
                source = root / f"{variant}.csv"
                frame.to_csv(source, index=False, float_format="%.17g")
                frame["y_ret_1d"] = pd.read_csv(source).y_ret_1d
                output = root / variant
                extract_truth_files(source, {output / "evaluate_input/测试集_Y.csv": (split.valid_start, split.valid_end)})
                with patch.object(runner, "get_split", return_value=split), patch.object(lgb.LGBMRegressor, "fit", capture):
                    result = runner.run_split(frame, build_baseline_v1_features(frame), "fixture", output_dir=output, reference=reference)
                self.assertEqual(result["prediction_sha256"], reference["prediction_sha256"])
                self.assertEqual(result["valid_prediction_rows"], int(valid.sum()))
            for x, y, kwargs in captured:
                pd.testing.assert_frame_equal(x, features.loc[train, FEATURE_COLUMNS])
                pd.testing.assert_series_equal(y, initial.loc[train, "y_ret_1d"].astype("float32"))
                self.assertEqual(tuple(kwargs["feature_name"]), FEATURE_COLUMNS)
                self.assertFalse(initial.loc[x.index, "trade_date"].eq(split.purge_date).any())
            self.assertEqual(len(captured), 2)

    def test_float32_training_label_overflow_fails_before_fit(self):
        frame = panel_fixture(stocks=110)
        dates = sorted(frame.trade_date.unique())
        split = TimeSplit("fixture", dates[0], dates[24], dates[25], dates[26], dates[-1])
        frame.loc[0, "y_ret_1d"] = 1e100
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "get_split", return_value=split), patch.object(lgb.LGBMRegressor, "fit") as fitted:
            with self.assertRaisesRegex(ValueError, "overflow"):
                runner.run_split(frame, build_baseline_v1_features(frame), "fixture", output_dir=Path(directory), reference={})
            fitted.assert_not_called()


class TestIndependentScoring(unittest.TestCase):
    def test_hand_scoring_and_official_sample_boundaries(self):
        for boundary in (29, 30, 99, 100, 101, 120):
            pred, truth, x = score_frames((120, boundary, 120, 120))
            actual = score_official(pred, truth, x)
            expected = independent_score(pred, truth, x)
            for name in expected:
                self.assertAlmostEqual(actual[name], expected[name], delta=1e-12)
            parity = runner._compare_with_official_evaluator(pred, truth, x, actual)
            self.assertLessEqual(parity["max_abs_difference"], 1e-12)

    def test_hand_scoring_missing_labels_limits_and_breaks(self):
        pred, truth, x = score_frames()
        truth.loc[120:129, "y_ret_1d"] = np.nan
        x.loc[140:164, "flag_limit_up"] = 1
        result = score_official(pred, truth, x)
        expected = independent_score(pred, truth, x)
        for name in expected:
            self.assertAlmostEqual(result[name], expected[name], delta=1e-12)
        self.assertLessEqual(runner._compare_with_official_evaluator(pred, truth, x, result)["max_abs_difference"], 1e-12)

    def test_no_ties_independent_shuffles_and_affine_rank_invariance(self):
        pred, truth, x = score_frames()
        original = score_official(pred, truth, x)
        changed = pred.sample(frac=1, random_state=1).copy()
        changed["pred"] = changed.pred * 2 + 3
        shuffled = score_official(changed, truth.sample(frac=1, random_state=2), x.sample(frac=1, random_state=3))
        for name in original:
            self.assertAlmostEqual(shuffled[name], original[name], delta=1e-12)

    def test_positive_and_negative_rank_direction(self):
        pred, truth, x = score_frames()
        truth["y_ret_1d"] = pred.pred / 10000
        positive = score_official(pred, truth, x)
        pred["pred"] *= -1
        negative = score_official(pred, truth, x)
        self.assertAlmostEqual(positive["ic_mean"], 1)
        self.assertAlmostEqual(negative["ic_mean"], -1)
        self.assertGreater(positive["annual_excess"], 0)
        self.assertLess(negative["annual_excess"], 0)

    def test_ties_match_official_and_have_measurable_order_sensitivity(self):
        pred, truth, x = score_frames(ties=True)
        evidence = audit.tie_sensitivity(pred, truth, x)
        self.assertGreater(evidence["duplicate_prediction_rows"], 0)
        self.assertGreater(evidence["metric_ranges"]["mean_turnover"], 0)
        self.assertTrue(all(r["official_max_abs_difference"] <= 1e-12 for r in evidence["runs"]))

    def test_missing_extra_duplicate_keys_in_every_input_fail(self):
        frames = score_frames()
        for i in range(3):
            for kind in ("missing", "extra", "duplicate"):
                modified = [f.copy() for f in frames]
                original = modified[i]
                if kind == "missing":
                    modified[i] = original.iloc[1:]
                else:
                    row = original.iloc[[0]].copy()
                    if kind == "extra":
                        row["ts_code"] = "EXTRA"
                    modified[i] = pd.concat([original, row], ignore_index=True)
                with self.assertRaises(DataContractError):
                    score_official(*modified)

    def test_nonfinite_predictions_and_invalid_limit_flags_fail(self):
        for value in (np.nan, np.inf, -np.inf):
            pred, truth, x = score_frames(); pred.loc[0, "pred"] = value
            with self.assertRaises(DataContractError):
                score_official(pred, truth, x)
        pred, truth, x = score_frames(); x.loc[0, "flag_limit_up"] = 2
        with self.assertRaises(DataContractError):
            score_official(pred, truth, x)


class TestAuditAndExport(unittest.TestCase):
    def test_recurring_control_alert_is_same_for_raw_and_residual_controls(self):
        self.assertTrue(audit.recurring_ic_alert([-.04] * 5, .021))
        self.assertFalse(audit.recurring_ic_alert([-.037, .020, -.013, .008, -.011], .021))
        self.assertTrue(audit.recurring_ic_alert([-.037, .025, -.022, .008, -.011], .021))

    def test_optimized_python_cannot_skip_audit_assertions(self):
        result = subprocess.run([sys.executable, "-B", "-O", str(audit.ROOT / "scripts/audit_baseline.py"), "--suite", "unit"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot run", result.stderr)

    def test_saved_control_hashes_use_final_status_and_reload(self):
        frame = panel_fixture(stocks=110)
        dates = sorted(frame.trade_date.unique())
        train = frame.trade_date.le(dates[24]); valid = frame.trade_date.ge(dates[26])
        features = build_baseline_v1_features(frame)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / "audit"; output.mkdir()
            fixture = root / "fixture"; fixture.mkdir()
            frame.loc[valid, [*audit.KEYS, "y_ret_1d"]].to_csv(fixture / "测试集_Y.csv", index=False, float_format="%.17g")
            frame.loc[valid, [*audit.KEYS, "flag_limit_up"]].to_csv(fixture / "测试集_X.csv", index=False)
            write_json(output / "provenance.json", {})
            with patch.object(audit, "ROOT", root), patch.object(runner, "EXPERIMENT_ROOT", root / "experiments"):
                result = audit.fit_saved_control(features.loc[train, FEATURE_COLUMNS], frame.loc[train, "y_ret_1d"],
                                                features.loc[valid, FEATURE_COLUMNS], frame.loc[valid, audit.KEYS], fixture, output, "fixture", 1)
            saved = root / result["run_directory"]
            self.assertEqual(result["status_sha256"], sha256_file(saved / "status.json"))
            self.assertEqual(result["summary_sha256"], sha256_file(saved / "summary.json"))
            self.assertNotIn("status.json", result["file_sha256"])
            self.assertTrue(result["model_reload_equal"])
            for name, expected in result["file_sha256"].items():
                self.assertEqual(sha256_file(saved / name), expected)
            with patch.object(audit, "ROOT", root):
                controls = {"controls": [result], "investigation": None}
                evidence = audit.verify_linked_artifacts({"runs": []}, controls)
                self.assertTrue(evidence["all_recorded_hashes_match_at_acceptance"])
                with (saved / "summary.json").open("ab") as handle:
                    handle.write(b" ")
                with self.assertRaisesRegex(AssertionError, "final artifact changed"):
                    audit.verify_linked_artifacts({"runs": []}, controls)

    def test_export_restores_input_order_and_rejects_bad_keys(self):
        pred, _, _ = score_frames()
        test = pred[audit.KEYS].sample(frac=1, random_state=10).reset_index(drop=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            export_submission(pred.sample(frac=1, random_state=3), test, path)
            actual = pd.read_csv(path)
            pd.testing.assert_frame_equal(actual[audit.KEYS], test)
            self.assertTrue(np.isfinite(actual.pred).all())
            with self.assertRaises(DataContractError):
                validate_submission(pred.iloc[1:], test)

    def test_release_requires_all_critical_checks(self):
        required = ("unit_tests", "dependencies", "raw_data_and_features", "full_repeatability", "controls_and_ties", "artifact_integrity", "snapshot_unchanged")
        checks = [{"name": n, "status": "passed", "blocking": True} for n in required]
        self.assertTrue(audit.release_decision(checks, "all")["can_start_feature_experiments"])
        self.assertFalse(audit.release_decision(checks, "unit")["can_start_feature_experiments"])
        self.assertFalse(audit.release_decision(checks[:-1], "all")["can_start_feature_experiments"])
        for state in ("failed", "unverified"):
            altered = copy.deepcopy(checks); altered[-1]["status"] = state
            self.assertFalse(audit.release_decision(altered, "all")["can_start_feature_experiments"])
        checks[-1].update(status="failed", blocking=False)
        self.assertFalse(audit.release_decision(checks, "all")["can_start_feature_experiments"])

    def test_source_or_data_changed_during_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); file = root / "input.csv"; file.write_text("original")
            before = {"data": {"sha256": "original"}, "source_sha256": {}, "dependencies": {}, "git": {},
                      "additional_inputs": {"input.csv": sha256_file(file)}}
            with patch.object(audit, "ROOT", root), patch.object(audit, "provenance", return_value=copy.deepcopy(before)):
                self.assertTrue(audit.assert_snapshot_unchanged(before)["source_dependencies_git_and_all_inputs_unchanged"])
                file.write_text("modified")
                with self.assertRaisesRegex(AssertionError, "input changed"):
                    audit.assert_snapshot_unchanged(before)
            after = copy.deepcopy(before); after["source_sha256"]["file.py"] = "new"
            with patch.object(audit, "provenance", return_value=after):
                with self.assertRaisesRegex(AssertionError, "snapshot changed"):
                    audit.assert_snapshot_unchanged(before)

    def test_failed_check_preserves_expected_actual_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            instance = audit.Audit(Path(directory))
            def fail():
                raise ValueError("deliberate bad input")
            self.assertIsNone(instance.check("fixture", "reject bad input", fail))
            self.assertEqual(instance.checks[0]["status"], "failed")
            evidence = json.loads((Path(directory) / "fixture.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["error_type"], "ValueError")

    def test_corrupt_saved_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture").mkdir()
            pred, _, _ = score_frames()
            file = root / "fixture/predictions.parquet"
            pred.to_parquet(file, index=False)
            write_json(root / "provenance.json", {})
            summary = {"splits": [{"split_name": "fixture", "file_sha256": {"predictions.parquet": sha256_file(file)},
                                    "prediction_sha256": prediction_hash(pred.pred), "valid_prediction_rows": len(pred)}],
                       "provenance_sha256": sha256_file(root / "provenance.json")}
            write_json(root / "status.json", {"status": "success"})
            write_json(root / "summary.json", summary)
            with patch.object(audit, "ROOT", root.parent):
                self.assertTrue(audit.verify_saved_run(root)["all_file_hashes_match"])
                with file.open("ab") as handle:
                    handle.write(b"corruption")
                with self.assertRaisesRegex(AssertionError, "corrupt"):
                    audit.verify_saved_run(root)

    def test_failed_experiment_cannot_have_success_status(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "score failed"):
                with experiment_run(Path(directory), "fixture", "failure") as output:
                    raise RuntimeError("score failed")
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")
            self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
