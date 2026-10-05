"""Boundary, alignment and causal behavior of the fixed prediction transforms."""
import copy

import numpy as np
import pandas as pd
import pytest

from scripts.validate_prediction_transforms import (
    CONFIG, blend, independent_smoothing, load_transitions, read, risk_summary, smooth,
    validate_config, validated,
)


def frame():
    return pd.DataFrame({"ts_code": ["A", "A", "A", "B", "B", "B"],
        "trade_date": [20230130, 20230131, 20230201] * 2,
        "pred": [1., 3., 9., 10., 30., 90.]})


DATES = [20230130, 20230131, 20230201]


def test_blend_aligns_by_key_and_preserves_first_input_order():
    a = frame()
    b = frame().iloc[::-1].reset_index(drop=True)
    b["pred"] *= 2
    got = blend(a, b)
    pd.testing.assert_frame_equal(got.iloc[:, :2], a.iloc[:, :2])
    np.testing.assert_array_equal(got.pred, a.pred * 1.5)


def test_blend_does_not_mutate_sources():
    a, b = frame(), frame()
    old_a, old_b = a.copy(), b.copy()
    blend(a, b)
    pd.testing.assert_frame_equal(a, old_a)
    pd.testing.assert_frame_equal(b, old_b)


def test_blend_missing_or_extra_key_rejected():
    a = frame()
    with pytest.raises(AssertionError):
        blend(a, a.iloc[:-1])
    b = a.copy()
    b.loc[0, "ts_code"] = "C"
    with pytest.raises(AssertionError):
        blend(a, b)


def test_duplicate_keys_rejected():
    a = pd.concat([frame(), frame().iloc[:1]], ignore_index=True)
    with pytest.raises(AssertionError):
        blend(a, a)
    with pytest.raises(AssertionError):
        smooth(a, DATES, ["m"] * len(a))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_rejected(bad):
    a = frame()
    a.loc[0, "pred"] = bad
    with pytest.raises(AssertionError):
        validated(a)


def test_smoothing_same_stock_original_lag_no_recursion_month_continuity():
    a = frame()
    got, trace = smooth(a, DATES, ["m"] * 6)
    np.testing.assert_array_equal(got.pred, [1., .8 * 3 + .2, .8 * 9 + .2 * 3,
                                           10., .8 * 30 + .2 * 10, .8 * 90 + .2 * 30])
    assert trace.used_previous_original.tolist() == [False, True, True, False, True, True]
    assert got.pred.iloc[2] != .8 * 9 + .2 * got.pred.iloc[1]
    assert trace.previous_market_date.iloc[2] == 20230131


def test_smoothing_arbitrary_input_order():
    a = frame().iloc[[5, 0, 4, 2, 1, 3]].reset_index(drop=True)
    got, _ = smooth(a, DATES, ["m"] * len(a))
    np.testing.assert_array_equal(got.pred, independent_smoothing(a, DATES, "m"))
    pd.testing.assert_frame_equal(got.iloc[:, :2], a.iloc[:, :2])


def test_missing_immediate_market_day_does_not_borrow_older_day():
    a = frame().drop(index=1).reset_index(drop=True)
    got, trace = smooth(a, DATES, ["m"] * len(a))
    assert got.loc[1, "pred"] == 9
    assert not trace.loc[1, "used_previous_original"]
    assert trace.loc[1, "previous_market_date"] == 20230131


def test_missing_entire_market_day_in_predictions_uses_full_calendar():
    a = frame()[frame().trade_date.ne(20230131)].reset_index(drop=True)
    got, trace = smooth(a, DATES, ["m"] * len(a))
    np.testing.assert_array_equal(got.pred, a.pred)
    assert not trace.used_previous_original.any()


def test_validation_first_day_ignores_calendar_previous_day_without_prediction():
    a = frame()
    got, trace = smooth(a, [20230120, *DATES], ["m"] * len(a))
    assert got.pred.iloc[0] == a.pred.iloc[0]
    assert trace.previous_market_date.iloc[0] == 20230120
    assert not trace.used_previous_original.iloc[0]


def test_annual_boundary_fallback_even_if_previous_model_id_same():
    a = pd.DataFrame({"ts_code": ["A", "A", "A"],
        "trade_date": [20211231, 20220104, 20220105], "pred": [5., 10., 20.]})
    got, trace = smooth(a, a.trade_date.tolist(), ["m"] * len(a))
    np.testing.assert_array_equal(got.pred, [5., 10., .8 * 20 + .2 * 10])
    assert not trace.used_previous_original.iloc[1]


def test_model_boundary_no_borrowing_other_model():
    a = frame()
    got, trace = smooth(a, DATES, ["old", "new", "new", "m", "m", "m"])
    assert got.pred.iloc[1] == 3
    assert got.pred.iloc[2] == .8 * 9 + .2 * 3
    assert not trace.used_previous_original.iloc[1]


def test_no_borrowing_other_stock_previous_day():
    a = pd.DataFrame({"ts_code": ["A", "B"], "trade_date": [20230130, 20230131], "pred": [100., 1.]})
    got, _ = smooth(a, DATES, ["m"] * len(a))
    np.testing.assert_array_equal(got.pred, a.pred)


@pytest.mark.parametrize("transform", ["F1", "S1"])
def test_future_perturbation_and_truncation_leave_prefix_equal(transform):
    a, b = frame(), frame()
    b.pred *= -3
    mask = a.trade_date.le(20230131)
    aa, bb = a.copy(), b.copy()
    aa.loc[~mask, "pred"], bb.loc[~mask, "pred"] = 999., -333.
    if transform == "F1":
        full, changed, short = blend(a, b), blend(aa, bb), blend(a[mask], b[mask])
    else:
        full, _ = smooth(a, DATES, ["m"] * len(a))
        changed, _ = smooth(aa, DATES, ["m"] * len(a))
        short, _ = smooth(a[mask], DATES[:2], ["m"] * int(mask.sum()))
    np.testing.assert_array_equal(full.loc[mask, "pred"], changed.loc[mask, "pred"])
    np.testing.assert_array_equal(full.loc[mask, "pred"], short.pred)


@pytest.mark.parametrize("calendar", [[20230131, 20230130, 20230201], [20230130, 20230130, 20230201], [20230130, 20230201]])
def test_invalid_calendar_rejected(calendar):
    with pytest.raises(AssertionError):
        smooth(frame(), calendar, ["m"] * 6)


def test_model_identity_length_and_missing_rejected():
    with pytest.raises(AssertionError):
        smooth(frame(), DATES, ["m"])
    with pytest.raises(AssertionError):
        smooth(frame(), DATES, [None] * 6)


def test_invalid_date_rejected():
    a = frame()
    a.loc[0, "trade_date"] = 20230230
    with pytest.raises(AssertionError):
        validated(a)


def test_fixed_config_refuses_extra_weight_and_2024():
    c = read(CONFIG)
    validate_config(c)
    altered = copy.deepcopy(c)
    altered["experiments"]["S1"]["weights"] = [.7, .3]
    with pytest.raises(AssertionError):
        validate_config(altered)
    altered = copy.deepcopy(c)
    altered["splits"].append("oos_2024")
    with pytest.raises(AssertionError):
        validate_config(altered)


def test_risk_strict_thresholds_and_direction_conflict():
    annual = pd.DataFrame({"after": ["F1"] * 3, "before": ["S4R_lean31_minus4"] * 3,
        "year": [2021, 2022, 2023], "final_score_delta": [-.005, .006, -.001]})
    monthly = pd.DataFrame({"after": ["F1"] * 4, "before": ["S4R_lean31_minus4"] * 4,
        "month": [202101, 202102, 202103, 202104], "final_score_delta": [-.005, -.0051, -.02, -.0201]})
    r = risk_summary(annual, monthly).iloc[0]
    assert r.mixed_annual_directions and r.small_change
    assert not r.annual_risk_vs27
    assert r.risk_months == 3 and r.severe_months == 1
    assert r.negative_months == 4


def test_empty_top_transition_lists_roundtrip_as_empty_sets(tmp_path):
    original = pd.DataFrame({"trade_date": [20230131, 20230201],
        "entered_codes": ["", "A"], "exited_codes": ["B", ""], "turnover": [0., .5]})
    path = tmp_path / "transitions.csv"
    original.to_csv(path, index=False, float_format="%.17g")
    pd.testing.assert_frame_equal(load_transitions(path), original, check_exact=True)
