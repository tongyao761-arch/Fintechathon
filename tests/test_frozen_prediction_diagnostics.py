"""Behavioral checks for frozen-prediction diagnostics."""
import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_frozen_predictions import (aggregate_daily, concentration,
    contributions, metric_difference, top_diagnostics, comparisons, IDS)


def fixture():
    rows=[]
    for date in [20240131,20240201,20240202]:
        for i in range(200):
            rows.append(dict(ts_code=f"s{i:03}",trade_date=date,pred=200-i,
                y_ret_1d=np.nan if i<20 else i/10000,flag_limit_up=0,is_price_valid=0 if i<20 else 1))
    return pd.DataFrame(rows)


def test_official_universes_and_price_filter():
    q,m,t,v,sets=top_diagnostics(fixture())
    assert sets['turnover'][20240131]=={f's{i:03}' for i in range(20)}
    assert sets['return'][20240131]=={f's{i:03}' for i in range(20,38)}
    assert sets['price_valid']==sets['return']
    assert q[q.top_type.eq('turnover')].missing_label_fraction.eq(1).all()
    assert q[q.top_type.eq('return')].missing_label_fraction.eq(0).all()
    assert t[t.trade_date.eq(20240201)].previous_trade_date.eq(20240131).all()
    assert v.price_valid_turnover.eq(0).all()


def test_label_availability_does_not_change_price_only_turnover():
    f=fixture();_,_,_,a,sa=top_diagnostics(f)
    f['y_ret_1d']=np.nan
    _,_,_,b,sb=top_diagnostics(f)
    pd.testing.assert_frame_equal(a,b)
    assert sa['price_valid']==sb['price_valid']


def test_month_boundary_retained_and_month_mean_not_annual_score():
    f=pd.DataFrame(dict(month=['202401','202402','202402'],year=[2024]*3,
        ic_mean=[.1,.2,.3],annual_excess=[1.,2.,4.],excess=[1/252,2/252,4/252],mean_turnover=[np.nan,.4,.8]))
    m=aggregate_daily(f,'month');a=aggregate_daily(f,'year')
    assert m.loc[1,'mean_turnover']==pytest.approx(.6)
    assert a.loc[0,'annual_excess']==pytest.approx(7/3)
    assert not np.isclose(m.final_score.mean(),a.final_score.iloc[0])


def test_daily_reset_on_insufficient_universe():
    f=fixture();f.loc[f.trade_date.eq(20240201),'flag_limit_up']=1
    _,_,t,v,_=top_diagnostics(f)
    assert t.empty and v.empty


def test_concentration_reports_positive_mass_and_net_cancellation():
    c=concentration([1.,.5,.2,-.8,-.7,.1])
    assert c['best_1_days_positive_mass_fraction']==pytest.approx(1/1.8)
    assert c['best_1_days_net_sum_fraction']>1
    assert c['best_1_days_removed_annual_excess']==pytest.approx((.5+.2-.8-.7+.1)/5*252)
    with pytest.raises(AssertionError):concentration([np.inf])


def test_compare_rejects_nonfinite_and_mismatched_names():
    with pytest.raises(AssertionError):metric_difference({'a':np.nan},{'a':0.})
    with pytest.raises(AssertionError):metric_difference({'a':1.},{'b':1.})


def test_pairwise_component_deltas_sum():
    f=contributions(pd.DataFrame(dict(candidate=IDS,year=[2024]*3,
        ic_mean=[.1,.2,.3],annual_excess=[.4,.8,.5],mean_turnover=[.2,.3,.4])))
    d=comparisons(f,['year'])
    np.testing.assert_allclose(d.final_score_delta,d.ic_contribution_delta+d.excess_contribution_delta+d.stability_contribution_delta,atol=1e-15)

