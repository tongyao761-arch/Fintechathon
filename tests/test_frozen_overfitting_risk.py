import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_frozen_overfitting_risk import half_spec, moving_indices, paired_bootstrap, fit_statistics
from scripts.diagnose_frozen_predictions import aggregate_daily
from src.metrics.official import score_official


def test_half_boundary_uses_observed_days():
    dates=[20180102,20240625,20240626,20240627,20240628,20240701,20240702,20241231]
    assert half_spec(dates,2024)==dict(train_start=20180102,train_end=20240627,purge_date=20240628,
                                       valid_start=20240701,valid_end=20241231)
    with pytest.raises(AssertionError):half_spec([20240701],2024)


def test_fixed_non_circular_joint_blocks():
    idx=moving_indices(47)
    assert idx.shape==(2000,47)
    assert idx.min()>=0 and idx.max()<47
    for start,end in [(0,20),(20,40),(40,47)]:
        np.testing.assert_array_equal(np.diff(idx[:,start:end],axis=1),np.ones((2000,end-start-1),dtype=int))
    np.testing.assert_array_equal(idx,moving_indices(47))
    with pytest.raises(AssertionError):moving_indices(19)


def test_bootstrap_own_missing_denominators_and_no_new_turnover():
    n=40
    a=pd.DataFrame(dict(trade_date=np.arange(n),ic_mean=np.linspace(.01,.2,n),
                        annual_excess=np.linspace(-1,2,n),mean_turnover=np.linspace(.1,.9,n)))
    b=a.copy();b.ic_mean-=.01;b.annual_excess-=.2;b.mean_turnover+=.02
    a.loc[0,'mean_turnover']=np.nan;b.loc[1,'ic_mean']=np.nan
    idx=moving_indices(n)
    names,sample=paired_bootstrap(a,b,idx)
    expected=np.stack([np.nanmean(a[m].to_numpy()[idx],axis=1)-np.nanmean(b[m].to_numpy()[idx],axis=1)
                       for m in names[:3]],axis=1)
    np.testing.assert_array_equal(sample[:,:3],expected)
    np.testing.assert_allclose(sample[:,-1],expected@np.array([.4,.3,-.3]),rtol=0,atol=1e-15)
    b.loc[0,'trade_date']=99
    with pytest.raises(AssertionError):paired_bootstrap(a,b,idx)


def test_invalid_zero_mse_is_uncomputable():
    result,_=fit_statistics(np.zeros(40),np.ones(40),np.ones(40,dtype=int))
    assert result['mse']==1 and result['normalized_mse'] is None and result['ic_days']==0
    result,_=fit_statistics(np.arange(40)/100,np.arange(40)/100,np.ones(40,dtype=int))
    assert result['normalized_mse']==0 and result['ic_mean']==1


def test_month_first_turnover_uses_annual_predecessor_and_half_resets():
    dates=[20240627,20240628,20240701,20240702]
    rows=[]
    for t,date in enumerate(dates):
        for stock in range(120):
            pred=float(stock if t<2 else 120-stock)
            rows.append((f'S{stock:03}',date,pred,float(stock)/100000,0))
    frame=pd.DataFrame(rows,columns=['ts_code','trade_date','pred','y_ret_1d','flag_limit_up'])
    result=score_official(frame[['ts_code','trade_date','pred']],frame[['ts_code','trade_date','y_ret_1d']],
                          frame[['ts_code','trade_date','flag_limit_up']],return_details=True)
    turns=result['details']['daily_turnover']
    july=turns[turns.trade_date.eq(20240701)].iloc[0]
    assert july.previous_trade_date==20240628 and july.turnover==1
    sub=frame[frame.trade_date.ge(20240701)]
    half=score_official(sub[['ts_code','trade_date','pred']],sub[['ts_code','trade_date','y_ret_1d']],
                        sub[['ts_code','trade_date','flag_limit_up']],return_details=True)
    assert half['mean_turnover']==0
    d=pd.DataFrame(dict(trade_date=dates,ic_mean=[.1]*4,annual_excess=[1]*4,
        mean_turnover=[np.nan,0,1,0],excess=[1/252]*4,month=[202406,202406,202407,202407]))
    m=aggregate_daily(d,'month')
    assert m.loc[m.month.eq(202407),'mean_turnover'].iloc[0]==.5
