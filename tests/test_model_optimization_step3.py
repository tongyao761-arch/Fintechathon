"""Guard fixed matrix, rolling dates, eligibility and serialized parameters."""
import copy
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from scripts.optimize_frozen27 import (
    CONFIG, MODEL_PARAMS, array_hash, check_model, compare, matrix,
    parameters, read, sample_evidence, validate_config,
)
from scripts.validate_prediction_transforms import METRICS, CONTROLS


def test_fixed_config_rejects_expansion():
    original = read(CONFIG)
    validate_config(original)
    for key,value in [('changes',{}),('splits',['oos_2024']),
                     ('budget',{'new_full_training':9}),('risk_flags',{})]:
        c=copy.deepcopy(original);c[key]=value
        with pytest.raises(AssertionError):validate_config(c)


def test_independent_changes_and_bad_original():
    for e,delta in [('M1',{}),('M2',{'n_estimators':200,'learning_rate':.025}),
                    ('M3',{'min_child_samples':500})]:
        p=parameters(MODEL_PARAMS,e)
        assert {k:v for k,v in p.items() if v!=MODEL_PARAMS[k]}==delta
    bad=MODEL_PARAMS.copy();bad['learning_rate']=.1
    with pytest.raises(AssertionError):parameters(bad,'M2')
    with pytest.raises(AssertionError):parameters(MODEL_PARAMS,'M4')


def test_matrix_actual_calendar_dates(monkeypatch):
    from scripts import optimize_frozen27 as runner
    from src.validation.splits import TimeSplit
    specs=read(runner.ROOT/'configs/features34_step3.json')['splits']
    monkeypatch.setattr(runner,'validate_dates',lambda dates,cfg:
        {n:TimeSplit(name=n,**v) for n,v in specs.items()})
    dates=[20180102,20190102,20200102,20210104,20220104,20230103]
    items=matrix(dates,MODEL_PARAMS)
    assert len(items)==9 and sum(not i['reuse'] for i in items)==8
    assert [i['split_spec']['train_start'] for i in items[:3]]==[20180102,20190102,20200102]
    for i in items:
        orig=specs[i['split']]
        for k in ['train_end','purge_date','valid_start','valid_end']:
            assert orig[k]==i['split_spec'][k]
        if i['experiment']!='M1':assert orig==i['split_spec']
    with pytest.raises(AssertionError):matrix([d for d in dates if d//10000!=2019],MODEL_PARAMS)


def test_training_masks_purge_and_float32_overflow():
    dates=[20201229,20201230,20201231,20210104,20210105]
    p=pd.DataFrame(dict(ts_code=['A']*5,trade_date=dates,is_trainable=[1,0,1,1,1],y_ret_1d=[.1]*5))
    f=pd.DataFrame({'x':np.array([1,2,3,4,5],dtype='float32')})
    i=dict(split='dev_2021',split_spec=dict(train_start=20201229,train_end=20201230,
        purge_date=20201231,valid_start=20210104,valid_end=20210105))
    train,valid,y,e=sample_evidence(p,f,i,['x'])
    assert train.tolist()==[True,False,False,False,False]
    assert valid.tolist()==[False,False,False,True,True]
    assert e['next_label_date_max']==20201230 and e['purge_trained_rows']==0
    p.loc[0,'y_ret_1d']=1e40
    with pytest.raises(AssertionError):sample_evidence(p,f,i,['x'])


def test_label_boundary_rejected():
    p=pd.DataFrame(dict(ts_code=['A']*2,trade_date=[20201230,20210104],is_trainable=[1,1],y_ret_1d=[.1,.2]))
    f=pd.DataFrame({'x':np.array([1,2],dtype='float32')})
    i=dict(split='dev_2021',split_spec=dict(train_start=20201230,train_end=20201230,
        purge_date=20201231,valid_start=20210104,valid_end=20210104))
    with pytest.raises(AssertionError):sample_evidence(p,f,i,['x'])


def test_saved_model_all_parameters_and_order():
    rng=np.random.default_rng(1)
    x=pd.DataFrame(rng.normal(size=(1000,2)),columns=['one','two'])
    y=x.one*.3+x.two*.2+rng.normal(size=1000)*.01
    for e in ['M1','M2','M3']:
        p=parameters(MODEL_PARAMS,e)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.txt'
            fitted=lgb.LGBMRegressor(**p).fit(x,y)
            fitted.booster_.save_model(str(path))
            saved=check_model(path,list(x.columns),p)
            np.testing.assert_array_equal(saved.predict(x),fitted.predict(x))
            with pytest.raises(AssertionError):check_model(path,list(reversed(x.columns)),p)


def test_comparison_all_controls_and_missing_periods():
    rows=[]
    for name in [*CONTROLS,'M1','M2','M3']:
        for year in [2021,2022,2023]:
            rows.append(dict(candidate=name,year=year,**{k:float(year) for k in METRICS}))
    frame=pd.DataFrame(rows)
    assert len(compare(frame,['year']))==27
    assert compare(frame,['year']).final_score_delta.eq(0).all()
    with pytest.raises(AssertionError):compare(frame.iloc[:-1],['year'])


def test_array_hash_includes_dtype_shape_and_values():
    a=np.array([1.,2.],dtype='float32')
    assert array_hash(a)!=array_hash(a.astype('float64'))
    assert array_hash(a)!=array_hash(a.reshape(1,2))
    assert array_hash(a)!=array_hash(a[::-1])
