"""Leakage-safe daily-origin forecasting features and official class scores."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
H=48
WEATHER=['cloud_cover','temperature','wind_speed','pressure','humidity','daylight','air_density']
DATA=pd.read_csv(ROOT/'train.csv',parse_dates=['ds'])
FUTURE=pd.read_csv(ROOT/'weather_forecast.csv',parse_dates=['ds'])
Y=DATA.y.to_numpy(float)
assert DATA.ds.is_unique and DATA.ds.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
assert np.isfinite(DATA.drop(columns='ds').to_numpy()).all()
assert len(FUTURE)==48 and FUTURE.ds.iloc[0]==DATA.ds.iloc[-1]+pd.Timedelta(hours=1)
assert 'y' not in FUTURE


def official():
    nb=json.loads((ROOT/'_project.ipynb').read_text())
    env={'np':np,'pd':pd,'H':48,'M':24,'FUTURE':FUTURE.ds}
    names={'rmse','mape','winkler','score','snaive','check_submission'}
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node],type_ignores=[]),'<official class function>','exec'),env)
    assert names.issubset(env)
    return env


SCORES=official()


def origin_index(date):
    i=int(DATA.ds.searchsorted(pd.Timestamp(date)))
    assert i<len(DATA) and DATA.ds.iloc[i]==pd.Timestamp(date)
    return i


def weather_at(origin):
    return DATA.iloc[origin:origin+48][['ds']+WEATHER].copy()


def stress(frame,origin,scenario):
    f=frame.copy()
    if scenario=='observed':return f
    if scenario in ['bias_high','bias_low']:
        direction=1 if scenario=='bias_high' else -1
        for name,bias in [('cloud_cover',.15),('wind_speed',1.),('temperature',1.5),('humidity',4.)]:
            f[name]=f[name].to_numpy(float)+direction*bias*np.repeat([1.,1.7],24)
    else:
        rng=np.random.default_rng(origin+840)
        severity=2. if scenario=='strong' else 1.
        horizon=np.repeat([1.,1.7],24)*severity
        for name,sd in [('cloud_cover',.12),('temperature',1.3),('wind_speed',.65),('pressure',.9),('humidity',3.),('air_density',.006)]:
            error=rng.normal(size=48)
            for j in range(1,48):error[j]=.75*error[j-1]+np.sqrt(1-.75**2)*error[j]
            f[name]=f[name].to_numpy(float)+sd*horizon*error
    f['cloud_cover']=f.cloud_cover.clip(0,1)
    f['wind_speed']=f.wind_speed.clip(0)
    f['humidity']=f.humidity.clip(0,100)
    return f


def context(origin):
    assert origin>=28*24
    past=Y[origin-28*24:origin].reshape(28,24)
    scale=max(30.,float(past[-3:].mean()))
    base=past[-1,np.arange(48)%24]
    return past,scale,base


def features(origin,future):
    # Explicit input firewall: future generation is never an input column.
    assert 'y' not in future.columns
    past,scale,base=context(origin)
    q=np.arange(48)%24
    hour=pd.to_datetime(future.ds).dt.hour.to_numpy()
    doy=pd.to_datetime(future.ds).dt.dayofyear.to_numpy()
    cols={}
    def add(name,value):
        v=np.asarray(value,float)
        cols[name]=np.repeat(v,48) if v.ndim==0 else v
    add('horizon',np.arange(1,49)/48)
    add('second_day',(np.arange(48)>=24).astype(float))
    for k in [1,2,3]:
        add('hour_sin_'+str(k),np.sin(hour*2*np.pi*k/24))
        add('hour_cos_'+str(k),np.cos(hour*2*np.pi*k/24))
    add('year_sin',np.sin(doy*2*np.pi/365.25))
    add('year_cos',np.cos(doy*2*np.pi/365.25))
    for days in [1,2,7]:add('profile_lag_'+str(days),past[-days,q]/scale)
    for days in [3,7,14,28]:add('profile_mean_'+str(days),past[-days:].mean(axis=0)[q]/scale)
    for days in [1,3,7,14,28]:
        add('level_'+str(days),past[-days:].mean()/scale)
        add('spread_'+str(days),past[-days:].std()/scale)
    add('profile_change',(past[-1,q]-past[-2,q])/scale)
    add('profile_trend',(past[-3:].mean(axis=0)[q]-past[-7:].mean(axis=0)[q])/scale)
    daily=past.mean(axis=1)
    add('daily_change',(daily[-1]-daily[-2])/scale)
    add('daily_trend_3',(daily[-1]-daily[-3])/(2*scale))
    add('level_ratio_7_28',(daily[-7:].mean()+10)/(daily.mean()+10))
    add('level_ratio_1_7',(daily[-1]+10)/(daily[-7:].mean()+10))
    add('last_observed',Y[origin-1]/scale)
    for n in [6,12]:add('recent_'+str(n),Y[origin-n:origin].mean()/scale)
    add('daily_difference_rms',np.sqrt(np.mean((past[-7:]-past[-8:-1])**2))/scale)
    add('daily_mean_volatility',daily[-7:].std()/scale)
    add('log_scale',np.log1p(scale))
    # Raw levels retained only for absolute-target/level diagnostics.
    add('raw_last_day',base)
    add('raw_level',scale)
    weather_columns=[]
    previous=DATA.iloc[origin-24:origin]
    history_weather=DATA.iloc[origin-7*24:origin]
    for name in WEATHER:
        add('weather_'+name,future[name].to_numpy(float));weather_columns.append('weather_'+name)
        add('weather_change_'+name,future[name].to_numpy(float)-previous[name].to_numpy(float)[q]);weather_columns.append('weather_change_'+name)
        add('weather_vs_week_'+name,future[name].to_numpy(float)-history_weather[name].mean());weather_columns.append('weather_vs_week_'+name)
    wind=future.wind_speed.to_numpy(float)
    add('weather_wind_squared',wind**2);weather_columns.append('weather_wind_squared')
    add('weather_wind_power',wind**3*future.air_density.to_numpy(float));weather_columns.append('weather_wind_power')
    add('weather_clear_daylight',future.daylight.to_numpy(float)*(1-future.cloud_cover.to_numpy(float)));weather_columns.append('weather_clear_daylight')
    for name in ['temperature','cloud_cover','wind_speed']:
        for harmonic in [1,2]:
            for kind in ['sin','cos']:
                label=f'weather_{name}_hour_{kind}_{harmonic}'
                add(label,future[name].to_numpy(float)*cols[f'hour_{kind}_{harmonic}']);weather_columns.append(label)
    names=list(cols)
    array=np.column_stack(list(cols.values())).astype(np.float32)
    assert array.shape==(48,len(names)) and np.isfinite(array).all()
    return array,names,np.array([i for i,n in enumerate(names) if n not in weather_columns])


def build_dataset():
    path=HERE/'training_examples.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            return {key:saved[key] for key in saved.files}
    origins=np.arange(672,len(DATA)-47,24,dtype=int)
    clean=[];noisy=[];target=[];scales=[];bases=[]
    for i,o in enumerate(origins):
        future=weather_at(int(o))
        a,names,nw=features(int(o),future)
        b,_,_=features(int(o),stress(future,int(o),'moderate'))
        clean.append(a);noisy.append(b)
        _,scale,base=context(int(o));target.append(Y[o:o+48]);scales.append(np.full(48,scale));bases.append(base)
        if (i+1)%300==0:print(f'Dataset origins {i+1}/{len(origins)}',flush=True)
    np.savez_compressed(path,X=np.concatenate(clean),X_noisy=np.concatenate(noisy),
        target=np.concatenate(target),scale=np.concatenate(scales),base=np.concatenate(bases),
        origins=np.repeat(origins,48),names=np.array(names),no_weather=nw)
    info={'origins':len(origins),'rows':len(origins)*48,'unique_target_hours':int(origins[-1]+48-origins[0]),
        'features':len(names),'stride_hours':24,'duplicate_targets_are_not_independent':True}
    (HERE/'dataset_summary.json').write_text(json.dumps(info,indent=2)+'\n')
    with np.load(path,allow_pickle=False) as saved:
        return {key:saved[key] for key in saved.files}


def interval_group(future):
    return 2*(np.arange(48)//24)+future.daylight.to_numpy(int)


def interval_scale(origin,point,future):
    start=max(24,origin-21*24)
    residual=Y[start:origin]-Y[start-24:origin-24]
    old_day=DATA.daylight.iloc[start:origin].to_numpy(int)
    v=np.array([np.sqrt(np.mean(residual[old_day==g]**2)) if np.any(old_day==g) else np.std(residual) for g in [0,1]])
    return 30.+.2*point+.5*np.maximum(v[future.daylight.to_numpy(int)],20.)


def prediction_interval(origin,point,future,records,model,rule,prob,pool_scenarios=None):
    pools={g:[] for g in range(4)}
    usable=[r for r in records if r['origin']+48<=origin][-16:]
    assert usable
    for r in usable:
        group=interval_group(r['future'])
        for scenario in (pool_scenarios or ['moderate']):
            p=r['predictions'][model][scenario]
            e=(r['truth']-p)/interval_scale(r['origin'],p,r['future'])
            for g in range(4):pools[g].extend(e[group==g])
    group=interval_group(future);scale=interval_scale(origin,point,future)
    lower=np.empty(48);upper=np.empty(48)
    for g in range(4):
        e=np.sort(pools[g]);n=len(e);assert n>0
        if rule=='absolute':
            k=min(n,int(np.ceil((n+1)*prob)))
            q=np.sort(np.abs(e))[k-1];ql,qu=-q,q
        else:
            tail=(1-prob)/2
            kl=max(1,min(n,int(np.floor((n+1)*tail))))
            ku=max(1,min(n,int(np.ceil((n+1)*(1-tail)))))
            ql,qu=min(e[kl-1],0),max(e[ku-1],0)
        mask=group==g
        lower[mask]=np.maximum(0,point[mask]+ql*scale[mask])
        upper[mask]=point[mask]+qu*scale[mask]
    return lower,upper
