"""Staged daily-origin model search with locked later historical tests."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import core as c

OUT=Path(__file__).resolve().parent
PROTOCOL=json.loads((OUT/'protocol.json').read_text())
SCENARIOS=PROTOCOL['weather_scenarios']
SPECS={
 'HGB absolute 150d':dict(kind='hgb',target='absolute',days=150,half=75,aug=False,weather=True),
 'HGB normalized 365d':dict(kind='hgb',target='normalized',days=365,half=90,aug=False,weather=True),
 'HGB delta all history':dict(kind='hgb',target='delta',days=None,half=180,aug=False,weather=True),
 'HGB log ratio 365d':dict(kind='hgb',target='logratio',days=365,half=90,aug=False,weather=True),
 'HGB log normalized':dict(kind='hgb',target='lognormalized',days=365,half=90,aug=False,weather=True),
 'HGB delta augmented':dict(kind='hgb',target='delta',days=365,half=90,aug=True,weather=True),
 'HGB delta no weather':dict(kind='hgb',target='delta',days=365,half=90,aug=False,weather=False),
 'HGB delta recent 90d':dict(kind='hgb',target='delta',days=90,half=30,aug=False,weather=True),
 'Ridge delta all history':dict(kind='ridge',target='delta',days=None,half=180,aug=True,weather=True)
}


def origins(block):
    first=c.origin_index(block['start'])
    ans=[first+i*block['step_days']*24 for i in range(block['count'])]
    assert max(ans)+48<=len(c.DATA)
    return ans


def fit_predict(data,origin,spec,scenario_frames=None):
    a=data['origins']
    mask=(a+48<=origin)
    if spec['days'] is not None:mask&=(a>=origin-spec['days']*24)
    assert mask.any() and np.all(a[mask]+48<=origin)
    columns=np.arange(len(data['names'])) if spec['weather'] else data['no_weather']
    if spec['target']!='absolute':
        columns=np.array([i for i in columns if str(data['names'][i]) not in ['raw_last_day','raw_level','log_scale']])
    X=data['X'][mask][:,columns]
    target=data['target'][mask];scale=data['scale'][mask];base=data['base'][mask]
    weight=np.exp(-np.log(2)*(origin-a[mask])/(24*spec['half']))
    weight=np.maximum(weight,.001);weight/=weight.mean()
    transform=spec['target']
    if transform=='normalized':target=target/scale
    elif transform=='delta':target=(target-base)/scale
    elif transform=='logratio':target=np.log((target+20)/(base+20))
    elif transform=='lognormalized':target=np.log1p(target/scale)
    if spec['aug']:
        noisy=data['X_noisy'][mask][:,columns]
        X=np.concatenate([X,noisy]);target=np.tile(target,2);weight=np.tile(weight/2,2)
    if spec['kind']=='ridge':
        scaler=StandardScaler();X=scaler.fit_transform(X,sample_weight=weight)
        model=Ridge(alpha=100)
    else:
        scaler=None
        model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,max_depth=5,
            learning_rate=.06,l2_regularization=5,min_samples_leaf=30,
            early_stopping=False,random_state=840)
    model.fit(X,target,sample_weight=weight)
    futures=scenario_frames or {s:c.stress(c.weather_at(origin),origin,s) for s in SCENARIOS}
    tests=[c.features(origin,future)[0][:,columns] for future in futures.values()]
    joined=np.concatenate(tests)
    if scaler is not None:joined=scaler.transform(joined)
    prediction=model.predict(joined).reshape(len(tests),48)
    _,current_scale,current_base=c.context(origin)
    result={}
    for i,s in enumerate(futures):
        p=prediction[i]
        if transform=='normalized':p=p*current_scale
        elif transform=='delta':p=current_base+p*current_scale
        elif transform=='logratio':p=(current_base+20)*np.exp(np.clip(p,-10,10))-20
        elif transform=='lognormalized':p=np.expm1(np.clip(p,-10,10))*current_scale
        result[s]=np.maximum(0,p)
    return result,{'training_origins':int(np.unique(a[mask]).size),'training_rows':int(mask.sum()),
        'last_target_index':int(a[mask].max()+47),'forecast_origin':origin}


def fallback(origin,forecast,future):
    past,_,base=c.context(origin)
    daily=past.mean(axis=1)
    ratio=(daily[-1]+10)/(daily[-7:].mean()+10)
    stable=daily[-3:].std()/(daily[-3:].mean()+10)<.1
    old=c.DATA.iloc[origin-24:origin]
    weather_close=(abs(future.wind_speed.mean()-old.wind_speed.mean())<.75 and
                   abs(future.cloud_cover.mean()-old.cloud_cover.mean())<.15)
    weight=.5 if (ratio<.65 or ratio>1.5) else (.75 if stable and weather_close else 0.)
    return (1-weight)*forecast+weight*base


def add_derived(record,recipe):
    for name,parts in recipe.items():
        if name.startswith('Equal ensemble'):
            record['predictions'][name]={s:(record['predictions'][parts[0]][s]+record['predictions'][parts[1]][s])/2 for s in SCENARIOS}
        else:
            record['predictions'][name]={s:fallback(record['origin'],record['predictions'][parts[0]][s],
                c.stress(record['future'],record['origin'],s)) for s in SCENARIOS}


def make_record(data,origin,names,period,recipe=None):
    future=c.weather_at(origin)
    frames={s:c.stress(future,origin,s) for s in SCENARIOS}
    point,lo,hi=c.SCORES['snaive'](c.Y[:origin])
    record={'origin':origin,'period':period,'future':future,'truth':c.Y[origin:origin+48],
        'predictions':{'Seasonal Naive':{s:point for s in SCENARIOS}},'training_audit':[]}
    for name in names:
        predictions,audit=fit_predict(data,origin,SPECS[name],frames)
        record['predictions'][name]=predictions
        record['training_audit'].append({'model':name,**audit})
    if recipe:add_derived(record,recipe)
    return record


def point_rows(records):
    rows=[]
    for r in records:
        for name,scenarios in r['predictions'].items():
            for s,p in scenarios.items():
                for label,sl in [('day 1',slice(0,24)),('day 2',slice(24,48)),('48 hours',slice(0,48))]:
                    rows.append({'period':r['period'],'origin':str(c.DATA.ds.iloc[r['origin']]),
                        'model':name,'scenario':s,'horizon':label,'MSE':float(np.mean((r['truth'][sl]-p[sl])**2)),
                        'MAPE':c.SCORES['mape'](r['truth'][sl],p[sl])})
    return pd.DataFrame(rows)


def summary(rows):
    numeric={name:(name,'mean') for name in ['MSE','MAPE','Winkler','coverage','width'] if name in rows}
    result=rows.groupby(['period','model','scenario','horizon']).agg(windows=('MSE','size'),**numeric).reset_index()
    result['RMSE']=np.sqrt(result.pop('MSE'))
    return result


def ranking(rows):
    table=summary(rows)
    d=table[(table.horizon=='day 2')&table.scenario.isin(['observed','moderate'])]
    base=d[d.model=='Seasonal Naive'].set_index(['period','scenario'])
    result=[]
    for name,g in d.groupby('model'):
        g=g.set_index(['period','scenario'])
        loss=float(np.log(g.RMSE/base.RMSE).mean())
        mape=float((g.MAPE/base.MAPE).mean())
        stress=table[(table.model==name)&(table.horizon=='day 2')&table.scenario.isin(['bias_high','bias_low'])]
        b=table[(table.model=='Seasonal Naive')&(table.horizon=='day 2')&table.scenario.isin(['bias_high','bias_low'])]
        worst=float((stress.set_index(['period','scenario']).RMSE/b.set_index(['period','scenario']).RMSE).max())
        result.append({'model':name,'selection_loss':loss,'MAPE_ratio':mape,'worst_biased_weather_RMSE_ratio':worst})
    result=pd.DataFrame(result)
    result['eligible']=(result.MAPE_ratio<=1)&(result.worst_biased_weather_RMSE_ratio<=1.05)
    return result.sort_values(['eligible','selection_loss'],ascending=[False,True])


def save_records(records,path):
    models=list(records[0]['predictions'])
    np.savez_compressed(OUT/path,origins=np.array([r['origin'] for r in records]),
        periods=np.array([r['period'] for r in records]),models=np.array(models),
        scenarios=np.array(SCENARIOS),
        predictions=np.array([[[r['predictions'][m][s] for s in SCENARIOS] for m in models] for r in records]))
    audit=[{'period':r['period'],**a} for r in records for a in r['training_audit']]
    pd.DataFrame(audit).to_csv(OUT/(Path(path).stem+'_training_boundaries.csv'),index=False)


def selection():
    data=c.build_dataset()
    development=[(b['name'],o) for b in PROTOCOL['development'] for o in origins(b)]
    scout=[x for i,x in enumerate(development) if i%12<4]
    records=[];started=time.monotonic()
    for i,(period,o) in enumerate(scout):
        records.append(make_record(data,o,list(SPECS),period))
        rows=point_rows(records);rows.to_csv(OUT/'scout_window_scores.csv',index=False)
        print(f'Scout {i+1}/{len(scout)}; elapsed {time.monotonic()-started:.1f}s',flush=True)
    scout_rows=point_rows(records);summary(scout_rows).to_csv(OUT/'scout_summary.csv',index=False)
    rank=ranking(scout_rows);rank.to_csv(OUT/'scout_ranking.csv',index=False)
    shortlist=list(rank[rank.model!='Seasonal Naive'].head(3).model)
    (OUT/'shortlist.json').write_text(json.dumps(shortlist,indent=2)+'\n')
    print('Shortlisted individuals:',shortlist,flush=True)
    all_records=[]
    for i,(period,o) in enumerate(development):
        old=next((r for r in records if r['origin']==o),None)
        if old is None:r=make_record(data,o,shortlist,period)
        else:r={**old,'predictions':{k:v for k,v in old['predictions'].items() if k in shortlist+['Seasonal Naive']}}
        all_records.append(r)
        print(f'Full development {i+1}/{len(development)}',flush=True)
    individual=ranking(point_rows(all_records));individual.to_csv(OUT/'individual_development_ranking.csv',index=False)
    top=list(individual[individual.model!='Seasonal Naive'].head(2).model)
    recipe={'Equal ensemble of top two':top,'Origin-time Naive fallback':[top[0]]}
    for r in all_records:add_derived(r,recipe)
    rows=point_rows(all_records);rows.to_csv(OUT/'development_window_scores.csv',index=False)
    summary(rows).to_csv(OUT/'development_summary.csv',index=False)
    rank=ranking(rows);rank.to_csv(OUT/'development_ranking.csv',index=False)
    # Seasonal Naive remains an honest candidate. A failure to beat it is not
    # hidden by excluding the benchmark from model selection.
    winner=str(rank.iloc[0].model)
    needed=recipe[winner] if winner in recipe else ([] if winner=='Seasonal Naive' else [winner])
    lock={'winner':winner,'base_models':needed,'recipe':{winner:recipe[winner]} if winner in recipe else {},
          'specifications':{n:SPECS[n] for n in needed},'selected_on':'development only; no later test scores',
          'time_utc':pd.Timestamp.now(tz='UTC').isoformat(),'test_opened':False}
    (OUT/'model_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    save_records(all_records,'development_predictions.npz')
    print('LOCKED point model:',json.dumps(lock),flush=True)


def predict_block(data,block,names,label,recipe):
    records=[]
    for i,o in enumerate(origins(block)):
        records.append(make_record(data,o,names,label,recipe))
        if (i+1)%4==0:print(f'{label}: {i+1}/{block["count"]}',flush=True)
    save_records(records,label+'_predictions.npz')
    return records


def calibration():
    data=c.build_dataset();lock=json.loads((OUT/'model_lock.json').read_text())
    winner=lock['winner'];names=lock['base_models'];recipe=lock['recipe']
    past=predict_block(data,PROTOCOL['calibration_warmup'],names,'calibration_warmup',recipe)
    records=predict_block(data,PROTOCOL['calibration'],names,'calibration',recipe)
    rules=[]
    for pool_name,pool in [('moderate only',['moderate']),('stress mixture',['moderate','bias_high','bias_low'])]:
        for method in ['absolute','signed']:
            for probability in [.95,.975]:
                vals=[]
                for r in records:
                    p=r['predictions'][winner]['moderate']
                    lo,hi=c.prediction_interval(r['origin'],p,r['future'],past,winner,method,probability,pool)
                    vals.append(c.SCORES['score'](r['truth'][24:],p[24:],lo[24:],hi[24:]))
                    past.append(r)
                past=past[:-len(records)]
                rules.append({'rule':method,'probability':probability,'pool':pool_name,
                    'Winkler':np.mean([v['Winkler'] for v in vals]),'coverage':np.mean([v['coverage'] for v in vals])})
    rules=pd.DataFrame(rules);rules.to_csv(OUT/'interval_rule_calibration.csv',index=False)
    good=rules[rules.coverage>=.95].sort_values('Winkler')
    best=good.iloc[0] if len(good) else rules.sort_values(['coverage','Winkler'],ascending=[False,True]).iloc[0]
    lock['interval']={'rule':str(best.rule),'probability':float(best.probability),
                      'pool_scenarios':['moderate'] if best.pool=='moderate only' else ['moderate','bias_high','bias_low'],
                      'calibration_coverage':float(best.coverage),'coverage_target_met':bool(best.coverage>=.95)}
    lock['implementation_sha256']={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['core.py','experiment.py']}
    lock['data_sha256']={name:hashlib.sha256((c.ROOT/name).read_bytes()).hexdigest() for name in ['train.csv','weather_forecast.csv']}
    lock['tree_parameters']={'max_iter':120,'max_leaf_nodes':15,'max_depth':5,'learning_rate':.06,
        'l2_regularization':5,'min_samples_leaf':30,'early_stopping':False,'random_state':840}
    lock['test_opened']=False
    (OUT/'final_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print('LOCKED interval:',json.dumps(lock['interval']),flush=True)


def load_records(path):
    z=np.load(OUT/path,allow_pickle=False)
    return [{'origin':int(o),'period':str(z['periods'][i]),'future':c.weather_at(int(o)),
        'truth':c.Y[o:o+48],'predictions':{str(m):{str(s):z['predictions'][i,j,k] for k,s in enumerate(z['scenarios'])}
                    for j,m in enumerate(z['models'])},'training_audit':[]} for i,o in enumerate(z['origins'])]


def evaluate_records(records,past,lock):
    rows=[];hourly=[]
    winner=lock['winner'];rule=lock['interval']['rule'];prob=lock['interval']['probability']
    for r in records:
        origin=r['origin'];past,earlier=[x for x in past if x['origin']<origin],None
        old,naive_lo,naive_hi=c.SCORES['snaive'](c.Y[:origin])
        known,_,_=c.context(origin);ratio=(known[-1].mean()+10)/(known[-7:].mean()+10)
        regime='recent drop' if ratio<.75 else ('recent rise' if ratio>1.25 else 'recent stable')
        actual_ratio=(r['truth'].mean()+10)/(known[-1].mean()+10)
        future_regime='future drop' if actual_ratio<.75 else ('future rise' if actual_ratio>1.25 else 'future stable')
        for scenario in SCENARIOS:
            for model in ['Seasonal Naive',winner] if winner!='Seasonal Naive' else ['Seasonal Naive']:
                p=r['predictions'][model][scenario]
                lo,hi=(naive_lo,naive_hi) if model=='Seasonal Naive' else c.prediction_interval(origin,p,r['future'],past,winner,rule,prob,lock['interval']['pool_scenarios'])
                for label,sl in [('day 1',slice(0,24)),('day 2',slice(24,48)),('48 hours',slice(0,48))]:
                    rows.append({'period':r['period'],'origin':str(c.DATA.ds.iloc[origin]),'origin_index':origin,
                        'model':model,'scenario':scenario,'horizon':label,'past_regime':regime,
                        'future_regime_diagnostic_only':future_regime,
                        **c.SCORES['score'](r['truth'][sl],p[sl],lo[sl],hi[sl]),
                        'MSE':float(np.mean((r['truth'][sl]-p[sl])**2)),'width':float(np.mean(hi[sl]-lo[sl]))})
                for h in range(48):
                    hourly.append({'period':r['period'],'origin_index':origin,'model':model,'scenario':scenario,
                        'horizon':h+1,'actual':r['truth'][h],'forecast':p[h],'lower_95':lo[h],'upper_95':hi[h]})
        past.append(r)
    return pd.DataFrame(rows),pd.DataFrame(hourly)


def tests():
    data=c.build_dataset();lock=json.loads((OUT/'final_lock.json').read_text())
    for name,digest in lock['implementation_sha256'].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest, 'Implementation changed after lock.'
    assert not (OUT/'tests_opened.json').exists(),'Tests already opened. Do not rerun model selection and relabel them fresh.'
    (OUT/'tests_opened.json').write_text(json.dumps({'final_lock_sha256':hashlib.sha256((OUT/'final_lock.json').read_bytes()).hexdigest(),
        'notice':'All later comparisons are evaluations of this frozen lock, never tuning inputs.'},indent=2)+'\n')
    names=lock['base_models'];recipe=lock['recipe']
    all_rows=[];all_hourly=[]
    for i,block in enumerate(PROTOCOL['independent_test_blocks']):
        if i==0:past=load_records('calibration_predictions.npz')
        else:past=predict_block(data,PROTOCOL['test_2025_warmup'],names,'test_2025_warmup',recipe)
        records=predict_block(data,block,names,block['name'],recipe)
        rows,hourly=evaluate_records(records,past,lock)
        all_rows.append(rows);all_hourly.append(hourly)
        rows.to_csv(OUT/(block['name']+'_window_scores.csv'),index=False)
        print(summary(rows)[lambda x:(x.scenario=='moderate')].to_string(index=False),flush=True)
    rows=pd.concat(all_rows,ignore_index=True);rows.to_csv(OUT/'independent_test_window_scores.csv',index=False)
    summary(rows).to_csv(OUT/'independent_test_summary.csv',index=False)
    pd.concat(all_hourly,ignore_index=True).to_csv(OUT/'independent_test_hourly.csv',index=False)
    # Recent data have already been used in older experiments. Their label
    # remains diagnostic, even if this frozen model performs very well.
    past=predict_block(data,PROTOCOL['recent_diagnostic_warmup'],names,'recent_warmup',recipe)
    block=PROTOCOL['recent_diagnostic'];records=predict_block(data,block,names,block['name'],recipe)
    rows,hourly=evaluate_records(records,past,lock)
    rows.to_csv(OUT/'recent_diagnostic_window_scores.csv',index=False)
    summary(rows).to_csv(OUT/'recent_diagnostic_summary.csv',index=False)
    hourly.to_csv(OUT/'recent_diagnostic_hourly.csv',index=False)
    print('Recent diagnostic (not a fresh test):',flush=True)
    print(summary(rows)[lambda x:(x.scenario=='moderate')].to_string(index=False),flush=True)


def final_forecast():
    data=c.build_dataset();lock=json.loads((OUT/'final_lock.json').read_text())
    names=lock['base_models'];recipe=lock['recipe'];origin=len(c.DATA)
    past=[]
    for o in range(origin-16*48,origin,48):past.append(make_record(data,o,names,'final calibration',recipe))
    frames={s:c.stress(c.FUTURE,origin,s) for s in SCENARIOS}
    naive,_,_=c.SCORES['snaive'](c.Y)
    record={'origin':origin,'future':c.FUTURE,'predictions':{'Seasonal Naive':{s:naive for s in SCENARIOS}}}
    for name in names:record['predictions'][name]=fit_predict(data,origin,SPECS[name],frames)[0]
    if recipe:add_derived(record,recipe)
    p=record['predictions'][lock['winner']]['observed']
    if lock['winner']=='Seasonal Naive':_,lo,hi=c.SCORES['snaive'](c.Y)
    else:lo,hi=c.prediction_interval(origin,p,c.FUTURE,past,lock['winner'],lock['interval']['rule'],lock['interval']['probability'],lock['interval']['pool_scenarios'])
    sub=pd.DataFrame({'ds':c.FUTURE.ds.dt.strftime('%Y-%m-%d %H:%M:%S'),'forecast':p,'lower_95':lo,'upper_95':hi})
    file=OUT/'TKACH_project.csv';sub.to_csv(file,index=False,float_format='%.6f')
    loaded=pd.read_csv(file);c.SCORES['check_submission'](loaded)
    assert np.isfinite(loaded.iloc[:,1:].to_numpy()).all()
    (OUT/'submission_check.json').write_text(json.dumps({'original_checker':'passed','rows':48,
        'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'root_csv_unchanged':True,
        'created_after_deadline':True,'no_timely_submission_claimed':True},indent=2)+'\n')
    print('Separate, late final model forecast saved and checked:',file,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['select','calibrate','test','forecast'])
    stage=parser.parse_args().stage
    {'select':selection,'calibrate':calibration,'test':tests,'forecast':final_forecast}[stage]()
