"""An optional pre-test, calibration-selected convex blend with Naive."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import argparse
import copy
import hashlib
import json
import numpy as np
import pandas as pd
import core as c
from experiment import load_records,summary,SCENARIOS

NAME='Adaptive forecast'


def blend_weights(origin,past,base,window,shrink):
    completed=[r for r in past if r['origin']+48<=origin][-window:]
    if len(completed)<4:return np.ones(2)
    weights=[]
    for sl in [slice(0,24),slice(24,48)]:
        numerator=0.;denominator=0.
        for r in completed:
            scale=c.context(r['origin'])[1]
            naive=r['predictions']['Seasonal Naive']['moderate'][sl]
            model=r['predictions'][base]['moderate'][sl]
            y=r['truth'][sl]
            difference=(model-naive)/scale
            numerator+=np.mean((y-naive)/scale*difference)
            denominator+=np.mean(difference**2)
        w=np.clip(numerator/max(denominator,1e-12),0,1)
        weights.append(shrink+(1-shrink)*w)
    return np.array(weights)


def add_adaptive(records,base,window,shrink):
    history=[];out=[]
    for original in sorted(records,key=lambda r:r['origin']):
        r={**original,'predictions':dict(original['predictions'])}
        weights=np.repeat(blend_weights(r['origin'],history,base,window,shrink),24)
        r['predictions'][NAME]={s:weights*r['predictions'][base][s]+(1-weights)*r['predictions']['Seasonal Naive'][s] for s in SCENARIOS}
        r['adaptive_weights']=weights[[0,24]].tolist()
        out.append(r);history.append(r)
    return out


def historical_files(include_test=False):
    files=['development_predictions.npz','calibration_warmup_predictions.npz','calibration_predictions.npz']
    if include_test:
        files+=['test_2024_autumn_predictions.npz','test_2025_warmup_predictions.npz',
            'test_2025_autumn_predictions.npz','recent_warmup_predictions.npz','diagnostic_2026_october_predictions.npz']
    answer=[]
    for name in files:answer.extend(load_records(name))
    assert len({r['origin'] for r in answer})==len(answer)
    return answer


def calibrate():
    main=json.loads((HERE/'final_lock.json').read_text());base=main['winner']
    raw=historical_files(False);choices=[];variants={}
    for window,shrink in [(8,0.),(8,.5),(16,0.),(16,.5)]:
        records=add_adaptive(raw,base,window,shrink)
        cal=[r for r in records if r['period']=='calibration']
        errors=[];percentage=[]
        for r in cal:
            p=r['predictions'][NAME]['moderate'][24:];y=r['truth'][24:]
            errors.append(np.mean((y-p)**2));percentage.append(c.SCORES['mape'](y,p))
        key=f'{window} blocks; shrink {shrink}'
        choices.append({'recipe':key,'window':window,'shrink':shrink,'RMSE':np.sqrt(np.mean(errors)),'MAPE':np.mean(percentage)})
        variants[key]=records
    cal=[r for r in raw if r['period']=='calibration']
    b_rmse=np.sqrt(np.mean([np.mean((r['truth'][24:]-r['predictions'][base]['moderate'][24:])**2) for r in cal]))
    b_mape=np.mean([c.SCORES['mape'](r['truth'][24:],r['predictions'][base]['moderate'][24:]) for r in cal])
    choices.append({'recipe':'base ensemble','window':0,'shrink':1.,'RMSE':b_rmse,'MAPE':b_mape})
    table=pd.DataFrame(choices)
    table['objective']=.7*np.log(table.RMSE/b_rmse)+.3*np.log(table.MAPE/b_mape)
    table=table.sort_values('objective');table.to_csv(HERE/'adaptive_point_calibration.csv',index=False)
    best=table.iloc[0]
    if best.recipe=='base ensemble':
        lock={'selected':'base ensemble','model':base,'window':0,'shrink':1.,'interval':main['interval']}
    else:
        records=variants[best.recipe];rules=[]
        for pool in [['moderate'],['moderate','bias_high','bias_low']]:
            for rule in ['absolute','signed']:
                for prob in [.95,.975]:
                    vals=[];past=[]
                    for r in records:
                        if r['period']=='calibration':
                            p=r['predictions'][NAME]['moderate']
                            lo,hi=c.prediction_interval(r['origin'],p,r['future'],past,NAME,rule,prob,pool)
                            vals.append(c.SCORES['score'](r['truth'][24:],p[24:],lo[24:],hi[24:]))
                        past.append(r)
                    rules.append({'rule':rule,'probability':prob,'pool':'+'.join(pool),
                        'coverage':np.mean([v['coverage'] for v in vals]),'Winkler':np.mean([v['Winkler'] for v in vals])})
        rules=pd.DataFrame(rules);rules.to_csv(HERE/'adaptive_interval_calibration.csv',index=False)
        valid=rules[rules.coverage>=.95].sort_values('Winkler')
        rule=valid.iloc[0] if len(valid) else rules.sort_values(['coverage','Winkler'],ascending=[False,True]).iloc[0]
        lock={'selected':str(best.recipe),'model':NAME,'base_model':base,'window':int(best.window),'shrink':float(best.shrink),
            'interval':{'rule':str(rule.rule),'probability':float(rule.probability),'pool_scenarios':str(rule.pool).split('+'),
                        'calibration_coverage':float(rule.coverage)}}
    lock['selected_from']='calibration point losses only; no historical test losses inspected'
    lock['locked_utc']=pd.Timestamp.now(tz='UTC').isoformat()
    lock['primary_lock_sha256']=hashlib.sha256((HERE/'final_lock.json').read_bytes()).hexdigest()
    lock['implementation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (HERE/'adaptive_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print(table.to_string(index=False),flush=True);print(json.dumps(lock,indent=2),flush=True)


def evaluate():
    lock=json.loads((HERE/'adaptive_lock.json').read_text());main=json.loads((HERE/'final_lock.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==lock['implementation_sha256']
    assert hashlib.sha256((HERE/'final_lock.json').read_bytes()).hexdigest()==lock['primary_lock_sha256']
    if lock['selected']=='base ensemble':
        (HERE/'adaptive_outcome.json').write_text(json.dumps({'selected':'base ensemble','adaptive_rule_did_not_win_calibration':True},indent=2)+'\n')
        print('Base ensemble retained by calibration. No test-based switch.');return
    records=add_adaptive(historical_files(True),main['winner'],lock['window'],lock['shrink'])
    past=[];rows=[];hourly=[];weights=[];settings=lock['interval']
    for r in records:
        if r['period'] in ['test_2024_autumn','test_2025_autumn','diagnostic_2026_october']:
            for scenario in SCENARIOS:
                p=r['predictions'][NAME][scenario]
                lo,hi=c.prediction_interval(r['origin'],p,r['future'],past,NAME,settings['rule'],settings['probability'],settings['pool_scenarios'])
                for label,sl in [('day 1',slice(0,24)),('day 2',slice(24,48)),('48 hours',slice(0,48))]:
                    rows.append({'period':r['period'],'origin':str(c.DATA.ds.iloc[r['origin']]),'origin_index':r['origin'],
                        'model':NAME,'scenario':scenario,'horizon':label,
                        **c.SCORES['score'](r['truth'][sl],p[sl],lo[sl],hi[sl]),
                        'MSE':float(np.mean((r['truth'][sl]-p[sl])**2)),'width':float(np.mean(hi[sl]-lo[sl]))})
                for h in range(48):hourly.append({'period':r['period'],'origin_index':r['origin'],'model':NAME,'scenario':scenario,
                    'horizon':h+1,'actual':r['truth'][h],'forecast':p[h],'lower_95':lo[h],'upper_95':hi[h]})
            weights.append({'period':r['period'],'origin':str(c.DATA.ds.iloc[r['origin']]),
                'day1_model_weight':r['adaptive_weights'][0],'day2_model_weight':r['adaptive_weights'][1]})
        past.append(r)
    scores=pd.DataFrame(rows);scores.to_csv(HERE/'adaptive_window_scores.csv',index=False)
    summary(scores).to_csv(HERE/'adaptive_summary.csv',index=False)
    pd.DataFrame(hourly).to_csv(HERE/'adaptive_hourly.csv',index=False)
    pd.DataFrame(weights).to_csv(HERE/'adaptive_origin_weights.csv',index=False)
    print(summary(scores)[lambda d:(d.scenario=='moderate')&(d.horizon=='day 2')].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['calibrate','evaluate'])
    {'calibrate':calibrate,'evaluate':evaluate}[p.parse_args().stage]()
