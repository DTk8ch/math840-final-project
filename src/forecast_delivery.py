"""Forecast the recipe selected on calibration before test losses were read."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import hashlib
import json
import numpy as np
import pandas as pd
import core as c
import forecast
import adaptive
from experiment import load_records,save_records,make_record


def main():
    choice=json.loads((HERE/'delivery_choice.json').read_text())
    lock=json.loads((HERE/'final_lock.json').read_text())
    gate=json.loads((HERE/'adaptive_lock.json').read_text())
    assert choice['primary_lock_sha256']==hashlib.sha256((HERE/'final_lock.json').read_bytes()).hexdigest()
    assert choice['adaptive_lock_sha256']==hashlib.sha256((HERE/'adaptive_lock.json').read_bytes()).hexdigest()
    assert gate['implementation_sha256']==hashlib.sha256((HERE/'adaptive.py').read_bytes()).hexdigest()
    forecast.main()
    primary=pd.read_csv(HERE/'primary/TKACH_project.csv')
    origin=len(c.DATA);base=lock['winner']
    if choice['model']==base:
        sub=primary;weights=np.ones(2);source='base ensemble selected on calibration'
    else:
        cache=HERE/'adaptive_base_history.npz';meta=HERE/'adaptive_base_history.json'
        required=16+gate['window'];wanted=list(range(origin-required*48,origin,48))
        source='stored completed historical forecasts'
        if cache.exists():
            info=json.loads(meta.read_text())
            assert info['primary_lock_sha256']==choice['primary_lock_sha256']
            assert info['sha256']==hashlib.sha256(cache.read_bytes()).hexdigest()
            records=load_records(cache.name)
        else:
            available={}
            for filename in ['recent_warmup_predictions.npz','diagnostic_2026_october_predictions.npz','final_calibration_predictions.npz']:
                if (HERE/filename).exists():
                    for r in load_records(filename):available[r['origin']]=r
            data=None;records=[]
            for o in wanted:
                if o in available:r=available[o]
                else:
                    if data is None:data=c.build_dataset()
                    r=make_record(data,o,lock['base_models'],'adaptive final history',lock['recipe'])
                    source='partly rebuilt from CSV'
                r={**r,'predictions':{k:v for k,v in r['predictions'].items() if k in ['Seasonal Naive',base]}}
                records.append(r)
            save_records(records,cache.name)
            boundary=HERE/(cache.stem+'_training_boundaries.csv')
            if boundary.stat().st_size<3:boundary.write_text('period,model,training_origins,training_rows,last_target_index,forecast_origin\n')
            meta.write_text(json.dumps({'primary_lock_sha256':choice['primary_lock_sha256'],
                'sha256':hashlib.sha256(cache.read_bytes()).hexdigest()},indent=2)+'\n')
        assert [r['origin'] for r in records]==wanted
        assert all(r['origin']+48<=origin for r in records)
        weights=adaptive.blend_weights(origin,records,base,gate['window'],gate['shrink'])
        naive=c.SCORES['snaive'](c.Y)[0]
        p=np.repeat(weights,24)*primary.forecast.to_numpy()+(1-np.repeat(weights,24))*naive
        past=adaptive.add_adaptive(records,base,gate['window'],gate['shrink'])[-16:]
        settings=gate['interval']
        lo,hi=c.prediction_interval(origin,p,c.FUTURE,past,adaptive.NAME,settings['rule'],settings['probability'],settings['pool_scenarios'])
        sub=pd.DataFrame({'ds':primary.ds,'forecast':p,'lower_95':lo,'upper_95':hi})
    file=HERE/'TKACH_project.csv';sub.to_csv(file,index=False,float_format='%.6f')
    saved=pd.read_csv(file);c.SCORES['check_submission'](saved)
    assert np.isfinite(saved.iloc[:,1:].to_numpy()).all()
    result={'model':choice['model'],'selected_from':'development/calibration only, before test losses inspected',
        'day1_base_model_weight':float(weights[0]),'day2_base_model_weight':float(weights[1]),
        'weight_history_weather':'moderate synthetic forecasts, not a real issued-forecast archive',
        'historical_calibration_source':source,'official_check_submission':'passed','rows':len(saved),
        'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'original_root_csv_overwritten':False,
        'created_after_deadline':True}
    (HERE/'submission_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
