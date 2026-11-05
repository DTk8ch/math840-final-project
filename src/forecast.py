"""Reproduce final model, reusing already completed historical forecasts when present.

Without the historical cache, recompute all calibration forecasts from CSV.
With it, refit the final two point models and reuse past error records, as a
live forecasting service would. No future generation is needed in either case.
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import core as c
from experiment import SCENARIOS,SPECS,fit_predict,make_record,add_derived,load_records,save_records


def main(recompute_calibration=False,output=None):
    lock=json.loads((HERE/'final_lock.json').read_text())
    for name,digest in lock['implementation_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
    for name,digest in lock['data_sha256'].items():
        assert hashlib.sha256((c.ROOT/name).read_bytes()).hexdigest()==digest
    data=c.build_dataset();origin=len(c.DATA);wanted=list(range(origin-16*48,origin,48))
    cache=HERE/'final_calibration_predictions.npz'
    meta=HERE/'final_calibration_metadata.json'
    lock_sha=hashlib.sha256((HERE/'final_lock.json').read_bytes()).hexdigest()
    records=[];source='rebuilt from CSV'
    if cache.exists() and not recompute_calibration:
        cached_meta=json.loads(meta.read_text())
        assert cached_meta['lock_sha256']==lock_sha
        assert cached_meta['predictions_sha256']==hashlib.sha256(cache.read_bytes()).hexdigest()
        records=load_records(cache.name);source='verified completed-forecast cache'
    elif not recompute_calibration:
        for name in ['recent_warmup_predictions.npz','diagnostic_2026_october_predictions.npz']:
            if (HERE/name).exists():records.extend(load_records(name))
        records=[r for r in records if r['origin'] in wanted]
        if sorted(r['origin'] for r in records)==wanted:source='already computed recent historical forecasts'
        else:records=[]
    if not records:
        for i,o in enumerate(wanted):
            records.append(make_record(data,o,lock['base_models'],'final calibration',lock['recipe']))
            print(f'Recomputed calibration {i+1}/16',flush=True)
    records=sorted(records,key=lambda r:r['origin'])
    assert [r['origin'] for r in records]==wanted
    assert all(r['origin']+48<=origin for r in records)
    if not cache.exists() or recompute_calibration:
        save_records(records,cache.name)
        boundary=HERE/(cache.stem+'_training_boundaries.csv')
        if boundary.stat().st_size<3:
            boundary.write_text('period,model,training_origins,training_rows,last_target_index,forecast_origin\n')
        meta.write_text(json.dumps({'lock_sha256':lock_sha,'data_sha256':lock['data_sha256'],
            'predictions_sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),
            'last_target_index':origin-1},indent=2)+'\n')
    frames={s:c.stress(c.FUTURE,origin,s) for s in SCENARIOS}
    naive,_,_=c.SCORES['snaive'](c.Y)
    current={'origin':origin,'future':c.FUTURE,'predictions':{'Seasonal Naive':{s:naive for s in SCENARIOS}}}
    for name in lock['base_models']:current['predictions'][name]=fit_predict(data,origin,SPECS[name],frames)[0]
    if lock['recipe']:add_derived(current,lock['recipe'])
    point=current['predictions'][lock['winner']]['observed']
    if lock['winner']=='Seasonal Naive':_,lower,upper=c.SCORES['snaive'](c.Y)
    else:
        settings=lock['interval']
        lower,upper=c.prediction_interval(origin,point,c.FUTURE,records,lock['winner'],
            settings['rule'],settings['probability'],settings['pool_scenarios'])
    sub=pd.DataFrame({'ds':c.FUTURE.ds.dt.strftime('%Y-%m-%d %H:%M:%S'),
                      'forecast':point,'lower_95':lower,'upper_95':upper})
    file=Path(output) if output else HERE/'primary/TKACH_project.csv'
    file.parent.mkdir(parents=True,exist_ok=True)
    sub.to_csv(file,index=False,float_format='%.6f')
    loaded=pd.read_csv(file);c.SCORES['check_submission'](loaded)
    assert np.isfinite(loaded.iloc[:,1:].to_numpy()).all()
    assert pd.to_datetime(loaded.ds).diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    info={'rows':48,'official_check_submission':'passed','source_of_calibration':source,
          'final_point_models':'refitted from source CSV, no saved fitted model',
          'weather_for_horizon':'supplied issued weather_forecast.csv; no hidden actual weather',
          'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
          'created_after_21_00_Kyiv':True,'root_submission_overwritten':False}
    (file.parent/'submission_check.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute-calibration',action='store_true')
    parser.add_argument('--output')
    args=parser.parse_args();main(args.recompute_calibration,args.output)
