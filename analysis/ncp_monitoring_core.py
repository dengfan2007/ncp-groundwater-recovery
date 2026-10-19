"""Fixed scientific primitives for the declared October 18 monitoring revision."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
import Rbeast as rb
import ncp_recovery_analysis as legacy
from ncp_signal_reconstruction import ssa_fill, reconstruct

ROOT = Path(__file__).resolve().parents[1]
STORY = ROOT/'outputs/story_revision/2026-10-17-r1'
REVIEW = ROOT/'outputs/review_revision/2026-10-15-r1'
VALIDATION = ROOT/'outputs/validation_revision/2026-10-17-r2'
EVIDENCE = Path(os.environ.get('NCP_MONITORING_EVIDENCE',str(ROOT/'outputs/monitoring_revision/2026-10-18-r1'))).resolve()
RUNTIME = Path(os.environ.get('NCP_MONITORING_RUN_DIR',str(ROOT/'outputs/runs/2026-10-18-monitoring-r1'))).resolve()
EMD = ROOT/'outputs/runs/2026-10-16-story-runtime-r1'
BOUNDARY = [f'g_{i}.0_36.5' for i in range(114, 120)]
ENDPOINTS = [f'{y}-12-01' for y in range(2019, 2024)] + ['2024-08-01']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_): return bool(value)
    return value

def json_write(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def seed(*parts):
    raw=json.dumps([20261018,*parts], separators=(',', ':'), ensure_ascii=True).encode()
    return 1+int.from_bytes(hashlib.sha256(raw).digest()[:8], 'little')%(2**31-2)

def panel(uniform=False):
    path=VALIDATION/'uniform/uniform_calendar_panel.csv' if uniform else ROOT/'data/raw/grid_monthly_panel.csv'
    return pd.read_csv(path, parse_dates=['date']).sort_values(['grid_id','date'])

def months(dates):
    d=pd.DatetimeIndex(dates)
    return d.year*12+d.month-1

def design(dates):
    t=(months(dates)-months(dates)[0])/12
    return np.column_stack([np.ones(len(t)),t,np.sin(2*np.pi*t),np.cos(2*np.pi*t),np.sin(4*np.pi*t),np.cos(4*np.pi*t)])

def deseason(y):
    y=np.asarray(y,float)
    fill=pd.Series(y).interpolate(limit_direction='both').to_numpy()
    season=STL(fill,period=12,robust=True).fit().seasonal
    return np.where(np.isfinite(y),y-season,np.nan)

def lin_slope(y, idx):
    idx=np.asarray(idx); valid=np.isfinite(y[idx]); n=int(valid.sum())
    return (float(stats.linregress(idx[valid]/12,y[idx][valid]).slope) if n>=18 else None),n

def select(candidates, probability=.5, gain=.5, rule='highest_probability', local=False):
    pool=[]
    for c in candidates:
        if c['date']<'2015-01-01' or c['probability']<probability: continue
        if local:
            qualifies=c.get('local_pre_slope') is not None and c.get('local_post_slope') is not None and c['local_pre_slope']<0<c['local_post_slope']
        else:
            a,b=c.get('pre_slope'),c.get('post_slope')
            qualifies=a is not None and b is not None and np.isfinite(a) and np.isfinite(b) and b>0 and b-a>gain
        if qualifies: pool.append(c)
    if not pool:return None
    return min(pool,key=lambda c:c['date']) if rule=='earliest' else max(pool,key=lambda c:(c['probability'],c['date']))

def fit(task):
    """One original input, A/D fits, full candidate pool, deterministic signal digests."""
    began=time.perf_counter();dates=pd.to_datetime(task['dates']);y=np.asarray(task['y'],float)
    q=deseason(y);signals={'A':q};diagnostics={}
    if 'D' in task.get('routes',['A','D']):
        filled,info=ssa_fill(y,window=36,rank=6,tolerance=1e-6,max_iterations=task.get('ssa_iterations',1000))
        diagnostics['ssa']=info
        signals['D'],info=reconstruct(filled,trials=50,noise_scale=.2,seed=task.get('ensemble_seed',42),min_period=24.)
        diagnostics['reconstruction']=info
    frame=pd.DataFrame({'date':dates,'deseasoned_gwa':q});candidates=[];selections=[];digests={};series=[]
    for route in task.get('routes',['A','D']):
        signal=signals[route]
        r=rb.beast(signal,start=dates[0].year+(dates[0].month-1)/12,deltat=1/12,period=1.,season='none',tseg_minlength=18,
                   quiet=1,mcmc_seed=task.get('beast_seed',42),print_param=False,print_progress=False,print_warning=False)
        pool=[]
        for cp,pr in zip(np.asarray(r.trend.cp).ravel(),np.asarray(r.trend.cpPr).ravel()):
            if not(np.isfinite(cp) and np.isfinite(pr)):continue
            d=legacy.maybe_date_from_cp(frame,float(cp));i=legacy.breakpoint_index(frame,d)
            a,b=legacy.fit_trends_for_breakpoint(frame,i)
            pre,npre=lin_slope(q,np.arange(max(0,i-36),i));post,npost=lin_slope(q,np.arange(i,min(len(q),i+36)))
            c=clean(dict(route=route,date=str(d.date()),probability=float(pr),pre_slope=a,post_slope=b,
                         local_pre_slope=pre,local_post_slope=post,local_pre_observations=npre,local_post_observations=npost,
                         post_calendar_months=min(len(q)-i,36),pre_calendar_months=min(i,36)))
            pool.append(c)
        candidates.extend(pool)
        for rule in ['highest_probability','earliest']:
            c=select(pool,rule=rule)
            selections.append(dict(route=route,rule=rule,qualifies=c is not None,selected_date=c['date'] if c else None))
        digests[route]={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in
                       [('input',y),('qualification',q),('signal',signal),('trend',np.asarray(r.trend.Y)),('cp',np.asarray(r.trend.cp)),('probability',np.asarray(r.trend.cpPr))]}
        if task.get('save_series'):
            series.extend(clean(dict(date=str(d.date()),route=route,observed=y[i],input_signal=signal[i],qualification_signal=q[i],fitted_trend=float(np.asarray(r.trend.Y).ravel()[i]),cp_probability=float(np.asarray(r.trend.cpOccPr).ravel()[i]))) for i,d in enumerate(dates))
    return dict(task_id=task['task_id'],metadata=task.get('metadata',{}),candidates=candidates,selections=selections,
                signal_digests=digests,series=series,diagnostics=diagnostics,elapsed_seconds=time.perf_counter()-began)

def checkpoint_worker(task, folder, source_sha):
    os.environ['NCP_EMD_RUNTIME']=str(EMD)
    signature=hashlib.sha256((json.dumps(clean(task),sort_keys=True,allow_nan=False)+source_sha).encode()).hexdigest()
    path=Path(folder)/(task['task_id']+'.json')
    if path.exists():
        r=json.loads(path.read_text(encoding='utf-8'))
        if r['task_signature']!=signature:raise RuntimeError('Checkpoint signature mismatch: '+str(path))
        return r['result']
    result=fit(task)
    json_write(path,dict(task_signature=signature,source_sha256=source_sha,result=result))
    return result

def run_tasks(tasks, folder, workers=8):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);source_sha=sha(__file__);started=time.time();results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(checkpoint_worker,t,str(folder),source_sha):t['task_id'] for t in tasks}
        for i,f in enumerate(as_completed(futures),1):
            results.append(f.result())
            if i%10==0 or i==len(tasks):print(f'{folder.name} {i}/{len(tasks)} elapsed={time.time()-started:.1f}s',flush=True)
    return sorted(results,key=lambda r:r['task_id'])

def export_results(results, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    for key in ['candidates','selections','series']:
        rows=[{**r['metadata'],'task_id':r['task_id'],**row} for r in results for row in r[key]]
        if rows:pd.DataFrame(rows).to_csv(folder/(key+'.csv'),index=False,lineterminator='\n')
    json_write(folder/'diagnostics.json',[{k:v for k,v in r.items() if k not in ['candidates','selections','series']} for r in results])

def frozen_actual():
    old=pd.read_csv(STORY/'actual_candidates.csv')
    old=old.loc[old.scope.eq('grid')&old.route.isin(['A','D'])&old.experiment.eq('actual')&old.endpoint.isin(ENDPOINTS)]
    new=pd.read_csv(VALIDATION/'fits/actual_candidates.csv')
    new=new.loc[new.scope.eq('grid')&new.experiment.eq('original')&new.endpoint.eq('2024-08-01')]
    return pd.concat([old,new],ignore_index=True)
