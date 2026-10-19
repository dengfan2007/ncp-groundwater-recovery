"""Stage runner for the frozen monitoring revision; historical evidence is read-only."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from ncp_monitoring_core import (ROOT,STORY,REVIEW,VALIDATION,EVIDENCE,RUNTIME,EMD,ENDPOINTS,
    panel,design,select,clean,sha,seed,fit,run_tasks,export_results,frozen_actual,json_write)

def actual_task(gid,g,endpoint,task_id=None):
    g=g.loc[g.date.le(endpoint)]
    return clean(dict(task_id=task_id or f'{gid}_{endpoint}',dates=g.date.astype(str).tolist(),y=g.gwa_cm.tolist(),
                      ssa_iterations=1000 if endpoint=='2024-08-01' else 200,save_series=gid in ['g_115.0_35.5','g_116.0_37.5'],
                      metadata=dict(grid_id=gid,region_tag=g.region_tag.iloc[0],endpoint=endpoint,experiment='baseline')))

@lru_cache(maxsize=1)
def synthetic_inputs():
    p=panel();dates=pd.date_range('2003-01-01','2024-08-01',freq='MS')
    observed=p.groupby('date').gwa_cm.count().reindex(dates).gt(0).to_numpy()
    return pd.read_csv(VALIDATION/'local/noise_quantiles.csv').set_index('quantile'),observed

def make_synthetic(experiment,scenario,regime,rep,split,onset='2018-01-01',weak=.6,mask='observed'):
    dates=pd.date_range('2003-01-01','2024-08-01',freq='MS');t=np.arange(len(dates))/12
    quantiles,observed=synthetic_inputs();q=quantiles.loc[{'low':.25,'median':.5,'high':.75}[regime]]
    # Mask and route are intentionally excluded from noise/algorithm seeds for pairing.
    key=[experiment,scenario,regime,rep,split,onset,weak]
    ns=seed(*key,'noise');bs=seed(*key,'beast');es=seed(*key,'ensemble');rng=np.random.default_rng(ns)
    z=rng.normal(size=len(t));noise=np.empty(len(t));phi=float(q.phi);sigma=float(q.innovation_sd)
    noise[0]=sigma*z[0]/np.sqrt(1-phi**2)
    for i in range(1,len(t)):noise[i]=phi*noise[i-1]+sigma*z[i]
    trend=-1.5*t;truth=None;acceleration=None
    if scenario=='temporary_gain':trend+=12*np.maximum(1-np.abs(t-18.5)/1.5,0)
    elif scenario=='sustained_reversal':trend+=4.5*np.maximum(t-16,0);truth='2019-01-01'
    elif scenario=='two_stage':
        k=(pd.Timestamp(onset).year-2003)+(pd.Timestamp(onset).month-1)/12
        trend+=(1.5+weak)*np.maximum(t-k,0)+(4-weak)*np.maximum(t-k-3,0)
        truth=onset;acceleration=str((pd.Timestamp(onset)+pd.DateOffset(months=36)).date())
    y=trend+float(q.annual_amplitude)*np.sin(2*np.pi*t)+noise
    if mask=='observed':
        y[~observed]=np.nan
    task_id=f'{experiment}_{split}_{scenario}_{regime}_{onset}_{weak:.1f}_{rep:04d}_{mask}'
    return clean(dict(task_id=task_id,dates=dates.astype(str).tolist(),y=y.tolist(),beast_seed=bs,ensemble_seed=es,ssa_iterations=1000,
                      metadata=dict(experiment=experiment,split=split,scenario=scenario,regime=regime,replicate=rep,mask=mask,onset=truth,
                                    acceleration=acceleration,weak_slope=weak,noise_seed=ns,beast_seed=bs,ensemble_seed=es)))

def baseline(workers):
    p=panel();expected=frozen_actual();frozen_selections=[]
    for keys,g in expected.groupby(['grid_id','region_tag','endpoint','route']):
        cs=clean(g.to_dict('records'))
        for rule in ['highest_probability','earliest']:
            c=select(cs,rule=rule)
            frozen_selections.append(dict(zip(['grid_id','region_tag','endpoint','route'],keys))|dict(rule=rule,qualifies=c is not None,selected_date=c['date'] if c else None))
    frozen=pd.DataFrame(frozen_selections)
    old=pd.read_csv(STORY/'actual_selections.csv');new=pd.read_csv(VALIDATION/'fits/actual_selections.csv')
    old=old.loc[old.scope.eq('grid')&old.experiment.eq('actual')&old.route.isin(['A','D'])&old.endpoint.isin(ENDPOINTS)]
    new=new.loc[new.scope.eq('grid')&new.experiment.eq('original')&new.endpoint.eq('2024-08-01')&new.rule.isin(['highest_probability','earliest'])]
    gold=pd.concat([old,new]);keys=['grid_id','endpoint','route','rule']
    z=frozen.merge(gold[keys+['qualifies','selected_date']],on=keys,suffixes=('_reselected','_frozen'),validate='one_to_one')
    assert len(z)==27*6*2*2
    assert (z.qualifies_reselected==z.qualifies_frozen).all()
    assert (z.selected_date_reselected.fillna('')==z.selected_date_frozen.fillna('')).all()
    slopes=[]
    for gid,g in p.loc[p.date.between('2019-01-01','2024-08-01')].groupby('grid_id'):
        X=design(g.date);ok=g.gwa_cm.notna().to_numpy();coef=np.linalg.lstsq(X[ok],g.gwa_cm.to_numpy()[ok],rcond=None)[0]
        slopes.append(dict(grid_id=gid,region_tag=g.region_tag.iloc[0],slope=coef[1]))
    slopes=pd.DataFrame(slopes);delta=slopes.loc[slopes.region_tag.eq('south'),'slope'].mean()-slopes.loc[slopes.region_tag.eq('north'),'slope'].mean()
    reference=pd.read_csv(ROOT/'outputs/figure_revision/2026-10-17-final/tables/fig02_trend_intervals.csv').iloc[0].estimate
    assert abs(delta-reference)<1e-6
    tasks=[actual_task(gid,g,end) for gid,g in p.groupby('grid_id') for end in ENDPOINTS]
    results=run_tasks(tasks,RUNTIME/'baseline/checkpoints',workers)
    rows=[{**r['metadata'],**s} for r in results for s in r['selections']];actual=pd.DataFrame(rows)
    z=actual.merge(frozen,on=keys,suffixes=('_rerun','_frozen'),validate='one_to_one')
    bad=z.loc[(z.qualifies_rerun!=z.qualifies_frozen)|(z.selected_date_rerun.fillna('')!=z.selected_date_frozen.fillna(''))]
    dest=EVIDENCE/'baseline';export_results(results,dest);z.to_csv(dest/'date_comparison.csv',index=False);bad.to_csv(dest/'discrepancies.csv',index=False)
    stable=[]
    for route in ['A','D']:
        a=frozen.loc[frozen.route.eq(route)&frozen.rule.eq('highest_probability')&frozen.endpoint.eq('2023-12-01')&frozen.qualifies]
        b=frozen.loc[frozen.route.eq(route)&frozen.rule.eq('highest_probability')&frozen.endpoint.eq('2024-08-01')&frozen.qualifies]
        m=a.merge(b,on='grid_id');d1=pd.to_datetime(m.selected_date_x);d2=pd.to_datetime(m.selected_date_y);diff=(d2.dt.year-d1.dt.year)*12+d2.dt.month-d1.dt.month
        stable.append(dict(route=route,n=len(m),within_six=int(diff.abs().le(6).sum())))
    assert stable==[dict(route='A',n=24,within_six=13),dict(route='D',n=23,within_six=13)]
    json_write(dest/'acceptance.json',dict(pass_gate=bad.empty,trend_delta=delta,reference=reference,absolute_error=abs(delta-reference),stability=stable,n_rerun_tasks=len(results),n_date_discrepancies=len(bad),declaration_commit='91bcd9e'))
    if not bad.empty:raise RuntimeError(f'Baseline gate failed: {len(bad)} date discrepancies; scientific stages stopped')
    print('Baseline gate passed',flush=True)

def repeatability():
    os.environ['NCP_EMD_RUNTIME']=str(EMD);p=panel();tasks=[]
    for i,(gid,end) in enumerate([('g_115.0_35.5','2023-12-01'),('g_116.0_37.5','2023-12-01'),('g_115.0_35.5','2024-08-01'),('g_116.0_37.5','2024-08-01')],1):
        tasks.append(actual_task(gid,p.loc[p.grid_id.eq(gid)],end,f'actual_{i:02d}'))
    for i,(scenario,regime) in enumerate([(s,r) for s in ['no_recovery','temporary_gain','sustained_reversal'] for r in ['low','high']],1):
        t=make_synthetic('repeatability',scenario,regime,0,'independent');t['task_id']=f'synthetic_{i:02d}';tasks.append(t)
    dest=EVIDENCE/'repeatability';dest.mkdir(parents=True,exist_ok=False);checks=[]
    for task in tasks:
        a,b=fit(task),fit(task)
        good=(a['signal_digests']==b['signal_digests'] and a['candidates']==b['candidates'] and a['selections']==b['selections'])
        checks.append(dict(task_id=task['task_id'],bitwise_identical=good,first=a,second=b));print(task['task_id'],good,flush=True)
        json_write(dest/(task['task_id']+'.json'),checks[-1])
        if not good:json_write(dest/'acceptance.json',dict(pass_gate=False,checks=checks));raise RuntimeError('Bitwise repeatability failed')
    json_write(dest/'acceptance.json',dict(pass_gate=True,n=10))

def gates():
    for stage in ['baseline','repeatability']:
        if not json.loads((EVIDENCE/stage/'acceptance.json').read_text(encoding='utf-8'))['pass_gate']:raise RuntimeError('Gate not passed: '+stage)

def simulation_tasks(kind,pilot=False):
    tasks=[]
    if kind=='calibration':
        for split in ['calibration','validation']:
            for scenario in ['no_recovery','temporary_gain','sustained_reversal']:
                for regime in ['low','median','high']:
                    for rep in range(20 if pilot else 500):
                        tasks.append(make_synthetic('calibration',scenario,regime,rep,'pilot_'+split if pilot else split))
    elif kind=='two_stage':
        configs=[(f'{year}-01-01',weak,'median') for year in [2016,2018,2019,2020] for weak in [.3,.6,1.2]]
        configs += [('2018-01-01',.6,r) for r in ['low','high']]
        for onset,weak,regime in configs:
            for rep in range(20 if pilot else 200):
                for mask in ['complete','observed']:
                    tasks.append(make_synthetic('two_stage','two_stage',regime,rep,'pilot' if pilot else 'formal',onset,weak,mask))
    return tasks

def simulation(stage,workers):
    gates();pilot=stage=='pilot';kinds=['calibration','two_stage'] if pilot else [stage]
    tasks=[t for kind in kinds for t in simulation_tasks(kind,pilot)]
    out=EVIDENCE/stage
    start=time.time();results=run_tasks(tasks,RUNTIME/stage/'checkpoints',workers);export_results(results,out)
    truth=pd.DataFrame([{'task_id':t['task_id'],**t['metadata'],'y_sha256':__import__('hashlib').sha256(np.asarray(t['y'],float).tobytes()).hexdigest()} for t in tasks]);truth.to_csv(out/'truth.csv',index=False)
    diagnostics=[r['diagnostics'].get('ssa',{}) for r in results]
    summary=dict(tasks=len(results),beast_fits=len(results)*2,elapsed_seconds=time.time()-start,workers=workers,
                 ssa_not_converged=sum(x.get('converged') is False for x in diagnostics),source_sha256=sha(__file__),core_sha256=sha(ROOT/'analysis/ncp_monitoring_core.py'),
                 unique_noise_seeds=truth.noise_seed.nunique(),pilot_excluded_from_formal=pilot)
    json_write(out/'manifest.json',summary);print(json.dumps(summary),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['baseline','repeatability','pilot','calibration','two_stage'],required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    EVIDENCE.mkdir(parents=True,exist_ok=True);RUNTIME.mkdir(parents=True,exist_ok=True)
    if a.stage=='baseline':baseline(a.workers)
    elif a.stage=='repeatability':repeatability()
    else:simulation(a.stage,a.workers)

if __name__=='__main__':main()
