"""Focused reconstruction/record-endpoint comparison and four truth scenarios.

Run with an isolated EMD-signal runtime, using a new output directory.
All specifications are fixed in SPEC; no numerical conclusion is a run target.
"""
from __future__ import annotations
import os
# Bound BLAS threads in worker processes, without changing the installed runtime.
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
import Rbeast as rb
import ncp_recovery_analysis as core
from ncp_signal_reconstruction import ssa_fill,reconstruct
from ncp_calendar_revision import calendar_detrend
from ncp_revision_analyses import pearson_or_nan,moving_block_indices

ROOT=Path(__file__).resolve().parents[1]
SPEC=dict(ssa_window=36,ssa_rank=6,ssa_tolerance=1e-6,ssa_iterations=200,
          ensemble_trials=50,ensemble_noise=.2,ensemble_seed=42,minimum_period_months=24,
          emd_sifting='PyEMD 1.10.0 defaults; cubic spline; nbsym=2; maximum 1000 iterations',
          beast_seed=42,minimum_real_observations_each_side=18,minimum_cp_probability=.5,
          slope_gain_cm_per_year=.5,earliest_reporting_date='2015-01-01',
          common_endpoint='2023-11-01',endpoints=['2019-12-01','2020-12-01','2021-12-01','2022-12-01','2023-12-01','2024-09-01'],
          simulation_replicates=100,simulation_ar1=.6,simulation_innovation_sd=1.5,
          qualification='Same original observed STL-deseasoned series for every route; no segmented fallback',
          routes={'A':'linear-filled STL seasonal estimate; original observation mask in BEAST',
                  'B':'SSA-filled STL seasonal estimate; original mask in BEAST',
                  'C':'SSA-filled STL; filled observations retained in BEAST',
                  'D':'SSA-filled improved adaptive-local-mean reconstruction; mean mode period >=24 months plus residue'},
          representatives=['g_115.0_35.5','g_116.0_37.5','g_114.0_37.5','g_118.0_39.5'],
          representative_selection='Before new results: southern late rise, northern 2020 candidate, unassigned northern cell, early northern candidate',
          stability_tolerances_months=[3,6,12])


def monthly_index(date):
    d=pd.Timestamp(date);return d.year*12+d.month-1


def report_candidates(dates,signal,qualification):
    dates=pd.DatetimeIndex(dates)
    result=rb.beast(signal,start=dates[0].year+(dates[0].month-1)/12,deltat=1/12,period=1.,season='none',
                    tseg_minlength=18,quiet=1,mcmc_seed=42,print_param=False,print_progress=False,print_warning=False)
    series=pd.DataFrame({'date':dates,'deseasoned_gwa':qualification})
    candidates=[]
    cps=np.asarray(result.trend.cp).reshape(-1);probs=np.asarray(result.trend.cpPr).reshape(-1)
    for cp,prob in zip(cps,probs):
        if not (np.isfinite(cp) and np.isfinite(prob)):continue
        date=core.maybe_date_from_cp(series,float(cp));i=core.breakpoint_index(series,date)
        pre,post=core.fit_trends_for_breakpoint(series,i)
        qualifies=bool(prob>=.5 and core.is_recovery_relevant_break(date,pre,post))
        candidates.append(dict(date=str(date.date()),probability=float(prob),pre_slope=pre,post_slope=post,qualifies=qualifies))
    pool=[r for r in candidates if r['qualifies']]
    choices=[]
    for rule in ['highest_probability','earliest']:
        chosen=(max(pool,key=lambda r:(r['probability'],r['date'])) if rule=='highest_probability' else min(pool,key=lambda r:r['date'])) if pool else None
        choices.append(dict(rule=rule,qualifies=bool(chosen),selected_date=None if chosen is None else chosen['date'],
                            probability=None if chosen is None else chosen['probability'],
                            pre_slope=None if chosen is None else chosen['pre_slope'],post_slope=None if chosen is None else chosen['post_slope'],
                            qualifying_candidates=len(pool),all_candidates=len(candidates)))
    return choices,candidates,np.asarray(result.trend.Y).reshape(-1)


def analyze(task):
    dates=pd.to_datetime(task['dates']);y=np.asarray(task['y'],float);observed=np.isfinite(y)
    linear=pd.Series(y).interpolate(limit_direction='both').to_numpy()
    season=STL(linear,period=12,robust=True).fit().seasonal
    qualification=np.where(observed,y-season,np.nan)
    signals={'A':qualification};info={};modes_info=None
    if any(r!='A' for r in task['routes']):
        filled,si=ssa_fill(y,window=36,rank=6,tolerance=1e-6,max_iterations=200);info['ssa']=si
        if any(r in task['routes'] for r in ['B','C']):
            sc=STL(filled,period=12,robust=True).fit().seasonal
            signals['B']=np.where(observed,filled-sc,np.nan);signals['C']=filled-sc
        if 'D' in task['routes']:
            signals['D'],modes_info=reconstruct(filled,trials=task.get('trials',50),noise_scale=task.get('noise_scale',.2),
                                               seed=task.get('ensemble_seed',42),min_period=task.get('min_period',24))
            info['reconstruction']=modes_info
    selections=[];candidates=[];series=[]
    metadata={k:task[k] for k in ['grid_id','region_tag','scope','endpoint','experiment']}
    for route in task['routes']:
        choices,cp,fitted=report_candidates(dates,signals[route],qualification)
        selections.extend([{**metadata,'route':route,**r,'real_observations':int(observed.sum())} for r in choices])
        candidates.extend([{**metadata,'route':route,**r} for r in cp])
        if task.get('save_series',False):
            series.extend([{**metadata,'route':route,'date':str(d.date()),'observed':float(y[i]),'input_signal':float(signals[route][i]),
                            'qualification_signal':float(qualification[i]),'beast_trend':float(fitted[i])} for i,d in enumerate(dates)])
    return dict(selections=selections,candidates=candidates,series=series,info={**metadata,**info})


def parallel_tasks(tasks,workers,label):
    results=[];start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(analyze,t):t for t in tasks}
        for i,f in enumerate(as_completed(futures),1):
            result=f.result();results.append(result)
            if i%10==0 or i==len(tasks):print(label,i,'/',len(tasks),'seconds',round(time.time()-start),flush=True)
    return results


def write_results(results,out,prefix):
    for key in ['selections','candidates','series']:
        rows=[r for result in results for r in result[key]]
        if rows:pd.DataFrame(rows).sort_values(['experiment','grid_id','endpoint','route']).to_csv(out/f'{prefix}_{key}.csv',index=False,lineterminator='\n')
    (out/f'{prefix}_diagnostics.json').write_text(json.dumps([r['info'] for r in results],indent=2,allow_nan=False)+'\n',encoding='utf-8')


def actual_tasks(panel):
    groups=[(gid,g,g.region_tag.iloc[0],'grid') for gid,g in panel.groupby('grid_id')]
    for region,g in [('all',panel),('north',panel[panel.region_tag.eq('north')]),('south',panel[panel.region_tag.eq('south')])]:
        groups.append(('region_'+region,g.groupby('date',as_index=False).gwa_cm.mean(),region,'regional'))
    tasks=[]
    for gid,g,region,scope in groups:
        for endpoint in [*SPEC['endpoints'],SPEC['common_endpoint']]:
            sub=g.loc[g.date<=endpoint]
            tasks.append(dict(grid_id=gid,region_tag=region,scope=scope,endpoint=endpoint,experiment='actual',
                              dates=sub.date.astype(str).tolist(),y=sub.gwa_cm.tolist(),routes=['A','B','C','D'] if endpoint==SPEC['common_endpoint'] else ['A','D'],
                              save_series=endpoint in [SPEC['common_endpoint'],'2024-09-01'] and (scope=='regional' or gid in SPEC['representatives'])))
    return tasks


def synthetic_tasks(panel,reps):
    dates=pd.date_range('2003-01-01','2024-09-01',freq='MS');t=np.arange(len(dates))/12
    mask=panel.groupby('date').gwa_cm.count().reindex(dates).gt(0).to_numpy()
    tasks=[];truth=[]
    for rep in range(reps):
        rng=np.random.default_rng(10000+rep);innov=rng.normal(0,1.5,len(t));noise=np.empty(len(t));noise[0]=innov[0]/np.sqrt(1-.6**2)
        for i in range(1,len(t)):noise[i]=.6*noise[i-1]+innov[i]
        for scenario in ['no_recovery','temporary_gain','sustained_reversal','reversal_then_acceleration']:
            trend=-1.5*t;onset=None;acceleration=None
            if scenario=='temporary_gain':trend=trend+12*np.maximum(1-np.abs(t-18.5)/1.5,0)
            elif scenario=='sustained_reversal':
                onset='2019-01-01';trend=trend+4.5*np.maximum(t-16,0)
            elif scenario=='reversal_then_acceleration':
                onset='2018-01-01';acceleration='2021-01-01';trend=trend+2.1*np.maximum(t-15,0)+3.4*np.maximum(t-18,0)
            y=trend+3*np.sin(2*np.pi*t)+noise;y[~mask]=np.nan
            gid=f'{scenario}_{rep:03d}'
            truth.append(dict(grid_id=gid,scenario=scenario,replicate=rep,true_onset=onset,true_acceleration=acceleration,sustained=onset is not None))
            tasks.append(dict(grid_id=gid,region_tag='synthetic',scope='synthetic',endpoint='2024-09-01',experiment=scenario,
                              dates=dates.astype(str).tolist(),y=y.tolist(),routes=['A','D'],save_series=rep==0))
    return tasks,pd.DataFrame(truth)


def wilson(success,n):
    z=1.959963984540054;p=success/n;denom=1+z*z/n
    mid=(p+z*z/(2*n))/denom;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return mid-half,mid+half


def summarize(out):
    actual=pd.read_csv(out/'actual_selections.csv')
    rows=[]
    for (endpoint,route,rule),g in actual.loc[actual.scope.eq('grid')].groupby(['endpoint','route','rule']):
        v=g.loc[g.qualifies].copy();v['month']=v.selected_date.map(monthly_index)
        delta=v.loc[v.region_tag.eq('south'),'month'].mean()-v.loc[v.region_tag.eq('north'),'month'].mean()
        median=v.month.median();median=None if pd.isna(median) else f'{int(round(median))//12:04d}-{int(round(median))%12+1:02d}'
        rows.append(dict(endpoint=endpoint,route=route,rule=rule,reportable=len(v),north=int(v.region_tag.eq('north').sum()),south=int(v.region_tag.eq('south').sum()),
                         median_date=median,timing_south_minus_north=delta))
    pd.DataFrame(rows).to_csv(out/'actual_summary.csv',index=False,lineterminator='\n')
    common=actual.loc[actual.endpoint.eq(SPEC['common_endpoint'])&actual.scope.eq('grid')]
    paired=[]
    for rule in ['highest_probability','earliest']:
        a=common.loc[common.route.eq('A')&common.rule.eq(rule)&common.qualifies]
        for route in ['B','C','D']:
            d=common.loc[common.route.eq(route)&common.rule.eq(rule)&common.qualifies]
            p=a.merge(d,on='grid_id',suffixes=('_a','_d'))
            for r in p.itertuples():paired.append(dict(grid_id=r.grid_id,region_tag=r.region_tag_a,route=route,rule=rule,A_date=r.selected_date_a,other_date=r.selected_date_d,
                                                       difference_months=monthly_index(r.selected_date_d)-monthly_index(r.selected_date_a)))
    pd.DataFrame(paired).to_csv(out/'paired_route_dates.csv',index=False,lineterminator='\n')
    transitions=[]
    for (gid,route,rule),g in actual.loc[actual.scope.eq('grid')&actual.endpoint.isin(SPEC['endpoints'])].groupby(['grid_id','route','rule']):
        g=g.sort_values('endpoint')
        for i in range(1,len(g)):
            before,after=g.iloc[i-1],g.iloc[i]
            delta=np.nan if not (before.qualifies and after.qualifies) else abs(monthly_index(after.selected_date)-monthly_index(before.selected_date))
            transitions.append(dict(grid_id=gid,route=route,rule=rule,from_endpoint=before.endpoint,to_endpoint=after.endpoint,previously_reportable=bool(before.qualifies),
                                    currently_reportable=bool(after.qualifies),withdrawn=bool(before.qualifies and not after.qualifies),revision_months=delta))
    pd.DataFrame(transitions).to_csv(out/'endpoint_transitions.csv',index=False,lineterminator='\n')
    sim=pd.read_csv(out/'simulation_selections.csv').merge(pd.read_csv(out/'simulation_truth.csv'),on='grid_id');rows=[]
    for (scenario,route,rule),g in sim.groupby(['scenario','route','rule']):
        v=g.loc[g.qualifies].copy();lo,hi=wilson(len(v),len(g));errors=[];accels=[]
        if g.sustained.iloc[0]:
            errors=[monthly_index(a)-monthly_index(b) for a,b in zip(v.selected_date,v.true_onset)]
            if scenario=='reversal_then_acceleration':accels=[abs(monthly_index(a)-monthly_index(b)) for a,b in zip(v.selected_date,v.true_acceleration)]
        rows.append(dict(scenario=scenario,route=route,rule=rule,n=len(g),qualified=len(v),qualification_rate=len(v)/len(g),wilson_low=lo,wilson_high=hi,
                         median_onset_error=np.median(errors) if errors else np.nan,mean_absolute_onset_error=np.mean(np.abs(errors)) if errors else np.nan,
                         selected_within12_onset=sum(abs(x)<=12 for x in errors),selected_within12_acceleration=sum(x<=12 for x in accels)))
    pd.DataFrame(rows).to_csv(out/'simulation_summary.csv',index=False,lineterminator='\n')


def component_intervals(out):
    e=ROOT/'outputs/review_revision/2026-10-15-r1'
    members=pd.read_csv(e/'well_cohort_membership.csv');common=set(members.loc[members.aquifer_group.eq('confined'),'grid_id'])&set(members.loc[members.aquifer_group.eq('unconfined'),'grid_id'])
    panel=pd.read_csv(ROOT/'data/raw/grid_monthly_panel.csv',parse_dates=['date'])
    series=pd.read_csv(e/'aquifer_validation_series.csv',parse_dates=['date']);rows=[]
    for aquifer in ['unconfined','confined']:
        for region in ['north','south']:
            p=panel.loc[panel.grid_id.isin(common)&panel.region_tag.eq(region)].groupby('date')[['twsa_cm','smsa_cm','gwa_cm']].mean()
            w=series.loc[series.aquifer.eq(aquifer)&series.support.eq('common')&series.region_tag.eq(region)].set_index('date').well_anomaly
            s=p.join(w).loc['2019-01-01':'2024-08-01']
            for treatment in ['raw','residual','difference']:
                values=s.to_numpy()
                if treatment=='residual':values=np.column_stack([calendar_detrend(values[:,i]) for i in range(4)])
                if treatment=='difference':values=np.diff(values,axis=0)
                rng=np.random.default_rng(42);boots=[]
                for _ in range(2000):
                    b=values[moving_block_indices(len(values),12,rng)]
                    boots.append([pearson_or_nan(b[:,i],b[:,3]) for i in range(3)])
                boots=np.asarray(boots)
                est=[pearson_or_nan(values[:,i],values[:,3]) for i in range(3)]
                for i,name in enumerate(['TWSA','SMSA','GWA']):
                    lo,hi=np.quantile(boots[:,i],[.025,.975]);rows.append(dict(aquifer=aquifer,region=region,treatment=treatment,metric=name,estimate=est[i],low=lo,high=hi))
                lo,hi=np.quantile(boots[:,2]-boots[:,0],[.025,.975]);rows.append(dict(aquifer=aquifer,region=region,treatment=treatment,metric='GWA_minus_TWSA_r',estimate=est[2]-est[0],low=lo,high=hi))
    pd.DataFrame(rows).to_csv(out/'component_correlation_intervals.csv',index=False,lineterminator='\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);p.add_argument('--emd-runtime',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--replicates',type=int,default=100);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    runtime=Path(a.emd_runtime).resolve();os.environ['NCP_EMD_RUNTIME']=str(runtime)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    (out/'specification.json').write_text(json.dumps({**SPEC,'run_replicates':a.replicates,'pilot':a.pilot},indent=2)+'\n',encoding='utf-8')
    panel=pd.read_csv(ROOT/'data/raw/grid_monthly_panel.csv',parse_dates=['date'])
    tasks=actual_tasks(panel)
    if a.pilot:tasks=[t for t in tasks if t['endpoint']==SPEC['common_endpoint'] and (t['scope']=='regional' or t['grid_id'] in SPEC['representatives'])]
    result=parallel_tasks(tasks,a.workers,'Actual');write_results(result,out,'actual')
    simulation,truth=synthetic_tasks(panel,a.replicates);truth.to_csv(out/'simulation_truth.csv',index=False,lineterminator='\n')
    result=parallel_tasks(simulation,a.workers,'Simulation');write_results(result,out,'simulation')
    summarize(out);component_intervals(out)
    sources=[Path(__file__),ROOT/'analysis/ncp_signal_reconstruction.py',ROOT/'analysis/ncp_recovery_analysis.py',ROOT/'analysis/ncp_calendar_revision.py',ROOT/'analysis/ncp_revision_analyses.py']
    inputs=[ROOT/'data/raw/grid_monthly_panel.csv',ROOT/'outputs/review_revision/2026-10-15-r1/aquifer_validation_series.csv',ROOT/'outputs/review_revision/2026-10-15-r1/well_cohort_membership.csv']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
                  inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},python=sys.version,emd_runtime=str(runtime),emd_signal='1.10.0',
                  wheel_sha256=sha(ROOT/'outputs/runs/2026-10-16-story-sources-r1/emd_signal-1.10.0-py3-none-any.whl'),
                  settings=SPEC,actual_tasks=len(tasks),simulation_tasks=len(simulation),strict_zhang_replication=False,
                  limitations=['Zhang S1 unavailable: working parameters and reconstruction rule are ours','EMD kernel differs from MATLAB reference','Single JPL product; no cross-product causal isolation','SSA estimates are not counted as independent observed months'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print('Complete',out,flush=True)


if __name__=='__main__':main()
