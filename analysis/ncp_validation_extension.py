"""Targeted validation of closure, endpoints, temporal uncertainty and well support.

Use the documented Anaconda runtime. Each stage requires a new output directory.
Existing scientific inputs, frozen scripts and earlier evidence are read-only.
"""
from __future__ import annotations
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[name]='1'
import argparse, hashlib, json, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
import Rbeast as rb
import ncp_recovery_analysis as core
from ncp_signal_reconstruction import ssa_fill, reconstruct
from ncp_calendar_revision import calendar_detrend
from ncp_revision_analyses import pearson_or_nan, moving_block_indices

ROOT=Path(__file__).resolve().parents[1]
STORY=ROOT/'outputs/story_revision/2026-10-17-r1'
REVIEW=ROOT/'outputs/review_revision/2026-10-15-r1'
SPEC=dict(seed=20261017,endpoint_dates=['2023-11-01','2023-12-01','2024-08-01','2024-09-01'],
          closure_tolerance_cm=1e-6,temporal_blocks=[6,12,18],bootstrap_replicates=5000,
          calibration='Observed residuals after 36-month hinge trends and annual/semiannual harmonics, full calendar',
          calibration_quantiles=[.25,.5,.75],simulation_replicates=50,
          simulation_extra_settings=['initial03','initial12','acceleration20','acceleration60','phi_low','phi_high'],
          local_constraint_months=36,local_constraint_minimum_observations=18,
          well_aggregation_months=[1,3,6,12],well_coverage_fraction=.8,well_lags=list(range(13)),
          well_time_bootstrap_replicates=2000,ssa_iterations=1000,
          reconstruction_trials=50,reconstruction_noise=.2,reconstruction_seed=42)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(df,path):df.to_csv(path,index=False,lineterminator='\n')
def month(x):
    d=pd.Timestamp(x);return d.year*12+d.month-1
def panel():
    p=pd.read_csv(ROOT/'data/raw/grid_monthly_panel.csv',parse_dates=['date'])
    p['component_gwa']=p.twsa_cm-p.smsa_cm-p.swea_cm-p.cwa_cm
    p['closure_delta']=p.gwa_cm-p.component_gwa
    return p
def meta(p):return p[['grid_id','grid_lon','grid_lat','region_tag','coverage_ratio']].drop_duplicates('grid_id')
def bad_ids(p):return set(p.groupby('grid_id').closure_delta.apply(lambda s:s.abs().max()).loc[lambda s:s>SPEC['closure_tolerance_cm']].index)
def paired_bootstrap_correlations(values,block,reps):
    rng=np.random.default_rng(SPEC['seed'])
    ix=np.array([moving_block_indices(len(values),block,rng) for _ in range(reps)])
    a,b=values[ix,0],values[ix,1];valid=np.isfinite(a)&np.isfinite(b)
    a=np.where(valid,a,np.nan);b=np.where(valid,b,np.nan)
    a=a-np.nanmean(a,axis=1,keepdims=True);b=b-np.nanmean(b,axis=1,keepdims=True)
    denom=np.sqrt(np.nansum(a*a,axis=1)*np.nansum(b*b,axis=1))
    return np.divide(np.nansum(a*b,axis=1),denom,out=np.full(reps,np.nan),where=denom>0)
def design(dates,hinges=False):
    t=np.arange(len(dates),dtype=float)/12
    cols=[np.ones(len(t)),t,np.sin(2*np.pi*t),np.cos(2*np.pi*t),np.sin(4*np.pi*t),np.cos(4*np.pi*t)]
    if hinges:cols += [np.maximum(t-k/12,0) for k in range(36,len(t)-18,36)]
    return np.column_stack(cols)
def fitted(y,X):
    ok=np.isfinite(y);coef=np.linalg.lstsq(X[ok],y[ok],rcond=None)[0]
    r=np.where(ok,y-X@coef,np.nan)
    return coef,X@coef,r

def closure_audit(p,out):
    records=[];series=[]
    for gid,g in p.groupby('grid_id'):
        g=g.sort_values('date');d=g.closure_delta.to_numpy();peak=g.loc[g.closure_delta.abs().idxmax()]
        for start in ['2003-01-01','2019-01-01','2022-01-01']:
            s=g.loc[g.date.ge(start)&g.date.le('2024-08-01')];X=design(s.date);c,_,_=fitted(s.closure_delta.to_numpy(),X)
            records.append(dict(grid_id=gid,region=g.region_tag.iloc[0],start=start,n=s.closure_delta.notna().sum(),mean_delta=s.closure_delta.mean(),sd_delta=s.closure_delta.std(),max_abs=s.closure_delta.abs().max(),delta_trend_cm_yr=c[1],peak_date=str(peak.date.date())))
        series.append(g[['grid_id','date','gwa_cm','component_gwa','closure_delta']])
    save(pd.DataFrame(records),out/'closure_summary.csv');save(pd.concat(series),out/'closure_series.csv')

def noise_calibration(p,out):
    rows=[]
    for gid,g in p.groupby('grid_id'):
        g=g.sort_values('date');y=g.gwa_cm.to_numpy();X=design(g.date,hinges=True);c,_,r=fitted(y,X)
        a,b=r[:-1],r[1:];ok=np.isfinite(a)&np.isfinite(b)
        phi=float(np.dot(a[ok],b[ok])/np.dot(a[ok],a[ok]));innovation=b[ok]-phi*a[ok]
        rows.append(dict(grid_id=gid,region=g.region_tag.iloc[0],n_observed=np.isfinite(r).sum(),n_consecutive=ok.sum(),phi=phi,innovation_sd=float(np.std(innovation,ddof=1)),residual_sd=float(np.nanstd(r,ddof=1)),annual_amplitude=float(np.hypot(c[2],c[3]))))
    cal=pd.DataFrame(rows);save(cal,out/'noise_calibration.csv')
    q=cal[['phi','innovation_sd','annual_amplitude']].quantile(SPEC['calibration_quantiles']).reset_index(names='quantile')
    save(q,out/'noise_quantiles.csv');return q

def temporal_uncertainty(p,out):
    rows=[];boot_rows=[];m=meta(p).set_index('grid_id');excluded=bad_ids(p)
    for sample in ['all27','closure22','component_means']:
        sub=p.loc[~p.grid_id.isin(excluded)] if sample=='closure22' else p
        value='component_gwa' if sample=='component_means' else 'gwa_cm'
        for start in ['2019-01-01','2022-01-01']:
            s=sub.loc[sub.date.between(start,'2024-08-01')];wide=s.pivot(index='date',columns='grid_id',values=value).sort_index()
            Y=wide.to_numpy();assert np.isfinite(Y).all();X=design(wide.index);pinv=np.linalg.pinv(X)
            coef=pinv@Y;fit=X@coef;residual=Y-fit;residual-=residual.mean(axis=0)
            mm=m.loc[wide.columns];north=mm.region_tag.eq('north').to_numpy();south=~north
            weights=south/south.sum()-north/north.sum();estimate=float(coef[1]@weights)
            for block in SPEC['temporal_blocks']:
                rng=np.random.default_rng(SPEC['seed']+block);slopes=[]
                for _ in range(SPEC['bootstrap_replicates']):
                    ix=moving_block_indices(len(Y),block,rng)
                    slopes.append(coef[1]+pinv[1]@residual[ix])
                slopes=np.asarray(slopes);d=slopes@weights
                lo,hi=np.quantile(d,[.025,.975]);rows.append(dict(sample=sample,start=start,method='synchronous_time',time_block=block,spatial_block=0,estimate=estimate,low=lo,high=hi,n_months=len(Y),n_cells=Y.shape[1],replicates=len(d)))
                for size in [2.,3.]:
                    blocks=list(zip(np.floor((mm.grid_lon.to_numpy()-112)/size),np.floor((mm.grid_lat.to_numpy()-34.5)/size)))
                    ids={b:np.flatnonzero(np.array([x==b for x in blocks])) for b in sorted(set(blocks))}
                    values=list(ids.values());draws=rng.integers(len(values),size=(len(slopes),len(values)));contrasts=[]
                    for j,draw in enumerate(draws):
                        ix=np.concatenate([values[k] for k in draw]);n=north[ix];z=slopes[j,ix]
                        if n.any() and (~n).any():contrasts.append(z[~n].mean()-z[n].mean())
                    lo,hi=np.quantile(contrasts,[.025,.975]);rows.append(dict(sample=sample,start=start,method='synchronous_time_and_space',time_block=block,spatial_block=size,estimate=estimate,low=lo,high=hi,n_months=len(Y),n_cells=Y.shape[1],replicates=len(contrasts)))
                boot_rows.extend(dict(sample=sample,start=start,time_block=block,replicate=i,contrast=x) for i,x in enumerate(d))
    save(pd.DataFrame(rows),out/'temporal_trend_intervals.csv');save(pd.DataFrame(boot_rows),out/'temporal_trend_draws.csv')

def well_series(p):
    members=pd.read_csv(REVIEW/'well_cohort_membership.csv');common=set(members.loc[members.aquifer_group.eq('confined'),'grid_id'])&set(members.loc[members.aquifer_group.eq('unconfined'),'grid_id'])
    raw=pd.read_csv(ROOT/'data/external/ncp_long2025_well_2005_2024.csv',parse_dates=['date'])
    raw=raw.loc[raw.well_id.isin(members.well_id)&raw.date.between('2018-01-01','2024-09-01')].copy()
    raw['well_anomaly']=raw.gw_level-raw.groupby('well_id').gw_level.transform('mean')
    cells=raw.groupby(['aquifer_group','grid_id','date'],as_index=False).well_anomaly.mean()
    counts=members.groupby(['aquifer_group','grid_id']).size();mm=meta(p).set_index('grid_id')
    dates=pd.date_range('2019-01-01','2024-08-01',freq='MS');results=[]
    for support in ['common18','closure_common']:
        allowed=common-bad_ids(p) if support=='closure_common' else common
        for aq in ['unconfined','confined']:
            for region in ['north','south']:
                ids=sorted(allowed&set(mm.index[mm.region_tag.eq(region)]))
                w=cells.loc[cells.aquifer_group.eq(aq)&cells.grid_id.isin(ids)].pivot(index='date',columns='grid_id',values='well_anomaly').reindex(index=dates,columns=ids)
                for weighting in ['equal_cell','area','cohort_count']:
                    weights=np.ones(len(ids)) if weighting=='equal_cell' else (mm.loc[ids,'coverage_ratio'].to_numpy()*np.cos(np.deg2rad(mm.loc[ids,'grid_lat'].to_numpy())) if weighting=='area' else np.array([counts[aq,gid] for gid in ids]))
                    # Fixed cell weights, renormalized only over present cell-months; record support.
                    def aggregate(wide):
                        a=wide.to_numpy();valid=np.isfinite(a);den=valid@weights
                        return np.divide(np.nansum(a*weights,axis=1),den,out=np.full(len(a),np.nan),where=den>0)
                    data={'date':dates,'well':aggregate(w),'well_cells':w.notna().sum(axis=1).to_numpy()}
                    for col in ['gwa_cm','component_gwa','twsa_cm','smsa_cm']:
                        v=p.loc[p.grid_id.isin(ids)].pivot(index='date',columns='grid_id',values=col).reindex(index=dates,columns=ids);data[col]=aggregate(v)
                    data=pd.DataFrame(data).assign(support=support,aquifer=aq,region=region,weighting=weighting,n_cells=len(ids))
                    results.append(data)
    return pd.concat(results,ignore_index=True)

def well_checks(p,out):
    series=well_series(p);save(series,out/'well_weighted_series.csv');records=[];lags=[];trends=[]
    for keys,g in series.groupby(['support','aquifer','region','weighting']):
        md=dict(zip(['support','aquifer','region','weighting'],keys));g=g.sort_values('date').set_index('date')
        for component in ['gwa_cm','component_gwa','twsa_cm','smsa_cm']:
            z=g[[component,'well']];X=design(z.index)
            for col in z:
                c,_,_=fitted(z[col].to_numpy(),X);trends.append({**md,'variable':col,'component_pair':component,'slope_per_year':c[1]})
            for size in SPEC['well_aggregation_months']:
                # Non-overlapping calendar bins; >=ceil(.8*size) observations separately in both variables.
                if size==1:agg=z.copy()
                else:
                    grouper=((np.arange(len(z)))//size);agg=z.groupby(grouper).mean();counts=z.groupby(grouper).count();length=z.groupby(grouper).size()
                    agg=agg.where(counts>=int(np.ceil(.8*size)));agg.loc[length<size,:]=np.nan
                for treatment in ['raw','difference']:
                    values=agg.to_numpy() if treatment=='raw' else np.diff(agg.to_numpy(),axis=0)
                    n=np.isfinite(values).all(axis=1).sum();r=pearson_or_nan(values[:,0],values[:,1]);boots=[]
                    if n>=4:
                        block=max(1,int(np.ceil(12/size)))
                        boots=paired_bootstrap_correlations(values,block,SPEC['well_time_bootstrap_replicates'])
                    lo,hi=np.nanquantile(boots,[.025,.975]) if len(boots) else [np.nan,np.nan]
                    records.append({**md,'component':component,'aggregation_months':size,'treatment':treatment,'n_pairs':n,'r':r,'low':lo,'high':hi})
            if keys[3]=='equal_cell':
                a=calendar_detrend(z[component].to_numpy());b=calendar_detrend(z.well.to_numpy())
                for lag in SPEC['well_lags']:
                    aa=a[:len(a)-lag] if lag else a;bb=b[lag:]
                    lags.append({**md,'component':component,'lag_months':lag,'r':pearson_or_nan(aa,bb),'n_pairs':np.isfinite(aa+bb).sum()})
    save(pd.DataFrame(records),out/'well_multiscale_correlations.csv');save(pd.DataFrame(lags),out/'well_lags_0_12.csv');save(pd.DataFrame(trends),out/'well_and_proxy_trends.csv')

def fit_worker(task):
    dates=pd.to_datetime(task['dates']);y=np.asarray(task['y'],float);observed=np.isfinite(y)
    linear=pd.Series(y).interpolate(limit_direction='both').to_numpy();season=STL(linear,period=12,robust=True).fit().seasonal;qual=np.where(observed,y-season,np.nan)
    signals={'A':qual};diagnostics={}
    if 'D' in task['routes']:
        complete,info=ssa_fill(y,window=36,rank=6,tolerance=1e-6,max_iterations=1000);diagnostics['ssa']=info
        signals['D'],info=reconstruct(complete);diagnostics['reconstruction']=info
    md={k:task[k] for k in ['grid_id','endpoint','experiment','region_tag','scope']};candidates=[];selections=[];posterior=[];series=[]
    for route in task['routes']:
        r=rb.beast(signals[route],start=dates[0].year+(dates[0].month-1)/12,deltat=1/12,period=1.,season='none',tseg_minlength=18,quiet=1,mcmc_seed=task.get('beast_seed',42),print_param=False,print_progress=False,print_warning=False)
        cp=np.asarray(r.trend.cp).ravel();prob=np.asarray(r.trend.cpPr).ravel();ci=np.asarray(r.trend.cpCI)
        pool=[]
        for k,(value,pp) in enumerate(zip(cp,prob)):
            if not (np.isfinite(value) and np.isfinite(pp)):continue
            date=core.maybe_date_from_cp(pd.DataFrame({'date':dates}),float(value));i=core.breakpoint_index(pd.DataFrame({'date':dates}),date)
            pre,post=core.fit_trends_for_breakpoint(pd.DataFrame({'date':dates,'deseasoned_gwa':qual}),i)
            qualifies=bool(pp>=.5 and core.is_recovery_relevant_break(date,pre,post))
            slopes=[]
            for ix in [np.arange(max(0,i-36),i),np.arange(i,min(len(y),i+36))]:
                valid=np.isfinite(qual[ix]);slopes.append(stats.linregress(ix[valid]/12,qual[ix][valid]).slope if valid.sum()>=18 else np.nan)
            local=bool(pp>=.5 and date>=pd.Timestamp('2015-01-01') and slopes[0]<0 and slopes[1]>0)
            item=dict(date=str(date.date()),probability=float(pp),pre_slope=pre,post_slope=post,qualifies=qualifies,local_reversal=local,local_pre_slope=slopes[0],local_post_slope=slopes[1],ci_low_decimal=float(ci[k,0]),ci_high_decimal=float(ci[k,1]))
            pool.append(item);candidates.append({**md,'route':route,**item})
        for rule in ['highest_probability','earliest','local_sign_reversal']:
            available=[x for x in pool if x['local_reversal' if rule=='local_sign_reversal' else 'qualifies']]
            chosen=(min(available,key=lambda x:x['date']) if rule in ['earliest','local_sign_reversal'] else max(available,key=lambda x:(x['probability'],x['date']))) if available else None
            selections.append({**md,'route':route,'rule':rule,'qualifies':chosen is not None,'selected_date':chosen['date'] if chosen else None,'real_observations':observed.sum(),'beast_seed':task.get('beast_seed',42)})
        if task.get('save_series'):
            for i,d in enumerate(dates):
                posterior.append({**md,'route':route,'date':str(d.date()),'cp_occurrence_probability':float(np.asarray(r.trend.cpOccPr).ravel()[i]),'slope_positive_probability':float(np.asarray(r.trend.slpSgnPosPr).ravel()[i])})
                series.append({**md,'route':route,'date':str(d.date()),'observed':y[i],'input_signal':signals[route][i],'qualification_signal':qual[i],'fitted_trend':float(np.asarray(r.trend.Y).ravel()[i])})
    return dict(selections=selections,candidates=candidates,posterior=posterior,series=series,diagnostics={**md,**diagnostics})

def run_tasks(tasks,out,workers,prefix):
    start=time.time();results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fit_worker,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            results.append(f.result())
            if i%10==0 or i==len(tasks):print(prefix,i,'/',len(tasks),'seconds',round(time.time()-start),flush=True)
    for key in ['selections','candidates','posterior','series']:
        data=[row for r in results for row in r[key]]
        if data:save(pd.DataFrame(data).sort_values(['experiment','grid_id','endpoint','route']),out/f'{prefix}_{key}.csv')
    (out/f'{prefix}_diagnostics.json').write_text(json.dumps([r['diagnostics'] for r in results],indent=2,allow_nan=False)+'\n',encoding='utf-8')

def actual_tasks(p):
    tasks=[];excluded=bad_ids(p)
    for sample in ['original','component_means']:
        groups=[(gid,g,g.region_tag.iloc[0],'grid') for gid,g in p.groupby('grid_id') if sample=='original' or gid in excluded]
        for region in ['all','north','south']:
            g=p if region=='all' else p.loc[p.region_tag.eq(region)]
            groups.append(('region_'+region,g.groupby('date',as_index=False)[['gwa_cm','component_gwa']].mean(),region,'regional'))
        for gid,g,region,scope in groups:
            ends=['2024-08-01'] if sample=='original' else SPEC['endpoint_dates']
            for end in ends:
                sub=g.loc[g.date.le(end)];tasks.append(dict(grid_id=gid,region_tag=region,scope=scope,endpoint=end,experiment=sample,dates=sub.date.astype(str).tolist(),y=sub['gwa_cm' if sample=='original' else 'component_gwa'].tolist(),routes=['A','D'],save_series=gid in ['g_115.0_35.5','g_116.0_37.5'] or scope=='regional'))
    # End-of-record posterior and seed repeatability on preselected examples.
    for gid in ['g_115.0_35.5','g_116.0_37.5','g_118.0_37.5']:
        for seed in [42,99,2026]:
            g=p.loc[p.grid_id.eq(gid)]
            tasks.append(dict(grid_id=gid,region_tag=g.region_tag.iloc[0],scope='grid',endpoint='2024-09-01',experiment='seed_'+str(seed),dates=g.date.astype(str).tolist(),y=g.gwa_cm.tolist(),routes=['A','D'],beast_seed=seed,save_series=seed==42))
    return tasks

def simulation_tasks(p,local,out):
    q=pd.read_csv(local/'noise_quantiles.csv').set_index('quantile');configs=[]
    for label,quantile in [('low',.25),('median',.5),('high',.75)]:
        r=q.loc[quantile];configs.append(dict(setting=label,phi=float(np.clip(r.phi,-.95,.95)),sigma=float(r.innovation_sd),amplitude=float(r.annual_amplitude),initial=.6,acceleration=4.,scenarios=['no_recovery','temporary_gain','sustained_reversal','reversal_then_acceleration']))
    median=configs[1]
    for label,field,value in [('initial03','initial',.3),('initial12','initial',1.2),('acceleration20','acceleration',2.),('acceleration60','acceleration',6.),('phi_low','phi',float(np.clip(q.loc[.25,'phi'],-.95,.95))),('phi_high','phi',float(np.clip(q.loc[.75,'phi'],-.95,.95)))]:
        configs.append({**median,'setting':label,field:value,'scenarios':['reversal_then_acceleration']})
    (out/'simulation_settings.json').write_text(json.dumps(configs,indent=2)+'\n',encoding='utf-8')
    dates=pd.date_range('2003-01-01','2024-09-01',freq='MS');t=np.arange(len(dates))/12;observed=p.groupby('date').gwa_cm.count().reindex(dates).gt(0).to_numpy();tasks=[];truth=[]
    for cfg in configs:
        for rep in range(SPEC['simulation_replicates']):
            rng=np.random.default_rng(10000+rep);z=rng.normal(size=len(t));noise=np.empty(len(t));noise[0]=cfg['sigma']*z[0]/np.sqrt(1-cfg['phi']**2)
            for k in range(1,len(t)):noise[k]=cfg['phi']*noise[k-1]+cfg['sigma']*z[k]
            for scenario in cfg['scenarios']:
                trend=-1.5*t;onset=None;accel=None
                if scenario=='temporary_gain':trend+=12*np.maximum(1-np.abs(t-18.5)/1.5,0)
                elif scenario=='sustained_reversal':trend+=4.5*np.maximum(t-16,0);onset='2019-01-01'
                elif scenario=='reversal_then_acceleration':trend+=(1.5+cfg['initial'])*np.maximum(t-15,0)+(cfg['acceleration']-cfg['initial'])*np.maximum(t-18,0);onset='2018-01-01';accel='2021-01-01'
                y=trend+cfg['amplitude']*np.sin(2*np.pi*t)+noise;y[~observed]=np.nan;gid=cfg['setting']+'_'+scenario+f'_{rep:03d}'
                truth.append(dict(grid_id=gid,setting=cfg['setting'],scenario=scenario,replicate=rep,true_onset=onset,true_acceleration=accel))
                tasks.append(dict(grid_id=gid,region_tag='synthetic',scope='synthetic',endpoint='2024-09-01',experiment=cfg['setting']+'_'+scenario,dates=dates.astype(str).tolist(),y=y.tolist(),routes=['A','D'],save_series=False))
    save(pd.DataFrame(truth),out/'simulation_truth.csv');return tasks

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['local','fits','simulation'],required=True);parser.add_argument('--output-dir',required=True);parser.add_argument('--local-evidence');parser.add_argument('--emd-runtime');parser.add_argument('--workers',type=int,default=4);a=parser.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);(out/'specification.json').write_text(json.dumps(SPEC,indent=2)+'\n',encoding='utf-8')
    if a.emd_runtime:os.environ['NCP_EMD_RUNTIME']=str(Path(a.emd_runtime).resolve())
    p=panel()
    if a.stage=='local':
        closure_audit(p,out);noise_calibration(p,out);temporal_uncertainty(p,out);well_checks(p,out)
    elif a.stage=='fits':run_tasks(actual_tasks(p),out,a.workers,'actual')
    else:
        assert a.local_evidence;run_tasks(simulation_tasks(p,Path(a.local_evidence),out),out,a.workers,'simulation')
    sources=[Path(__file__),ROOT/'analysis/ncp_signal_reconstruction.py',ROOT/'analysis/ncp_recovery_analysis.py'];inputs=[ROOT/'data/raw/grid_monthly_panel.csv',ROOT/'data/external/ncp_long2025_well_2005_2024.csv',REVIEW/'well_cohort_membership.csv']
    manifest=dict(stage=a.stage,git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.version,specification=SPEC,sources={str(x.relative_to(ROOT)):sha(x) for x in sources},inputs={str(x.relative_to(ROOT)):sha(x) for x in inputs},source_support_reextraction=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print('Complete',a.stage,str(out),flush=True)

if __name__=='__main__':main()
