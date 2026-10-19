"""Targeted review experiments, preserving the October 12 evidence and raw data.

Run with the documented Anaconda runtime and a NEW --output-dir.
All cutoffs refit seasonality using only observations available at that cutoff.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL

import ncp_recovery_analysis as core
import ncp_revision_analyses as rev
from ncp_calendar_revision import calendar_detrend, validation
from ncp_well_utils import standardize_well_data

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/calendar_revision/2026-10-12-r4'
SEED = 42


def seasonal_series(g):
    s = core.calendar_series(g)
    s['deseasoned_gwa'] = s.gwa_cm - STL(s.gwa_cm.interpolate(limit_direction='both'), period=12, robust=True).fit().seasonal
    return s


def harmonics(t):
    return np.column_stack([f(2*np.pi*k*t/12) for k in (1, 2) for f in (np.sin, np.cos)])


def spatial_contrast(frame, value='slope', size=2., dx=0., dy=0., boundary=None, n_boot=5000):
    d = frame.dropna(subset=[value]).copy()
    if boundary is not None:
        d['region_tag'] = np.where(d.grid_lat >= boundary, 'north', 'south')
    d['block'] = list(zip(np.floor((d.grid_lon-112-dx)/size), np.floor((d.grid_lat-34.5-dy)/size)))
    grouped = list(d.groupby('block'))
    sums = np.array([[g.loc[g.region_tag.eq(r), value].sum() for r in ['north','south']] for _, g in grouped])
    counts = np.array([[g.region_tag.eq(r).sum() for r in ['north','south']] for _, g in grouped])
    ix = np.random.default_rng(SEED).integers(len(grouped), size=(n_boot,len(grouped)))
    n, total = counts[ix].sum(axis=1), sums[ix].sum(axis=1)
    valid = (n>0).all(axis=1)
    delta = (total[valid]/n[valid])[:,1]-(total[valid]/n[valid])[:,0]
    ci = np.quantile(delta,[.025,.975]) if len(delta) else [np.nan,np.nan]
    return dict(estimate=d.loc[d.region_tag.eq('south'),value].mean()-d.loc[d.region_tag.eq('north'),value].mean(),
                ci_low=ci[0],ci_high=ci[1],blocks=len(grouped),valid_replicates=int(valid.sum()),
                n_north=int(d.region_tag.eq('north').sum()),n_south=int(d.region_tag.eq('south').sum()),
                block_degrees=size,origin_dx=dx,origin_dy=dy,boundary=boundary)


def fit_shape(dates, y, model, fixed_date=None):
    """Joint seasonal + trend regression; BIC is nominal (iid working model).

    Searching a break adds one parameter. Hinge is continuous; step has a common
    slope; step_hinge allows independent slopes AND an instantaneous level shift.
    """
    dates = pd.DatetimeIndex(dates)
    t = np.arange(len(dates),dtype=float)/12
    ok = np.isfinite(y)
    base = np.column_stack([np.ones(len(t)),t,harmonics(t*12)])
    candidates = [None] if model == 'linear' else ([int(dates.get_loc(pd.Timestamp(fixed_date)))] if fixed_date is not None else range(len(t)))
    best = None
    for i in candidates:
        if i is not None and (ok[:i].sum()<18 or ok[i:].sum()<18):
            continue
        extra = []
        if model in ['step','step_hinge']:
            extra.append((t>=t[i]).astype(float))
        if model in ['hinge','step_hinge']:
            extra.append(np.maximum(t-t[i],0))
        design = np.column_stack([base,*extra])
        coef = np.linalg.lstsq(design[ok],np.asarray(y)[ok],rcond=None)[0]
        residual = np.asarray(y)[ok]-design[ok]@coef
        k = design.shape[1]+int(i is not None and fixed_date is None)
        bic = ok.sum()*np.log(max(np.sum(residual**2)/ok.sum(),1e-12))+k*np.log(ok.sum())
        if best is None or bic < best['bic']:
            r = np.full(len(t),np.nan); r[ok] = residual
            best = dict(model=model,bic=float(bic),date=None if i is None else dates[i].strftime('%Y-%m-%d'),
                        pre_slope=float(coef[1]),post_slope=float(coef[1]+(coef[-1] if model in ['hinge','step_hinge'] else 0)),
                        step_cm=float(coef[6]) if model in ['step','step_hinge'] else 0.,
                        residual_lag1=rev.pearson_or_nan(r[:-1],r[1:]),n=int(ok.sum()),
                        fitted=design@coef)
    return best


def rolling(panel,meta,out):
    rows, summaries, candidates = [],[],[]
    original = core.rb.beast
    # Capture all full-record candidate dates without changing selection logic.
    for endpoint in ['2019-12-01','2020-12-01','2021-12-01','2022-12-01','2023-12-01','2024-09-01']:
        print('Rolling endpoint',endpoint,flush=True)
        subset = panel.loc[panel.date<=endpoint]
        captured=[]
        def capture(*args,**kwargs):
            result=original(*args,**kwargs); captured.append(result); return result
        with patch.object(core.rb,'beast',side_effect=capture):
            fits=core.detect_breakpoints(subset).merge(meta,on='grid_id')
        fits['endpoint']=endpoint; rows.append(fits)
        valid=fits.loc[fits.is_recovery_relevant]
        summaries.append(dict(endpoint=endpoint,n_reportable=len(valid),north=int(valid.region_tag.eq('north').sum()),south=int(valid.region_tag.eq('south').sum()),median_date=pd.to_datetime(valid.onset_date).median()))
        if endpoint=='2024-09-01':
            for (gid,g),result in zip(subset.groupby('grid_id'),captured):
                s=seasonal_series(g)
                for date,prob in zip(np.atleast_1d(result.trend.cp),np.atleast_1d(result.trend.cpPr)):
                    if not np.isfinite(date): continue
                    d=core.maybe_date_from_cp(s,float(date)); idx=core.breakpoint_index(s,d)
                    pre,post=core.fit_trends_for_breakpoint(s,idx)
                    candidates.append(dict(grid_id=gid,date=d,probability=float(prob),pre_slope=pre,post_slope=post,
                                           qualifies=core.is_recovery_relevant_break(d,pre,post) and prob>=.5))
    allfits=pd.concat(rows,ignore_index=True)
    allfits.to_csv(out/'rolling_breakpoints.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'rolling_summary.csv',index=False)
    pd.DataFrame(candidates).to_csv(out/'beast_candidates.csv',index=False)
    full=allfits.loc[allfits.endpoint.eq('2024-09-01')]
    old=pd.read_csv(BASE/'breakpoints_calendar_stl.csv')
    assert full.onset_date.fillna('').tolist()==old.onset_date.fillna('').tolist(), 'Baseline dates changed'
    assert np.allclose(full.post_trend_cm_per_year,old.post_trend_cm_per_year,equal_nan=True), 'Baseline slopes changed'
    stability=[]
    for gid,g in allfits.groupby('grid_id'):
        v=g.loc[g.is_recovery_relevant].copy()
        dates=pd.to_datetime(v.onset_date)
        m=dates.dt.year*12+dates.dt.month
        stability.append(dict(grid_id=gid,first_reporting_endpoint=None if v.empty else v.endpoint.iloc[0],
                              first_selected_date=None if v.empty else v.onset_date.iloc[0],
                              final_selected_date=g.onset_date.iloc[-1],reportable_endpoints=len(v),
                              selected_date_range_months=np.nan if v.empty else m.max()-m.min()))
    pd.DataFrame(stability).to_csv(out/'rolling_stability.csv',index=False)
    eligibility=[]
    for row in old.loc[old.is_recovery_relevant].itertuples():
        sub=panel.loc[panel.grid_id.eq(row.grid_id)&panel.date.le('2020-12-01')]
        n=int(sub.loc[sub.date.ge(row.onset_date),'gwa_cm'].notna().sum())
        eligibility.append(dict(grid_id=row.grid_id,full_record_date=row.onset_date,post_observations_through_2020=n,eligible=n>=18))
    pd.DataFrame(eligibility).to_csv(out/'endpoint_eligibility.csv',index=False)


def well_checks(panel,meta,out):
    raw=pd.read_csv(ROOT/'data/external/ncp_long2025_well_2005_2024.csv',parse_dates=['date'])
    raw=raw.loc[raw.grid_id.isin(meta.grid_id)&raw.aquifer_group.isin(['unconfined','confined'])].copy()
    assert not raw.duplicated(['well_id','date']).any()
    assert np.allclose(raw.gw_level,-raw.gw_depth_m,equal_nan=True)
    assert raw.groupby('well_id').grid_id.nunique().max()==1
    cohorts={}; inventory=[]; memberships=[]
    for aquifer in ['unconfined','confined']:
        g=raw.loc[raw.aquifer_group.eq(aquifer)&raw.date.between('2018-01-01','2024-09-01')].dropna(subset=['gw_level']).copy()
        g['period']=np.where(g.date<'2019-01-01','pre','post')
        counts=g.groupby(['well_id','period']).date.nunique().unstack(fill_value=0).reindex(columns=['pre','post'],fill_value=0)
        ids=counts.index[(counts.pre>=.8*12)&(counts.post>=.8*69)]
        selected=g.loc[g.well_id.isin(ids)]
        cohorts[aquifer]=standardize_well_data(selected,dataset_label=aquifer,value_kind='level',value_col='gw_level')
        member=selected.drop_duplicates('well_id')[['well_id','grid_id','aquifer_group','aquifer_layer','type']].merge(meta,on='grid_id')
        memberships.append(member)
        for r in ['north','south']:
            u=member.loc[member.region_tag.eq(r)]
            inventory.append(dict(aquifer=aquifer,region=r,wells=len(u),cells=u.grid_id.nunique()))
    pd.concat(memberships).to_csv(out/'well_cohort_membership.csv',index=False)
    pd.DataFrame(inventory).to_csv(out/'well_cohort_inventory.csv',index=False)
    common=set(cohorts['unconfined'].grid_id)&set(cohorts['confined'].grid_id)
    metrics=[]; lags=[]; allseries=[]
    for aquifer,wells in cohorts.items():
        for support,ids in [('own',set(wells.grid_id)),('common',common)]:
            print('Well validation',aquifer,support,flush=True)
            folder=out/f'{aquifer}_{support}'; folder.mkdir()
            w=wells.loc[wells.grid_id.isin(ids)]
            metrics.append(validation(panel.loc[panel.grid_id.isin(ids)],meta,w,folder).assign(aquifer=aquifer,support=support))
            s=pd.read_csv(folder/'validation_series.csv',parse_dates=['date']).assign(aquifer=aquifer,support=support)
            allseries.append(s)
            for region,g in s.loc[s.date.ge('2019-01-01')].groupby('region_tag'):
                a=calendar_detrend(g.gwa_cm.to_numpy()); b=calendar_detrend(g.well_anomaly.to_numpy())
                for lag in range(-3,4):
                    # Positive lag compares earlier GWA with later well anomalies.
                    bv=pd.Series(b).shift(-lag).to_numpy()
                    lags.append(dict(aquifer=aquifer,support=support,region=region,lag=lag,r=rev.pearson_or_nan(a,bv),n=int((np.isfinite(a)&np.isfinite(bv)).sum())))
    pd.concat(metrics).to_csv(out/'aquifer_validation_metrics.csv',index=False)
    pd.concat(allseries).to_csv(out/'aquifer_validation_series.csv',index=False)
    pd.DataFrame(lags).to_csv(out/'well_lag_diagnostics.csv',index=False)


def fixed_windows(panel,meta,out):
    rows=[]
    for start in ['2019-01-01','2022-01-01']:
        for gid,g in panel.loc[panel.date.between(start,'2024-08-01')].groupby('grid_id'):
            s=core.calendar_series(g); t=np.arange(len(s))/12; h=harmonics(t*12)
            y=s.gwa_cm.to_numpy(); ok=np.isfinite(y)
            design=np.column_stack([np.ones(len(t)),t,h]); coef=np.linalg.lstsq(design[ok],y[ok],rcond=None)[0]
            clean=y-h@coef[2:]
            for method,slope in [('OLS',coef[1]),('Theil-Sen',stats.theilslopes(clean[ok],t[ok])[0])]:
                rows.append(dict(grid_id=gid,start=start,method=method,slope=slope,n=int(ok.sum())))
    cells=pd.DataFrame(rows).merge(meta,on='grid_id'); cells.to_csv(out/'fixed_window_cell_slopes.csv',index=False)
    contrasts=[]
    for (start,method),g in cells.groupby(['start','method']):
        for size in [2.,3.]:
            for dx in [0.,size/2]:
                for dy in [0.,size/2]:
                    contrasts.append(dict(start=start,method=method,**spatial_contrast(g,size=size,dx=dx,dy=dy)))
        for boundary in [36.,37.,38.]:
            contrasts.append(dict(start=start,method=method,**spatial_contrast(g,boundary=boundary)))
    pd.DataFrame(contrasts).to_csv(out/'fixed_window_spatial_contrasts.csv',index=False)
    old=pd.read_csv(BASE/'breakpoints_calendar_stl.csv'); old=old.loc[old.is_recovery_relevant].copy()
    old['slope']=old.post_trend_cm_per_year
    pd.DataFrame([spatial_contrast(old,size=size,dx=dx,dy=dy) for size in [2.,3.] for dx in [0.,size/2] for dy in [0.,size/2]]).to_csv(out/'selected_slope_origin_sensitivity.csv',index=False)


def shapes_and_climate(panel,out):
    rows=[]; fitted=[]
    for gid,g in panel.loc[panel.date.between('2015-01-01','2024-08-01')].groupby('grid_id'):
        s=core.calendar_series(g)
        for model in ['linear','hinge','step','step_hinge']:
            fit=fit_shape(s.date,s.gwa_cm.to_numpy(),model)
            fitted.append(pd.DataFrame(dict(grid_id=gid,date=s.date,model=model,observed=s.gwa_cm,fitted=fit.pop('fitted'))))
            rows.append(dict(grid_id=gid,**fit))
    pd.DataFrame(rows).to_csv(out/'shape_comparison.csv',index=False)
    pd.concat(fitted).to_csv(out/'shape_fitted_series.csv',index=False)
    seasonal=[]
    for gid,g in panel.groupby('grid_id'):
        s=seasonal_series(g); seasonal.append(s[['date','gwa_cm','deseasoned_gwa']].assign(grid_id=gid,region_tag=g.region_tag.iloc[0]))
    pd.concat(seasonal).to_csv(out/'deseasoned_series.csv',index=False)
    climate=[]
    for region,p in [('all',panel),('north',panel.loc[panel.region_tag.eq('north')]),('south',panel.loc[panel.region_tag.eq('south')])]:
        monthly=p.groupby('date')[['precip_mm','et_mm']].mean()
        for variable in ['precip_mm','et_mm']:
            annual=monthly[variable].groupby(monthly.index.year).agg(['sum','count'])
            annual=annual.loc[annual['count'].eq(12)]
            reference=annual.loc[2003:2014,'sum']; mean,sd=reference.mean(),reference.std()
            for year,r in annual.iterrows():
                climate.append(dict(region=region,variable=variable,year=year,total_mm=r['sum'],baseline_mean=mean,baseline_sd=sd,anomaly_mm=r['sum']-mean,z=(r['sum']-mean)/sd))
    pd.DataFrame(climate).to_csv(out/'annual_climate.csv',index=False)


def gap_experiment(panel,out,n_reps=20):
    """Paired illustrative simulations, not a calibration of real-world coverage."""
    dates=pd.date_range('2003-01-01','2024-09-01',freq='MS'); t=np.arange(len(dates))
    observed=panel.groupby('date').gwa_cm.count().reindex(dates).gt(0).to_numpy()
    rows=[]; rng=np.random.default_rng(SEED)
    for true_date in ['2017-12-01','2019-08-01','2021-08-01']:
        k=int(dates.get_loc(true_date))
        for rep in range(n_reps):
            e=rng.normal(0,1.5,len(t)); noise=np.empty(len(t)); noise[0]=e[0]/np.sqrt(1-.6**2)
            for j in range(1,len(t)): noise[j]=.6*noise[j-1]+e[j]
            y=-1.5*t/12+4.5*np.maximum(t-k,0)/12+3*np.sin(2*np.pi*t/12)+noise
            for mask_name,mask in [('complete',np.ones(len(t),bool)),('observed_mask',observed)]:
                data=pd.DataFrame(dict(grid_id='synthetic',date=dates,gwa_cm=np.where(mask,y,np.nan)))
                b=core.detect_breakpoints(data).iloc[0]
                d=pd.to_datetime(b.onset_date)
                error=np.nan if pd.isna(d) else (d.year-dates[k].year)*12+d.month-dates[k].month
                rows.append(dict(true_date=true_date,replicate=rep,mask=mask_name,detected=bool(b.is_recovery_relevant),selected_date=b.onset_date,error_months=error,method=b.selected_method))
        print('Gap simulations',true_date,flush=True)
    pd.DataFrame(rows).to_csv(out/'gap_simulations.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',required=True); args=parser.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=False)
    panel=core.load_panel(ROOT/'data/raw/grid_monthly_panel.csv')
    meta=panel[['grid_id','grid_lon','grid_lat','region_tag']].drop_duplicates('grid_id')
    rolling(panel,meta,out)
    well_checks(panel,meta,out)
    fixed_windows(panel,meta,out)
    shapes_and_climate(panel,out)
    gap_experiment(panel,out)
    sources=[Path(__file__),*[ROOT/'analysis'/f for f in ['ncp_recovery_analysis.py','ncp_revision_analyses.py','ncp_calendar_revision.py','ncp_well_utils.py']]]
    inputs=[ROOT/'data/raw/grid_monthly_panel.csv',ROOT/'data/external/ncp_long2025_well_2005_2024.csv',BASE/'breakpoints_calendar_stl.csv']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(seed=SEED,git_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.version,
                  sources={str(p.relative_to(ROOT)):sha(p) for p in sources},inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
                  packages={n:getattr(importlib.import_module(n),'__version__','unknown') for n in ['numpy','pandas','scipy','statsmodels','Rbeast']},
                  baseline_reproduced=True,gap_simulation=dict(replicates_per_date=20,ar1=.6,innovation_sd_cm=1.5,pre_slope=-1.5,post_slope=3.,seasonal_amplitude_cm=3.),
                  notes=['No new input downloads','Shape BIC assumes iid errors and is descriptive','Fixed-window block intervals condition on cell fits','Cohort membership fixed; monthly availability can vary'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Complete:',out,flush=True)


if __name__=='__main__': main()
