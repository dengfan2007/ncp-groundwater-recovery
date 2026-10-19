"""Calendar-preserving event, boundary and paired component diagnostics."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from ncp_monitoring_core import ROOT,REVIEW,VALIDATION,EVIDENCE,RUNTIME,BOUNDARY,panel,design,seed,json_write

VERSIONS=['complete','end_2023_06','exclude_2021','exclude_2023','exclude_both']

def keep_dates(dates,version):
    dates=pd.DatetimeIndex(dates);keep=np.ones(len(dates),bool)
    if version=='end_2023_06':keep&=dates<=pd.Timestamp('2023-06-01')
    if version in ['exclude_2021','exclude_both']:keep&=~((dates>=pd.Timestamp('2021-07-01'))&(dates<=pd.Timestamp('2022-06-01')))
    if version in ['exclude_2023','exclude_both']:keep&=~((dates>=pd.Timestamp('2023-07-01'))&(dates<=pd.Timestamp('2023-12-01')))
    return keep

def runs(keep):
    ix=np.flatnonzero(keep)
    return np.split(ix,np.flatnonzero(np.diff(ix)>1)+1) if len(ix) else []

def block_indices(keep,n=5000,block=12,rng=None):
    """Sample within each retained calendar run, truncate at either boundary."""
    rng=np.random.default_rng(42) if rng is None else rng
    segments=runs(keep);out=np.tile(np.arange(len(keep)),(n,1));truncated=total=0
    for target in segments:
        for j in range(n):
            filled=0
            while filled<len(target):
                start=int(rng.integers(len(target)));length=min(block,len(target)-start,len(target)-filled)
                out[j,target[filled:filled+length]]=target[start:start+length]
                filled+=length;total+=1;truncated+=length<block
    return out,dict(run_lengths=[len(s) for s in segments],blocks=total,truncated_blocks=truncated,truncated_fraction=truncated/total if total else None)

def spatial_weights(meta,size,n,rng):
    codes=list(zip(np.floor((meta.grid_lon.to_numpy()-112)/size).astype(int),np.floor((meta.grid_lat.to_numpy()-34.5)/size).astype(int)))
    unique=sorted(set(codes));lookup={c:i for i,c in enumerate(unique)};codes=np.array([lookup[c] for c in codes]);k=len(unique)
    draws=rng.integers(k,size=(n,k));counts=np.zeros((n,k),int)
    for j in range(k):counts+=(draws==j).sum(axis=1)[:,None]*(np.arange(k)==j)
    mult=counts[:,codes];south=meta.region_tag.eq('south').to_numpy();north=~south
    ns=mult[:,south].sum(axis=1);nn=mult[:,north].sum(axis=1);valid=(ns>0)&(nn>0)
    weights=np.divide(mult*south,ns[:,None],out=np.zeros_like(mult,dtype=float),where=ns[:,None]>0)-np.divide(mult*north,nn[:,None],out=np.zeros_like(mult,dtype=float),where=nn[:,None]>0)
    return weights,valid,k

def quantiles(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return [float(v) for v in np.quantile(x,[.025,.975])] if len(x)>=100 else [None,None]

def trend_diagnostics(data,folder):
    rows=[];cells=[];blocks=[]
    for start in ['2019-01-01','2022-01-01']:
        base=data.loc[data.date.between(start,'2024-08-01')];dates=pd.date_range(start,'2024-08-01',freq='MS')
        for boundary in ['original','flip','exclude']:
            d=base.loc[~base.grid_id.isin(BOUNDARY)].copy() if boundary=='exclude' else base.copy()
            if boundary=='flip':d.loc[d.grid_id.isin(BOUNDARY),'region_tag']='north'
            meta=d[['grid_id','grid_lon','grid_lat','region_tag']].drop_duplicates('grid_id').set_index('grid_id').sort_index()
            assert meta.index.is_unique and len(meta)<=27
            for version in VERSIONS:
                Y=d.pivot(index='date',columns='grid_id',values='gwa_cm').reindex(index=dates,columns=meta.index).to_numpy()
                keep=keep_dates(dates,version)&np.isfinite(Y).all(axis=1);X=design(dates);pinv=np.linalg.pinv(X[keep]);coef=pinv@Y[keep]
                residual=np.full_like(Y,np.nan);residual[keep]=Y[keep]-X[keep]@coef;residual[keep]-=residual[keep].mean(axis=0)
                harmonic=X[:,2:]@coef[2:];t=(dates.year*12+dates.month-(dates[0].year*12+dates[0].month))/12
                sen=np.array([stats.theilslopes((Y-harmonic)[keep,j],t[keep]).slope for j in range(Y.shape[1])])
                tag=dict(model=data.model.iloc[0],start=start,boundary=boundary,version=version,n_months=int(keep.sum()),n_cells=len(meta),n_north=int(meta.region_tag.eq('north').sum()),n_south=int(meta.region_tag.eq('south').sum()),descriptive_only=int(keep.sum())<24)
                ix,info=block_indices(keep,rng=np.random.default_rng(seed('trend',start,boundary,version)))
                boots=coef[1]+np.einsum('n,bnc->bc',pinv[1],residual[ix[:,keep]],optimize=True)
                blocks.append({**tag,**info});south=meta.region_tag.eq('south').to_numpy()
                for method,slopes in [('OLS',coef[1]),('Theil-Sen',sen)]:
                    for gid,value in zip(meta.index,slopes):cells.append({**tag,'method':method,'grid_id':gid,'slope':value,'region_tag':meta.loc[gid,'region_tag']})
                    for size in [2,3]:
                        weights,valid,k=spatial_weights(meta,size,5000,np.random.default_rng(seed('space',size,start,boundary,version)))
                        estimates=np.sum((boots if method=='OLS' else slopes)*weights,axis=1);lo,hi=quantiles(estimates[valid])
                        rows.append({**tag,'method':method,'block_degrees':size,'interval_type':'time_and_space' if method=='OLS' else 'spatial_conditional',
                                     'north':float(slopes[~south].mean()),'south':float(slopes[south].mean()),'estimate':float(slopes[south].mean()-slopes[~south].mean()),'low':lo,'high':hi,'n_blocks':k,'valid_replicates':int(valid.sum())})
        print('Trends',data.model.iloc[0],start,flush=True)
    pd.DataFrame(rows).to_csv(folder/'trend_contrasts.csv',index=False);pd.DataFrame(cells).to_csv(folder/'cell_trends.csv',index=False)
    pd.DataFrame([{**x,'run_lengths':';'.join(map(str,x['run_lengths']))} for x in blocks]).to_csv(folder/'block_support.csv',index=False)

def well_cells():
    members=pd.read_csv(REVIEW/'well_cohort_membership.csv');raw=pd.read_csv(ROOT/'data/external/ncp_long2025_well_2005_2024.csv',parse_dates=['date'])
    raw=raw.loc[raw.well_id.isin(members.well_id)&raw.date.between('2018-01-01','2024-09-01')].copy()
    raw['well']=raw.gw_level-raw.groupby('well_id').gw_level.transform('mean')
    cells=raw.groupby(['aquifer_group','grid_id','date'],as_index=False).well.mean()
    common=sorted(set(members.loc[members.aquifer_group.eq('unconfined'),'grid_id'])&set(members.loc[members.aquifer_group.eq('confined'),'grid_id']))
    return cells,common

def correlate(a,b):
    a=np.asarray(a);b=np.asarray(b);ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3 or np.std(a[ok])==0 or np.std(b[ok])==0:return np.nan
    return float(np.corrcoef(a[ok],b[ok])[0,1])

def boot_corr(a,b,ix):
    a=np.asarray(a)[ix];b=np.asarray(b)[ix];ok=np.isfinite(a)&np.isfinite(b)
    a=np.where(ok,a,np.nan);b=np.where(ok,b,np.nan)
    a-=np.nanmean(a,axis=1,keepdims=True);b-=np.nanmean(b,axis=1,keepdims=True)
    den=np.sqrt(np.nansum(a*a,axis=1)*np.nansum(b*b,axis=1))
    return np.divide(np.nansum(a*b,axis=1),den,out=np.full(len(ix),np.nan),where=den>0)

def well_diagnostics(data,folder):
    cells,common=well_cells();dates=pd.date_range('2019-01-01','2024-08-01',freq='MS');metadata=data[['grid_id','region_tag']].drop_duplicates('grid_id').set_index('grid_id');rows=[];series=[];paired=[]
    for aq in ['unconfined','confined']:
        for region in ['north','south']:
            ids=[gid for gid in common if metadata.loc[gid,'region_tag']==region]
            well=cells.loc[cells.aquifer_group.eq(aq)].pivot(index='date',columns='grid_id',values='well').reindex(index=dates,columns=ids)
            sat={c:data.pivot(index='date',columns='grid_id',values=c).reindex(index=dates,columns=ids) for c in ['gwa_cm','twsa_cm','smsa_cm']}
            for support in ['fixed_cells','monthly_matched']:
                w=well.mean(axis=1).to_numpy();values={c:(v.where(well.notna()) if support=='monthly_matched' else v).mean(axis=1).to_numpy() for c,v in sat.items()}
                series.extend(pd.DataFrame(dict(date=dates,well=w,well_cells=well.notna().sum(axis=1).to_numpy(),**values)).assign(model=data.model.iloc[0],aquifer=aq,region=region,support=support).to_dict('records'))
                for version in VERSIONS:
                    keep=keep_dates(dates,version);ix,_=block_indices(keep,n=2000,rng=np.random.default_rng(seed('correlation',aq,region,support,version)))
                    for treatment in ['raw','residual','difference']:
                        signals={**values,'well':w};transformed={}
                        for name,v in signals.items():
                            v=v.copy();v[~keep]=np.nan
                            if treatment=='residual':
                                X=design(dates);ok=np.isfinite(v);v=v-X@np.linalg.lstsq(X[ok],v[ok],rcond=None)[0]
                            elif treatment=='difference':v=np.r_[np.nan,np.diff(v)]
                            transformed[name]=v
                        boot={};ww=transformed['well']
                        for component in values:
                            v=transformed[component];n=int(np.isfinite(v+ww).sum());b=boot_corr(v,ww,ix[:,keep]);boot[component]=b
                            lo,hi=quantiles(b)
                            rows.append(dict(model=data.model.iloc[0],aquifer=aq,region=region,support=support,version=version,treatment=treatment,component=component,n_pairs=n,r=correlate(v,ww),low=lo,high=hi,descriptive_only=n<24))
                        delta=boot['twsa_cm']-boot['gwa_cm'];lo,hi=quantiles(delta)
                        paired.append(dict(model=data.model.iloc[0],aquifer=aq,region=region,support=support,version=version,treatment=treatment,
                                           estimate=correlate(transformed['twsa_cm'],ww)-correlate(transformed['gwa_cm'],ww),low=lo,high=hi))
    pd.DataFrame(rows).to_csv(folder/'well_correlations.csv',index=False);pd.DataFrame(series).to_csv(folder/'well_series.csv',index=False);pd.DataFrame(paired).to_csv(folder/'paired_component_correlation_differences.csv',index=False)

def timing_boundary(folder):
    actual=pd.read_csv(STORY_PATH/'actual_selections.csv')
    actual=actual.loc[actual.scope.eq('grid')&actual.experiment.eq('actual')&actual.endpoint.eq('2023-11-01')&actual.route.isin(['A','D'])]
    primary=pd.read_csv(VALIDATION/'fits/actual_selections.csv');primary=primary.loc[primary.scope.eq('grid')&primary.experiment.eq('original')&primary.endpoint.eq('2024-08-01')&primary.rule.isin(['highest_probability','earliest'])]
    meta=panel()[['grid_id','grid_lon','grid_lat','region_tag']].drop_duplicates('grid_id').set_index('grid_id');rows=[]
    for endpoint,d in [('2023-11-01',actual),('2024-08-01',primary)]:
        common=set.intersection(*(set(d.loc[d.route.eq(route)&d.rule.eq('highest_probability')&d.qualifies,'grid_id']) for route in ['A','D']))
        for boundary in ['original','flip','exclude']:
            ids=sorted(common-set(BOUNDARY)) if boundary=='exclude' else sorted(common);mm=meta.loc[ids].copy()
            if boundary=='flip':mm.loc[mm.index.isin(BOUNDARY),'region_tag']='north'
            south=mm.region_tag.eq('south').to_numpy()
            for route in ['A','D']:
                for rule in ['highest_probability','earliest']:
                    v=d.loc[d.route.eq(route)&d.rule.eq(rule)&d.grid_id.isin(ids)].set_index('grid_id').loc[ids];dt=pd.to_datetime(v.selected_date);m=(dt.dt.year*12+dt.dt.month).to_numpy()
                    for size in [2,3]:
                        weights,valid,k=spatial_weights(mm,size,5000,np.random.default_rng(seed('timing_boundary',endpoint,size,boundary,route,rule)));lo,hi=quantiles((weights*m).sum(axis=1)[valid])
                        rows.append(dict(endpoint=endpoint,boundary=boundary,route=route,rule=rule,n=len(ids),n_north=int((~south).sum()),n_south=int(south.sum()),estimate=float(m[south].mean()-m[~south].mean()),low=lo,high=hi,block_degrees=size,n_blocks=k))
    pd.DataFrame(rows).to_csv(folder/'timing_boundary.csv',index=False)

STORY_PATH=ROOT/'outputs/story_revision/2026-10-17-r1'

def observation_support(folder):
    p=panel();rows=[]
    for gid,g in p.loc[p.date.between('2015-01-01','2018-12-01')].groupby('grid_id'):
        for year,s in g.groupby(g.date.dt.year):
            missing=runs(s.gwa_cm.isna().to_numpy());rows.append(dict(grid_id=gid,year=year,n_observed=int(s.gwa_cm.notna().sum()),longest_missing=max(map(len,missing),default=0)))
    pd.DataFrame(rows).to_csv(folder/'observation_support_2015_2018.csv',index=False)

def model_panels():
    p=panel();p['model']='archived';yield p
    p=panel(True);p['model']='uniform_noah';yield p
    source=EVIDENCE/'extraction/monthly_anomalies.csv'
    if not source.exists():return
    raw=pd.read_csv(source,parse_dates=['date']).set_index(['grid_id','date'])
    original=panel().set_index(['grid_id','date'])
    for model,gwa,soil,snow,canopy in [('shared_noah','noah_gwa_cm','sms_anomaly','swe_anomaly','cwa_anomaly'),
                                     ('era2','era_gwa2_cm','era_soil2_cm_anomaly','era_snow_cm_anomaly','era_skin_cm_anomaly'),
                                     ('era289','era_gwa289_cm','era_soil289_cm_anomaly','era_snow_cm_anomaly','era_skin_cm_anomaly')]:
        d=original.copy()
        for dest,src in [('gwa_cm',gwa),('twsa_cm','tws_anomaly'),('smsa_cm',soil),('swea_cm',snow),('cwa_cm',canopy),('precip_mm','precip_mm')]:d[dest]=raw[src].reindex(d.index)
        d['model']=model;yield d.reset_index()

def climate(folder):
    path=EVIDENCE/'extraction/monthly_anomalies.csv'
    if not path.exists():return
    e=pd.read_csv(path,parse_dates=['date']);e['month']=e.date.dt.month
    base=e.loc[e.date.between('2003-01-01','2014-12-01')].groupby(['grid_id','month']).precip_mm.mean()
    e['precip_anomaly_mm']=e.precip_mm-e.set_index(['grid_id','month']).index.map(base)
    meta=panel()[['grid_id','region_tag','coverage_ratio','grid_lat']].drop_duplicates('grid_id');e=e.merge(meta,on='grid_id',validate='many_to_one');e['weight']=e.coverage_ratio*np.cos(np.deg2rad(e.grid_lat))
    archived=panel()[['grid_id','date','precip_mm']].rename(columns={'precip_mm':'imerg_mm'});e=e.merge(archived,on=['grid_id','date'])
    for c in ['precip_mm','precip_anomaly_mm','imerg_mm']:e[c+'_weighted']=e[c]*e.weight
    reg=e.groupby(['region_tag','date']).agg(weight=('weight','sum'),**{c:(c+'_weighted','sum') for c in ['precip_mm','precip_anomaly_mm','imerg_mm']}).reset_index()
    for c in ['precip_mm','precip_anomaly_mm','imerg_mm']:reg[c]/=reg.weight
    reg.to_csv(folder/'regional_precipitation.csv',index=False)
    event=[]
    for name,start,end in [('2021','2021-07-01','2022-06-01'),('2023','2023-07-01','2023-12-01')]:
        for region,g in reg.loc[reg.date.between(start,end)].groupby('region_tag'):event.append(dict(event=name,region=region,months=len(g),precip_mm=g.precip_mm.sum(),anomaly_mm=g.precip_anomaly_mm.sum(),imerg_mm=g.imerg_mm.sum()))
    pd.DataFrame(event).to_csv(folder/'event_precipitation.csv',index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='*');a=p.parse_args()
    for data in model_panels():
        model=data.model.iloc[0]
        if a.models and model not in a.models:continue
        folder=EVIDENCE/'diagnostics'/model;folder.mkdir(parents=True,exist_ok=False)
        trend_diagnostics(data,folder);well_diagnostics(data,folder);print('Completed',model,flush=True)
    folder=EVIDENCE/'diagnostics';folder.mkdir(parents=True,exist_ok=True)
    timing_boundary(folder);observation_support(folder);climate(folder)

if __name__=='__main__':main()
