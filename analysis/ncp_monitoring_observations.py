"""Independent fixed-well dates and the observational coverage of simulation settings."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from ncp_monitoring_core import ROOT,REVIEW,VALIDATION,EVIDENCE,RUNTIME,panel,select,clean,run_tasks,export_results,seed,json_write

def tasks(start='2005-01-01'):
    members=pd.read_csv(REVIEW/'well_cohort_membership.csv');common=sorted(set(members.loc[members.aquifer_group.eq('unconfined'),'grid_id'])&set(members.loc[members.aquifer_group.eq('confined'),'grid_id']))
    raw=pd.read_csv(ROOT/'data/external/ncp_long2025_well_2005_2024.csv',parse_dates=['date']);raw=raw.loc[raw.well_id.isin(members.well_id)&raw.grid_id.isin(common)&raw.date.ge(start)]
    p=panel();result=[];support=[];all_series=[]
    for endpoint in ['2023-12-01','2024-08-01']:
        dates=pd.date_range(start,endpoint,freq='MS');r=raw.loc[raw.date.le(endpoint)].copy();r['well']=r.gw_level-r.groupby('well_id').gw_level.transform('mean')
        cell=r.groupby(['aquifer_group','grid_id','date'],as_index=False).agg(well=('well','mean'),n_wells=('well_id','nunique'))
        for gid in common:
            region=p.loc[p.grid_id.eq(gid),'region_tag'].iloc[0]
            records=[]
            for aq in ['unconfined','confined']:
                g=cell.loc[cell.grid_id.eq(gid)&cell.aquifer_group.eq(aq)].set_index('date').reindex(dates)
                records.append((aq,g.well.to_numpy(),g.n_wells.to_numpy()))
            g=p.loc[p.grid_id.eq(gid)].set_index('date').reindex(dates);records.append(('gwa',g.gwa_cm.to_numpy(),np.full(len(dates),np.nan)))
            for aq,y,nw in records:
                coverage=float(np.isfinite(y).mean());tid=f'{aq}_{gid}_{endpoint}';eligible=coverage>=.8
                valid=np.isfinite(y);positions=[i for i in range(len(y)) if valid[max(0,i-36):i].sum()>=18 and valid[i:min(len(y),i+36)].sum()>=18]
                md=dict(grid_id=gid,region_tag=region,endpoint=endpoint,aquifer=aq,coverage=coverage,coverage_eligible=eligible,latest_assessable=str(dates[max(positions)].date()) if positions else None)
                support.append(dict(task_id=tid,**md,n_observed=int(valid.sum()),calendar_months=len(y)))
                all_series.extend(pd.DataFrame(dict(date=dates,value=y,n_wells=nw)).assign(task_id=tid,**md).to_dict('records'))
                if eligible:result.append(clean(dict(task_id=tid,dates=dates.astype(str).tolist(),y=y.tolist(),routes=['A'],beast_seed=42,metadata=md)))
    return result,pd.DataFrame(support),pd.DataFrame(all_series)

def mean_spatial_interval(values,meta,size):
    if len(values)<3:return np.nan,np.nan,0
    codes=list(zip(np.floor((meta.grid_lon.to_numpy()-112)/size),np.floor((meta.grid_lat.to_numpy()-34.5)/size)));unique=sorted(set(codes));lookup={c:i for i,c in enumerate(unique)}
    if len(unique)<2:return np.nan,np.nan,len(unique)
    sums=np.zeros(len(unique));counts=sums.copy()
    for c,v in zip(codes,values):sums[lookup[c]]+=v;counts[lookup[c]]+=1
    rng=np.random.default_rng(seed('well_date_fraction',size,meta.index.tolist()));ix=rng.integers(len(unique),size=(5000,len(unique)));b=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    return *np.quantile(b,[.025,.975]),len(unique)

def wells(workers,start='2005-01-01'):
    name='well_dates' if start=='2005-01-01' else 'well_dates_2018'
    jobs,support,series=tasks(start);results=run_tasks(jobs,RUNTIME/name/'checkpoints',workers);dest=EVIDENCE/name;export_results(results,dest)
    support.to_csv(dest/'support.csv',index=False);series.to_csv(dest/'input_series.csv',index=False);rows=[];lookup={r['task_id']:r for r in results}
    for r in support.to_dict('records'):
        tid=r['task_id'];cp=lookup[tid]['candidates'] if tid in lookup else []
        for rule in ['highest_probability','earliest']:
            c=select(cp,rule=rule,local=True)
            if not r['coverage_eligible']:reason='coverage_below_80_percent'
            elif c:reason='assigned'
            elif not any(x['date']>='2015-01-01' and x['probability']>=.5 for x in cp):reason='no_posterior_candidate'
            elif not any(x.get('local_pre_slope') is not None and x.get('local_post_slope') is not None for x in cp if x['date']>='2015-01-01' and x['probability']>=.5):reason='insufficient_local_observations'
            else:reason='no_local_sign_reversal_candidate'
            rows.append({**r,'rule':rule,'qualifies':c is not None,'selected_date':c['date'] if c else None,'reason':reason,'limited_followup':bool(c and c['post_calendar_months']<36),
                         'post_calendar_months':c['post_calendar_months'] if c else None,'post_real_observations':c['local_post_observations'] if c else None})
    selected=pd.DataFrame(rows);selected.to_csv(dest/'local_selections.csv',index=False);paired=[];summaries=[];meta=panel()[['grid_id','grid_lon','grid_lat','region_tag']].drop_duplicates('grid_id').set_index('grid_id')
    for aq in ['unconfined','confined']:
        for endpoint in ['2023-12-01','2024-08-01']:
            for rule in ['highest_probability','earliest']:
                a=selected.loc[selected.aquifer.eq(aq)&selected.endpoint.eq(endpoint)&selected.rule.eq(rule)];b=selected.loc[selected.aquifer.eq('gwa')&selected.endpoint.eq(endpoint)&selected.rule.eq(rule)]
                m=a.merge(b,on='grid_id',suffixes=('_well','_gwa'),validate='one_to_one');w=pd.to_datetime(m.selected_date_well);g=pd.to_datetime(m.selected_date_gwa)
                m['difference_months']=(w.dt.year-g.dt.year)*12+w.dt.month-g.dt.month;m['aquifer']=aq;m['endpoint']=endpoint;m['rule']=rule;paired.append(m)
                for region in ['all','north','south']:
                    s=m if region=='all' else m.loc[m.region_tag_well.eq(region)];eligible=s.loc[s.difference_months.notna()];mm=meta.loc[eligible.grid_id]
                    for size in [2,3]:
                        match=eligible.difference_months.abs().le(12).to_numpy();lo,hi,k=mean_spatial_interval(match.astype(float),mm,size)
                        summaries.append(dict(aquifer=aq,endpoint=endpoint,rule=rule,region=region,total_cells=len(s),joint_dates=len(eligible),within_six=int(eligible.difference_months.abs().le(6).sum()),within_twelve=int(match.sum()),
                                              fraction_twelve=match.mean() if len(match) else np.nan,low=lo,high=hi,block_degrees=size,n_blocks=k,median_difference=eligible.difference_months.median(),
                                              support_sufficient_for_description=len(match)>=10 and match.mean()>=.5 if len(match) else False))
    pd.concat(paired).to_csv(dest/'paired_dates.csv',index=False);pd.DataFrame(summaries).to_csv(dest/'date_agreement.csv',index=False)
    json_write(dest/'manifest.json',dict(tasks=len(jobs),expected_tasks=108,start=start,selection='shared local signs; no head/storage amplitude conversion',wells_are_not_storage_truth=True))

def coverage():
    p=panel();noise=pd.read_csv(VALIDATION/'local/noise_calibration.csv').set_index('grid_id');q=pd.read_csv(VALIDATION/'local/noise_quantiles.csv').set_index('quantile')
    choices=pd.read_csv(EVIDENCE/'baseline/selections.csv');cp=pd.read_csv(EVIDENCE/'baseline/candidates.csv');rows=[]
    for r in choices.loc[choices.endpoint.eq('2024-08-01')&choices.rule.eq('highest_probability')].itertuples():
        g=p.loc[p.grid_id.eq(r.grid_id)].sort_values('date');observed=g.gwa_cm.notna().to_numpy();missing=np.flatnonzero(~observed);chunks=np.split(missing,np.flatnonzero(np.diff(missing)>1)+1) if len(missing) else []
        c=cp.loc[cp.grid_id.eq(r.grid_id)&cp.endpoint.eq(r.endpoint)&cp.route.eq(r.route)&cp.date.eq(r.selected_date)]
        n=noise.loc[r.grid_id];noise_inside=q.loc[.25,'innovation_sd']<=n.innovation_sd<=q.loc[.75,'innovation_sd']
        row=dict(grid_id=r.grid_id,route=r.route,selected_date=r.selected_date,full_mask_sha256=__import__('hashlib').sha256(observed.tobytes()).hexdigest(),n_observed=int(observed.sum()),longest_missing=max(map(len,chunks),default=0),innovation_sd=n.innovation_sd,phi=n.phi,true_first_stage_unknown=True)
        if not len(c):row['coverage_class']='cannot_determine_no_selected_candidate'
        else:
            c=c.iloc[0];candidate=pd.Timestamp(c.date);near=g.loc[g.date.between(candidate-pd.DateOffset(months=36),candidate+pd.DateOffset(months=35))];miss=near.gwa_cm.isna().to_numpy();chunks=np.split(np.flatnonzero(miss),np.flatnonzero(np.diff(np.flatnonzero(miss))>1)+1) if miss.any() else []
            row.update(local_pre_observations=c.local_pre_observations,local_post_observations=c.local_post_observations,local_pre_slope=c.local_pre_slope,local_post_slope=c.local_post_slope,local_longest_missing=max(map(len,chunks),default=0))
            slope_inside=pd.notna(c.local_post_slope) and .3<=c.local_post_slope<=1.2
            if c.local_pre_observations<18 or c.local_post_observations<18:row['coverage_class']='partially_covered_observation_support'
            elif not noise_inside or not slope_inside:row['coverage_class']='outside_tested_descriptor_range'
            else:row['coverage_class']='measured_descriptors_within_range_true_onset_unknown'
        rows.append(row)
    pd.DataFrame(rows).to_csv(EVIDENCE/'inference/real_simulation_coverage.csv',index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['wells','coverage'],required=True);p.add_argument('--start',choices=['2005-01-01','2018-01-01'],default='2005-01-01');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.stage=='wells':wells(a.workers,a.start)
    else:coverage()

if __name__=='__main__':main()
