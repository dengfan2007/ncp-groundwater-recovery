"""Frozen rule search, held-out validation and retrospective real-data application."""
from __future__ import annotations
import argparse
from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import norm
from ncp_monitoring_core import EVIDENCE,VALIDATION,ENDPOINTS,frozen_actual,seed,json_write,panel

PROBABILITIES=[.5,.7,.9,.95,.99]
GAINS=[.5,1.,1.5,2.]

def wilson(k,n,one_sided=False):
    z=norm.ppf(.95 if one_sided else .975);p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return float(center-half),float(center+half)

def choose(candidates,probability,gain,rule='highest_probability'):
    d=candidates.loc[candidates.date.ge('2015-01-01')&candidates.probability.ge(probability)&candidates.post_slope.gt(0)&(candidates.post_slope-candidates.pre_slope).gt(gain)].copy()
    if rule=='highest_probability':d=d.sort_values(['task_id','probability','date'],ascending=[True,False,False])
    else:d=d.sort_values(['task_id','date'])
    return d.drop_duplicates('task_id')[['task_id','date']].rename(columns={'date':'selected_date'})

def month_number(values):
    d=pd.to_datetime(values);return d.dt.year*12+d.dt.month

def evaluate(candidates,truth,probability,gain,rule='highest_probability'):
    chosen=choose(candidates,probability,gain,rule);d=truth.merge(chosen,on='task_id',how='left',validate='one_to_one')
    d['qualifies']=d.selected_date.notna();d['error_months']=month_number(d.selected_date)-month_number(d.onset)
    d['onset_hit']=d.error_months.abs().le(12);d['acceleration_hit']=False
    if 'acceleration' in d:d['acceleration_hit']=(month_number(d.selected_date)-month_number(d.acceleration)).abs().le(12)
    return d

def summarize_trials(d,keys):
    rows=[]
    for values,g in d.groupby(keys,dropna=False):
        if not isinstance(values,tuple):values=(values,)
        n=len(g);k=int(g.qualifies.sum());hit=int(g.onset_hit.sum());lo,hi=wilson(k,n);hlo,hhi=wilson(hit,n)
        rows.append({**dict(zip(keys,values)),'n':n,'qualified':k,'fraction':k/n,'low':lo,'high':hi,'upper_onesided95':wilson(k,n,True)[1],
                     'onset_hits':hit,'onset_hit_rate':hit/n,'onset_low':hlo,'onset_high':hhi,'median_error_months':g.error_months.median(),
                     'median_absolute_error_months':g.error_months.abs().median(),'acceleration_hits':int(g.acceleration_hit.sum()),'acceleration_hit_rate':g.acceleration_hit.mean()})
    return pd.DataFrame(rows)

def calibrate():
    folder=EVIDENCE/'calibration';truth=pd.read_csv(folder/'truth.csv');candidates=pd.read_csv(folder/'candidates.csv')
    a=set(truth.loc[truth.split.eq('calibration'),'noise_seed']);b=set(truth.loc[truth.split.eq('validation'),'noise_seed']);assert a.isdisjoint(b)
    for field in ['noise_seed','beast_seed','ensemble_seed']:
        assert set(truth.loc[truth.split.eq('calibration'),field]).isdisjoint(set(truth.loc[truth.split.eq('validation'),field])),field
    out=EVIDENCE/'inference';out.mkdir(parents=True,exist_ok=False);grid=[];rules={};validation=[];trial_rows=[];original_validation=[]
    for route in ['A','D']:
        cc=candidates.loc[candidates.route.eq(route)];cal=truth.loc[truth.split.eq('calibration')];scores=[]
        original=summarize_trials(evaluate(cc,truth.loc[truth.split.eq('validation')],.5,.5),['scenario','regime']);original['route']=route;original_validation.append(original)
        for probability,gain in product(PROBABILITIES,GAINS):
            d=evaluate(cc,cal,probability,gain);s=summarize_trials(d,['scenario','regime']);s['route']=route;s['probability_threshold']=probability;s['gain_threshold']=gain;grid.append(s)
            null=s.loc[s.scenario.ne('sustained_reversal')];positive=s.loc[s.scenario.eq('sustained_reversal')]
            feasible=bool(null.upper_onesided95.le(.05).all())
            scores.append(dict(probability=probability,gain=gain,calibration_feasible=feasible,
                               minimum_onset_hit_rate=positive.onset_hit_rate.min(),minimum_candidate_detection=positive.fraction.min(),
                               pooled_median_absolute_error=d.loc[d.scenario.eq('sustained_reversal'),'error_months'].abs().median()))
        feasible=[x for x in scores if x['calibration_feasible']]
        if not feasible:rules[route]=dict(status='no_feasible_calibration_rule',searched_settings=20);continue
        def order(x):
            err=x['pooled_median_absolute_error'];err=err if np.isfinite(err) else float('inf')
            return (-x['minimum_onset_hit_rate'],-x['minimum_candidate_detection'],err,x['probability'],x['gain'])
        winner=sorted(feasible,key=order)[0];held=evaluate(cc,truth.loc[truth.split.eq('validation')],winner['probability'],winner['gain']);s=summarize_trials(held,['scenario','regime']);s['route']=route;validation.append(s)
        passed=bool(s.loc[s.scenario.ne('sustained_reversal'),'upper_onesided95'].le(.05).all())
        rules[route]={**winner,'status':'heldout_error_control_passed' if passed else 'heldout_error_control_failed','minimum_heldout_onset_hit_rate':float(s.loc[s.scenario.eq('sustained_reversal'),'onset_hit_rate'].min())}
        for rule in ['highest_probability','earliest']:
            d=evaluate(cc,truth,winner['probability'],winner['gain'],rule);d['route']=route;d['rule']=rule;trial_rows.append(d)
    pd.concat(grid).to_csv(out/'calibration_search.csv',index=False)
    pd.concat(original_validation).to_csv(out/'original_rule_validation.csv',index=False)
    if validation:pd.concat(validation).to_csv(out/'heldout_validation.csv',index=False)
    if trial_rows:pd.concat(trial_rows).to_csv(out/'selected_rule_trials.csv',index=False)
    json_write(out/'rules.json',rules);json_write(out/'seed_audit.json',dict(calibration_validation_noise_disjoint=True,algorithm_seeds_disjoint=True,n_calibration=len(a),n_validation=len(b),maximum_false_reports_n500=16))
    capped=[]
    for r in __import__('json').loads((folder/'diagnostics.json').read_text()):
        if r['diagnostics'].get('ssa',{}).get('converged') is False:capped.append({'task_id':r['task_id'],**r['metadata'],**r['diagnostics']['ssa']})
    pd.DataFrame(capped).to_csv(out/'ssa_capped_calibration_tasks.csv',index=False)
    print(rules,flush=True)

def apply_real():
    rules=__import__('json').loads((EVIDENCE/'inference/rules.json').read_text());c=frozen_actual();rows=[]
    for route in ['A','D']:
        data=c.loc[c.route.eq(route)].copy();data['task_id']=data.grid_id+'_'+data.endpoint
        truth=data[['task_id','grid_id','region_tag','endpoint']].drop_duplicates('task_id')
        settings=[('original',.5,.5)]
        if rules[route]['status']=='heldout_error_control_passed':settings.append(('calibrated',rules[route]['probability'],rules[route]['gain']))
        for label,probability,gain in settings:
            for rule in ['highest_probability','earliest']:
                d=truth.merge(choose(data,probability,gain,rule),on='task_id',how='left');d['qualifies']=d.selected_date.notna();d['route']=route;d['rule']=rule;d['setting']=label
                d['reason']=np.where(d.qualifies,'assigned','no_qualifying_candidate');rows.append(d)
    out=EVIDENCE/'inference';all_dates=pd.concat(rows);all_dates.to_csv(out/'real_selections.csv',index=False)
    counts=all_dates.groupby(['route','rule','setting','endpoint']).agg(reportable=('qualifies','sum'),total=('qualifies','size')).reset_index()
    regional=all_dates.loc[all_dates.qualifies].groupby(['route','rule','setting','endpoint','region_tag']).size().unstack(fill_value=0).reset_index();counts=counts.merge(regional,how='left').fillna(0);counts.to_csv(out/'real_reportability.csv',index=False)
    differences=[];stability=[]
    for (route,rule),d in all_dates.groupby(['route','rule']):
        old=d.loc[d.setting.eq('original')];new=d.loc[d.setting.eq('calibrated')]
        if len(new):
            match=old.merge(new,on=['grid_id','endpoint'],suffixes=('_original','_calibrated'));match['revision_months']=month_number(match.selected_date_calibrated)-month_number(match.selected_date_original)
            differences.append(match)
        for setting,s in d.groupby('setting'):
            for before,after in zip(ENDPOINTS[:-1],ENDPOINTS[1:]):
                a=s.loc[s.endpoint.eq(before)].set_index('grid_id');b=s.loc[s.endpoint.eq(after)].set_index('grid_id');common=a.index[a.qualifies&b.qualifies]
                delta=month_number(b.loc[common,'selected_date'])-month_number(a.loc[common,'selected_date'])
                stable=dict(route=route,rule=rule,setting=setting,before=before,after=after,common_reportable=len(common),within_six=int(delta.abs().le(6).sum()),gained=int((~a.qualifies&b.qualifies).sum()),lost=int((a.qualifies&~b.qualifies).sum()),unassigned_both=int((~a.qualifies&~b.qualifies).sum()))
                if len(new):
                    other=d.loc[d.setting.eq('calibrated' if setting=='original' else 'original')];oa=other.loc[other.endpoint.eq(before)].set_index('grid_id');ob=other.loc[other.endpoint.eq(after)].set_index('grid_id');shared=common.intersection(oa.index[oa.qualifies&ob.qualifies])
                    dd=month_number(b.loc[shared,'selected_date'])-month_number(a.loc[shared,'selected_date']);stable.update(shared_all_rules=len(shared),shared_within_six=int(dd.abs().le(6).sum()))
                stability.append(stable)
    if differences:pd.concat(differences).to_csv(out/'original_calibrated_date_differences.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'real_endpoint_stability.csv',index=False)

def summarize_two_stage():
    import json
    folder=EVIDENCE/'two_stage';truth=pd.read_csv(folder/'truth.csv');c=pd.read_csv(folder/'candidates.csv');rules=json.loads((EVIDENCE/'inference/rules.json').read_text());summaries=[];trials=[]
    for route in ['A','D']:
        settings=[('original',.5,.5)]
        if rules[route]['status']=='heldout_error_control_passed':settings.append(('calibrated',rules[route]['probability'],rules[route]['gain']))
        for setting,p,g in settings:
            for rule in ['highest_probability','earliest']:
                d=evaluate(c.loc[c.route.eq(route)],truth,p,g,rule);d['route']=route;d['setting']=setting;d['rule']=rule;trials.append(d)
                s=summarize_trials(d,['regime','onset','weak_slope','mask']);s['route']=route;s['setting']=setting;s['rule']=rule;summaries.append(s)
    out=EVIDENCE/'inference';data=pd.concat(trials);data.to_csv(out/'two_stage_trials.csv',index=False);summary=pd.concat(summaries)
    medlow=[];medhigh=[]
    for row in summary.itertuples():
        q=data.loc[data.route.eq(row.route)&data.setting.eq(row.setting)&data.rule.eq(row.rule)&data.regime.eq(row.regime)&data.onset.eq(row.onset)&data.weak_slope.eq(row.weak_slope)&data['mask'].eq(row.mask)].error_months.dropna().to_numpy()
        if len(q):
            rng=np.random.default_rng(seed('median_ci',row.route,row.setting,row.rule,row.regime,row.onset,row.weak_slope,row.mask));b=np.median(q[rng.integers(len(q),size=(2000,len(q)))],axis=1);lo,hi=np.quantile(b,[.025,.975])
        else:lo=hi=np.nan
        medlow.append(lo);medhigh.append(hi)
    summary['median_error_low']=medlow;summary['median_error_high']=medhigh;summary['late_acceleration_short_followup']=summary.onset.eq('2020-01-01');summary.to_csv(out/'two_stage_summary.csv',index=False)
    differences=[]
    for keys,g in data.groupby(['route','setting','rule','regime','onset','weak_slope']):
        m=g.loc[g['mask'].eq('observed')].merge(g.loc[g['mask'].eq('complete')],on='replicate',suffixes=('_observed','_complete'),validate='one_to_one');rng=np.random.default_rng(seed('paired_gap',*keys));ix=rng.integers(len(m),size=(2000,len(m)))
        row=dict(zip(['route','setting','rule','regime','onset','weak_slope'],keys));row['n_paired_realizations']=len(m)
        for metric in ['qualifies','onset_hit','acceleration_hit']:
            delta=m[metric+'_observed'].to_numpy(dtype=float)-m[metric+'_complete'].to_numpy(dtype=float);b=delta[ix].mean(axis=1);lo,hi=np.quantile(b,[.025,.975]);row.update({metric+'_difference':delta.mean(),metric+'_low':lo,metric+'_high':hi})
        both=m.error_months_observed.notna()&m.error_months_complete.notna();delta=(m.loc[both,'error_months_observed']-m.loc[both,'error_months_complete']).to_numpy();row['both_reportable']=int(both.sum())
        if len(delta):
            b=np.median(delta[rng.integers(len(delta),size=(2000,len(delta)))],axis=1);lo,hi=np.quantile(b,[.025,.975]);row.update(median_date_difference=np.median(delta),median_date_difference_low=lo,median_date_difference_high=hi)
        differences.append(row)
    pd.DataFrame(differences).to_csv(out/'paired_gap_effects.csv',index=False)

def endpoint_intervals():
    """Compute date intervals in the analysis stage, before render-only plotting."""
    from ncp_monitoring_diagnostics import spatial_weights,quantiles
    d=pd.read_csv(EVIDENCE/'inference/real_selections.csv',parse_dates=['endpoint','selected_date'])
    meta=panel()[['grid_id','grid_lon','grid_lat','region_tag']].drop_duplicates('grid_id').set_index('grid_id');rows=[]
    d=d.loc[d.setting.eq('original')&d.rule.eq('highest_probability')&d.qualifies]
    for (route,end),g in d.groupby(['route','endpoint']):
        mm=meta.loc[g.grid_id];south=mm.region_tag.eq('south').to_numpy();n=int((~south).sum());s=int(south.sum())
        if n<2 or s<2:continue
        m=(g.selected_date.dt.year*12+g.selected_date.dt.month).to_numpy();w,valid,k=spatial_weights(mm,3,5000,np.random.default_rng(seed('figure_endpoint',route,str(end))));lo,hi=quantiles((w*m).sum(axis=1)[valid])
        rows.append(dict(route=route,endpoint=str(end.date()),n_north=n,n_south=s,estimate=m[south].mean()-m[~south].mean(),low=lo,high=hi,block_degrees=3,n_blocks=k,valid_replicates=int(valid.sum())))
    pd.DataFrame(rows).to_csv(EVIDENCE/'inference/endpoint_date_contrasts.csv',index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['calibrate','apply','two_stage','endpoint_intervals'],required=True);a=p.parse_args()
    if a.stage=='calibrate':calibrate();apply_real()
    elif a.stage=='apply':apply_real()
    elif a.stage=='two_stage':summarize_two_stage()
    else:endpoint_intervals()

if __name__=='__main__':main()
