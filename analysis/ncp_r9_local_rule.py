"""r9 check 3: local negative-to-positive slope rule on the archived simulation pools.

No BEAST refits.  Candidate pools from the formal October 18 experiments already contain
36-month local slopes (each side requires >=18 real observations).  Thresholds are searched
on the calibration split only and checked once on the held-out split, as for the original
rule.  The same feasibility criterion (one-sided 95% Wilson upper bound <=0.05 in all six
null scenario-residual combinations) is used.
"""
from __future__ import annotations
import argparse, json
from itertools import product
import numpy as np
import pandas as pd
from ncp_monitoring_core import ROOT,EVIDENCE,frozen_actual,json_write,seed
from ncp_monitoring_inference import wilson,month_number,summarize_trials,evaluate as evaluate_original

PROBABILITIES=[.5,.7,.9,.95,.99]
GAINS=[0.,1.,2.,4.,6.]   # local post-minus-pre slope gain (cm/yr); 0 = sign change only

def choose_local(c,probability,gain,rule):
    ok=(c.date.ge('2015-01-01')&c.probability.ge(probability)&c.local_pre_slope.lt(0)&c.local_post_slope.gt(0)
        &(c.local_post_slope-c.local_pre_slope).gt(gain))
    d=c.loc[ok].copy()
    d=d.sort_values(['task_id','date']) if rule=='earliest' else d.sort_values(['task_id','probability','date'],ascending=[True,False,False])
    return d.drop_duplicates('task_id')[['task_id','date']].rename(columns={'date':'selected_date'})

def evaluate(c,truth,probability,gain,rule):
    d=truth.merge(choose_local(c,probability,gain,rule),on='task_id',how='left',validate='one_to_one')
    d['qualifies']=d.selected_date.notna();d['error_months']=month_number(d.selected_date)-month_number(d.onset)
    d['onset_hit']=d.error_months.abs().le(12)
    d['acceleration_hit']=(month_number(d.selected_date)-month_number(d.acceleration)).abs().le(12) if 'acceleration' in d else False
    return d

def calibration(out):
    folder=EVIDENCE/'calibration';truth=pd.read_csv(folder/'truth.csv');cand=pd.read_csv(folder/'candidates.csv')
    grid=[];feasible=[];held=[]
    for route,rule in product(['A','D'],['earliest','highest_probability']):
        cc=cand.loc[cand.route.eq(route)]
        for p,g in product(PROBABILITIES,GAINS):
            for split in ['calibration','validation']:
                s=summarize_trials(evaluate(cc,truth.loc[truth.split.eq(split)],p,g,rule),['scenario','regime'])
                s=s.assign(route=route,rule=rule,probability_threshold=p,gain_threshold=g,split=split);grid.append(s)
                if split=='calibration':
                    null=s.loc[s.scenario.ne('sustained_reversal')];pos=s.loc[s.scenario.eq('sustained_reversal')]
                    feasible.append(dict(route=route,rule=rule,probability=p,gain=g,feasible=bool(null.upper_onesided95.le(.05).all()),
                        max_null_fraction=float(null.fraction.max()),max_null_upper=float(null.upper_onesided95.max()),
                        min_onset_hit=float(pos.onset_hit_rate.min()),min_detection=float(pos.fraction.min())))
    grid=pd.concat(grid);grid.to_csv(out/'local_rule_grid.csv',index=False)
    f=pd.DataFrame(feasible);f.to_csv(out/'local_rule_feasibility.csv',index=False)
    chosen={}
    for (route,rule),g in f.groupby(['route','rule']):
        ok=g.loc[g.feasible].sort_values(['min_onset_hit','min_detection','probability','gain'],ascending=[False,False,True,True])
        if len(ok):
            w=ok.iloc[0];v=grid.loc[grid.route.eq(route)&grid.rule.eq(rule)&grid.split.eq('validation')&grid.probability_threshold.eq(w.probability)&grid.gain_threshold.eq(w.gain)]
            chosen[f'{route}_{rule}']=dict(status='feasible',probability=float(w.probability),gain=float(w.gain),calibration_min_onset_hit=float(w.min_onset_hit),
                heldout_max_null_fraction=float(v.loc[v.scenario.ne('sustained_reversal'),'fraction'].max()),
                heldout_max_null_upper=float(v.loc[v.scenario.ne('sustained_reversal'),'upper_onesided95'].max()),
                heldout_null_passed=bool(v.loc[v.scenario.ne('sustained_reversal'),'upper_onesided95'].le(.05).all()),
                heldout_min_onset_hit=float(v.loc[v.scenario.eq('sustained_reversal'),'onset_hit_rate'].min()),
                heldout_min_detection=float(v.loc[v.scenario.eq('sustained_reversal'),'fraction'].min()))
        else:
            b=g.sort_values('max_null_upper').iloc[0]
            chosen[f'{route}_{rule}']=dict(status='no_feasible_setting',lowest_null_upper=float(b.max_null_upper),at_probability=float(b.probability),at_gain=float(b.gain))
    return chosen,grid

def two_stage(out,settings):
    folder=EVIDENCE/'two_stage';truth=pd.read_csv(folder/'truth.csv');cand=pd.read_csv(folder/'candidates.csv');rows=[];diffs=[]
    for route in ['A','D']:
        cc=cand.loc[cand.route.eq(route)]
        for name,(p,g,rule,local) in settings.items():
            d=evaluate(cc,truth,p,g,rule) if local else evaluate_original(cc,truth,p,g,rule)
            s=summarize_trials(d,['regime','onset','weak_slope','mask']).assign(route=route,setting=name);rows.append(s)
            for keys,x in d.loc[d['mask'].eq('observed')&d.regime.eq('median')].groupby(['onset','weak_slope']):
                a=x.acceleration_hit.to_numpy(float)-x.onset_hit.to_numpy(float);rng=np.random.default_rng(seed('r9_local_stage',route,name,*keys))
                b=a[rng.integers(len(a),size=(2000,len(a)))].mean(axis=1);lo,hi=np.quantile(b,[.025,.975])
                diffs.append(dict(route=route,setting=name,onset=keys[0],weak_slope=keys[1],first_hit=x.onset_hit.mean(),accel_hit=x.acceleration_hit.mean(),
                                  qualified=x.qualifies.mean(),difference=a.mean(),low=lo,high=hi,median_error=x.error_months.median()))
    pd.concat(rows).to_csv(out/'two_stage_local_summary.csv',index=False);d=pd.DataFrame(diffs);d.to_csv(out/'two_stage_local_stage_hits.csv',index=False);return d

def real(out,settings):
    # The baseline rerun reproduces every frozen date (acceptance gate) and, unlike the frozen
    # pre-2024 pools, stores local slopes at every endpoint.
    c=pd.read_csv(EVIDENCE/'baseline/candidates.csv');c['task_id']=c.grid_id+'_'+c.endpoint;rows=[]
    truth=c[['task_id','grid_id','region_tag','endpoint']].drop_duplicates('task_id')
    for route in ['A','D']:
        for name,(p,g,rule,local) in settings.items():
            if not local:continue
            ch=choose_local(c.loc[c.route.eq(route)],p,g,rule);d=truth.merge(ch,on='task_id',how='left');d['route']=route;d['setting']=name;rows.append(d)
    d=pd.concat(rows);d.to_csv(out/'real_local_selections.csv',index=False);st=[]
    for (route,name),g in d.groupby(['route','setting']):
        a=g.loc[g.endpoint.eq('2023-12-01')].set_index('grid_id').selected_date;b=g.loc[g.endpoint.eq('2024-08-01')].set_index('grid_id').selected_date.reindex(a.index)
        ok=a.notna()&b.notna();dd=(month_number(b[ok])-month_number(a[ok])).abs()
        st.append(dict(route=route,setting=name,dated_dec2023=int(a.notna().sum()),dated_aug2024=int(b.notna().sum()),paired=int(ok.sum()),within6=int((dd<=6).sum()),
                       median_date_aug2024=str(pd.to_datetime(b.dropna()).sort_values().iloc[len(b.dropna())//2].date()) if b.notna().any() else None))
    s=pd.DataFrame(st);s.to_csv(out/'real_local_stability.csv',index=False);return s

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--label',required=True);args=a.parse_args()
    out=ROOT/'outputs/r9_checks'/args.label/'local_rule';out.mkdir(parents=True,exist_ok=False)
    chosen,_=calibration(out)
    settings={'original_H':(.5,.5,'highest_probability',False),'original_E':(.5,.5,'earliest',False),
              'local_E_p050':(.5,0.,'earliest',True),'local_H_p050':(.5,0.,'highest_probability',True)}
    for k,v in chosen.items():
        if v['status']=='feasible':route,rule=k.split('_',1);settings[f'local_{"E" if rule=="earliest" else "H"}_calibrated_{route}']=(v['probability'],v['gain'],rule,True)
    stage=two_stage(out,settings);realst=real(out,settings)
    json_write(out/'manifest.json',dict(probabilities=PROBABILITIES,gains=GAINS,chosen=chosen,settings={k:list(v) for k,v in settings.items()},
               source_sha256=__import__('ncp_monitoring_core').sha(__file__)))
    print(json.dumps(chosen,indent=1));print(realst.to_string(index=False))

if __name__=='__main__':main()
