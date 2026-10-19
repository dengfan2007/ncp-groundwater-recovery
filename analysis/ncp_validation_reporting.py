"""Promote validation evidence, summarize paired comparisons, and draw validation figures."""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ncp_validation_extension import ROOT,STORY,month

def save(d,p):d.to_csv(p,index=False,lineterminator='\n')
def ci(v):
    v=np.asarray(v,float);rng=np.random.default_rng(20261017)
    if not len(v):return [np.nan,np.nan]
    return np.quantile(np.median(v[rng.integers(len(v),size=(2000,len(v)))],axis=1),[.025,.975])
def month_col(s):
    d=pd.to_datetime(s);return d.dt.year*12+d.dt.month-1
def wilson(k,n):
    p=k/n;z=1.959963984540054;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h

def summarize(e):
    old=pd.read_csv(STORY/'actual_selections.csv');fit=pd.read_csv(e/'fits/actual_selections.csv');uni=pd.read_csv(e/'uniform/uniform_selections.csv');rows=[]
    primary=pd.concat([old.loc[old.scope.eq('grid')&old.endpoint.ne('2024-09-01')],fit.loc[fit.scope.eq('grid')&fit.experiment.eq('original')]],ignore_index=True)
    for route in ['A','D']:
        for rule in ['highest_probability','earliest']:
            for label,before,after in [('Dec_to_Aug',old.loc[old.endpoint.eq('2023-12-01')],fit.loc[fit.experiment.eq('original')]),('Aug_to_missing_Sept',fit.loc[fit.experiment.eq('original')],old.loc[old.endpoint.eq('2024-09-01')]),('archived_to_uniform_Aug',fit.loc[fit.experiment.eq('original')],uni.loc[uni.endpoint.eq('2024-08-01')])]:
                a=before.loc[before.scope.eq('grid')&before.route.eq(route)&before.rule.eq(rule)];b=after.loc[after.scope.eq('grid')&after.route.eq(route)&after.rule.eq(rule)]
                j=a.merge(b,on='grid_id',suffixes=('_a','_b'));paired=j.loc[j.qualifies_a&j.qualifies_b].copy();d=month_col(paired.selected_date_b)-month_col(paired.selected_date_a)
                rows.append(dict(comparison=label,route=route,rule=rule,before_qualified=int(a.qualifies.sum()),after_qualified=int(b.qualifies.sum()),paired=len(paired),within6=int(d.abs().le(6).sum()),median_absolute_revision=float(d.abs().median()),max_absolute_revision=float(d.abs().max())))
    save(pd.DataFrame(rows),e/'endpoint_and_support_summary.csv');save(primary,e/'primary_endpoint_selections.csv')
    routes=[];common=old.loc[old.endpoint.eq('2023-11-01')&old.scope.eq('grid')]
    for a,b in [('A','B'),('B','C'),('C','D'),('A','D')]:
        for rule in ['highest_probability','earliest']:
            x=common.loc[common.route.eq(a)&common.rule.eq(rule)];y=common.loc[common.route.eq(b)&common.rule.eq(rule)];j=x.merge(y,on='grid_id',suffixes=('_a','_b'));p=j.loc[j.qualifies_a&j.qualifies_b];d=month_col(p.selected_date_b)-month_col(p.selected_date_a)
            routes.append(dict(step=a+' to '+b,rule=rule,paired=len(p),median_absolute_shift=d.abs().median(),within6=int(d.abs().le(6).sum()),reportability_changed=int((j.qualifies_a!=j.qualifies_b).sum())))
    save(pd.DataFrame(routes),e/'ablation_step_summary.csv')
    # Same-vintage extraction control separates product retrieval from statistical support.
    original=pd.read_csv(ROOT/'data/raw/grid_monthly_panel.csv');uniform=pd.read_csv(e/'extraction/uniform_support_panel.csv');legacy=pd.read_csv(e/'legacy/legacy_repeat_panel.csv');j=original.merge(legacy,on=['grid_id','date'],suffixes=('_archived','_legacy')).merge(uniform,on=['grid_id','date']);records=[]
    for gid,g in j.groupby('grid_id'):
        original_delta=g.gwa_cm_archived-g.twsa_cm_archived+g.smsa_cm_archived+g.swea_cm_archived+g.cwa_cm_archived
        legacy_delta=g.gwa_cm_legacy-g.twsa_cm_legacy+g.smsa_cm_legacy+g.swea_cm_legacy+g.cwa_cm_legacy
        records.append(dict(grid_id=gid,original_closure_max=original_delta.abs().max(),legacy_closure_max=legacy_delta.abs().max(),uniform_closure_max=g.closure_delta.abs().max(),archived_vs_repeat_gwa_max=(g.gwa_cm_archived-g.gwa_cm_legacy).abs().max(),archived_vs_uniform_gwa_max=(g.gwa_cm_archived-g.gwa_cm).abs().max(),repeat_vs_uniform_gwa_max=(g.gwa_cm_legacy-g.gwa_cm).abs().max()))
    save(pd.DataFrame(records),e/'extraction_comparison.csv')
    truth=pd.read_csv(e/'simulation/simulation_truth.csv');sim=pd.read_csv(e/'simulation/simulation_selections.csv').merge(truth,on='grid_id');sim['error']=month_col(sim.selected_date)-month_col(sim.true_onset);records=[];pairs=[]
    for keys,g in sim.groupby(['setting','scenario','route','rule']):
        q=g.loc[g.qualifies];err=q.error.dropna().to_numpy();lo,hi=ci(err);wl,wh=wilson(len(q),len(g))
        records.append(dict(zip(['setting','scenario','route','rule'],keys),n=len(g),qualified=len(q),rate=len(q)/len(g),rate_low=wl,rate_high=wh,median_error=np.median(err) if len(err) else np.nan,median_low=lo,median_high=hi,mean_absolute_error=np.mean(abs(err)) if len(err) else np.nan))
    for keys,g in sim.groupby(['setting','scenario','rule']):
        a=g.loc[g.route.eq('A')&g.qualifies];d=g.loc[g.route.eq('D')&g.qualifies];p=a.merge(d,on='replicate',suffixes=('_a','_d'));v=abs(p.error_d)-abs(p.error_a);v=v.dropna().to_numpy();lo,hi=ci(v)
        pairs.append(dict(zip(['setting','scenario','rule'],keys),paired=len(v),median_D_minus_A_absolute_error=np.median(v) if len(v) else np.nan,low=lo,high=hi))
    save(pd.DataFrame(records),e/'calibrated_simulation_summary.csv');save(pd.DataFrame(pairs),e/'calibrated_simulation_paired_errors.csv')
    seeds=fit.loc[fit.experiment.str.startswith('seed_')&fit.rule.eq('highest_probability')];rows=[]
    for (gid,route),g in seeds.groupby(['grid_id','route']):
        v=month_col(g.loc[g.qualifies,'selected_date']);rows.append(dict(grid_id=gid,route=route,seeds=len(g),qualified=int(g.qualifies.sum()),date_range_months=float(v.max()-v.min()) if len(v) else np.nan))
    save(pd.DataFrame(rows),e/'seed_repeatability.csv')

def figures(e,out):
    out.mkdir(exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
    primary=pd.read_csv(e/'primary_endpoint_selections.csv');primary=primary.loc[primary.endpoint.ne('2023-11-01')]
    dates=sorted(primary.endpoint.unique());fig,axs=plt.subplots(1,2,figsize=(11.5,4.2))
    for route,color in [('A','#2166ac'),('D','#b2182b')]:
        for rule,style in [('highest_probability','-'),('earliest','--')]:
            counts=[];diff=[]
            for endpoint in dates:
                g=primary.loc[primary.endpoint.eq(endpoint)&primary.route.eq(route)&primary.rule.eq(rule)&primary.qualifies].copy();g['m']=month_col(g.selected_date);counts.append(len(g));n=g.loc[g.region_tag.eq('north'),'m'];s=g.loc[g.region_tag.eq('south'),'m'];diff.append(s.mean()-n.mean() if len(n)>=2 and len(s)>=2 else np.nan)
            if rule=='highest_probability':axs[0].plot(range(6),counts,'o-',color=color,label=route)
            axs[1].plot(range(6),diff,'o'+style,color=color,label=f'{route}: '+('highest' if style=='-' else 'earliest'))
    for ax in axs:ax.set_xticks(range(6),['2019','2020','2021','2022','2023','2024*']);ax.set_xlabel('Record endpoint')
    axs[0].set_ylabel('Qualifying cells');axs[0].set_title('(a) Reporting extent');axs[0].legend(frameon=False);axs[1].axhline(0,color='gray',lw=.7);axs[1].set_ylabel('South minus north (months)');axs[1].set_title('(b) Mean selected-date contrast');axs[1].legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(out/'validation_03_endpoint_stability.png');plt.close(fig)
    # Matched wells: uniform components, archived storage proxy closely corroborated by re-extraction.
    w=pd.read_csv(e/'uniform/well_weighted_series.csv',parse_dates=['date']);c=pd.read_csv(e/'uniform/well_multiscale_correlations.csv');fig,axs=plt.subplots(3,1,figsize=(9.5,8),gridspec_kw={'height_ratios':[1,1,.8]})
    for ax,region in zip(axs[:2],['north','south']):
        for aq,color in [('unconfined','#2166ac'),('confined','#b2182b')]:
            g=w.loc[w.support.eq('common18')&w.weighting.eq('equal_cell')&w.region.eq(region)&w.aquifer.eq(aq)].sort_values('date');y=g.well;ax.plot(g.date,(y-y.mean())/y.std(),color=color,lw=1,label=aq)
        y=g.gwa_cm;ax.plot(g.date,(y-y.mean())/y.std(),color='black',lw=1,label='GWA');ax.set_title(region.capitalize()+': matched fixed cohorts');ax.set_ylabel('Standardized anomaly');ax.legend(frameon=False,ncol=3,fontsize=8)
    for region,color,shift in [('north','#2166ac',-.06),('south','#b2182b',.06)]:
        for i,component in enumerate(['twsa_cm','smsa_cm','gwa_cm']):
            r=c.loc[c.support.eq('common18')&c.weighting.eq('equal_cell')&c.aquifer.eq('unconfined')&c.region.eq(region)&c.component.eq(component)&c.aggregation_months.eq(1)&c.treatment.eq('difference')].iloc[0];axs[2].errorbar(i+shift,r.r,yerr=[[r.r-r.low],[r.high-r.r]],fmt='o',color=color,capsize=2)
    axs[2].set_xticks(range(3),['TWSA','SMSA (subtracted)','GWA']);axs[2].axhline(0,color='gray',ls=':',lw=.8);axs[2].set_ylabel('Monthly-change correlation');fig.tight_layout();fig.savefig(out/'validation_06_observation_support.png');plt.close(fig)
    d=pd.read_csv(e/'extraction_comparison.csv');fig,axs=plt.subplots(1,2,figsize=(11,4));bad=d.loc[d.original_closure_max.gt(1e-6)]
    x=np.arange(len(bad));axs[0].bar(x,bad.original_closure_max,color='#777777',label='Archived components');axs[0].plot(x,bad.uniform_closure_max,'o',color='#2166ac',label='Uniform support');axs[0].set_xticks(x,bad.grid_id.str.removeprefix('g_'),rotation=30);axs[0].set_ylabel('Maximum closure error (cm)');axs[0].legend(frameon=False,fontsize=8)
    u=pd.read_csv(e/'endpoint_and_support_summary.csv');u=u.loc[u.rule.eq('highest_probability')&u.comparison.isin(['Dec_to_Aug','Aug_to_missing_Sept'])];labels=[]
    for i,r in enumerate(u.itertuples()):axs[1].bar(i,r.median_absolute_revision,color='#2166ac' if r.route=='A' else '#b2182b');labels.append(r.route+'\n'+('Dec to Aug' if r.comparison=='Dec_to_Aug' else 'Aug to Sep'))
    axs[1].set_xticks(range(len(labels)),labels);axs[1].set_ylabel('Median absolute date revision (months)');fig.tight_layout();fig.savefig(out/'validation_s1_closure_endpoint.png');plt.close(fig)
    s=pd.read_csv(e/'calibrated_simulation_summary.csv');s=s.loc[s.setting.isin(['low','median','high'])&s.rule.eq('highest_probability')];fig,axs=plt.subplots(1,2,figsize=(11,4.3))
    for route,color in [('A','#2166ac'),('D','#b2182b')]:
        for setting,marker,offset in [('low','o',-.12),('median','s',0),('high','^',.12)]:
            for i,scenario in enumerate(['no_recovery','temporary_gain','sustained_reversal','reversal_then_acceleration']):
                r=s.loc[s.route.eq(route)&s.setting.eq(setting)&s.scenario.eq(scenario)].iloc[0];xx=i+offset+(-.04 if route=='A' else .04);axs[0].plot(xx,r.rate,marker,color=color,ms=5,label=f'{route} {setting}' if i==0 else None)
            r=s.loc[s.route.eq(route)&s.setting.eq(setting)&s.scenario.eq('reversal_then_acceleration')].iloc[0];xx=['low','median','high'].index(setting)+(-.06 if route=='A' else .06);axs[1].errorbar(xx,r.median_error,yerr=[[r.median_error-r.median_low],[r.median_high-r.median_error]],fmt=marker,color=color,capsize=3)
    axs[0].set_xticks(range(4),['Decline','Temporary\ngain','Sustained\nreversal','Two-stage']);axs[0].set_ylabel('Qualifying fraction');axs[0].legend(frameon=False,fontsize=7,ncol=2);axs[1].set_xticks(range(3),['Low','Median','High']);axs[1].axhline(36,color='gray',ls=':',label='Later acceleration');axs[1].set_ylabel('Median delay from first reversal (months)');axs[1].set_xlabel('Empirical residual regime');fig.tight_layout();fig.savefig(out/'validation_s2_calibrated_simulation.png');plt.close(fig)
    p=pd.read_csv(e/'fits/actual_posterior.csv',parse_dates=['date']);fig,axs=plt.subplots(2,2,figsize=(11,6),sharex=True)
    for row,gid in enumerate(['g_115.0_35.5','g_116.0_37.5']):
        for col,route in enumerate(['A','D']):
            ax=axs[row,col]
            for end,style,color in [('2024-08-01','-','#2166ac'),('2024-09-01','--','#b2182b')]:
                z=p.loc[p.grid_id.eq(gid)&p.route.eq(route)&p.endpoint.eq(end)];z=z.loc[z.date.ge('2014-01-01')];ax.plot(z.date,z.cp_occurrence_probability,style,color=color,lw=1,label=end[:7])
            ax.set_title(gid+' / '+route);ax.set_ylim(0,1);ax.legend(frameon=False,fontsize=8);ax.set_ylabel('Change occurrence probability')
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(out/'validation_s3_posterior.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,9),sharey=True);ids=sorted(primary.grid_id.unique())
    for ax,route in zip(axs,['A','D']):
        z=primary.loc[primary.route.eq(route)&primary.rule.eq('highest_probability')].copy();z['year']=pd.to_datetime(z.selected_date).dt.year+(pd.to_datetime(z.selected_date).dt.month-1)/12
        matrix=z.pivot(index='grid_id',columns='endpoint',values='year').reindex(index=ids,columns=dates);im=ax.imshow(matrix.to_numpy(),aspect='auto',vmin=2015,vmax=2023,cmap='viridis');ax.set_xticks(range(6),['2019','2020','2021','2022','2023','Aug24'],rotation=45);ax.set_title('Route '+route);ax.set_yticks(range(27),[x.removeprefix('g_') for x in ids],fontsize=8)
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.6,label='Selected year');fig.savefig(out/'validation_s4_date_heatmap.png',bbox_inches='tight');plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();e=Path(a.output_dir).resolve();e.mkdir(parents=True,exist_ok=False)
    folders={'local':'2026-10-17-validation-local-r2','fits':'2026-10-17-validation-fits-r1','simulation':'2026-10-17-validation-simulation-r1','reference':'2026-10-17-validation-reference-r1','uniform':'2026-10-17-validation-uniform-r1','extraction':'2026-10-17-validation-ee-r1','legacy':'2026-10-17-validation-ee-legacy-r1'}
    promoted=[]
    for name,folder in folders.items():
        src=ROOT/'outputs/runs'/folder;dest=e/name;dest.mkdir()
        assert (src/'manifest.json').exists(),str(src)
        for f in src.iterdir():
            if f.suffix not in ['.csv','.json'] or f.name.startswith('year_') or f.name=='temporal_trend_draws.csv':continue
            shutil.copy2(f,dest/f.name);promoted.append(dict(path=str((dest/f.name).relative_to(ROOT)).replace('\\','/'),origin=str(f.relative_to(ROOT)).replace('\\','/'),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    snaps=e/'source_snapshots';snaps.mkdir();shutil.copy2(ROOT/'outputs/runs/2026-10-17-validation-source-r1/ncp_validation_extension_v1.py',snaps/'ncp_validation_extension_v1.py')
    summarize(e);figures(e,e/'figures')
    (e/'manifest.json').write_text(json.dumps(dict(promoted_files=promoted,main_endpoint='2024-08-01',last_missing_month_sensitivity='2024-09-01',archived_inputs_preserved=True),indent=2)+'\n');print('Promoted',len(promoted),'files',flush=True)
if __name__=='__main__':main()
