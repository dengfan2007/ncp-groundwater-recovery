"""Bounded reconstruction-setting checks and paired-sample summaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
from ncp_story_analysis import SPEC,parallel_tasks,write_results,monthly_index
from ncp_review_revision import spatial_contrast

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--emd-runtime',required=True);a=p.parse_args()
    e=Path(a.evidence).resolve();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    os.environ['NCP_EMD_RUNTIME']=str(Path(a.emd_runtime).resolve())
    panel=pd.read_csv(ROOT/'data/raw/grid_monthly_panel.csv',parse_dates=['date']);panel=panel.loc[panel.date<=SPEC['common_endpoint']]
    tasks=[]
    configurations=[dict(label='period36',min_period=36),dict(label='period48',min_period=48),dict(label='ensemble100',trials=100),dict(label='seed99',ensemble_seed=99),dict(label='noise010',noise_scale=.1)]
    groups=[(gid,g,g.region_tag.iloc[0],'grid') for gid,g in panel.loc[panel.grid_id.isin(SPEC['representatives'])].groupby('grid_id')]
    for r in ['all','north','south']:
        g=panel if r=='all' else panel.loc[panel.region_tag.eq(r)]
        groups.append(('region_'+r,g.groupby('date',as_index=False).gwa_cm.mean(),r,'regional'))
    for gid,g,r,scope in groups:
        for cfg in configurations:
            tasks.append(dict(grid_id=gid,region_tag=r,scope=scope,endpoint=SPEC['common_endpoint'],experiment=cfg['label'],dates=g.date.astype(str).tolist(),y=g.gwa_cm.tolist(),routes=['D'],save_series=False,**{k:v for k,v in cfg.items() if k!='label'}))
    result=parallel_tasks(tasks,2,'Setting checks');write_results(result,out,'settings')
    choices=pd.read_csv(e/'actual_selections.csv');meta=panel[['grid_id','grid_lon','grid_lat','region_tag']].drop_duplicates('grid_id')
    rows=[]
    for endpoint in [SPEC['common_endpoint'],'2024-09-01']:
        for rule in ['highest_probability','earliest']:
            v=choices.loc[choices.scope.eq('grid')&choices.endpoint.eq(endpoint)&choices.rule.eq(rule)&choices.qualifies]
            common=set(v.loc[v.route.eq('A'),'grid_id'])&set(v.loc[v.route.eq('D'),'grid_id'])
            for sample in ['own','common_AD']:
                for route in ['A','D']:
                    sub=v.loc[v.route.eq(route)].copy()
                    if sample=='common_AD':sub=sub.loc[sub.grid_id.isin(common)]
                    sub=sub.drop(columns='region_tag').merge(meta,on='grid_id');sub['date_month']=sub.selected_date.map(monthly_index)
                    for size in [2.,3.]:rows.append(dict(endpoint=endpoint,rule=rule,sample=sample,route=route,**spatial_contrast(sub,value='date_month',size=size)))
    pd.DataFrame(rows).to_csv(out/'timing_contrasts.csv',index=False,lineterminator='\n')
    # Repeat contrasts after excluding every cell with a non-negligible exported identity discrepancy.
    audit=pd.read_csv(ROOT/'outputs/review_revision/2026-10-15-r1/diagnostics/component_aggregation_audit.csv')
    ids=set(audit.loc[audit.max_abs.le(1e-6),'grid_id']);rows=[]
    for (endpoint,route,rule),sub in choices.loc[choices.scope.eq('grid')&choices.grid_id.isin(ids)&choices.endpoint.isin([SPEC['common_endpoint'],'2024-09-01'])&choices.route.isin(['A','D'])&choices.qualifies].groupby(['endpoint','route','rule']):
        sub=sub.drop(columns='region_tag').merge(meta,on='grid_id');sub['date_month']=sub.selected_date.map(monthly_index)
        rows.append(dict(endpoint=endpoint,route=route,rule=rule,**spatial_contrast(sub,value='date_month',size=2.)))
    pd.DataFrame(rows).to_csv(out/'mask_exclusion_timing.csv',index=False,lineterminator='\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=[Path(__file__),ROOT/'analysis/ncp_story_analysis.py',ROOT/'analysis/ncp_signal_reconstruction.py',ROOT/'analysis/ncp_review_revision.py']
    inputs=[e/'actual_selections.csv',ROOT/'data/raw/grid_monthly_panel.csv',ROOT/'outputs/review_revision/2026-10-15-r1/diagnostics/component_aggregation_audit.csv']
    manifest=dict(configurations=configurations,representatives=SPEC['representatives'],sources={str(p.relative_to(ROOT)):sha(p) for p in sources},inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
