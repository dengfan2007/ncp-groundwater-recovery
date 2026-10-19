"""Same-support, same-configuration land-component timing sensitivity."""
import argparse
import numpy as np
import pandas as pd
from ncp_monitoring_core import ROOT,EVIDENCE,RUNTIME,clean,run_tasks,export_results,json_write,sha
from ncp_monitoring_diagnostics import model_panels

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args();tasks=[]
    for data in model_panels():
        model=data.model.iloc[0]
        if model not in ['shared_noah','era2','era289']:continue
        for gid,g in data.groupby('grid_id'):
            for end in ['2023-11-01','2023-12-01','2024-08-01']:
                s=g.loc[g.date.le(end)]
                tasks.append(clean(dict(task_id=f'{model}_{gid}_{end}',dates=s.date.astype(str).tolist(),y=s.gwa_cm.tolist(),ssa_iterations=1000,save_series=gid in ['g_115.0_35.5','g_116.0_37.5'],
                                       metadata=dict(model=model,grid_id=gid,region_tag=g.region_tag.iloc[0],endpoint=end))))
    results=run_tasks(tasks,RUNTIME/'land_models/checkpoints',a.workers);dest=EVIDENCE/'land_models';export_results(results,dest)
    s=pd.DataFrame([{**r['metadata'],**v} for r in results for v in r['selections']]);pairs=[]
    for model in ['era2','era289']:
        m=s.loc[s.model.eq('shared_noah')].merge(s.loc[s.model.eq(model)],on=['grid_id','endpoint','route','rule'],suffixes=('_noah','_era'))
        x=pd.to_datetime(m.selected_date_noah);y=pd.to_datetime(m.selected_date_era);m['date_difference_months']=(y.dt.year-x.dt.year)*12+y.dt.month-x.dt.month;m['comparison']=model;pairs.append(m)
    pd.concat(pairs).to_csv(dest/'land_model_date_comparisons.csv',index=False)
    summary=[]
    for keys,g in pd.concat(pairs).groupby(['comparison','endpoint','route','rule']):
        assigned=g.date_difference_months.notna();d=g.loc[assigned,'date_difference_months']
        summary.append(dict(zip(['comparison','endpoint','route','rule'],keys))|dict(n_noah=int(g.qualifies_noah.sum()),n_era=int(g.qualifies_era.sum()),n_joint=int(assigned.sum()),within_six=int(d.abs().le(6).sum()),median_absolute_difference=d.abs().median(),maximum_absolute_difference=d.abs().max()))
    pd.DataFrame(summary).to_csv(dest/'land_model_timing_summary.csv',index=False)
    absolute=pd.read_csv(EVIDENCE/'extraction/monthly_anomalies.csv');old=pd.read_csv(ROOT/'outputs/validation_revision/2026-10-17-r2/extraction/uniform_support_panel.csv');m=absolute.merge(old,on=['grid_id','date']);m['support_ratio']=m.support_area_m2_x/m.support_area_m2_y
    audit=m.groupby('grid_id').agg(support_ratio=('support_ratio','first'),tws_maximum_difference=('tws_anomaly',lambda x:0))
    audit['tws_maximum_difference']=m.assign(delta=(m.tws_anomaly-m.twsa_cm).abs()).groupby('grid_id').delta.max();audit['noah_gwa_maximum_difference']=m.assign(delta=(m.noah_gwa_cm-m.gwa_cm).abs()).groupby('grid_id').delta.max()
    audit.to_csv(dest/'common_mask_changes.csv')
    json_write(dest/'manifest.json',dict(tasks=len(tasks),beast_fits=2*len(tasks),source_sha256=sha(__file__),common_support_across_models=True,prior_support_changed_cells=int(audit.support_ratio.lt(.999999).sum()),minimum_support_ratio=audit.support_ratio.min()))

if __name__=='__main__':main()
