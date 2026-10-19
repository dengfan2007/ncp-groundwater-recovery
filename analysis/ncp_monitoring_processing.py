"""Save complete November-2023 traces for all cells, without replacing old fits."""
from ncp_monitoring_core import panel,clean,run_tasks,export_results,EVIDENCE,RUNTIME

def main():
    p=panel();tasks=[]
    for gid,g in p.loc[p.date.le('2023-11-01')].groupby('grid_id'):
        tasks.append(clean(dict(task_id=gid,dates=g.date.astype(str).tolist(),y=g.gwa_cm.tolist(),ssa_iterations=200,save_series=True,
                               metadata=dict(grid_id=gid,region_tag=g.region_tag.iloc[0],endpoint='2023-11-01'))))
    r=run_tasks(tasks,RUNTIME/'processing/checkpoints',4);export_results(r,EVIDENCE/'processing')

if __name__=='__main__':main()
