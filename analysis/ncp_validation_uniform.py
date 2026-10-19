"""Apply completed uniform-support extraction to targeted validation."""
import argparse,os,json,hashlib
from pathlib import Path
import pandas as pd
from ncp_validation_extension import ROOT,panel,run_tasks,well_checks,temporal_uncertainty,noise_calibration

def main():
    p=argparse.ArgumentParser();p.add_argument('--extraction',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--emd-runtime',required=True);a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);os.environ['NCP_EMD_RUNTIME']=str(Path(a.emd_runtime).resolve())
    original=panel();new=pd.read_csv(Path(a.extraction)/'uniform_support_panel.csv',parse_dates=['date'])
    data=original.drop(columns=['component_gwa','closure_delta']).set_index(['grid_id','date']);replacement=new.set_index(['grid_id','date'])
    columns=['gwa_cm','twsa_cm','smsa_cm','swea_cm','cwa_cm'];data.loc[replacement.index,columns]=replacement[columns]
    data=data.reset_index();data['component_gwa']=data.twsa_cm-data.smsa_cm-data.swea_cm-data.cwa_cm;data['closure_delta']=data.gwa_cm-data.component_gwa
    data.to_csv(out/'uniform_calendar_panel.csv',index=False,lineterminator='\n')
    tasks=[]
    for gid,g in data.groupby('grid_id'):
        for end in ['2023-11-01','2023-12-01','2024-08-01']:
            sub=g.loc[g.date.le(end)];tasks.append(dict(grid_id=gid,region_tag=g.region_tag.iloc[0],scope='grid',endpoint=end,experiment='uniform_support',dates=sub.date.astype(str).tolist(),y=sub.gwa_cm.tolist(),routes=['A','D'],save_series=gid in ['g_115.0_35.5','g_116.0_37.5']))
    run_tasks(tasks,out,4,'uniform');well_checks(data,out);temporal_uncertainty(data,out);noise_calibration(data,out)
    m=dict(extraction_manifest_sha256=hashlib.sha256((Path(a.extraction)/'manifest.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256((Path(a.extraction)/'uniform_support_panel.csv').read_bytes()).hexdigest(),observed_months_only_replaced=True,archived_panel_unchanged=True)
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('Uniform validation complete',flush=True)
if __name__=='__main__':main()
