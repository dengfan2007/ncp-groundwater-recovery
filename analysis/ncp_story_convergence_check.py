"""Check the two SSA completions that reached the prespecified iteration cap."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from ncp_signal_reconstruction import ssa_fill,reconstruct
from ncp_story_analysis import report_candidates

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--emd-runtime',required=True);a=p.parse_args()
    e=Path(a.evidence).resolve();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);os.environ['NCP_EMD_RUNTIME']=str(Path(a.emd_runtime).resolve())
    diag=json.loads((e/'actual_diagnostics.json').read_text());panel=pd.read_csv(ROOT/'data/raw/grid_monthly_panel.csv',parse_dates=['date']);base=pd.read_csv(e/'actual_selections.csv');rows=[]
    for case in diag:
        if case['ssa']['converged']:continue
        s=panel.loc[panel.grid_id.eq(case['grid_id'])&panel.date.le(case['endpoint'])];y=s.gwa_cm.to_numpy()
        filled,info=ssa_fill(y,max_iterations=1000);reconstructed,_=reconstruct(filled)
        q=y-STL(pd.Series(y).interpolate(limit_direction='both'),period=12,robust=True).fit().seasonal.to_numpy()
        chosen,_,_=report_candidates(s.date,reconstructed,q)
        old=base.loc[base.grid_id.eq(case['grid_id'])&base.endpoint.eq(case['endpoint'])&base.route.eq('D')].set_index('rule')
        for r in chosen:
            before=old.loc[r['rule']].selected_date;before=None if pd.isna(before) else before
            rows.append(dict(grid_id=case['grid_id'],endpoint=case['endpoint'],rule=r['rule'],original_date=before,rechecked_date=r['selected_date'],
                             unchanged=before==r['selected_date'],iterations=info['iterations'],converged=info['converged'],relative_update=info['relative_update']))
    pd.DataFrame(rows).to_csv(out/'ssa_convergence_check.csv',index=False,lineterminator='\n')
    inputs=[e/'actual_diagnostics.json',e/'actual_selections.csv',ROOT/'data/raw/grid_monthly_panel.csv'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs}),indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
