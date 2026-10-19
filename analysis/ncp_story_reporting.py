"""Reproducible descriptive summaries of endpoint stability; no new model fits."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    e=Path(a.evidence).resolve();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    path=e/'endpoint_transitions.csv';data=pd.read_csv(path);rows=[]
    for (route,rule,end),g in data.groupby(['route','rule','to_endpoint']):
        for tolerance in [3,6,12]:
            both=int(g.revision_months.notna().sum());stable=int(g.revision_months.le(tolerance).sum())
            rows.append(dict(route=route,rule=rule,endpoint=end,tolerance_months=tolerance,total_cells=len(g),previously_reportable=int(g.previously_reportable.sum()),
                             currently_reportable=int(g.currently_reportable.sum()),withdrawn=int(g.withdrawn.sum()),both_reportable=both,within_tolerance=stable,
                             conditional_fraction=stable/both if both else None))
    pd.DataFrame(rows).to_csv(out/'stability_tolerances.csv',index=False,lineterminator='\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),inputs={str(path.relative_to(ROOT)):sha(path)}),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
