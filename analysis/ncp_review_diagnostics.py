"""Component and candidate-selection diagnostics from the completed review run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ncp_calendar_revision import calendar_detrend
from ncp_revision_analyses import pearson_or_nan
from ncp_review_revision import spatial_contrast

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',required=True); parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(); e=Path(args.evidence).resolve(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=False)
    panel=pd.read_csv(ROOT/'data/raw/grid_monthly_panel.csv',parse_dates=['date'])
    # Raster subtraction precedes separate band aggregation in the exporter.
    # Distinct component masks can therefore prevent an exact cell-mean identity.
    panel['aggregation_residual']=panel.gwa_cm-(panel.twsa_cm-panel.smsa_cm-panel.swea_cm-panel.cwa_cm)
    audit=panel.groupby('grid_id').aggregation_residual.agg(mean='mean',sd='std',max_abs=lambda x:x.abs().max())
    audit.to_csv(out/'component_aggregation_audit.csv')
    frame=pd.read_csv(e/'fixed_window_cell_slopes.csv')
    sensitivity=[]
    for tolerance in [1e-6,.1]:
        ids=audit.index[audit.max_abs<=tolerance]
        for (start,method),g in frame.loc[frame.grid_id.isin(ids)].groupby(['start','method']):
            for size in [2.,3.]:
                sensitivity.append(dict(tolerance_cm=tolerance,start=start,method=method,**spatial_contrast(g,size=size)))
    pd.DataFrame(sensitivity).to_csv(out/'component_support_sensitivity.csv',index=False)
    members=pd.read_csv(e/'well_cohort_membership.csv')
    common=set(members.loc[members.aquifer_group.eq('confined'),'grid_id'])&set(members.loc[members.aquifer_group.eq('unconfined'),'grid_id'])
    series=pd.read_csv(e/'aquifer_validation_series.csv',parse_dates=['date'])
    rows=[]
    for aquifer in ['unconfined','confined']:
        for region in ['north','south']:
            p=panel.loc[panel.grid_id.isin(common)&panel.region_tag.eq(region)].groupby('date')[['gwa_cm','twsa_cm','smsa_cm','swea_cm','cwa_cm','aggregation_residual']].mean()
            w=series.loc[series.aquifer.eq(aquifer)&series.support.eq('common')&series.region_tag.eq(region)].set_index('date').well_anomaly
            s=p.join(w).loc['2019-01-01':'2024-08-01']
            assert np.allclose(s.gwa_cm,s.twsa_cm-s.smsa_cm-s.swea_cm-s.cwa_cm+s.aggregation_residual,equal_nan=True,atol=1e-8)
            for column in ['gwa_cm','twsa_cm','smsa_cm','swea_cm','cwa_cm','aggregation_residual']:
                a=s[column].to_numpy(); b=s.well_anomaly.to_numpy()
                rows.append(dict(aquifer=aquifer,region=region,component=column,raw_r=pearson_or_nan(a,b),
                                 residual_r=pearson_or_nan(calendar_detrend(a),calendar_detrend(b)),
                                 difference_r=pearson_or_nan(np.diff(a),np.diff(b)),
                                 difference_sd=float(np.nanstd(np.diff(a),ddof=1))))
    pd.DataFrame(rows).to_csv(out/'component_well_diagnostics.csv',index=False)
    c=pd.read_csv(e/'beast_candidates.csv',parse_dates=['date'])
    meta=panel[['grid_id','region_tag']].drop_duplicates('grid_id')
    summaries=[]; choices=[]
    # These reselect stored BEAST candidates only; no segmented fallback or refit.
    for margin,prob,rule in [(0.,.5,'highest_probability'),(.5,.5,'highest_probability'),(1.,.5,'highest_probability'),(.5,.75,'highest_probability'),(.5,.5,'earliest')]:
        v=c.loc[c.date.ge('2015-01-01')&c.post_slope.gt(0)&(c.post_slope-c.pre_slope).gt(margin)&c.probability.ge(prob)]
        v=v.sort_values(['probability','date'],ascending=False) if rule=='highest_probability' else v.sort_values('date')
        v=v.drop_duplicates('grid_id').merge(meta,on='grid_id')
        m=v.date.dt.year*12+v.date.dt.month-1
        median=int(np.rint(m.median()))
        delta=m[v.region_tag.eq('south')].mean()-m[v.region_tag.eq('north')].mean()
        summaries.append(dict(slope_margin=margin,probability_threshold=prob,rule=rule,n=len(v),median=f'{median//12}-{median%12+1:02d}',timing_south_minus_north=delta))
        choices.append(v.assign(slope_margin=margin,probability_threshold=prob,rule=rule))
    pd.DataFrame(summaries).to_csv(out/'candidate_rule_summary.csv',index=False)
    pd.concat(choices).to_csv(out/'candidate_rule_choices.csv',index=False)
    inputs=[ROOT/'data/raw/grid_monthly_panel.csv',e/'well_cohort_membership.csv',e/'aquifer_validation_series.csv',e/'beast_candidates.csv',e/'fixed_window_cell_slopes.csv']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs}),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
