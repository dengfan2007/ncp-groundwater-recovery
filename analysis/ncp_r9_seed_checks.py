"""r9 checks 0-1: input-perturbation audit and multi-seed / longer-chain refits.

Reuses the frozen primitive ``ncp_monitoring_core.fit`` unchanged.  Longer chains are
injected by wrapping ``Rbeast.beast`` inside worker processes only.  Archived evidence is
read-only; checkpoints go to outputs/runs/<label>/ and tracked evidence to
outputs/r9_checks/<label>/.  The seed-42 tasks are a gate: they must reproduce the frozen
archived and uniform-support dates exactly before any summary is written.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_key]='1'
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import Rbeast as rb
import ncp_monitoring_core as core

ROOT=core.ROOT
ENDS=['2023-12-01','2024-08-01']
N_SEEDS=10
LONG=dict(mcmc_chains=6,mcmc_samples=24000,mcmc_burbin=2000)   # Rbeast spells burn-in 'burbin'
_ORIGINAL_BEAST=rb.beast
_EXTRA={}

def _beast(*args,**kwargs):
    kwargs.update(_EXTRA)
    return _ORIGINAL_BEAST(*args,**kwargs)
rb.beast=_beast   # core imports the same module object

def seeds():
    return [42]+[core.seed('r9_seed',k) for k in range(1,N_SEEDS)]

def worker(task,folder):
    os.environ['NCP_EMD_RUNTIME']=str(core.EMD)
    path=Path(folder)/(task['task_id']+'.json')
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    _EXTRA.clear();_EXTRA.update(task.get('mcmc',{}))
    try:r=core.fit(task)
    finally:_EXTRA.clear()
    out=dict(task_id=task['task_id'],metadata=task['metadata'],selections=r['selections'],
             n_candidates={k:sum(c['route']==k for c in r['candidates']) for k in ['A','D']},
             ssa=r['diagnostics'].get('ssa',{}),elapsed=r['elapsed_seconds'])
    core.json_write(path,out);return out

def tasks():
    panels={'archived':core.panel(),'uniform':core.panel(uniform=True)}
    plan=[]
    for k,s in enumerate(seeds()):
        plan.append(('joint',s,s,{}))
        if k:plan.append(('beast_only',s,42,{}))
        if k<5:plan.append(('long_beast_only',s,42,LONG))
    out=[]
    for source,p in panels.items():
        for gid,g in p.groupby('grid_id'):
            for end in (ENDS if source=='archived' else ['2024-08-01']):
                sub=g.loc[g.date.le(end)]
                for variant,bs,es,mcmc in plan:
                    if source=='uniform' and variant.startswith('long'):continue
                    tid=f'{source}_{gid}_{end}_{variant}_{bs}_{es}'
                    out.append(core.clean(dict(task_id=tid,dates=sub.date.astype(str).tolist(),y=sub.gwa_cm.tolist(),
                        ssa_iterations=1000 if end=='2024-08-01' else 200,beast_seed=bs,ensemble_seed=es,mcmc=mcmc,
                        metadata=dict(source=source,grid_id=gid,region_tag=g.region_tag.iloc[0],endpoint=end,
                                      variant=variant,beast_seed=bs,ensemble_seed=es))))
    return out

def run(label,workers):
    folder=ROOT/'outputs/runs'/label/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
    todo=tasks();started=time.time();results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,t,str(folder)) for t in todo]
        for i,f in enumerate(as_completed(futures),1):
            results.append(f.result())
            if i%100==0 or i==len(todo):print(f'{i}/{len(todo)} {time.time()-started:.0f}s',flush=True)
    rows=[{**r['metadata'],**s,'ssa_converged':r['ssa'].get('converged')} for r in results for s in r['selections']]
    return pd.DataFrame(rows),dict(tasks=len(todo),elapsed_seconds=time.time()-started,workers=workers)

def gate(sel):
    """Seed-42 joint fits must equal the frozen archived and uniform-support selections."""
    s=sel.loc[sel.variant.eq('joint')&sel.beast_seed.eq(42)].copy()
    frozen=core.frozen_actual();rows=[]
    for (gid,end,route),g in frozen.loc[frozen.endpoint.isin(ENDS)].groupby(['grid_id','endpoint','route']):
        cs=core.clean(g.to_dict('records'))
        for rule in ['highest_probability','earliest']:
            c=core.select(cs,rule=rule);rows.append(dict(source='archived',grid_id=gid,endpoint=end,route=route,rule=rule,frozen=c['date'] if c else ''))
    u=pd.read_csv(core.VALIDATION/'uniform/uniform_selections.csv')
    u=u.loc[u.endpoint.eq('2024-08-01')&u.rule.isin(['highest_probability','earliest'])]
    rows+= [dict(source='uniform',grid_id=r.grid_id,endpoint=r.endpoint,route=r.route,rule=r.rule,frozen=r.selected_date if isinstance(r.selected_date,str) else '') for r in u.itertuples()]
    ref=pd.DataFrame(rows);z=ref.merge(s,on=['source','grid_id','endpoint','route','rule'],how='left',validate='one_to_one')
    z['rerun']=z.selected_date.fillna('');bad=z.loc[z.rerun.ne(z.frozen)]
    return dict(compared=len(z),discrepancies=len(bad)),bad

def month(x):
    d=pd.to_datetime(pd.Series(x));return (d.dt.year*12+d.dt.month).to_numpy(dtype=float)

def agreement(a,b):
    """Within-six fraction and count over cells dated in both selection vectors."""
    ok=a.notna()&b.notna();d=np.abs(month(a[ok])-month(b[ok]))
    return int(ok.sum()),int((d<=6).sum())

def summarize(sel,out):
    h=sel.copy();h['key']=h.source+'|'+h.endpoint
    wide=h.pivot_table(index=['variant','beast_seed','ensemble_seed','route','rule','grid_id'],columns='key',values='selected_date',aggfunc='first')
    rows=[];spread=[];pert=[]
    a_dec,a_aug,u_aug='archived|2023-12-01','archived|2024-08-01','uniform|2024-08-01'
    for (variant,route,rule),g in wide.groupby(level=['variant','route','rule']):
        seedsets=sorted(set(zip(g.index.get_level_values('beast_seed'),g.index.get_level_values('ensemble_seed'))))
        def col(s,key):
            x=g.xs((variant,s[0],s[1],route,rule),level=['variant','beast_seed','ensemble_seed','route','rule'])
            return x[key] if key in x else pd.Series(dtype=object)
        for s in seedsets:
            for name,(k1,k2) in dict(record_growth=(a_dec,a_aug),perturbation=(a_aug,u_aug)).items():
                x,y=col(s,k1),col(s,k2)
                if len(x) and len(y):
                    n,w=agreement(x,y.reindex(x.index));rows.append(dict(variant=variant,route=route,rule=rule,comparison=name,beast_seed=s[0],ensemble_seed=s[1],paired=n,within6=w,
                        dated_before=int(x.notna().sum()),dated_after=int(y.notna().sum())))
        for key in [a_dec,a_aug,u_aug]:
            for s1,s2 in combinations(seedsets,2):
                x,y=col(s1,key),col(s2,key)
                if len(x) and len(y):
                    n,w=agreement(x,y.reindex(x.index));rows.append(dict(variant=variant,route=route,rule=rule,comparison='cross_seed_'+key.replace('|','_'),beast_seed=s1[0],ensemble_seed=s1[1],
                        beast_seed_2=s2[0],ensemble_seed_2=s2[1],paired=n,within6=w))
            # per-cell spread across seeds
            m=pd.DataFrame({s:col(s,key) for s in seedsets if len(col(s,key))})
            for gid,r in m.iterrows():
                v=month(r.dropna()) if r.notna().any() else np.array([])
                spread.append(dict(variant=variant,route=route,rule=rule,input=key,grid_id=gid,n_seeds=m.shape[1],n_dated=len(v),
                    range_months=float(v.max()-v.min()) if len(v) else np.nan,
                    modal_date=r.dropna().mode().iloc[0] if len(v) else None,modal_share=float(r.dropna().value_counts().iloc[0]/m.shape[1]) if len(v) else 0.))
    pairs=pd.DataFrame(rows);pairs['fraction']=pairs.within6/pairs.paired
    pairs.to_csv(out/'seed_pair_agreement.csv',index=False);pd.DataFrame(spread).to_csv(out/'cell_seed_spread.csv',index=False)
    summary=pairs.groupby(['variant','route','rule','comparison']).agg(n_pairs=('fraction','size'),median_fraction=('fraction','median'),
        min_fraction=('fraction','min'),max_fraction=('fraction','max'),median_paired=('paired','median')).reset_index()
    summary.to_csv(out/'agreement_summary.csv',index=False);return summary

def regional(sel,out):
    rows=[]
    for keys,g in sel.loc[sel.qualifies].groupby(['source','endpoint','variant','beast_seed','ensemble_seed','route','rule']):
        n=g.region_tag.eq('north');m=month(g.selected_date)
        if n.sum()>=2 and (~n).sum()>=2:
            rows.append(dict(zip(['source','endpoint','variant','beast_seed','ensemble_seed','route','rule'],keys),n_north=int(n.sum()),n_south=int((~n).sum()),
                             contrast_months=float(m[~n.to_numpy()].mean()-m[n.to_numpy()].mean())))
    d=pd.DataFrame(rows);d.to_csv(out/'regional_contrast_by_seed.csv',index=False)
    s=d.groupby(['source','endpoint','variant','route','rule']).contrast_months.agg(['size','min','median','max']).reset_index()
    s['sign_changes']=d.groupby(['source','endpoint','variant','route','rule']).contrast_months.apply(lambda x:bool((x>0).any() and (x<0).any())).to_numpy()
    s.to_csv(out/'regional_contrast_summary.csv',index=False)

def perturbation_audit(sel,out):
    a=core.panel();u=core.panel(uniform=True)
    z=a[['grid_id','date','gwa_cm']].merge(u[['grid_id','date','gwa_cm']],on=['grid_id','date'],suffixes=('_archived','_uniform'))
    z=z.loc[z.date.le('2024-08-01')&z.gwa_cm_archived.notna()]
    d=z.assign(diff=(z.gwa_cm_uniform-z.gwa_cm_archived).abs()).groupby('grid_id')['diff'].agg(max_abs_cm='max',mean_abs_cm='mean').reset_index()
    s=sel.loc[sel.variant.eq('joint')&sel.beast_seed.eq(42)&sel.endpoint.eq('2024-08-01')]
    w=s.pivot_table(index=['grid_id','route','rule'],columns='source',values='selected_date',aggfunc='first').reset_index()
    w['shift_months']=month(w.uniform)-month(w.archived)
    w=w.merge(d,on='grid_id');w.to_csv(out/'perturbation_cell_audit.csv',index=False)
    return dict(cells=len(d),cells_changed=int((d.max_abs_cm>1e-9).sum()),max_abs_cm=float(d.max_abs_cm.max()),
                min_of_cell_max_cm=float(d.max_abs_cm.min()),median_cell_max_cm=float(d.max_abs_cm.median()))

def local_worker(task,folder):
    """Same fit as ``worker``; returns local-rule selections (core.select(local=True))."""
    os.environ['NCP_EMD_RUNTIME']=str(core.EMD)
    path=Path(folder)/(task['task_id']+'.json')
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    r=core.fit(task);sel=[]
    for route in ['A','D']:
        pool=[c for c in r['candidates'] if c['route']==route]
        for rule in ['highest_probability','earliest']:
            c=core.select(pool,rule=rule,local=True);sel.append(dict(route=route,rule=rule,qualifies=c is not None,selected_date=c['date'] if c else None))
    out=dict(task_id=task['task_id'],metadata=task['metadata'],selections=sel);core.json_write(path,out);return out

def local_seeds(label,workers):
    """Local-rule dates across the ten joint seeds on the archived input."""
    out=ROOT/'outputs/r9_checks'/label/'seeds_local_rule';out.mkdir(parents=True,exist_ok=False)
    folder=ROOT/'outputs/runs'/label/'checkpoints_local';folder.mkdir(parents=True,exist_ok=True)
    todo=[t for t in tasks() if t['metadata']['source']=='archived' and t['metadata']['variant']=='joint'];results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(local_worker,t,str(folder)) for t in todo]),1):
            results.append(f.result())
            if i%100==0 or i==len(todo):print(f'local {i}/{len(todo)}',flush=True)
    sel=pd.DataFrame([{**r['metadata'],**s} for r in results for s in r['selections']]);sel.to_csv(out/'selections.csv',index=False)
    # Gate: seed 42 equals local selections recomputed from the frozen baseline candidate pool.
    b=pd.read_csv(core.EVIDENCE/'baseline/candidates.csv');ref=[]
    for (gid,end,route),g in b.loc[b.endpoint.isin(ENDS)].groupby(['grid_id','endpoint','route']):
        cs=core.clean(g.to_dict('records'))
        for rule in ['highest_probability','earliest']:
            c=core.select(cs,rule=rule,local=True);ref.append(dict(grid_id=gid,endpoint=end,route=route,rule=rule,frozen=c['date'] if c else ''))
    z=pd.DataFrame(ref).merge(sel.loc[sel.beast_seed.eq(42)],on=['grid_id','endpoint','route','rule'],how='left',validate='one_to_one')
    bad=int((z.selected_date.fillna('')!=z.frozen).sum())
    rows=[]
    for (route,rule),g in sel.groupby(['route','rule']):
        w=g.pivot_table(index=['beast_seed','grid_id'],columns='endpoint',values='selected_date',aggfunc='first')
        S=sorted(g.beast_seed.unique())
        for s in S:
            x=w.loc[s];n,k=agreement(x.get('2023-12-01',pd.Series(dtype=object)),x.get('2024-08-01',pd.Series(dtype=object)));rows.append(dict(route=route,rule=rule,comparison='record_growth',seed=s,paired=n,within6=k))
        for end in ENDS:
            for s1,s2 in combinations(S,2):
                n,k=agreement(w.loc[s1][end],w.loc[s2][end].reindex(w.loc[s1].index));rows.append(dict(route=route,rule=rule,comparison='cross_seed_'+end,seed=s1,seed_2=s2,paired=n,within6=k))
    d=pd.DataFrame(rows);d['fraction']=d.within6/d.paired;d.to_csv(out/'seed_pair_agreement.csv',index=False)
    s=d.groupby(['route','rule','comparison']).fraction.agg(['size','median','min','max']).reset_index();s.to_csv(out/'agreement_summary.csv',index=False)
    core.json_write(out/'manifest.json',dict(tasks=len(todo),gate_discrepancies=bad,seeds=seeds()))
    print('gate discrepancies',bad);print(s.to_string(index=False))
    if bad:raise SystemExit('local gate failed')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',required=True);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--local-only',action='store_true');a=p.parse_args()
    if a.local_only:return local_seeds(a.label,a.workers)
    out=ROOT/'outputs/r9_checks'/a.label/'seeds';out.mkdir(parents=True,exist_ok=False)
    sel,info=run(a.label,a.workers);sel.to_csv(out/'selections.csv',index=False)
    g,bad=gate(sel);bad.to_csv(out/'gate_discrepancies.csv',index=False)
    report=dict(run=info,gate=g,seeds=seeds(),long_chain=LONG,endpoints=ENDS,core_sha256=core.sha(ROOT/'analysis/ncp_monitoring_core.py'),source_sha256=core.sha(__file__))
    if g['discrepancies']:
        core.json_write(out/'manifest.json',report);raise SystemExit(f"Gate failed: {g['discrepancies']} discrepancies")
    report['perturbation']=perturbation_audit(sel,out);summarize(sel,out);regional(sel,out)
    report['ssa_not_converged']=int((sel.ssa_converged==False).sum())
    core.json_write(out/'manifest.json',report);print(json.dumps(report['gate']),json.dumps(report['perturbation']))

if __name__=='__main__':main()
