"""Independent outer-loop translation of the author's ceemdan.m, common EMD kernel.

Checks adaptive-local-mean iteration, not equivalence to MATLAB/Rilling sifting.
"""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import patch
import ncp_signal_reconstruction as implementation

ROOT=Path(__file__).resolve().parents[1]

def reference(x,noise_imfs,noise_sd=.2):
    # Literal sequence corresponding to ceemdan.m (25 February 2015), SNRFlag=2.
    scale=np.std(x,ddof=1);x=x/scale;emd=implementation.emd_instance();nr=len(noise_imfs)
    average=np.zeros(len(x))
    for noise in noise_imfs:
        xi=x+noise_sd*noise[0]/np.std(noise[0],ddof=1)
        emd.emd(xi,max_imf=1);imfs,residual=emd.get_imfs_and_residue()
        temp=imfs[0] if len(imfs) else np.zeros(len(x))
        average+=(xi-temp)/nr
    modes=[x-average];means=[average.copy()];k=1
    while k<10:
        emd.emd(means[-1],max_imf=1);imfs,_=emd.get_imfs_and_residue()
        if not len(imfs):break
        average=np.zeros(len(x))
        for noise in noise_imfs:
            perturbation=np.zeros(len(x))
            if len(noise)>=k+1:perturbation=noise_sd*noise[k]/np.std(noise[k],ddof=1)
            xi=means[-1]+np.std(means[-1],ddof=1)*perturbation
            emd.emd(xi,max_imf=1);_,residual=emd.get_imfs_and_residue()
            average+=residual/nr
        modes.append(means[-1]-average);means.append(average.copy());k+=1
    return np.asarray(modes)*scale,means[-1]*scale

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--emd-runtime',required=True);a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);os.environ['NCP_EMD_RUNTIME']=str(Path(a.emd_runtime).resolve())
    t=np.arange(261)/12;rng=np.random.default_rng(1729)
    signals={'decline':-1.5*t,'seasonal_decline':-1.5*t+3*np.sin(2*np.pi*t),'single_reversal':-1.5*t+4.5*np.maximum(t-16,0),
             'two_stage':-1.5*t+2.1*np.maximum(t-15,0)+3.4*np.maximum(t-18,0),'pulse':-1.5*t+12*np.maximum(1-abs(t-18.5)/1.5,0),
             'noisy_two_stage':-1.5*t+2.1*np.maximum(t-15,0)+3.4*np.maximum(t-18,0)+rng.normal(0,3,len(t))}
    panel=pd.read_csv(ROOT/'data/raw/grid_monthly_panel.csv')
    for gid in ['g_115.0_35.5','g_116.0_37.5','g_118.0_37.5']:
        y=panel.loc[panel.grid_id.eq(gid),'gwa_cm'].to_numpy();signals[gid]=implementation.ssa_fill(y,max_iterations=1000)[0]
    # Pass precisely the same precomputed noise modes to both outer loops.
    noises=implementation.noise_modes(261,50,42);rows=[]
    for name,y in signals.items():
        with patch.object(implementation,'noise_modes',return_value=noises):m,r=implementation.improved_ensemble(y)
        mr,rr=reference(y,noises)
        periods=np.array([implementation.mode_period(v) for v in m]);pr=np.array([implementation.mode_period(v) for v in mr])
        trend=r+m[periods>=24].sum(axis=0);tr=rr+mr[pr>=24].sum(axis=0)
        rows.append(dict(signal=name,implementation_modes=len(m),reference_modes=len(mr),max_mode_difference=float(np.max(np.abs(m-mr))) if m.shape==mr.shape else np.nan,
                         max_residue_difference=float(np.max(abs(r-rr))),max_long_signal_difference=float(np.max(abs(trend-tr))),
                         identity_error=float(np.max(abs(m.sum(axis=0)+r-y))),same_selected_modes=bool(np.array_equal(periods>=24,pr>=24))))
    pd.DataFrame(rows).to_csv(out/'outer_iteration_comparison.csv',index=False,lineterminator='\n')
    source=ROOT/'outputs/runs/2026-10-16-story-sources-r1/ceemdan.m'
    report=dict(reference_url='https://github.com/macolominas/CEEMDAN/blob/main/ceemdan.m',reference_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                reference_configuration='Improved adaptive local mean, SNRFlag=2; same 50 precomputed noise-mode sets; same PyEMD EMD kernel; at most 10 modes',
                implementation_sha256=hashlib.sha256((ROOT/'analysis/ncp_signal_reconstruction.py').read_bytes()).hexdigest(),test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                matlab_available=False,octave_available=False,matlab_sifting_equivalence_tested=False,signals=len(rows),maximum_long_signal_difference=max(x['max_long_signal_difference'] for x in rows))
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
