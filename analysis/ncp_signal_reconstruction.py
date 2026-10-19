"""Transparent signal reconstruction for a controlled method-family comparison.

The adaptive-local-mean ensemble follows Colominas et al. (2014) and the
author's 2015 improved CEEMDAN reference (SNRFlag=2). EMD sifting is supplied
by PyEMD, so this is not a bitwise port of the author's MATLAB/Rilling EMD,
nor a reproduction of unpublished Zhang et al. settings.
"""
from __future__ import annotations
import os
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy import linalg


def configure_emd(runtime=None):
    runtime=runtime or os.environ.get('NCP_EMD_RUNTIME')
    if runtime:
        resolved=str(Path(runtime).resolve())
        if resolved not in sys.path: sys.path.insert(0,resolved)
    from PyEMD import EMD
    return EMD


def diagonal_mean(matrix):
    n=sum(matrix.shape)-1
    rows,cols=np.indices(matrix.shape)
    return np.bincount((rows+cols).ravel(),weights=matrix.ravel(),minlength=n)/np.bincount((rows+cols).ravel(),minlength=n)


def ssa_fill(y,window=36,rank=6,tolerance=1e-6,max_iterations=200):
    """Iterative low-rank Hankel completion, retaining every observed value.

    Explicit working settings, not parameters attributed to Zhang/Yi.
    """
    y=np.asarray(y,dtype=float); observed=np.isfinite(y)
    if observed.sum()<12: raise ValueError('At least 12 observations required')
    if observed.all():return y.copy(),dict(iterations=0,converged=True,relative_update=0.)
    t=np.arange(len(y)); filled=np.interp(t,t[observed],y[observed])
    window=min(window,len(y)//2); rank=min(rank,window-1)
    change=np.inf
    for iteration in range(1,max_iterations+1):
        trajectory=np.lib.stride_tricks.sliding_window_view(filled,window).T.copy()
        _,vectors=linalg.eigh(trajectory@trajectory.T,subset_by_index=(window-rank,window-1),check_finite=False)
        recon=diagonal_mean(vectors@(vectors.T@trajectory))
        old=filled[~observed].copy();filled[~observed]=recon[~observed]
        change=float(np.linalg.norm(filled[~observed]-old)/max(np.linalg.norm(old),1.))
        if change<tolerance:break
    assert np.array_equal(filled[observed],y[observed])
    return filled,dict(iterations=iteration,converged=change<tolerance,relative_update=change)


def emd_instance():
    EMD=configure_emd()
    return EMD(spline_kind='cubic',nbsym=2,MAX_ITERATION=1000,
               std_thr=.2,svar_thr=.001,energy_ratio_thr=.2,range_thr=.001,total_power_thr=.005)


@lru_cache(maxsize=24)
def noise_modes(length,trials,seed):
    rng=np.random.default_rng(seed);emd=emd_instance()
    return tuple(emd.emd(rng.normal(size=length)).copy() for _ in range(trials))


def first_mode_and_residue(y,emd):
    emd.emd(y,max_imf=1)
    imfs,residue=emd.get_imfs_and_residue()
    return (imfs[0] if len(imfs) else np.zeros_like(y)),residue.copy(),bool(len(imfs))


def improved_ensemble(y,trials=50,noise_scale=.2,seed=42,max_modes=10):
    """Adaptive local-mean iteration; sum(modes)+residue reconstructs input."""
    y=np.asarray(y,dtype=float)
    if not np.isfinite(y).all():raise ValueError('Ensemble input must be complete')
    scale=float(np.std(y,ddof=1))
    if scale<1e-12:return np.empty((0,len(y))),y.copy()
    x=y/scale; emd=emd_instance(); noises=noise_modes(len(y),trials,seed)
    means=[]
    for mode in noises:
        noise=mode[0]/max(np.std(mode[0],ddof=1),1e-12)
        _,residue,_=first_mode_and_residue(x+noise_scale*noise,emd)
        means.append(residue)
    residual=np.mean(means,axis=0); modes=[x-residual]
    for k in range(1,max_modes):
        _,_,has_imf=first_mode_and_residue(residual,emd)
        if not has_imf:break
        means=[]; local_scale=np.std(residual,ddof=1)
        for mode in noises:
            if len(mode)>k:
                noise=mode[k]/max(np.std(mode[k],ddof=1),1e-12)
                perturbed=residual+noise_scale*local_scale*noise
            else:perturbed=residual
            _,local_mean,_=first_mode_and_residue(perturbed,emd)
            means.append(local_mean)
        new_residual=np.mean(means,axis=0)
        modes.append(residual-new_residual);residual=new_residual
    modes=np.asarray(modes)*scale;residual=residual*scale
    assert np.allclose(modes.sum(axis=0)+residual,y,rtol=1e-10,atol=1e-10)
    return modes,residual


def mode_period(mode):
    centered=mode-np.mean(mode)
    signs=np.sign(centered); signs=signs[signs!=0]
    crossings=np.sum(signs[1:]!=signs[:-1])
    return float(2*len(mode)/crossings) if crossings else float('inf')


def reconstruct(y,trials=50,noise_scale=.2,seed=42,min_period=24.):
    modes,residual=improved_ensemble(y,trials,noise_scale,seed)
    periods=np.array([mode_period(m) for m in modes])
    keep=periods>=min_period
    trend=residual+(modes[keep].sum(axis=0) if keep.any() else 0)
    info=dict(mode_periods=[float(p) if np.isfinite(p) else None for p in periods],selected_modes=(np.flatnonzero(keep)+1).tolist(),
              reconstruction_error=float(np.max(np.abs(modes.sum(axis=0)+residual-y))),trials=trials,
              noise_scale=noise_scale,seed=seed,min_period=min_period)
    return trend,info
