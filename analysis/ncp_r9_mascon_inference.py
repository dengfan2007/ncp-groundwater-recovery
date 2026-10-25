"""r9 check 2 (analysis): south-minus-north storage trends on native JPL mascons.

Inputs: the read-only GEE extraction from ee_r9_mascon_membership.py, the archived cell
panel, the same-mask land-model panels and the fixed-well cell trends.  Pixels with
identical non-CRI series define native mascons.  Three estimators are reported for
January 2019--August 2024 (OLS with annual/semiannual harmonics, observed months only):

* cell contrast with mascon-cluster resampling (replaces the 3-degree blocks);
* equal-weight mascon contrast, using cell series averaged by lowland area;
* the same with native non-CRI mascon TWSA, minus lowland-weighted land-model storage.

With five native mascons (two north, three south) an exact permutation test has ten
labelings; its smallest attainable one-sided p value is 0.1.
"""
from __future__ import annotations
import argparse, json
from itertools import combinations
import numpy as np
import pandas as pd
from ncp_monitoring_core import ROOT,EVIDENCE,panel,design,seed,json_write
from ncp_monitoring_diagnostics import block_indices,quantiles

START,END='2019-01-01','2024-08-01'

def mascons(folder):
    s=pd.read_csv(folder/'pixel_series.csv');a=pd.read_csv(folder/'pixel_lowland_area.csv')
    s['date']=pd.to_datetime(s.band.str[:8]).dt.to_period('M').dt.to_timestamp()
    w=s.loc[s['product'].eq('lwe_noncri')].pivot_table(index=['pixel_lon','pixel_lat'],columns='band',values='lwe_cm')
    key=w.round(5).apply(lambda r:tuple(r.values),axis=1);order={k:i for i,k in enumerate(pd.unique(key))}
    pix=key.map(order).rename('mascon').reset_index();a=a.merge(pix,on=['pixel_lon','pixel_lat'],validate='many_to_one')
    share=a.groupby(['grid_id','mascon']).lowland_m2.sum().unstack(fill_value=0);share=share.div(share.sum(axis=1),axis=0)
    assert (share.max(axis=1)>0.999).all(),'a cell spans more than one mascon'
    member=share.idxmax(axis=1).rename('mascon').reset_index()
    native=s.merge(pix,on=['pixel_lon','pixel_lat']).groupby(['product','mascon','date']).lwe_cm.mean().reset_index()
    return member,native,a

def ols(dates,y):
    X=design(dates);ok=np.isfinite(y);c=np.linalg.lstsq(X[ok],y[ok],rcond=None)[0];return c[1],X,ok

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',required=True);a=p.parse_args()
    folder=ROOT/'outputs/r9_checks'/a.label/'mascon_r2';out=folder/'inference';out.mkdir(exist_ok=False)
    member,native,pix=mascons(folder);member.to_csv(out/'cell_mascon_membership.csv',index=False)
    cells=panel();cells=cells.merge(member,on='grid_id');dates=pd.date_range(START,END,freq='MS')
    meta=cells[['grid_id','region_tag','mascon']].drop_duplicates('grid_id').set_index('grid_id').sort_index()
    regions=meta.groupby('mascon').region_tag.agg(lambda x:x.unique().tolist())
    assert all(len(v)==1 for v in regions),'a mascon contains both regions';mreg=regions.str[0]
    area=pix.groupby('grid_id').lowland_m2.sum();rows=[];cell_rows=[]
    # Land-model components on the same-mask Noah and ERA5-Land supports.
    lsm={'archived_noah':cells}
    ext=pd.read_csv(EVIDENCE/'extraction/monthly_anomalies.csv',parse_dates=['date']).merge(member,on='grid_id')
    ext=ext.merge(meta.region_tag.reset_index(),on='grid_id');ext.loc[~ext.grace_observed.astype(bool),['tws_anomaly']]=np.nan
    for name,soil,other,gwa in [('same_mask_noah','sms_anomaly',['swe_anomaly','cwa_anomaly'],'noah_gwa_cm'),
                                ('era5_2m','era_soil2_cm_anomaly',['era_snow_cm_anomaly','era_skin_cm_anomaly'],'era_gwa2_cm'),
                                ('era5_289m','era_soil289_cm_anomaly',['era_snow_cm_anomaly','era_skin_cm_anomaly'],'era_gwa289_cm')]:
        d=ext[['grid_id','date','mascon','region_tag']].copy();d['twsa_cm']=ext.tws_anomaly;d['smsa_cm']=ext[soil]
        d['swea_cm']=ext[other[0]];d['cwa_cm']=ext[other[1]];d['gwa_cm']=ext[gwa].where(ext.tws_anomaly.notna())
        lsm[name]=d
    def cell_slopes(d,col):
        w=d.pivot(index='date',columns='grid_id',values=col).reindex(index=dates,columns=meta.index).to_numpy()
        return np.array([ols(dates,w[:,j])[0] for j in range(w.shape[1])]),w
    south=meta.region_tag.eq('south').to_numpy()
    # (1) cell contrast, mascon-cluster resampling with synchronous temporal blocks.
    for name,d in lsm.items():
        if 'gwa_cm' not in d:continue
        slopes,Y=cell_slopes(d,'gwa_cm');X=design(dates);keep=np.isfinite(Y).all(axis=1);pinv=np.linalg.pinv(X[keep]);coef=pinv@Y[keep]
        res=np.full_like(Y,np.nan);res[keep]=Y[keep]-X[keep]@coef;res[keep]-=res[keep].mean(axis=0)
        ix,_=block_indices(keep,rng=np.random.default_rng(seed('r9_mascon_time',name)))
        boots=coef[1]+np.einsum('n,bnc->bc',pinv[1],res[ix[:,keep]],optimize=True)
        codes=meta.mascon.to_numpy();u=np.unique(codes);rng=np.random.default_rng(seed('r9_mascon_space',name));est=[]
        for b in range(len(boots)):
            draw=rng.choice(u,size=len(u));mult=np.array([(draw==c).sum() for c in codes])
            ns,nn=(mult*south).sum(),(mult*~south).sum()
            if ns and nn:est.append((boots[b]*mult*south).sum()/ns-(boots[b]*mult*~south).sum()/nn)
        lo,hi=quantiles(est);rows.append(dict(input=name,estimator='cell_mean_mascon_cluster_bootstrap',estimate=float(slopes[south].mean()-slopes[~south].mean()),low=lo,high=hi,valid_draws=len(est),n_units=len(u)))
        for gid,v in zip(meta.index,slopes):cell_rows.append(dict(input=name,grid_id=gid,mascon=meta.loc[gid,'mascon'],region_tag=meta.loc[gid,'region_tag'],gwa_slope=v))
    # (2)-(3) mascon-level series and exact permutation.
    m_rows=[]
    for name,d in lsm.items():
        g=d.merge(area.rename('w').reset_index(),on='grid_id')
        cols=[c for c in ['gwa_cm','twsa_cm','smsa_cm','swea_cm','cwa_cm'] if c in g]
        agg=g.groupby(['mascon','date']).apply(lambda x:pd.Series({c:np.average(x[c],weights=x.w) if x[c].notna().all() else np.nan for c in cols}),include_groups=False).reset_index()
        nat=native.loc[native['product'].eq('lwe_noncri')].pivot(index='date',columns='mascon',values='lwe_cm')
        base=nat.loc['2003-01-01':'2014-12-01'].mean();nat=nat-base
        for mid,x in agg.groupby('mascon'):
            x=x.set_index('date').reindex(dates);obs=x['gwa_cm'].notna().to_numpy()
            land=sum(x[c] for c in ['smsa_cm','swea_cm','cwa_cm'] if c in x) if 'smsa_cm' in x else None
            out_row=dict(input=name,mascon=mid,region_tag=mreg[mid],n_cells=int((meta.mascon==mid).sum()),gwa_cell_avg=ols(dates,x['gwa_cm'].to_numpy())[0])
            if 'twsa_cm' in x:out_row['twsa_cri_cell_avg']=ols(dates,x['twsa_cm'].to_numpy())[0]
            if land is not None:
                tw=nat[mid].reindex(dates).where(obs).to_numpy();out_row['twsa_native']=ols(dates,tw)[0];out_row['gwa_native']=ols(dates,tw-land.to_numpy())[0]
            m_rows.append(out_row)
    mt=pd.DataFrame(m_rows);mt.to_csv(out/'mascon_trends.csv',index=False)
    for name,g in mt.groupby('input'):
        g=g.set_index('mascon')
        for col in [c for c in ['gwa_cell_avg','gwa_native','twsa_native','twsa_cri_cell_avg'] if c in g and g[c].notna().all()]:
            v=g[col];s=g.region_tag.eq('south');obs=v[s].mean()-v[~s].mean();perm=[]
            for north in combinations(g.index,int((~s).sum())):
                lab=g.index.isin(north);perm.append(v[~lab].mean()-v[lab].mean())
            perm=np.array(perm);rows.append(dict(input=name,estimator='mascon_equal_weight_'+col,estimate=float(obs),n_units=len(g),n_labelings=len(perm),
                exact_p_one_sided=float((perm>=obs-1e-12).mean()),minimum_attainable_p=float(1/len(perm)),rank_of_observed=int((perm>obs+1e-12).sum()+1)))
    res=pd.DataFrame(rows);res.to_csv(out/'mascon_contrasts.csv',index=False);pd.DataFrame(cell_rows).to_csv(out/'cell_gwa_slopes.csv',index=False)
    # Fixed-well trends by mascon (18 common cells; descriptive).
    w=pd.read_csv(ROOT/'outputs/monitoring_revision/2026-10-19-cell-trends/cell_well_trends.csv').merge(member,on='grid_id')
    wm=w.groupby(['aquifer','mascon']).agg(region_tag=('region_tag','first'),cells=('grid_id','size'),well_m_per_yr=('well_m_per_yr','mean'),twsa_cm_per_yr=('twsa_cm_per_yr','mean'),gwa_cm_per_yr=('gwa_cm_per_yr','mean')).reset_index()
    wm.to_csv(out/'mascon_well_trends.csv',index=False)
    # Native footprint of each mascon and its lowland share within the analysis domain.
    fp=pd.read_csv(folder/'footprint_pixels.csv');R=6371.0088
    fp['area_km2']=R*R*np.deg2rad(.5)*(np.sin(np.deg2rad(fp.lat+.5))-np.sin(np.deg2rad(fp.lat)));fp['lowland_km2']=fp.area_km2*fp.frac
    s=pd.read_csv(folder/'pixel_series.csv');s=s.loc[s['product'].eq('lwe_noncri')];foot=[]
    first=pix.merge(member,on=['grid_id','mascon']).drop_duplicates('mascon')
    for r in first.itertuples():
        x=s.loc[(s.pixel_lon==r.pixel_lon)&(s.pixel_lat==r.pixel_lat)].set_index('band').lwe_cm
        v1=x[[k for k in x.index if k.startswith('201001')]].iloc[0];v2=x[[k for k in x.index if k.startswith('202006')]].iloc[0]
        g=fp.loc[(fp.v1-v1).abs().lt(1e-3)&(fp.v2-v2).abs().lt(1e-3)]
        foot.append(dict(mascon=r.mascon,region_tag=mreg[r.mascon],pixels=len(g),area_km2=g.area_km2.sum(),lowland_in_domain_km2=g.lowland_km2.sum(),
                         lowland_share=g.lowland_km2.sum()/g.area_km2.sum(),lon_min=g.lon.min(),lon_max=g.lon.max()+.5,lat_min=g.lat.min(),lat_max=g.lat.max()+.5))
    foot=pd.DataFrame(foot).sort_values('mascon');foot.to_csv(out/'mascon_footprints.csv',index=False);print(foot.round(2).to_string(index=False))
    json_write(out/'manifest.json',dict(n_mascons=int(meta.mascon.nunique()),mascons_north=int((mreg=='north').sum()),mascons_south=int((mreg=='south').sum()),
        cells_per_mascon=meta.mascon.value_counts().sort_index().to_dict(),window=[START,END],source_sha256=__import__('ncp_monitoring_core').sha(__file__)))
    pd.set_option('display.width',250);print(res.to_string(index=False));print(mt.round(2).to_string(index=False));print(wm.round(2).to_string(index=False))

if __name__=='__main__':main()
