"""Re-extract storage components on explicitly shared support into a new directory.

No uploads, authentication changes, or replacement of archived observations.
Retains the archived 218-month GRACE mask to distinguish support from added months.
"""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,time
from pathlib import Path
import ee
import pandas as pd
from ee_ncp_common import build_ncp_context

ROOT=Path(__file__).resolve().parents[1]
CONFIG=dict(project='fire-risk-460605',crs='EPSG:4326',scale_m=10000,resampling='Earth Engine default nearest neighbour',
            baseline_start='2003-01-01',baseline_end='2015-01-01',bbox=[112.,34.5,120.,40.6],
            province_names=['Beijing Shi','Tianjin Shi','Hebei Sheng','Henan Sheng','Shandong Sheng'],
            elevation_max_m=200,grace='NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI',gldas='NASA/GLDAS/V021/NOAH/G025/T3H',gain_factors_applied=False)

def build():
    archived=pd.read_csv(ROOT/'data/raw/grid_monthly_panel.csv',parse_dates=['date'])
    obs=sorted(archived.loc[archived.gwa_cm.notna(),'date'].dt.strftime('%Y-%m').unique())
    geometry=ee.Geometry.Rectangle(CONFIG['bbox'],geodesic=False)
    land=build_ncp_context(CONFIG['bbox'],CONFIG['province_names'],200)['lowland_mask']
    grace=ee.ImageCollection(CONFIG['grace']).filterDate('2003-01-01','2024-09-01').map(lambda i:i.set('month_tag',i.date().format('YYYY-MM')))
    grace=grace.filter(ee.Filter.inList('month_tag',obs))
    gldas=ee.ImageCollection(CONFIG['gldas'])
    projection=ee.Projection(CONFIG['crs']).atScale(CONFIG['scale_m'])
    def raw(month):
        month=ee.Date(month);end=month.advance(1,'month');lsm=gldas.filterDate(month,end).mean()
        sms=lsm.select(['SoilMoi0_10cm_inst','SoilMoi10_40cm_inst','SoilMoi40_100cm_inst','SoilMoi100_200cm_inst']).reduce(ee.Reducer.sum()).divide(10).rename('sms')
        components=ee.Image.cat([sms,lsm.select('SWE_inst').divide(10).rename('swe'),lsm.select('CanopInt_inst').divide(10).rename('cwa')])
        tws=ee.Image(ee.Algorithms.If(grace.filterDate(month,end).size().gt(0),grace.filterDate(month,end).select('lwe_thickness').mean().rename('tws'),ee.Image.constant(0).rename('tws').updateMask(ee.Image.constant(0))))
        # Set identical output projection explicitly before arithmetic or reduction.
        result=ee.Image.cat([tws,components]).reproject(projection).clip(geometry)
        return result.set('system:time_start',month.millis()).set('month_tag',month.format('YYYY-MM'))
    dates=pd.date_range('2003-01-01','2024-09-01',freq='MS')
    monthly=ee.ImageCollection.fromImages([raw(d.strftime('%Y-%m-%d')) for d in dates])
    selected=monthly.filter(ee.Filter.inList('month_tag',obs))
    # Static intersection of all component masks over the retained GRACE months.
    common=selected.map(lambda i:i.mask().reduce(ee.Reducer.min()).unmask(0).rename('valid')).min().gt(0)
    shared=common.And(land.unmask(0).reproject(projection).gt(0)).selfMask().reproject(projection)
    def mask(i):return i.updateMask(shared)
    monthly=monthly.map(mask)
    baseline=monthly.filterDate(CONFIG['baseline_start'],CONFIG['baseline_end']).mean()
    features=[]
    for row in archived[['grid_id','region_tag']].drop_duplicates().itertuples():
        _,lon,lat=row.grid_id.split('_');lon,lat=float(lon),float(lat)
        features.append(ee.Feature(ee.Geometry.Rectangle([lon,lat,lon+1,lat+1],geodesic=False),dict(grid_id=row.grid_id,region_tag=row.region_tag)))
    grid=ee.FeatureCollection(features)
    def summarize(i):
        d=i.subtract(baseline).rename(['twsa_cm','smsa_cm','swea_cm','cwa_cm']).updateMask(shared)
        gwa=d.select('twsa_cm').subtract(d.select('smsa_cm')).subtract(d.select('swea_cm')).subtract(d.select('cwa_cm')).rename('gwa_cm')
        image=ee.Image.cat([d,gwa]);area=ee.Image.pixelArea().updateMask(shared).rename('support_area_m2')
        weighted=image.multiply(area).addBands(area)
        def finish(f):
            a=ee.Number(f.get('support_area_m2'));props={b:ee.Number(f.get(b)).divide(a) for b in ['twsa_cm','smsa_cm','swea_cm','cwa_cm','gwa_cm']}
            return ee.Feature(None,props).set('support_area_m2',a).set('grid_id',f.get('grid_id')).set('region_tag',f.get('region_tag')).set('date',i.date().format('YYYY-MM-dd'))
        return weighted.reduceRegions(collection=grid,reducer=ee.Reducer.sum(),crs=CONFIG['crs'],scale=CONFIG['scale_m'],tileScale=4).map(finish)
    return monthly.filter(ee.Filter.inList('month_tag',obs)),summarize,obs

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--years',nargs='*',type=int);a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);CONFIG['project']=a.project
    ee.Initialize(project=a.project);ee.data.setDeadline(180000)
    source=Path(__file__).read_bytes();(out/'specification.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
    collection,summarize,obs=build();years=a.years or list(range(2003,2025));start=time.time()
    def fetch(year):
        sub=collection.filterDate(f'{year}-01-01',f'{year+1}-01-01');count=sub.size().getInfo()
        if not count:return pd.DataFrame()
        images=sub.toList(count);fc=ee.FeatureCollection([summarize(ee.Image(images.get(k))) for k in range(count)]).flatten()
        payload=fc.getInfo();frame=pd.DataFrame([x['properties'] for x in payload['features']]);frame['query_year']=year
        frame.to_csv(out/f'year_{year}.csv',index=False,lineterminator='\n');return frame
    frames=[];errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(fetch,y):y for y in years}
        for future in concurrent.futures.as_completed(futures):
            year=futures[future]
            try:frame=future.result();frames.append(frame);print('Extracted',year,len(frame),'rows seconds',round(time.time()-start),flush=True)
            except Exception as e:errors.append(dict(year=year,error=str(e)));print('Failed',year,str(e)[:250],flush=True)
    if frames:
        frame=pd.concat(frames).sort_values(['grid_id','date']);frame['closure_delta']=frame.gwa_cm-frame.twsa_cm+frame.smsa_cm+frame.swea_cm+frame.cwa_cm
        frame.to_csv(out/'uniform_support_panel.csv',index=False,lineterminator='\n')
        check=dict(max_abs_closure=float(frame.closure_delta.abs().max()),n_rows=len(frame),support_relative_range=float(frame.groupby('grid_id').support_area_m2.apply(lambda s:(s.max()-s.min())/s.mean()).max()))
    else:check={}
    manifest=dict(config=CONFIG,requested_years=years,archived_observation_months=obs,checks=check,errors=errors,source_sha256=hashlib.sha256(source).hexdigest(),input_sha256=hashlib.sha256((ROOT/'data/raw/grid_monthly_panel.csv').read_bytes()).hexdigest(),retrieval_date='2026-10-17',vintage='Current Earth Engine collection; archived month mask retained; not assumed identical numerical vintage')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(check),flush=True)
    if errors:raise RuntimeError(f'{len(errors)} year batches failed; preserve this run and retry separately')

if __name__=='__main__':main()
