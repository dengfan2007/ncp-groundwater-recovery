"""Repeat archived storage extraction with current GEE inputs and archived month mask.

Provides a same-vintage control for the uniform-support re-extraction.
"""
import argparse,concurrent.futures,hashlib,json,time
from pathlib import Path
import ee
import pandas as pd
from ee_ncp_common import build_ncp_context
from ee_ncp_closure_reextract import CONFIG
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);ee.Initialize(project=a.project);ee.data.setDeadline(180000)
    archived=pd.read_csv(ROOT/'data/raw/grid_monthly_panel.csv',parse_dates=['date']);obs=sorted(archived.loc[archived.gwa_cm.notna(),'date'].dt.strftime('%Y-%m').unique())
    land=build_ncp_context(CONFIG['bbox'],CONFIG['province_names'],200)['lowland_mask'];bbox=ee.Geometry.Rectangle(CONFIG['bbox'],geodesic=False)
    grace=ee.ImageCollection(CONFIG['grace']).filterDate('2003-01-01','2024-09-01').map(lambda i:i.set('month_tag',i.date().format('YYYY-MM'))).filter(ee.Filter.inList('month_tag',obs));gldas=ee.ImageCollection(CONFIG['gldas'])
    def raw(text):
        d=ee.Date(text);end=d.advance(1,'month');g=gldas.filterDate(d,end).mean();t=grace.filterDate(d,end)
        tws=ee.Image(ee.Algorithms.If(t.size().gt(0),t.select('lwe_thickness').mean().rename('tws'),ee.Image.constant(0).rename('tws').updateMask(ee.Image.constant(0))))
        sms=g.select(['SoilMoi0_10cm_inst','SoilMoi10_40cm_inst','SoilMoi40_100cm_inst','SoilMoi100_200cm_inst']).reduce(ee.Reducer.sum()).divide(10).rename('sms')
        return ee.Image.cat([tws,sms,g.select('SWE_inst').divide(10).rename('swe'),g.select('CanopInt_inst').divide(10).rename('cwa')]).updateMask(land).clip(bbox).set('system:time_start',d.millis()).set('month_tag',d.format('YYYY-MM'))
    dates=pd.date_range('2003-01-01','2024-09-01',freq='MS');monthly=ee.ImageCollection.fromImages([raw(d.strftime('%Y-%m-%d')) for d in dates]);baseline=monthly.filterDate('2003-01-01','2015-01-01').mean()
    grid=[]
    for r in archived[['grid_id','region_tag']].drop_duplicates().itertuples():
        _,x,y=r.grid_id.split('_');x,y=float(x),float(y);grid.append(ee.Feature(ee.Geometry.Rectangle([x,y,x+1,y+1],geodesic=False),dict(grid_id=r.grid_id,region_tag=r.region_tag)))
    grid=ee.FeatureCollection(grid)
    def reduce(i):
        d=i.subtract(baseline).rename(['twsa_cm','smsa_cm','swea_cm','cwa_cm']);gwa=d.select('twsa_cm').subtract(d.select('smsa_cm')).subtract(d.select('swea_cm')).subtract(d.select('cwa_cm')).rename('gwa_cm')
        # Deliberately retain original per-band masks and implicit first-band projection.
        result=ee.Image.cat([gwa,d]).reduceRegions(collection=grid,reducer=ee.Reducer.mean(),scale=10000,tileScale=4)
        return result.map(lambda f:ee.Feature(None,f.toDictionary(['grid_id','region_tag','gwa_cm','twsa_cm','smsa_cm','swea_cm','cwa_cm'])).set('date',i.date().format('YYYY-MM-dd')))
    monthly=monthly.filter(ee.Filter.inList('month_tag',obs));start=time.time()
    def fetch(year):
        c=monthly.filterDate(f'{year}-01-01',f'{year+1}-01-01');n=c.size().getInfo();images=c.toList(n);fc=ee.FeatureCollection([reduce(ee.Image(images.get(i))) for i in range(n)]).flatten();data=pd.DataFrame([f['properties'] for f in fc.getInfo()['features']]);data.to_csv(out/f'year_{year}.csv',index=False,lineterminator='\n');return data
    frames=[];errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(fetch,y):y for y in range(2003,2025)}
        for f in concurrent.futures.as_completed(futures):
            year=futures[f]
            try:frames.append(f.result());print('Legacy control',year,'seconds',round(time.time()-start),flush=True)
            except Exception as e:errors.append(dict(year=year,error=str(e)));print('Failed',year,str(e)[:150],flush=True)
    if frames:pd.concat(frames).sort_values(['grid_id','date']).to_csv(out/'legacy_repeat_panel.csv',index=False,lineterminator='\n')
    manifest=dict(project=a.project,original_extraction_scale_m=10000,explicit_reprojection=False,shared_mask_enforced=False,gain_factors_applied=False,archived_observation_months=obs,errors=errors,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),retrieval_date='2026-10-17')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if errors:raise RuntimeError('Incomplete batches; inspect manifest')
if __name__=='__main__':main()
