"""ERA5-Land/Noah extraction on one explicit static support; no credential edits."""
from __future__ import annotations
import argparse
import concurrent.futures
import time
from pathlib import Path
import ee
import pandas as pd
import numpy as np
from ee_ncp_closure_reextract import build as old_support, CONFIG
from ncp_monitoring_core import ROOT,EVIDENCE,RUNTIME,json_write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',default='fire-risk-460605');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    for gate in ['baseline','repeatability']:
        import json
        assert json.loads((EVIDENCE/gate/'acceptance.json').read_text())['pass_gate']
    dest=EVIDENCE/'extraction';dest.mkdir(parents=True,exist_ok=False)
    ee.Initialize(project=a.project);ee.data.setDeadline(180000)
    CONFIG['project']=a.project
    original,_,obs=old_support();projection=ee.Projection('EPSG:4326').atScale(10000)
    shared=ee.Image(original.first()).select('tws').mask()
    source=ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').filterDate('2003-01-01','2024-09-01')
    def convert(i):
        s=[i.select(f'volumetric_soil_water_layer_{j}') for j in range(1,5)]
        soil2=s[0].multiply(7).add(s[1].multiply(21)).add(s[2].multiply(72)).add(s[3].multiply(100)).rename('era_soil2_cm')
        soil289=soil2.add(s[3].multiply(89)).rename('era_soil289_cm')
        r=ee.Image.cat([soil2,soil289,i.select('snow_depth_water_equivalent').multiply(100).rename('era_snow_cm'),
                        i.select('skin_reservoir_content').multiply(100).rename('era_skin_cm'),
                        i.select('total_precipitation_sum').multiply(1000).rename('precip_mm')]).reproject(projection)
        return r.set('system:time_start',i.date().millis()).set('month_tag',i.date().format('YYYY-MM'))
    era=source.map(convert)
    valid=era.map(lambda i:i.mask().reduce(ee.Reducer.min()).unmask(0)).min().gt(0)
    shared=shared.And(valid).selfMask().reproject(projection)
    archived=pd.read_csv(ROOT/'data/raw/ncp_grid_definition.csv');features=[]
    for row in archived.itertuples():
        _,lon,lat=row.grid_id.split('_');lon,lat=float(lon),float(lat)
        features.append(ee.Feature(ee.Geometry.Rectangle([lon,lat,lon+1,lat+1],geodesic=False),{'grid_id':row.grid_id}))
    grid=ee.FeatureCollection(features);area=ee.Image.pixelArea().updateMask(shared).rename('support_area_m2')
    bands=['era_soil2_cm','era_soil289_cm','era_snow_cm','era_skin_cm','precip_mm','tws','sms','swe','cwa']
    empty=ee.Image.constant(0).rename('tws').updateMask(ee.Image.constant(0))
    gldas=ee.ImageCollection(CONFIG['gldas'])
    def summarize(i):
        match=original.filter(ee.Filter.eq('month_tag',i.get('month_tag')))
        tws=ee.Image(ee.Algorithms.If(match.size().gt(0),ee.Image(match.first()).select('tws'),empty))
        lsm=gldas.filterDate(i.date(),i.date().advance(1,'month')).mean()
        sms=lsm.select(['SoilMoi0_10cm_inst','SoilMoi10_40cm_inst','SoilMoi40_100cm_inst','SoilMoi100_200cm_inst']).reduce(ee.Reducer.sum()).divide(10).rename('sms')
        noah=ee.Image.cat([sms,lsm.select('SWE_inst').divide(10).rename('swe'),lsm.select('CanopInt_inst').divide(10).rename('cwa')]).reproject(projection)
        im=i.addBands(tws).addBands(noah).updateMask(shared)
        def finish(f):
            return ee.Feature(None,{b:f.get(b) for b in bands+['support_area_m2']}).set('grid_id',f.get('grid_id')).set('date',i.date().format('YYYY-MM-dd'))
        return im.multiply(area).addBands(area).reduceRegions(collection=grid,reducer=ee.Reducer.sum(),crs='EPSG:4326',scale=10000,tileScale=4).map(finish)
    config=dict(project=a.project,collection='ECMWF/ERA5_LAND/MONTHLY_AGGR',crs='EPSG:4326',scale=10000,
                static_support='intersection of original shared GRACE/Noah/lowland support and all ERA5-Land monthly component masks',
                baseline='2003-01 through 2014-12; GRACE observed months for TWS, all calendar months for land states',
                gain_factors=False,source_sha256=sha(__file__),input_sha256=sha(ROOT/'data/raw/grid_monthly_panel.csv'))
    json_write(dest/'specification.json',config);start=time.time();frames=[];errors=[]
    def fetch(year):
        collection=era.filterDate(f'{year}-01-01',f'{year+1}-01-01');n=collection.size().getInfo();images=collection.toList(n)
        records=ee.FeatureCollection([summarize(ee.Image(images.get(k))) for k in range(n)]).flatten().getInfo()
        frame=pd.DataFrame([r['properties'] for r in records['features']])
        for band in bands:frame[band]=pd.to_numeric(frame.get(band),errors='coerce')/frame.support_area_m2
        frame.to_csv(dest/f'year_{year}.csv',index=False,lineterminator='\n');return frame
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(fetch,y):y for y in range(2003,2025)}
        for future in concurrent.futures.as_completed(futures):
            year=futures[future]
            try:frames.append(future.result());print('ERA5/Noah',year,'seconds',round(time.time()-start),flush=True)
            except Exception as e:errors.append(dict(year=year,error=str(e)));print('Failed',year,str(e)[:220],flush=True)
    if frames:
        raw=pd.concat(frames).sort_values(['grid_id','date']);raw['date']=pd.to_datetime(raw.date)
        # Earth Engine sum over an entirely masked band can return zero. Such
        # values must never become observations or enter the GRACE baseline.
        raw['grace_observed']=raw.date.dt.strftime('%Y-%m').isin(obs)
        raw.loc[~raw.grace_observed,'tws']=np.nan
        raw.to_csv(dest/'monthly_absolute.csv',index=False,lineterminator='\n')
        baseline=raw.loc[raw.date.between('2003-01-01','2014-12-01')].groupby('grid_id')[bands[:-1]+['cwa']].mean(numeric_only=True)
        for name in bands:
            if name!='precip_mm':raw[name+'_anomaly']=raw[name]-raw.grid_id.map(baseline[name])
        raw['era_gwa2_cm']=raw.tws_anomaly-raw.era_soil2_cm_anomaly-raw.era_snow_cm_anomaly-raw.era_skin_cm_anomaly
        raw['era_gwa289_cm']=raw.tws_anomaly-raw.era_soil289_cm_anomaly-raw.era_snow_cm_anomaly-raw.era_skin_cm_anomaly
        raw['noah_gwa_cm']=raw.tws_anomaly-raw.sms_anomaly-raw.swe_anomaly-raw.cwa_anomaly
        raw.to_csv(dest/'monthly_anomalies.csv',index=False,lineterminator='\n')
        config['checks']=dict(rows=len(raw),unique_months=raw.date.nunique(),negative_precipitation=int(raw.precip_mm.lt(0).sum()),
                              support_relative_range=float(raw.groupby('grid_id').support_area_m2.apply(lambda s:(s.max()-s.min())/s.mean()).max()))
    config['errors']=errors;json_write(dest/'manifest.json',config)
    if errors:raise RuntimeError('Incomplete extraction; failed years preserved, no scientific replacement')

if __name__=='__main__':main()
