"""r9 check 2 (extraction): native JPL mascon membership of the 27 analysis cells.

Read-only Earth Engine queries; nothing is uploaded.  For each 0.5-degree native pixel
inside a retained cell, record its lowland (<=200 m, five provinces) area and the full
non-CRI JPL RL06.3Mv04 mascon series.  Pixels with identical non-CRI series belong to the
same native mascon.  Output goes to a new directory.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import ee
import pandas as pd
from ee_ncp_common import build_ncp_context

ROOT=Path(__file__).resolve().parents[1]
CONFIG=dict(project='fire-risk-460605',bbox=[112.,34.5,120.,40.6],province_names=['Beijing Shi','Tianjin Shi','Hebei Sheng','Henan Sheng','Shandong Sheng'],
            elevation_max_m=200,mascon='NASA/GRACE/MASS_GRIDS_V04/MASCON',mascon_cri='NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI',area_scale_m=1000)

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--label',required=True);args=a.parse_args()
    out=ROOT/'outputs/r9_checks'/args.label/'mascon_r2';out.mkdir(parents=True,exist_ok=False)
    ee.Initialize(project=CONFIG['project'])
    cells=pd.read_csv(ROOT/'data/raw/grid_monthly_panel.csv',usecols=['grid_id','region_tag']).drop_duplicates('grid_id')
    land=build_ncp_context(CONFIG['bbox'],CONFIG['province_names'],CONFIG['elevation_max_m'])['lowland_mask']
    feats=[]
    for r in cells.itertuples():
        _,lon,lat=r.grid_id.split('_');lon,lat=float(lon),float(lat)
        for dx in (0,.5):
            for dy in (0,.5):
                feats.append(ee.Feature(ee.Geometry.Rectangle([lon+dx,lat+dy,lon+dx+.5,lat+dy+.5],proj='EPSG:4326',geodesic=False),
                    dict(grid_id=r.grid_id,region_tag=r.region_tag,pixel_lon=lon+dx,pixel_lat=lat+dy)))
    fc=ee.FeatureCollection(feats)
    area=ee.Image.pixelArea().updateMask(land.mask()).rename('lowland_m2')
    a_rows=area.reduceRegions(collection=fc,reducer=ee.Reducer.sum(),scale=CONFIG['area_scale_m'],tileScale=4).getInfo()['features']
    pix=pd.DataFrame([f['properties'] for f in a_rows]).rename(columns={'sum':'lowland_m2'})
    pts=ee.FeatureCollection([ee.Feature(ee.Geometry.Point([x+.25,y+.25]),dict(pixel_lon=x,pixel_lat=y)) for x,y in pix[['pixel_lon','pixel_lat']].drop_duplicates().itertuples(index=False)])
    series=[]
    for name,key in [('mascon','lwe_noncri'),('mascon_cri','lwe_cri')]:
        img=ee.ImageCollection(CONFIG[name]).filterDate('2002-04-01','2024-10-01').select('lwe_thickness').toBands()
        v=img.sampleRegions(pts,scale=55000,geometries=False).getInfo()['features']
        for f in v:
            p=f['properties'];lon,lat=p.pop('pixel_lon'),p.pop('pixel_lat')
            for band,value in p.items():series.append(dict(pixel_lon=lon,pixel_lat=lat,product=key,band=band.split('_lwe')[0],lwe_cm=value))
    pix.to_csv(out/'pixel_lowland_area.csv',index=False);pd.DataFrame(series).to_csv(out/'pixel_series.csv',index=False)
    # Full native footprints: fingerprint each 0.5-degree pixel in a wider window by two
    # non-CRI solutions (Jan 2010, Jun 2020; band names start with the solution's first day)
    # and record the lowland fraction inside the analysis domain.
    col=ee.ImageCollection(CONFIG['mascon']).select('lwe_thickness')
    def first(prefix):return col.filter(ee.Filter.stringStartsWith('system:index',prefix)).first()
    st=ee.Image.cat([first('201001').rename('v1'),first('202006').rename('v2'),land.mask().unmask(0).rename('frac')])
    import numpy as np
    wide=[ee.Feature(ee.Geometry.Rectangle([float(x),float(y),float(x)+.5,float(y)+.5],proj='EPSG:4326',geodesic=False),dict(lon=float(x),lat=float(y)))
          for x in np.arange(108,124,.5) for y in np.arange(31,44,.5)]
    f=st.reduceRegions(ee.FeatureCollection(wide),reducer=ee.Reducer.mean(),scale=2000,tileScale=8).getInfo()['features']
    pd.DataFrame([x['properties'] for x in f]).to_csv(out/'footprint_pixels.csv',index=False)
    (out/'extraction_manifest.json').write_text(json.dumps(dict(config=CONFIG,pixels=len(pix),series_rows=len(series),gain_factors_available_in_gee=False),indent=2)+'\n',encoding='utf-8')
    print(len(pix),len(series))

if __name__=='__main__':main()
