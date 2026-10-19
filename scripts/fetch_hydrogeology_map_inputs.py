"""Download public cartographic inputs into an isolated, auditable run directory."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'ne_10m_rivers_lake_centerlines.zip': 'https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_rivers_lake_centerlines.zip',
    'ne_10m_admin_0_boundary_lines_land.zip': 'https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_boundary_lines_land.zip',
    'ne_10m_land.zip': 'https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip',
    'ne_terms.html': 'https://www.naturalearthdata.com/about/terms-of-use/',
    'whymap_capabilities.xml': 'https://services.bgr.de/wms/grundwasser/whymap/?SERVICE=WMS&REQUEST=GetCapabilities',
    'whymap_maps_data.html': 'https://www.bgr.bund.de/whymap/EN/Maps_Data/maps_data_node_en.html',
    'un_china.html': 'https://www.un.org/geospatial/content/china',
    'un_m49.html': 'https://unstats.un.org/unsd/methodology/m49/',
}
MAP_SOURCES = {
    'ne_10m_land.zip': SOURCES['ne_10m_land.zip'],
    'whymap_service.json': 'https://services.bgr.de/arcgis/rest/services/grundwasser/whymap_gwr/MapServer?f=pjson',
    'whymap_layer11.json': 'https://services.bgr.de/arcgis/rest/services/grundwasser/whymap_gwr/MapServer/11?f=pjson',
    'whymap_aquifers.geojson': 'https://services.bgr.de/arcgis/rest/services/grundwasser/whymap_gwr/MapServer/11/query?where=1%3D1&geometry=103%2C32%2C134%2C51&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=*&returnGeometry=true&outSR=4326&maxAllowableOffset=0.005&f=geojson',
}
UN_SOURCES = {
    'un_eastern_asia_page.html': 'https://www.un.org/geospatial/node/2695',
    'un_eastern_asia_2023.pdf': 'https://www.un.org/geospatial/sites/www.un.org.geospatial/files/files/documents/2023/Mar/2023_om_un_reg_easternasia_4658.pdf',
    'un_eastern_asia_2023_alt.pdf': 'https://www.un.org/geospatial/sites/www.un.org.geospatial/files/2023-03/2023_om_un_reg_easternasia_4658.pdf',
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--url', help='Fetch one additional public source')
    p.add_argument('--name', help='Local name for --url')
    p.add_argument('--map-stage', action='store_true')
    p.add_argument('--un-stage', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('Output must be inside the project')
    out.mkdir(parents=True, exist_ok=True)
    sources = {args.name: args.url} if args.url and args.name else (UN_SOURCES if args.un_stage else MAP_SOURCES if args.map_stage else SOURCES)
    manifest_path = out / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    for name, url in sources.items():
        target = out / name
        if target.exists():
            print(f'Already present: {name}', flush=True)
            continue
        try:
            response = requests.get(url, timeout=55)
            response.raise_for_status()
            target.write_bytes(response.content)
            manifest[name] = {'url': url, 'final_url': response.url,
                              'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                              'size_bytes': len(response.content),
                              'sha256': hashlib.sha256(response.content).hexdigest()}
            print(f'{name}: {len(response.content)} bytes', flush=True)
        except requests.RequestException as exc:
            manifest[name] = {'url': url, 'error': str(exc)}
            print(f'{name}: {type(exc).__name__}: {exc}', flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
