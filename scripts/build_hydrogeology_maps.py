"""Create hydrogeological location map (HLM) and study area map (SAM) from analysis geometry and public GIS data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import fitz
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from shapely.geometry import box

from prepare_hydrogeology_submission import build_pdf

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'submission/hydrogeology-journal/2026-10-19-hydrogeology-r6'
GRID = ROOT / 'outputs/figure_revision/2026-10-17-final/tables/fig01_analysis_cells.csv'
WELLS = ROOT / 'outputs/figure_revision/2026-10-17-final/tables/fig01_fixed_wells.csv'
LOWLAND = ROOT / 'manuscripts/assets/jhrs-v30/figure_01_lowland_mask_200m.geojson'
BLUE, RED, BLACK = '#286294', '#a64242', '#222222'
HYDRO_COLORS = {'major groundwater basin': '#c7dce9',
                'complex hydrogeological structures': '#b7b7b7',
                'local and shallow aquifers': '#e5e5e5'}
LOCATOR = (104.0, 34.0, 129.0, 51.0)
DETAIL = (112.5, 34.2, 120.6, 41.0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, text):
    Path(p).write_text(text.rstrip() + '\n', encoding='utf-8', newline='\n')


def dump(p, obj):
    write(p, json.dumps(obj, ensure_ascii=False, indent=2))


def clip_frame(frame, bounds):
    out = frame.loc[frame.intersects(box(*bounds))].copy()
    out.geometry = out.geometry.make_valid().intersection(box(*bounds))
    return out.loc[~out.is_empty & out.geometry.notna()].copy()


def configure(ax, bounds, small=False):
    ax.set(xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]))
    ax.set_aspect(1 / np.cos(np.deg2rad((bounds[1] + bounds[3]) / 2)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}°E'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}°N'))
    ax.tick_params(labelsize=8, length=2.5, pad=2)
    ax.set_facecolor('#f5f8fa')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.55)
        spine.set_color('#555555')
    ax.set_xticks([105, 115, 125] if small else [114, 116, 118, 120])
    ax.set_yticks([35, 40, 45, 50] if small else [35, 37, 39, 41])


def label(ax, x, y, value, color=BLACK, fontsize=8, **kwargs):
    return ax.text(x, y, value, color=color, fontsize=fontsize, zorder=25,
                   path_effects=[pe.withStroke(linewidth=1.5, foreground='white', alpha=.9)],
                   **kwargs)


def draw_maps(dest, inputs):
    assets = dest / '06_map_sources'
    assets.mkdir(exist_ok=True)
    grids = pd.read_csv(GRID)
    wells = pd.read_csv(WELLS)
    assert len(grids) == 27
    assert grids.groupby('region_tag').size().to_dict() == {'north': 11, 'south': 16}
    assert wells.well_id.is_unique
    assert wells.groupby('aquifer_group').size().to_dict() == {'confined': 284, 'unconfined': 310}
    common = set(wells.loc[wells.aquifer_group.eq('confined'), 'grid_id']) & set(wells.loc[wells.aquifer_group.eq('unconfined'), 'grid_id'])
    assert len(common) == 18
    assert grids.loc[grids.grid_id.isin(common)].groupby('region_tag').size().to_dict() == {'north': 10, 'south': 8}
    low = gpd.read_file(LOWLAND).to_crs(4326)
    land = clip_frame(gpd.read_file(inputs / 'ne_10m_land.zip', bbox=LOCATOR), LOCATOR)
    rivers = clip_frame(gpd.read_file(inputs / 'ne_10m_rivers_lake_centerlines.zip', bbox=LOCATOR), LOCATOR)
    borders = clip_frame(gpd.read_file(inputs / 'ne_10m_admin_0_boundary_lines_land.zip', bbox=LOCATOR), LOCATOR)
    # Deliberately exclude overlay limits; the Korean military demarcation line
    # has a separate dashed convention and is not a normal international border.
    borders = borders.loc[borders.FEATURECLA.str.startswith(('International boundary', 'Line of control'))].copy()
    hydro = clip_frame(gpd.read_file(inputs / 'whymap_aquifers.geojson').to_crs(4326), LOCATOR)
    assert set(hydro.aquif_type) == set(HYDRO_COLORS)
    # Dissolve recharge subclasses ONLY for regional cartographic context.
    hydro = hydro[['aquif_type', 'geometry']].dissolve('aquif_type', as_index=False)
    hydro.geometry = hydro.geometry.intersection(land.geometry.union_all())
    selected = rivers.loc[rivers.name_en.isin(['Yellow', 'Hai', 'Yongding', 'Wei', 'Luan', 'Zhang', 'Qin He'])].copy()
    for name, frame in [('regional_land', land), ('regional_land_borders', borders),
                        ('regional_aquifer_environments', hydro), ('context_rivers', selected)]:
        frame.to_file(assets / (name + '.geojson'), driver='GeoJSON')
    shutil.copy2(LOWLAND, assets / 'lowland_context_200m.geojson')
    shutil.copy2(GRID, assets / 'analysis_cells.csv')
    shutil.copy2(WELLS, assets / 'fixed_wells.csv')
    for name in ['download_manifest.json', 'whymap_layer11.json', 'whymap_service.json']:
        shutil.copy2(inputs / name, assets / name)
    cell_geo = gpd.GeoDataFrame(grids, geometry=[box(r.west, r.south, r.east, r.north) for r in grids.itertuples()], crs=4326)
    cell_geo.to_file(assets / 'analysis_cells.geojson', driver='GeoJSON')
    gpd.GeoDataFrame(wells, geometry=gpd.points_from_xy(wells.lon, wells.lat), crs=4326).to_file(assets / 'fixed_wells.geojson', driver='GeoJSON')

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none', 'hatch.linewidth': .45,
                         'axes.unicode_minus': False})
    # Final publication width: 176 mm. Coordinates reserve explicit space for
    # legends and labels instead of shrinking all map labels after drawing.
    fig = plt.figure(figsize=(176 / 25.4, 155 / 25.4), facecolor='white')
    a = fig.add_axes([.070, .650, .43, .310])
    b = fig.add_axes([.070, .105, .415, .430])
    c = fig.add_axes([.565, .105, .415, .430])
    for ax in [a, b, c]:
        configure(ax, LOCATOR if ax is a else DETAIL, small=ax is a)
        land.plot(ax=ax, color='#eeeeee' if ax is a else '#f0f0f0', edgecolor='#b0b0b0', linewidth=.4, zorder=1)
    for kind, color in HYDRO_COLORS.items():
        hydro.loc[hydro.aquif_type.eq(kind)].plot(ax=a, color=color, edgecolor='none', zorder=2)
    # Clip generalized hydro polygons to independently sourced land; the
    # physical coastline remains visible even where global data have gaps.
    land.boundary.plot(ax=a, color='#929292', linewidth=.35, zorder=3)
    borders.loc[borders.FEATURECLA.str.startswith('International')].plot(ax=a, color=BLACK, linewidth=.55, zorder=4)
    borders.loc[borders.FEATURECLA.str.startswith('Line of control')].plot(ax=a, color=BLACK, linewidth=.55, linestyle='--', zorder=4)
    selected.loc[selected.name_en.eq('Yellow')].plot(ax=a, color=BLUE, linewidth=.45, zorder=5)
    a.add_patch(Rectangle((112.0, 34.5), 8.0, 6.1, fill=False, lw=1.05, edgecolor=RED, zorder=8))
    label(a, 110.1, 39.5, 'China', weight='bold')
    label(a, 109.0, 47.0, 'Mongolia')
    label(a, 120.7, 49.1, 'Russian\nFederation', ha='center', linespacing=1.05)
    label(a, 126.4, 40.5, 'PRK', ha='center')
    label(a, 127.1, 35.7, 'KOR', ha='center')
    label(a, 120.6, 37.2, 'b, c', color=RED, weight='bold')
    a.set_title('(a) Regional hydrogeological setting', loc='left', fontsize=9, pad=5)

    for ax in [b, c]:
        low.plot(ax=ax, color='#e0e0e0', edgecolor='#747474', linewidth=.5, zorder=2)
        selected.plot(ax=ax, color=BLUE, linewidth=.55, zorder=6)
        for row in grids.itertuples():
            col = BLUE if row.region_tag == 'north' else RED
            if ax is b:
                ax.add_patch(Rectangle((row.west, row.south), 1, 1,
                                      facecolor=col, alpha=.11, edgecolor='none', zorder=3))
            ax.add_patch(Rectangle((row.west, row.south), 1, 1, facecolor='none',
                                  edgecolor=col if ax is b else '#b4b4b4', lw=.55, zorder=4))
            if ax is b and abs((row.south + row.north) / 2 - 37) < .01:
                ax.add_patch(Rectangle((row.west, row.south), 1, 1, fill=False,
                                      hatch='///', edgecolor='#9a7777', lw=0, zorder=5))
            if ax is c and row.grid_id in common:
                ax.add_patch(Rectangle((row.west, row.south), 1, 1, fill=False,
                                      edgecolor=BLACK, lw=.85, zorder=5))
        ax.axhline(37, c=BLACK, ls=(0, (4, 2.5)), lw=.8, zorder=9)
        configure(ax, DETAIL)
    b.set_title('(b) Study domain and regional grouping\n11 northern / 16 southern cells', loc='left', fontsize=9, pad=6)
    c.set_title('(c) Matched fixed well cohorts\n18 common cells: 10 north / 8 south', loc='left', fontsize=9, pad=6)
    for x, y, name, dx, dy in [(116.40,39.90,'Beijing',.15,.20), (117.20,39.12,'Tianjin',.23,-.08)]:
        b.scatter(x, y, marker='o', s=7, c=BLACK, zorder=22)
        label(b, x+dx, y+dy, name)
    label(b, 114.85, 38.65, 'Hebei', color='#4e4e4e', fontstyle='italic')
    b.scatter(114.51, 38.04, marker='o', s=6, c=BLACK, zorder=22)
    label(b, 114.67, 38.03, 'Shijiazhuang')
    label(b, 114.15, 35.15, 'Henan', color='#4e4e4e', fontstyle='italic')
    label(b, 118.2, 35.8, 'Shandong', color='#4e4e4e', fontstyle='italic')
    label(b, 113.45, 37.85, 'Taihang Mts', rotation=69, ha='center', color='#555555')
    label(b, 115.0, 40.7, 'Yanshan Mts', color='#555555')
    label(b, 119.5, 39.3, 'Bohai\nSea', color=BLUE, ha='center', linespacing=1.05)
    label(b, 117.35, 36.6, 'Yellow River', rotation=15, color=BLUE)
    # Hai is short at this scale: a leader locates the actual river near Tianjin.
    b.annotate('Hai River', xy=(117.45, 38.98), xytext=(118.25, 38.1), fontsize=8,
               color=BLUE, arrowprops={'arrowstyle':'-', 'color':BLUE, 'lw':.55},
               zorder=24, path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
    for aq, marker, face, size, alpha in [('unconfined','o','#555555',5,.65),
                                          ('confined','^','none',9,.8)]:
        q = wells.loc[wells.aquifer_group.eq(aq)]
        c.scatter(q.lon, q.lat, s=size, marker=marker, facecolors=face,
                  edgecolors=BLACK, linewidths=.35, alpha=alpha, zorder=15)

    # Shared legend in the otherwise unused upper-right space.
    leg = fig.add_axes([.555, .615, .435, .36]); leg.axis('off')
    leg.text(0, .97, 'Legend', fontsize=9, weight='bold', va='top')
    entries = [Patch(facecolor=HYDRO_COLORS['major groundwater basin'], edgecolor='#777777', lw=.35, label='Major groundwater basin'),
               Patch(facecolor=HYDRO_COLORS['complex hydrogeological structures'], edgecolor='#777777', lw=.35, label='Complex hydrogeological structures'),
               Patch(facecolor=HYDRO_COLORS['local and shallow aquifers'], edgecolor='#777777', lw=.35, label='Local and shallow aquifers'),
               Rectangle((0,0),1,1,fill=False,edgecolor=RED,lw=1,label='Study bounding box (a)'),
               Line2D([],[],color=BLACK,lw=.55,label='Land border (a)'),
               Line2D([],[],color=BLACK,ls='--',lw=.55,label='Military demarcation line (a)'),
               Line2D([],[],color=BLUE,lw=.65,label='Selected rivers'),
               Patch(facecolor='#e0e0e0',edgecolor='#747474',lw=.5,label='Lowland context, ≤200 m (b, c)'),
               Line2D([],[],color=BLACK,ls='--',lw=.8,label='37°N analysis grouping line'),
               Patch(facecolor='none',edgecolor='#9a7777',hatch='///',lw=.35,label='37°N-centered row: south (b)')]
    leg.legend(handles=entries, loc='upper left', bbox_to_anchor=(-.02,.88), frameon=False,
               fontsize=8, handlelength=1.6, handletextpad=.6, labelspacing=.3, borderaxespad=0)
    leg.text(0,.14,'WHYMAP: regional context at 1:25 million;\nrecharge subclasses merged by aquifer type.',
             fontsize=8, va='top', linespacing=1.25)
    bottom = fig.add_axes([.07, .012, .91, .055]); bottom.axis('off')
    handles = [Patch(facecolor='#e8eff5',edgecolor=BLUE,lw=.6,label='North'),
               Patch(facecolor='#f6e8e8',edgecolor=RED,lw=.6,label='South'),
               Line2D([],[],marker='o',ls='none',markerfacecolor='#555555',markeredgecolor=BLACK,markersize=3,label='Unconfined: 310'),
               Line2D([],[],marker='^',ls='none',markerfacecolor='none',markeredgecolor=BLACK,markersize=4,label='Confined: 284')]
    bottom.legend(handles=handles,loc='center',ncol=4,frameon=False,fontsize=8,
                  columnspacing=1.1,handletextpad=.4,handlelength=1.5)
    for ext in ['pdf','svg','png']:
        fig.savefig(dest / '01_manuscript' / ('Fig01.'+ext), dpi=600, facecolor='white')
    plt.close(fig)
    # Export individual vector maps as convenient review artifacts.
    p = fitz.open(dest / '01_manuscript/Fig01.pdf')
    w, h = p[0].rect.width, p[0].rect.height
    for name, rect in [('HLM',fitz.Rect(0,0,w,h*.406)), ('SAM',fitz.Rect(0,h*.406,w,h))]:
        out = fitz.open(); page=out.new_page(width=rect.width,height=rect.height)
        page.show_pdf_page(page.rect,p,0,clip=rect)
        out.save(assets / (name+'.pdf'))
        page.get_pixmap(matrix=fitz.Matrix(3,3)).save(assets/(name+'.png'))
        out.close()
    p.close()
    result = {'analysis_cells':27,'north_cells':11,'south_cells':16,'common_cells':18,
              'unconfined_wells':310,'confined_wells':284,'common_north_cells':10,'common_south_cells':8,
              'hydrogeological_context_only':True,'recharge_subclasses_merged':True,
              'geographic_crs':'EPSG:4326','locator_bbox':LOCATOR,'detail_bbox':DETAIL,
              'figure_width_mm':176,'figure_height_mm':155,'minimum_label_pt':8,
              'inputs':{p.relative_to(ROOT).as_posix():sha(p) for p in [GRID,WELLS,LOWLAND]},
              'figure_sha256':sha(dest/'01_manuscript/Fig01.pdf')}
    dump(dest/'05_provenance_and_checks/map_validation.json',result)
    return result


CAPTION = r'''Hydrogeological setting, analysis domain and fixed well cohorts. (a) Regional aquifer environments generalized from WHYMAP \citep{bgr2008whymap}; the red box locates panels b--c. PRK and KOR denote the Democratic People's Republic of Korea and Republic of Korea. (b) The 27 one-degree cells (11 northern, blue; 16 southern, red), with a contextual lowland outline ($\leq$200 m). The dashed line is the 37$^{\circ}$ N analysis division, not a hydrogeological boundary; the six hatched cells centered on it belong to the south and are reassigned or excluded in sensitivity analyses. (c) The 18 common cells (ten northern, eight southern; black outlines), containing 310 unconfined wells (filled circles) and 284 confined wells (open triangles); temporal coverage is in Table~S19. Other cells have gray outlines. Physical geography and land borders use Natural Earth \citep{naturalearth2026maps}; country names follow United Nations M49 conventions. WHYMAP (\copyright{} BGR Hannover and UNESCO Paris) provides regional context, not local aquifer limits or depth-specific well classifications.'''


NEW_BIB = r'''
@misc{bgr2008whymap,
  author = {{BGR} and {UNESCO}},
  title = {Groundwater Resources of the World, 1:25,000,000 ({WHYMAP})},
  year = {2008},
  howpublished = {BGR, Hannover and UNESCO, Paris},
  url = {https://services.bgr.de/arcgis/rest/services/grundwasser/whymap_gwr/MapServer},
  note = {Aquifer-type context from GIS service layer 11; accessed 19 October 2026}
}
@misc{naturalearth2026maps,
  author = {{Natural Earth}},
  title = {1:10m land, rivers and lake centerlines, and land boundary lines},
  year = {2026},
  url = {https://www.naturalearthdata.com/downloads/},
  note = {Public-domain cartographic data; accessed 19 October 2026}
}
'''


def update_text(dest):
    p = dest/'01_manuscript/manuscript.tex'
    text = p.read_text(encoding='utf-8')
    start = text.index(r'\caption{Study domain and fixed well cohorts.')
    end = text.index('\n\\label{fig:study-area}',start)
    text = text[:start]+r'\caption{'+CAPTION+'}'+text[end:]
    key = 'The grouping is therefore a geographic latitude boundary rather than a hydrogeological one, and several cells share a native satellite footprint.'
    text = text.replace(key,key+' Regional aquifer environments are shown as cartographic context in Fig.~\\ref{fig:study-area}a; these generalized units do not define the north--south groups or the aquifer classes of individual wells.')
    write(p,text)
    with (dest/'01_manuscript/references.bib').open('a',encoding='utf-8') as f:
        f.write(NEW_BIB)


def compile_main(dest, label):
    work=ROOT/'outputs/build'/label/'hydrogeology-journal/main'
    if work.exists():
        raise FileExistsError('Build label already exists')
    shutil.copytree(dest/'01_manuscript',work)
    p=work/'manuscript.tex'
    original=p.read_text(encoding='utf-8')
    write(p,original.replace(r'\input{references.bbl}',r'\bibliography{references}'))
    build_pdf(work,'manuscript',bib=True)
    shutil.copy2(work/'manuscript.bbl',work/'references.bbl')
    write(p,original)
    report=build_pdf(work,'manuscript')
    for name in ['manuscript.pdf','references.bbl']:
        shutil.copy2(work/name,dest/'01_manuscript'/name)
    doc=fitz.open(dest/'01_manuscript/manuscript.pdf')
    report.update({'pages':len(doc),'figure1_pdf_pages':[i+1 for i,p in enumerate(doc) if 'Fig. 1 Hydrogeological' in p.get_text()]})
    doc.close()
    return report


def package(dest):
    for folder,filename in [('01_manuscript','manuscript_latex_source.zip')]:
        with zipfile.ZipFile(dest/filename,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted((dest/folder).iterdir()):
                if p.suffix in ['.tex','.bib','.bbl','.bst','.pdf'] and p.name not in ['manuscript.pdf','ESM1.pdf']:
                    z.write(p,p.name)
    shutil.copy2(BASE/'ESM1_latex_source.zip',dest/'ESM1_latex_source.zip')
    dump(dest/'package_manifest.json',{'schema_version':1,'label':dest.name,
         'created_at_utc':datetime.now(timezone.utc).isoformat(),'submission_ready':False,
         'note':'Manifest excludes itself. Isolated r7 maps; author fields remain incomplete.',
         'files':[{'path':p.relative_to(dest).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha(p)}
                  for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='package_manifest.json']})


def finalize_metadata(dest, inputs, result, build_label):
    """Replace inherited r6 status records with evidence specific to this map revision."""
    record=dest/'05_provenance_and_checks'
    r6hashes=json.loads((record/'r6_preservation_hashes.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in r6hashes.items()), 'The r6 package changed during map work'
    protected=json.loads((record/'protected_source_hashes.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in protected.items()), 'A canonical manuscript or shared asset changed'
    old=(BASE/'01_manuscript/manuscript.tex').read_text(encoding='utf-8')
    new=(dest/'01_manuscript/manuscript.tex').read_text(encoding='utf-8')
    def strip_caption(s):
        return re.sub(r'\\caption\{(?:Study domain and fixed well cohorts\.|Hydrogeological setting, analysis domain and fixed well cohorts\.).*?\}\n(?=\\label\{fig:study-area\})','',s,flags=re.S)
    context=' Regional aquifer environments are shown as cartographic context in Fig.~\\ref{fig:study-area}a; these generalized units do not define the north--south groups or the aquifer classes of individual wells.'
    assert strip_caption(old)==strip_caption(new.replace(context,'')), 'Unexpected prose edits outside map scope'
    unchanged=[]
    for folder in ['02_supplement','03_submission_documents']:
        for p in (BASE/folder).rglob('*'):
            if p.is_file():
                assert sha(p)==sha(dest/p.relative_to(BASE))
                unchanged.append(p.relative_to(BASE).as_posix())
    for i in range(2,7):
        name=f'01_manuscript/Fig{i:02}.pdf'
        assert sha(BASE/name)==sha(dest/name)
        unchanged.append(name)
    report=json.loads((record/'build_validation.json').read_text(encoding='utf-8'))
    report.update({'label':dest.name,'base_label':BASE.name,'map_build_label':build_label,
                   'base_git_commit':subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   'main':result['main_build'],'cited_reference_count':40,
                   'protected_sources_unchanged':True,'r6_package_unchanged':True,
                   'map_related_prose_only':True,'unchanged_files':unchanged,
                   'map_requirement':'HLM and SAM supplied in Fig01; UN M49 names and a regional visual boundary comparison recorded in 06_map_sources/README.md',
                   'main_payload_bytes':sum(p.stat().st_size for p in (dest/'01_manuscript').iterdir() if p.is_file()),
                   'outstanding':['Other author emails and available ORCIDs','Author-verified declarations',
                                  'Public data/code archival identifier','Verified AI tool names, versions and roles']})
    dump(record/'build_validation.json',report)
    source_map=json.loads((record/'source_mapping.json').read_text(encoding='utf-8'))
    for entry in source_map:
        if entry.get('package_path')=='01_manuscript/Fig01.pdf':
            entry.update({'source':'scripts/build_hydrogeology_maps.py',
                          'source_kind':'new figure from recorded vector inputs',
                          'sha256':sha(dest/'01_manuscript/Fig01.pdf'),
                          'input_manifest':'06_map_sources/download_manifest.json'})
    dump(record/'source_mapping.json',source_map)
    # Keep the two original base checks as explicit history, not current evidence.
    for name in ['pdf_and_content_validation.json','portable_source_validation.json']:
        p=record/name
        base=record/('r6_'+name)
        if p.exists() and not base.exists():
            shutil.copy2(p,base)
    sources=dest/'06_map_sources'
    m49=json.loads((inputs/'download_manifest.json').read_text(encoding='utf-8'))
    dump(sources/'un_reference_check.json',{
        'checked_on_local':'2026-10-19',
        'names':'UN Statistics Division M49: China, Mongolia, Russian Federation, PRK and KOR',
        'names_url':'https://unstats.un.org/unsd/methodology/m49/',
        'border_reference':'United Nations, Map of the World, Map No.4651 Rev.4, July 2026',
        'official_landing_page':'https://www.un.org/geospatial/content/map-world-1',
        'reference_file':m49.get('un_map4651rev4_july2026_reference.jpg'),
        'limitation':'Official UN file endpoints returned HTTP403/SSL errors. The UN-authored reference image was viewed from a public mirror, with UN title, map number and date visible. This is a visual regional convention check, not coordinate-level validation or verification of mirror authenticity by the UN.',
        'checks':['China-Mongolia and Mongolia-Russian Federation adjacency',
                  'China-Russian Federation and China-PRK boundaries in the visible northeast region',
                  'Korean military demarcation line distinguished from international land borders',
                  'No offshore country boundaries or distant disconnected territories drawn'],
        'un_map_artwork_reproduced_in_figure':False})
    write(sources/'README.md','''# Figure 1 Sources and Usage Notes

Figure 1 comprises a regional hydrogeological location map (HLM, panel a), a detailed study area map (SAM, panel b), and monitoring well locations (panel c). The main manuscript incorporates 01_manuscript/Fig01.pdf. Standalone maps are provided as HLM.pdf and SAM.pdf, with PNG previews. Vector artwork is available as 01_manuscript/Fig01.svg. Cartographic layers are stored as EPSG:4326 GeoJSON. Maps are generated via scripts/build_hydrogeology_maps.py, and public data are retrieved via scripts/fetch_hydrogeology_map_inputs.py.

## Data Sources and Licensing

| Layer | Source and Processing | Usage Terms |
| --- | --- | --- |
| Regional aquifer system types | BGR/UNESCO, Groundwater Resources of the World (WHYMAP), 2008, 1:25,000,000 (layer 11 of the official GIS service). Aquifer types merged and clipped to the regional extent. | Used and cited per BGR/UNESCO terms (https://www.bgr.bund.de/whymap/EN/Maps_Data/maps_data_node_en.html). |
| Land, coastlines, rivers, international boundaries | Natural Earth 1:10m. Relevant land boundaries retained; Korean military demarcation line shown as a dashed line; offshore maritime boundaries omitted. | Public domain (https://www.naturalearthdata.com/about/terms-of-use/). |
| Lowland elevation contour | Lowland mask (<= 200 m elevation) derived from SRTM/GAUL2015. | Display contour for regional topography. |
| Grid cells and wells | Study analysis coordinates: 27 cells (11 North, 16 South; 18 overlapping cells); 310 unconfined wells, 284 confined wells. | Study analysis grid and monitoring network. |
| Country names and codes | UN M49 standard (https://unstats.un.org/unsd/methodology/m49/). | Country naming and coding conventions. |
| Boundary convention reference | UN Map of the World, No. 4651 Rev. 4 (July 2026). Used for visual comparison of visible land boundaries and adjacency. | Visual geographic reference. |

## Cartographic Interpretation Boundaries

- WHYMAP serves solely as regional hydrogeological background and should not be interpreted as local aquifer boundaries, shallow/deep aquifer contact lines, or well screen depths. No unverified flow directions, cross-sections, or thicknesses are depicted.
- The 37°N line represents an analytical division for study grouping; the six cells centered on 37°N are assigned to the southern group. Neither 37°N nor the lowland contour represents a physical aquifer boundary.
- Panels b and c show an extent slightly wider than the study domain to present complete 1° grid cells; the red rectangle in panel a denotes the study extent (112–120°E, 34.5–40.6°N).
- Rivers provide generalized geographic reference and do not represent actual streamflow rates, recharge paths, or causal mechanisms during the study period.
- City, mountain, and provincial names serve as geographic labels.
''')
    readme_path = BASE / 'README.md'
    if readme_path.exists():
        readme = readme_path.read_text(encoding='utf-8')
        write(dest / 'README.md', readme)
    checklist_path = BASE / '04_journal_requirements/submission_checklist.md'
    if checklist_path.exists():
        text = checklist_path.read_text(encoding='utf-8')
        write(dest / '04_journal_requirements/submission_checklist.md', text)
    write(record / 'map_revision_log.md', f'''# HLM/SAM Map Revision Record

Date: 2026-10-19. Baseline: {BASE.name}. Build: {build_label}.

- Hydrogeological location map (HLM) added from WHYMAP; regional context, UN M49 country designations, and international land boundaries updated. SAM retains 27 cells, 37°N grouping, and 594 fixed wells.
- Geographic labels added for Shijiazhuang, mountain ranges, and rivers. No inferred subsurface flow arrows, depths, or causal paths added.
- Vector data directly sourced from Natural Earth and WHYMAP.
''')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label',required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--initialize',action='store_true')
    parser.add_argument('--build-label')
    args=parser.parse_args()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}-[a-z0-9-]+',args.label):
        raise ValueError('Invalid label')
    dest=ROOT/'submission/hydrogeology-journal'/args.label
    if args.initialize:
        if dest.exists():
            raise FileExistsError('Destination already exists')
        shutil.copytree(BASE,dest)
        baseline={p.relative_to(ROOT).as_posix():sha(p) for p in BASE.rglob('*') if p.is_file()}
        dump(dest/'05_provenance_and_checks/r6_preservation_hashes.json',baseline)
        update_text(dest)
    elif not dest.is_dir():
        raise FileNotFoundError('Initialize the isolated package first')
    else:
        p=dest/'01_manuscript/manuscript.tex'
        text=p.read_text(encoding='utf-8')
        start=text.index(r'\caption{Hydrogeological setting, analysis domain and fixed well cohorts.')
        end=text.index('\n\\label{fig:study-area}',start)
        write(p,text[:start]+r'\caption{'+CAPTION+'}'+text[end:])
    result=draw_maps(dest,args.inputs.resolve())
    if args.build_label:
        result['main_build']=compile_main(dest,args.build_label)
        dump(dest/'05_provenance_and_checks/map_validation.json',result)
        finalize_metadata(dest,args.inputs.resolve(),result,args.build_label)
    package(dest)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
