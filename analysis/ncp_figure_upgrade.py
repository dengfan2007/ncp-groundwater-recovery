"""Render the six-figure paper from frozen evidence; never fit research models.

Example (existing output directories are refused)::
  python analysis/ncp_figure_upgrade.py --evidence outputs/validation_revision/2026-10-17-r2 \
    --story outputs/story_revision/2026-10-17-r1 --review outputs/review_revision/2026-10-15-r1 \
    --output-dir outputs/runs/2026-10-17-figures-r1
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE, RED, BLACK, GRAY = '#286090', '#b33e36', '#222222', '#777777'
ROUTES = [('A', BLUE), ('D', RED)]
SCENARIOS = ['no_recovery', 'temporary_gain', 'sustained_reversal', 'reversal_then_acceleration']
NAMES = ['No recovery', 'Temporary gain', 'Sustained reversal', 'Two-stage']
MM = 1 / 25.4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def month(series):
    d = pd.to_datetime(series)
    return d.dt.year * 12 + d.dt.month - 1


class Renderer:
    def __init__(self, args):
        self.e, self.s, self.r = map(Path, [args.evidence, args.story, args.review])
        self.baseline=Path(args.baseline_dir) if args.baseline_dir else ROOT/'manuscripts/assets'
        self.out = Path(args.output_dir)
        self.out.mkdir(parents=True, exist_ok=False)
        for name in ['figures', 'tables', 'before']:
            (self.out / name).mkdir()
        self.inputs, self.panels, self.checks = {}, [], {}
        plt.rcParams.update({
            'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.titlesize': 9.5,
            'axes.labelsize': 9.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
            'legend.fontsize': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
            'lines.linewidth': 1.15, 'pdf.fonttype': 42, 'ps.fonttype': 42,
            'savefig.dpi': 300, 'axes.unicode_minus': True,
        })

    def read(self, path, **kwargs):
        path = path.resolve()
        self.inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        return pd.read_csv(path, **kwargs)

    def table(self, name, data):
        data.to_csv(self.out / 'tables' / (name + '.csv'), index=False, lineterminator='\n')

    def panel(self, figure, panels, sources, selection, units, uncertainty):
        self.panels.append(dict(figure=figure, panels=panels, sources=sources,
                                selection=selection, units=units, uncertainty=uncertainty))

    def save(self, fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(self.out / 'figures' / (name + '.' + ext), facecolor='white')
        old = self.baseline / (name + '.png')
        if old.exists():
            shutil.copy2(old, self.out / 'before' / old.name)
        plt.close(fig)

    def subplots(self, rows=1, cols=1, height=100, **kwargs):
        return plt.subplots(rows, cols, figsize=(160 * MM, height * MM), layout='constrained', **kwargs)

    def grid(self, ax, coverage=None, region_fill=True):
        # Analysis-cell geometry comes from archived bounds, not a synthetic coastline.
        ax.set_facecolor('#fafafa')
        for row in self.grids.itertuples():
            col = BLUE if row.region_tag == 'north' else RED
            ax.add_patch(Rectangle((row.west, row.south), row.east-row.west, row.north-row.south,
                                  facecolor=col if region_fill else 'white',
                                  alpha=.12 if region_fill else 1, edgecolor='none'))
            ax.add_patch(Rectangle((row.west, row.south), row.east-row.west, row.north-row.south,
                                  fill=False, edgecolor='#a0a0a0', lw=.55))
            if coverage is not None and row.grid_id in coverage:
                ax.add_patch(Rectangle((row.west, row.south), row.east-row.west, row.north-row.south,
                                      fill=False, edgecolor=BLACK, lw=1))
        ax.axhline(37, c=BLACK, ls='--', lw=.9)
        ax.set(xlim=(112.7, 120.15), ylim=(34.3, 40.7), xlabel='Longitude (°E)', ylabel='Latitude (°N)')
        ax.set_xticks([114, 116, 118, 120]); ax.set_yticks([35, 37, 39])
        ax.set_aspect(1 / np.cos(np.deg2rad(37.5)))

    def wells(self):
        self.grids = self.read(ROOT / 'outputs/figure_tables_en/fig01_grid_map.csv')
        self.meta = self.grids.set_index('grid_id')
        m = self.read(self.r / 'well_cohort_membership.csv')
        raw = self.read(ROOT / 'data/external/ncp_long2025_well_2005_2024.csv')
        raw = raw.loc[raw.grid_id.isin(self.grids.grid_id) & raw.aquifer_group.isin(['unconfined','confined']) & raw.gw_level.notna()].copy()
        assert raw.groupby('well_id')[['lon', 'lat']].nunique().max().max() == 1
        coords = raw.drop_duplicates('well_id')[['well_id', 'lon', 'lat', 'aquifer_group', 'grid_id']]
        common = set(m.loc[m.aquifer_group.eq('unconfined'), 'grid_id']) & set(m.loc[m.aquifer_group.eq('confined'), 'grid_id'])
        fixed = coords.loc[coords.well_id.isin(m.well_id) & coords.grid_id.isin(common)].copy()
        assert len(common) == 18 and fixed.well_id.is_unique
        assert fixed.groupby('aquifer_group').size().to_dict() == {'confined': 284, 'unconfined': 310}
        self.fixed, self.coords, self.common = fixed, coords, common
        self.table('fig01_fixed_wells', fixed)
        counts = fixed.groupby(['grid_id', 'aquifer_group']).size().rename('wells').reset_index()
        self.table('fig01_well_counts', counts)
        self.table('fig01_analysis_cells', self.grids)
        self.checks['fixed_wells'] = fixed.groupby('aquifer_group').size().to_dict()
        self.checks['common_cells'] = len(common)
        fig, axs = self.subplots(1, 2, 112)
        self.grid(axs[0]); self.grid(axs[1], common, False)
        axs[0].set_title('(a) Analysis domain', loc='left')
        axs[0].text(114, 40.2, 'North: 11 cells', color=BLUE, fontsize=9)
        axs[0].text(114, 34.65, 'South: 16 cells', color=RED, fontsize=9)
        axs[0].text(118.9, 37.12, '37°N', ha='right', fontsize=8.5)
        axs[0].scatter([116.4, 117.2], [39.9, 39.1], s=9, c=BLACK)
        axs[0].text(116.55, 40, 'Beijing', fontsize=8.5)
        axs[0].text(117.35, 39.1, 'Tianjin', fontsize=8.5)
        axs[1].set_title('(b) Matched fixed well cohorts', loc='left')
        for aq, marker, col, label in [('unconfined', 'o', BLACK, 'Unconfined: 310'), ('confined', '^', GRAY, 'Confined: 284')]:
            q = fixed.loc[fixed.aquifer_group.eq(aq)]
            axs[1].scatter(q.lon, q.lat, s=6, marker=marker, c=col, alpha=.55, linewidths=0, label=label)
        axs[1].legend(loc='lower left', frameon=False, markerscale=1.7)
        axs[1].text(.02, .98, '18 common cells\n10 north / 8 south', transform=axs[1].transAxes, va='top', fontsize=8.5)
        self.save(fig, 'figure_01_study_area')
        fig, axs = self.subplots(2, 2, 190)
        coverage_rows = []
        for i, aq in enumerate(['unconfined', 'confined']):
            for j, (label, source) in enumerate([('Full network', coords), ('Matched fixed cohort', fixed)]):
                ax = axs[i, j]; q = source.loc[source.aquifer_group.eq(aq)]
                self.grid(ax, set(q.grid_id), False)
                ax.scatter(q.lon, q.lat, s=5, c=BLACK if aq == 'unconfined' else GRAY, marker='o' if i == 0 else '^', alpha=.5, linewidths=0)
                ax.set_title(f'({chr(97+i*2+j)}) {label}\n{aq.capitalize()}: {len(q):,} wells / {q.grid_id.nunique()} cells', loc='left')
                coverage_rows.append(dict(network=label, aquifer=aq, wells=len(q), cells=q.grid_id.nunique()))
        self.table('supplement_well_coverage', pd.DataFrame(coverage_rows))
        self.table('supplement_full_well_locations', coords)
        assert coords.groupby('aquifer_group').size().to_dict() == {'confined':755,'unconfined':1257}
        self.save(fig, 'upgrade_s5_well_coverage')
        self.panel('figure_01_study_area; upgrade_s5_well_coverage', 'all',
                   ['fig01_grid_map.csv', 'well_cohort_membership.csv', 'ncp_long2025_well_2005_2024.csv'],
                   'Unique wells, fixed member IDs, common aquifer-cell intersection; full network only in supplement',
                   'geographic degrees; unique wells and cells', 'No spatial representativeness or causal claims')

    def trends(self):
        series = self.read(self.r / 'deseasoned_series.csv', parse_dates=['date'])
        series = series.loc[series.date.le('2024-08-01')]
        rows = []
        for region in ['all', 'north', 'south']:
            q = series if region == 'all' else series.loc[series.region_tag.eq(region)]
            z = q.groupby('date').deseasoned_gwa.mean().reindex(pd.date_range('2003-01-01', '2024-08-01', freq='MS'))
            rows.append(pd.DataFrame(dict(date=z.index, region=region, gwa_cm=z.values)))
        regional = pd.concat(rows)
        self.regional = regional
        self.table('fig02_regional_series', regional)
        spatial = self.read(self.r / 'fixed_window_spatial_contrasts.csv')
        spatial = spatial.loc[spatial.origin_dx.eq(0) & spatial.origin_dy.eq(0) & spatial.boundary.isna()].copy()
        temporal = self.read(self.e / 'local/temporal_trend_intervals.csv')
        temporal = temporal.loc[temporal['sample'].eq('all27')].copy()
        self.temporal = temporal
        a = spatial.rename(columns={'ci_low': 'low', 'ci_high': 'high', 'block_degrees': 'spatial_block'}).copy()
        a['interval'] = 'Spatial only'; a['time_block'] = 0
        b = temporal.loc[temporal.method.eq('synchronous_time_and_space') & temporal.time_block.eq(12)].copy()
        b['interval'] = 'Time + space'; b['method'] = 'OLS'
        plotted = pd.concat([a, b], ignore_index=True)
        self.table('fig02_trend_intervals', plotted)
        focus = b.loc[b.start.eq('2019-01-01') & b.spatial_block.eq(3)].iloc[0]
        assert np.isclose(focus.estimate, 3.3216018997351235) and focus.low < 0 < focus.high
        self.checks['joint_3degree'] = focus[['estimate', 'low', 'high']].to_dict()
        fig, axs = self.subplots(2, 1, 157, gridspec_kw={'height_ratios': [1.15, 1]})
        self.regional_axis(axs[0])
        axs[0].set_title('(a) Regional storage-proxy evolution', loc='left')
        labels = []; idx = 0
        for start in ['2019-01-01', '2022-01-01']:
            for interval, size, col, marker in [('Spatial only', 2, GRAY, 'o'), ('Spatial only', 3, GRAY, 's'), ('Time + space', 2, BLACK, 'o'), ('Time + space', 3, BLACK, 's')]:
                row = plotted.loc[plotted.start.eq(start) & plotted.method.eq('OLS') & plotted.interval.eq(interval) & plotted.spatial_block.eq(size)].iloc[0]
                self.interval(axs[1], row, idx, col, marker)
                labels.append(('2019+' if start.startswith('2019') else '2022+') + f'  {size}°  ' + interval)
                idx += 1
            idx += .5
        axs[1].set(yticks=[0,1,2,3,4.5,5.5,6.5,7.5], yticklabels=labels, xlabel='South − North OLS trend (cm/yr)')
        axs[1].invert_yaxis(); axs[1].axvline(0, c=GRAY, ls=':', lw=.8)
        axs[1].set_title('(b) Common windows ending Aug 2024; 95% intervals', loc='left')
        self.save(fig, 'story_02_regional_evidence')
        fig, axs = self.subplots(2, 1, 165)
        for ax, start in zip(axs, ['2019-01-01', '2022-01-01']):
            labels = []
            q = plotted.loc[plotted.start.eq(start)]
            for i, row in enumerate(q.itertuples()):
                self.interval(ax, row, i, BLACK if row.interval == 'Time + space' else GRAY, 'o' if row.method == 'OLS' else '^')
                labels.append(f'{row.method}, {int(row.spatial_block)}°: {row.interval}')
            offset = len(q) + .5
            q2 = temporal.loc[temporal.start.eq(start) & temporal.method.eq('synchronous_time_and_space') & temporal.time_block.ne(12)]
            for i, row in enumerate(q2.itertuples()):
                self.interval(ax, row, offset+i, BLUE if row.time_block == 6 else RED, 's')
                labels.append(f'OLS, {int(row.spatial_block)}°: {row.time_block}-month + space')
            ax.set(yticks=list(range(len(q))) + list(offset+np.arange(len(q2))), yticklabels=labels, xlabel='South − North trend (cm/yr)')
            ax.invert_yaxis(); ax.axvline(0, c=GRAY, ls=':', lw=.8)
            ax.set_title(('(a) 2019' if start.startswith('2019') else '(b) 2022') + '–Aug 2024', loc='left')
        self.table('supplement_temporal_intervals', temporal)
        self.save(fig, 'upgrade_s6_trend_uncertainty')
        self.panel('story_02_regional_evidence; upgrade_s6_trend_uncertainty', 'a: regional; b: intervals',
                   ['deseasoned_series.csv', 'fixed_window_spatial_contrasts.csv', 'local/temporal_trend_intervals.csv'],
                   'All27; archived GWA; original block origin; spatial-only OLS/Theil-Sen; joint OLS with 12-month blocks in main',
                   'GWA cm water-equivalent; slope cm/yr', 'Stored 95% spatial or joint intervals, no refitting')

    def regional_axis(self, ax):
        for region, col, name in [('all', BLACK, 'All NCP'), ('north', BLUE, 'North'), ('south', RED, 'South')]:
            q = self.regional.loc[self.regional.region.eq(region)]
            ax.plot(q.date, q.gwa_cm, c=col, label=name)
        ax.axvspan(pd.Timestamp('2017-06-01'), pd.Timestamp('2018-05-01'), color='.91', zorder=-1)
        ax.set(ylabel='Deseasoned GWA (cm)')
        ax.xaxis.set_major_locator(mdates.YearLocator(4)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.legend(frameon=False, ncol=3, loc='lower left', columnspacing=1)

    @staticmethod
    def interval(ax, row, y, col, marker):
        ax.errorbar(row.estimate, y, xerr=[[row.estimate-row.low], [row.high-row.estimate]], fmt=marker, c=col, ms=4, capsize=2, lw=1)

    def endpoints(self):
        p = self.read(self.e / 'primary_endpoint_selections.csv')
        p = p.loc[p.endpoint.ne('2023-11-01') & p.scope.eq('grid')].copy()
        dates = sorted(p.endpoint.unique()); assert dates[-1] == '2024-08-01' and len(dates) == 6
        summary = []
        for (end, route, rule), g in p.groupby(['endpoint', 'route', 'rule']):
            q = g.loc[g.qualifies].copy(); q['m'] = month(q.selected_date)
            n = q.loc[q.region_tag.eq('north'), 'm']; s = q.loc[q.region_tag.eq('south'), 'm']
            summary.append(dict(endpoint=end, route=route, rule=rule, qualified=len(q), north=len(n), south=len(s),
                                contrast=s.mean()-n.mean() if min(len(n),len(s)) >= 2 else np.nan))
        summary = pd.DataFrame(summary); self.table('fig03_endpoint_summary', summary)
        revisions = []
        for route, _ in ROUTES:
            q = p.loc[p.route.eq(route) & p.rule.eq('highest_probability')]
            j = q.loc[q.endpoint.eq('2023-12-01')].merge(q.loc[q.endpoint.eq('2024-08-01')], on='grid_id', suffixes=('_before', '_after'))
            j = j.loc[j.qualifies_before & j.qualifies_after].copy()
            j['revision_months'] = month(j.selected_date_after)-month(j.selected_date_before)
            j['route'] = route; revisions.append(j)
            count = int(j.revision_months.abs().le(6).sum())
            assert count == 13 and len(j) == (24 if route == 'A' else 23)
            self.checks['stability_'+route] = dict(within6=count, paired=len(j))
        rev = pd.concat(revisions); self.rev = rev; self.primary = p
        self.table('fig03_paired_revisions', rev)
        fig, axs = self.subplots(3, 1, 190, gridspec_kw={'height_ratios': [1, 1.25, 1]})
        endlabels = ['2019', '2020', '2021', '2022', '2023', 'Aug 2024']
        for route, col in ROUTES:
            q = summary.loc[summary.route.eq(route) & summary.rule.eq('highest_probability')].set_index('endpoint').reindex(dates)
            axs[0].plot(range(6), q.qualified, 'o-', c=col, label=route+(': deseasoned' if route == 'A' else ': reconstructed'))
            for rule, style, shift in [('highest_probability', '-', -.035), ('earliest', '--', .035)]:
                z = summary.loc[summary.route.eq(route) & summary.rule.eq(rule)].set_index('endpoint').reindex(dates)
                axs[1].plot(np.arange(6)+shift, z.contrast, marker='o' if rule == 'highest_probability' else '^', ls=style, c=col, ms=3.5,
                            label=route+(': highest' if rule == 'highest_probability' else ': earliest'))
        counts=[]
        for route,_ in ROUTES:
            q=summary.loc[summary.route.eq(route)&summary.rule.eq('highest_probability')].set_index('endpoint').reindex(dates)
            counts.append([f'{row.north}/{row.south}' for row in q.itertuples()])
        axs[0].set(ylabel='Qualifying cells', ylim=(-1, 30), title='(a) Reporting extent; table: north/south counts')
        axs[0].legend(frameon=False, ncol=2, loc='upper left')
        tab=axs[0].table(cellText=counts,rowLabels=['A N/S','D N/S'],colLabels=endlabels,
                         cellLoc='center',bbox=[0,-.43,1,.35],edges='open')
        tab.auto_set_font_size(False);tab.set_fontsize(8.5)
        for (row,col),cell in tab.get_celld().items():
            if row in [1,2]:cell.get_text().set_color(BLUE if row==1 else RED)
        axs[1].set(ylabel='South − North (months)', title='(b) Mean selected-date contrast; positive = south later')
        axs[1].axhline(0, c=GRAY, lw=.8); axs[1].legend(frameon=False, ncol=2, loc='lower left', columnspacing=1)
        axs[0].set_xticks(range(6),[])
        axs[1].set_xticks(range(6),endlabels);axs[1].set_xlabel('Record endpoint')
        # Fixed deterministic offsets expose coincident integer revisions without random jitter.
        for i, (route, col) in enumerate(ROUTES):
            q = rev.loc[rev.route.eq(route)].sort_values(['revision_months', 'grid_id'])
            offsets = np.linspace(-.16, .16, len(q))
            axs[2].scatter(q.revision_months, i+offsets, c=col, s=12, alpha=.8)
        axs[2].axvspan(-6, 6, color='.91', zorder=-1)
        for v in [-6, 6]: axs[2].axvline(v, c=GRAY, ls=':', lw=.8)
        axs[2].axvline(0, c=BLACK, lw=.6)
        axs[2].set(yticks=[0,1], yticklabels=['A: 13/24 within 6 mo', 'D: 13/23 within 6 mo'], ylim=(-.45,1.45),
                   xlabel='Aug 2024 date − Dec 2023 date (months)', title='(c) Revisions on cells qualifying at both endpoints')
        self.save(fig, 'validation_03_endpoint_stability')
        self.panel('validation_03_endpoint_stability', 'a,b,c', ['primary_endpoint_selections.csv'],
                   'Grid scope; primary endpoints; route-specific samples a/b; paired qualifying cells c; highest rule counts equal earliest counts',
                   'cells; months', 'Descriptive contrasts; no interval inferred from cell spread')

    def processing(self):
        s = self.read(self.s / 'actual_series.csv', parse_dates=['date'])
        s = s.loc[s.endpoint.eq('2023-11-01') & s.route.isin(['A','D'])]
        c = self.read(self.s / 'actual_candidates.csv', parse_dates=['date'])
        c = c.loc[c.endpoint.eq('2023-11-01') & c.qualifies & c.route.isin(['A','D'])]
        picks = self.read(self.s / 'actual_selections.csv')
        picks = picks.loc[picks.endpoint.eq('2023-11-01') & picks.route.isin(['A','D']) & picks.qualifies]
        ids = ['g_115.0_35.5', 'g_116.0_37.5']
        self.table('fig04_signals', s.loc[s.grid_id.isin(ids)])
        self.table('fig04_candidates', c.loc[c.grid_id.isin(ids)])
        self.table('fig04_selections', picks.loc[picks.grid_id.isin(ids)])
        fig, axs = self.subplots(2,2,157, sharex=True)
        for i, gid in enumerate(ids):
            visible = s.loc[s.grid_id.eq(gid) & s.date.ge('2014-01-01')]
            lo = visible[['observed','input_signal','beast_trend']].min().min()-2
            hi = visible[['observed','input_signal','beast_trend']].max().max()+2
            for j, (route, col) in enumerate(ROUTES):
                ax = axs[i,j]; q = visible.loc[visible.route.eq(route)]
                ax.scatter(q.date, q.observed, s=4, c='.72', alpha=.65, linewidths=0, zorder=1)
                ax.plot(q.date, q.input_signal, c=col, lw=1, zorder=2)
                ax.plot(q.date, q.beast_trend, c=BLACK, lw=1.1, zorder=3)
                chosen = picks.loc[picks.grid_id.eq(gid) & picks.route.eq(route)]
                h = pd.Timestamp(chosen.loc[chosen.rule.eq('highest_probability'),'selected_date'].iloc[0])
                e = pd.Timestamp(chosen.loc[chosen.rule.eq('earliest'),'selected_date'].iloc[0])
                for d in c.loc[c.grid_id.eq(gid) & c.route.eq(route), 'date']:
                    if d not in [h,e]: ax.axvline(d, c=GRAY, ls=':', lw=.8)
                for d in sorted(set([h,e])):
                    ax.axvline(d, c=col, ls='--' if d == h else '-.', lw=1)
                    ax.text(d, hi-1, 'H=E' if h == e else ('H' if d == h else 'E'), color=col, ha='center', va='top', fontsize=8.5)
                ax.set(ylim=(lo,hi), xlim=(pd.Timestamp('2014-01-01'),pd.Timestamp('2024-01-01')),
                       title=f'({chr(97+i*2+j)}) '+('South' if i == 0 else 'North')+f' / route {route}', ylabel='GWA (cm)')
                ax.xaxis.set_major_locator(mdates.YearLocator(4)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                if i == 1: ax.set_xlabel('Year')
        handles=[Line2D([],[],marker='.',ls='',c='.65',label='Observed'),Line2D([],[],c=GRAY,label='Route input'),Line2D([],[],c=BLACK,label='BEAST trend')]
        fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False)
        self.save(fig, 'story_04_processing_comparison')
        self.panel('story_04_processing_comparison','a–d',['actual_series.csv','actual_candidates.csv','actual_selections.csv'],
                   'Frozen example IDs; common November 2023 endpoint; display 2014 onward; H highest/E earliest',
                   'cm water-equivalent; dates', 'Stored fitted mean trends and qualifying candidates, no refit')

    def simulation(self):
        cal = self.read(self.e / 'calibrated_simulation_summary.csv')
        cal = cal.loc[cal.setting.isin(['low','median','high']) & cal.rule.eq('highest_probability')]
        assert cal.n.eq(50).all()
        self.table('fig05_calibrated_summary', cal)
        sel = self.read(self.s / 'simulation_selections.csv')
        truth = self.read(self.s / 'simulation_truth.csv')
        z = sel.merge(truth, on='grid_id'); z = z.loc[z.scenario.isin(SCENARIOS[2:]) & z.qualifies].copy()
        z['error_months'] = month(z.selected_date)-month(z.true_onset)
        self.table('fig05_baseline_errors', z)
        fig, axs = self.subplots(2,2,177)
        dates = pd.date_range('2003-01-01','2024-09-01',freq='MS'); t=np.arange(len(dates))/12
        curves = [-1.5*t, -1.5*t+12*np.maximum(1-abs(t-18.5)/1.5,0),
                  -1.5*t+4.5*np.maximum(t-16,0), -1.5*t+2.1*np.maximum(t-15,0)+3.4*np.maximum(t-18,0)]
        self.table('fig05_truth_trends', pd.DataFrame({'date':dates, **dict(zip(SCENARIOS,curves))}))
        for y, name, col, style in zip(curves,NAMES,[GRAY,GRAY,BLACK,BLACK],[':', '--', '-', '-.']):
            axs[0,0].plot(dates,y,c=col,ls=style,label=name)
        axs[0,0].set(xlim=(pd.Timestamp('2014-01-01'),dates[-1]),ylabel='True trend (cm)',title='(a) Four prescribed trajectories')
        axs[0,0].xaxis.set_major_locator(mdates.YearLocator(4)); axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axs[0,0].legend(frameon=False,loc='upper left',handlelength=1.5)
        axs[0,0].annotate('First reversal\n(two-stage)',xy=(pd.Timestamp('2018-01-01'),-22.5),xytext=(pd.Timestamp('2014-03-01'),-32),
                          fontsize=8.5,arrowprops={'arrowstyle':'-','lw':.6})
        axs[0,0].annotate('Later\nacceleration',xy=(pd.Timestamp('2021-01-01'),-20.7),xytext=(pd.Timestamp('2021-01-01'),-32),
                          fontsize=8.5,arrowprops={'arrowstyle':'-','lw':.6})
        for route,col in ROUTES:
            for regime,marker,offset in [('low','o',-.18),('median','s',0),('high','^',.18)]:
                q=cal.loc[cal.route.eq(route)&cal.setting.eq(regime)].set_index('scenario').reindex(SCENARIOS)
                x=np.arange(4)+offset+(-.035 if route=='A' else .035)
                axs[0,1].errorbar(x,q.rate,yerr=[q.rate-q.rate_low,q.rate_high-q.rate],fmt=marker,c=col,ms=3.2,lw=.65,capsize=1)
        axs[0,1].set(xticks=range(4),xticklabels=['No\nrecovery','Temporary\ngain','Sustained\nreversal','Two-\nstage'],ylabel='Qualifying fraction',ylim=(-.04,1.08),title='(b) Empirical residuals: n = 50 each')
        axs[0,1].legend(handles=[Line2D([],[],c=col,label='Route '+route) for route,col in ROUTES]+[Line2D([],[],ls='',marker=m,c=GRAY,label=l) for m,l in [('o','Low'),('s','Median'),('^','High')]],frameon=False,ncol=2,loc='lower right',columnspacing=.7,handlelength=1.2)
        boxes=[]; positions=[]; cols=[]
        for i, scenario in enumerate(SCENARIOS[2:]):
            for j,(route,rule) in enumerate([('A','highest_probability'),('A','earliest'),('D','highest_probability'),('D','earliest')]):
                boxes.append(z.loc[z.scenario.eq(scenario)&z.route.eq(route)&z.rule.eq(rule),'error_months'])
                positions.append(i*5+j); cols.append(BLUE if route=='A' else RED)
        bp=axs[1,0].boxplot(boxes,positions=positions,widths=.6,patch_artist=True,showfliers=False,medianprops={'color':BLACK})
        for box,col in zip(bp['boxes'],cols): box.set_facecolor(col); box.set_alpha(.6)
        axs[1,0].set(xticks=positions,xticklabels=['AH','AE','DH','DE']*2,ylabel='Date − first reversal (months)',title='(c) Baseline: 100 realizations / scenario')
        axs[1,0].axhline(0,c=GRAY,ls=':',lw=.8)
        axs[1,0].set_xlabel('Sustained reversal       Two-stage\nH: highest; E: earliest')
        for route,col in ROUTES:
            q=cal.loc[cal.route.eq(route)&cal.scenario.eq(SCENARIOS[-1])].set_index('setting').reindex(['low','median','high'])
            axs[1,1].errorbar(np.arange(3)+(-.06 if route=='A' else .06),q.median_error,
                              yerr=[q.median_error-q.median_low,q.median_high-q.median_error],fmt='o',c=col,capsize=2,label='Route '+route)
        axs[1,1].axhline(0,c=GRAY,ls=':',lw=.8)
        axs[1,1].axhline(36,c=GRAY,ls='--',lw=.8)
        axs[1,1].set(xticks=range(3),xticklabels=['Low','Median','High'],ylabel='Median onset delay (months)',
                     xlabel='Empirical residual regime\nDashed: acceleration at +36 months',title='(d) Two-stage: highest-probability date')
        axs[1,1].legend(frameon=False,loc='center',bbox_to_anchor=(.5,.28))
        self.save(fig,'story_05_detection_scenarios')
        self.checks['empirical_realizations_per_setting']=50
        self.panel('story_05_detection_scenarios','a: prescribed; b,d: empirical; c: baseline',
                   ['simulation_truth.csv','simulation_selections.csv','calibrated_simulation_summary.csv'],
                   'b/d three original residual regimes, highest rule; c qualifying baseline realizations, both rules',
                   'cm; fraction; months', 'b Wilson Monte Carlo intervals; c IQR/1.5IQR whiskers; d stored bootstrap median intervals')

    def observations(self):
        w=self.read(self.e/'uniform/well_weighted_series.csv',parse_dates=['date'])
        w=w.loc[w.support.eq('common18')&w.weighting.eq('equal_cell')].copy()
        c=self.read(self.e/'uniform/well_multiscale_correlations.csv')
        c=c.loc[c.support.eq('common18')&c.weighting.eq('equal_cell')&c.aquifer.eq('unconfined')&c.treatment.eq('difference')].copy()
        self.table('fig06_matched_series',w); self.table('fig06_correlations',c)
        fig=plt.figure(figsize=(160*MM,185*MM),layout='constrained'); gs=fig.add_gridspec(3,2,height_ratios=[1,1,1.15])
        axs=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,:]),fig.add_subplot(gs[2,0]),fig.add_subplot(gs[2,1])]
        allz=[]
        for i,region in enumerate(['north','south']):
            ax=axs[i]
            for aq,style in [('unconfined','--'),('confined',':')]:
                g=w.loc[w.region.eq(region)&w.aquifer.eq(aq)].sort_values('date')
                v=(g.well-g.well.mean())/g.well.std(); allz.extend(v)
                ax.plot(g.date,v,c=GRAY,ls=style,lw=1.35,label=aq.capitalize())
            a=w.loc[w.region.eq(region)&w.aquifer.eq('unconfined')].sort_values('date')
            b=w.loc[w.region.eq(region)&w.aquifer.eq('confined')].sort_values('date')
            assert np.allclose(a.gwa_cm,b.gwa_cm,equal_nan=True)
            v=(a.gwa_cm-a.gwa_cm.mean())/a.gwa_cm.std();allz.extend(v)
            ax.plot(a.date,v,c=BLACK,label='GWA')
            ax.set(ylabel='Standardized anomaly',title=f'({chr(97+i)}) {region.capitalize()}: matched fixed cohorts')
            ax.xaxis.set_major_locator(mdates.YearLocator(1)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        for ax in axs[:2]:ax.set_ylim(np.nanmin(allz)-.15,np.nanmax(allz)+.15)
        axs[0].legend(frameon=False,ncol=3,loc='upper left',columnspacing=1)
        for region,col,offset in [('north',BLUE,-.075),('south',RED,.075)]:
            q=c.loc[c.region.eq(region)&c.aggregation_months.eq(1)].set_index('component').reindex(['twsa_cm','smsa_cm','gwa_cm'])
            axs[2].errorbar(np.arange(3)+offset,q.r,yerr=[q.r-q.low,q.high-q.r],fmt='o',c=col,capsize=2,ms=4,label=region.capitalize())
            q=c.loc[c.region.eq(region)&c.component.eq('gwa_cm')].set_index('aggregation_months').reindex([1,3,6,12])
            axs[3].errorbar(np.arange(4)+offset,q.r,yerr=[q.r-q.low,q.high-q.r],fmt='o-',c=col,capsize=2,ms=4)
        axs[2].set(xticks=range(3),xticklabels=['TWSA','SMSA','GWA'],ylabel='Monthly-change r',ylim=(-1.02,1.02),title='(c) Component subtraction')
        axs[2].legend(frameon=False,loc='lower left')
        axs[3].set(xticks=range(4),xticklabels=['1\n(n=67)','3\n(n=21)','6\n(n=10)','12\n(n=4)'],xlabel='Aggregation (months)',ylabel='GWA–well change r',ylim=(-1.02,1.02),title='(d) Scale dependence')
        for ax in axs[2:]:ax.axhline(0,c=GRAY,ls=':',lw=.8)
        self.save(fig,'validation_06_observation_support')
        self.panel('validation_06_observation_support','a,b: series; c,d: correlations',
                   ['uniform/well_weighted_series.csv','uniform/well_multiscale_correlations.csv'],
                   'common18; equal_cell; Jan2019–Aug2024; unconfined differences c/d; same component support',
                   'separately standardized anomalies; Pearson r', 'Stored paired 12-calendar-month block intervals, annual n=4 descriptive')

    def supplements(self):
        d=self.read(self.e/'extraction_comparison.csv');u=self.read(self.e/'endpoint_and_support_summary.csv')
        bad=d.loc[d.original_closure_max.gt(1e-6)]
        fig,axs=self.subplots(1,2,100)
        x=np.arange(len(bad));axs[0].bar(x,bad.original_closure_max,color='.6',label='Archived components')
        axs[0].plot(x,bad.uniform_closure_max,'o',c=BLACK,label='Uniform support')
        axs[0].set(xticks=x,xticklabels=[g.removeprefix('g_').replace('_',' / ') for g in bad.grid_id],ylabel='Maximum closure error (cm)',title='(a) Five non-closing archived cells')
        axs[0].tick_params(axis='x',rotation=60)
        handles,labels=axs[0].get_legend_handles_labels()
        fig.legend(handles,labels,frameon=False,ncol=2,loc='outside lower center')
        rows=[]
        for route,col in ROUTES:
            q=u.loc[u.route.eq(route)&u.rule.eq('highest_probability')&u.comparison.isin(['Dec_to_Aug','Aug_to_missing_Sept'])]
            for row in q.itertuples():
                i=len(rows);axs[1].bar(i,row.median_absolute_revision,color=col,width=.65)
                rows.append(route+'\n'+('Dec→Aug' if row.comparison=='Dec_to_Aug' else 'Aug→Sep'))
        axs[1].set(xticks=range(4),xticklabels=rows,ylabel='Median absolute revision (months)',title='(b) Observed vs missing extension')
        self.save(fig,'validation_s1_closure_endpoint');self.table('supplement_closure',d)
        # Full empirical-rate intervals, including the reporting denominator.
        q=self.read(self.e/'calibrated_simulation_summary.csv');q=q.loc[q.setting.isin(['low','median','high'])&q.rule.eq('highest_probability')]
        fig,axs=self.subplots(2,2,155)
        for ax,scenario,letter in zip(axs.flat,SCENARIOS,'abcd'):
            for route,col in ROUTES:
                z=q.loc[q.route.eq(route)&q.scenario.eq(scenario)].set_index('setting').reindex(['low','median','high'])
                ax.errorbar(np.arange(3)+(-.05 if route=='A' else .05),z.rate,yerr=[z.rate-z.rate_low,z.rate_high-z.rate],fmt='o',c=col,capsize=2,label='Route '+route)
            ax.set(xticks=range(3),xticklabels=['Low','Median','High'],ylim=(-.03,1.05),ylabel='Qualifying fraction',title=f'({letter}) '+NAMES[SCENARIOS.index(scenario)])
        axs[0,0].legend(frameon=False);self.save(fig,'validation_s2_calibrated_simulation')
        p=self.read(self.e/'fits/actual_posterior.csv',parse_dates=['date'])
        fig,axs=self.subplots(2,2,150,sharex=True)
        for i,gid in enumerate(['g_115.0_35.5','g_116.0_37.5']):
            for j,route in enumerate(['A','D']):
                ax=axs[i,j]
                for end,style,col,label in [('2024-08-01','-',BLACK,'Aug: observed end'),('2024-09-01','--',GRAY,'Sep: missing end')]:
                    z=p.loc[p.grid_id.eq(gid)&p.route.eq(route)&p.endpoint.eq(end)&p.date.ge('2014-01-01')]
                    ax.plot(z.date,z.cp_occurrence_probability,c=col,ls=style,label=label)
                ax.set(ylim=(0,1.05),ylabel='Occurrence probability',title=f'({chr(97+i*2+j)}) '+('South' if i==0 else 'North')+f' / route {route}')
                ax.xaxis.set_major_locator(mdates.YearLocator(4));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axs[0,0].legend(frameon=False);self.save(fig,'validation_s3_posterior')
        self.table('supplement_occurrence_profiles',p)
        order=self.grids.sort_values(['region_tag','grid_lat','grid_lon'],ascending=[True,False,True]).grid_id.tolist()
        assert self.grids.loc[self.grids.grid_id.isin(order[:11]),'region_tag'].eq('north').all()
        fig,axs=self.subplots(1,2,195,sharey=True)
        dates=sorted(self.primary.endpoint.unique());matrices=[]
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
        for ax,route,letter in zip(axs,['A','D'],'ab'):
            z=self.primary.loc[self.primary.route.eq(route)&self.primary.rule.eq('highest_probability')].copy()
            # A stored candidate date is never visible unless it passes qualification.
            z['selected_year']=np.where(z.qualifies,pd.to_datetime(z.selected_date).dt.year+(pd.to_datetime(z.selected_date).dt.month-1)/12,np.nan)
            self.table('supplement_heatmap_route_'+route,z)
            matrix=z.pivot(index='grid_id',columns='endpoint',values='selected_year').reindex(index=order,columns=dates)
            matrices.append(matrix)
            image=ax.pcolormesh(np.arange(7)-.5,np.arange(28)-.5,np.ma.masked_invalid(matrix.to_numpy()),vmin=2015,vmax=2024+7/12,cmap=cmap,shading='flat')
            ax.set_ylim(26.5,-.5);ax.set_xlim(-.5,5.5)
            ax.set(xticks=range(6),xticklabels=['2019','2020','2021','2022','2023','Aug24'],yticks=range(27),yticklabels=[x.removeprefix('g_').replace('_',' / ') for x in order],title=f'({letter}) Route {route}')
            ax.tick_params(axis='x',rotation=45);ax.axhline(10.5,c=BLACK,lw=1)
            ax.set_xlabel('Record endpoint')
        axs[0].set_ylabel('North (top 11) / South (bottom 16): cell west / south')
        colorbar=fig.colorbar(image,ax=axs.ravel().tolist(),label='Selected calendar year',shrink=.6,pad=.025)
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_edgecolor('face')
        self.save(fig,'validation_s4_date_heatmap')
        self.checks['heatmap_unqualified_masked']=all(not np.isfinite(z.to_numpy()[:,1]).any() for z in matrices)
        self.panel('validation_s1–s4','all',['extraction_comparison.csv','endpoint_and_support_summary.csv','calibrated_simulation_summary.csv','fits/actual_posterior.csv','primary_endpoint_selections.csv'],
                   'V1 highest rule; V2 three residual regimes/highest; V3 frozen two examples; V4 qualified primary dates only',
                   'cm; months; fraction; occurrence probability; year', 'V2 Wilson intervals; V3 marginal occurrence probabilities, not unique-onset posteriors')

    def graphical(self):
        fig,axs=plt.subplots(1,3,figsize=(240*MM,96*MM),layout='constrained')
        self.regional_axis(axs[0]);axs[0].set_title('Regional improvement',loc='left',fontweight='bold')
        axs[0].legend(frameon=False,fontsize=8.5,loc='lower left');axs[0].set_xlabel('Observed proxy; gaps retained')
        for i,(route,col) in enumerate([('A',BLACK),('D',GRAY)]):
            q=self.rev.loc[self.rev.route.eq(route)]
            axs[1].scatter(q.revision_months,i+np.linspace(-.13,.13,len(q)),s=9,c=col,marker='o' if route=='A' else '^',zorder=2)
        axs[1].axvspan(-6,6,color='.92',zorder=-1);axs[1].axvline(0,c=GRAY,lw=.7)
        axs[1].set(yticks=[0,1],yticklabels=['A: 13/24','D: 13/23'],ylim=(-.5,1.5),xlabel='Date revision (months)\nCounts within ±6 months',title='Date stability')
        c=self.read(self.e/'uniform/well_multiscale_correlations.csv')
        c=c.loc[c.support.eq('common18')&c.weighting.eq('equal_cell')&c.aquifer.eq('unconfined')&c.treatment.eq('difference')&c.component.eq('gwa_cm')&c.aggregation_months.isin([1,6])]
        for region,col,offset in [('north',BLUE,-.04),('south',RED,.04)]:
            q=c.loc[c.region.eq(region)].sort_values('aggregation_months')
            axs[2].errorbar(np.arange(2)+offset,q.r,yerr=[q.r-q.low,q.high-q.r],fmt='o-',c=col,label=region.capitalize(),capsize=2)
        axs[2].axhline(0,c=GRAY,ls=':',lw=.8);axs[2].set(xticks=[0,1],xticklabels=['1 month','6 months'],ylabel='GWA–well change r',ylim=(-1,1),title='Observation support',xlabel='Matched cells and fixed wells')
        axs[2].legend(frameon=False,loc='upper left')
        self.save(fig,'graphical_abstract_1328x531')
        self.panel('graphical_abstract_1328x531','all',['fig02_regional_series.csv','fig03_paired_revisions.csv','uniform/well_multiscale_correlations.csv'],
                   'Same subsets as main figures; no new estimands','cm; months; r','Stored correlation intervals')

    def run(self):
        self.wells();self.trends();self.endpoints();self.processing();self.simulation();self.observations();self.supplements();self.graphical()
        assert self.checks['heatmap_unqualified_masked']
        self.inputs[Path(__file__).relative_to(ROOT).as_posix()]=digest(__file__)
        ledger=dict(schema_version=1,render_only=True,model_fits_run=0,primary_endpoint='2024-08-01',
                    missing_endpoint_sensitivity='2024-09-01',inputs=self.inputs,panels=self.panels,checks=self.checks,
                    style=dict(width_mm=160,font='DejaVu Sans',min_font_pt=8.5,png_dpi=300,pdf_fonttype=42),
                    files={p.relative_to(self.out).as_posix():digest(p) for folder in ['figures','tables','before'] for p in sorted((self.out/folder).iterdir())})
        (self.out/'figure_evidence_ledger.json').write_text(json.dumps(ledger,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps(self.checks,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',required=True);p.add_argument('--story',required=True);p.add_argument('--review',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--baseline-dir',help='Optional frozen before-images directory for review comparisons')
    Renderer(p.parse_args()).run()


if __name__=='__main__':main()
