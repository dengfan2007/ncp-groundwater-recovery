"""Render HJ r7 endpoint evidence from frozen selections and spatial intervals.

No model fits, interval recomputation, date jitter, or canonical-source writes.
Use the existing project Anaconda runtime and a fresh output directory.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'outputs/monitoring_revision/2026-10-18-r1/inference'
BLUE, RED = '#286294', '#a64242'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def month_number(dates):
    return dates.dt.year * 12 + dates.dt.month - 1


def render(output):
    output.mkdir(parents=True, exist_ok=False)
    paths = [EVIDENCE / x for x in ['real_selections.csv', 'endpoint_date_contrasts.csv']]
    selections = pd.read_csv(paths[0], parse_dates=['endpoint', 'selected_date'])
    contrasts = pd.read_csv(paths[1], parse_dates=['endpoint'])
    selected = selections.loc[selections.setting.eq('original') &
                              selections.rule.eq('highest_probability')].copy()
    assert not selected.duplicated(['route', 'endpoint', 'grid_id']).any()
    assert selected.groupby(['route', 'endpoint']).size().eq(27).all()
    endpoints = sorted(selected.endpoint.unique())
    dates = [pd.Timestamp(x) for x in endpoints]
    xdates = mdates.date2num(dates)
    counts = (selected.loc[selected.qualifies].groupby(['route', 'endpoint', 'region_tag'])
              .size().reindex(pd.MultiIndex.from_product(
                  [['A', 'D'], dates, ['north', 'south']],
                  names=['route', 'endpoint', 'region_tag']), fill_value=0)
              .unstack('region_tag').reset_index())
    counts.to_csv(output / 'reportable_counts.csv', index=False)
    # Match the archived interval's conditioning sample exactly.
    for r in contrasts.itertuples():
        n = counts.loc[counts.route.eq(r.route) & counts.endpoint.eq(r.endpoint)].iloc[0]
        assert n.north == r.n_north and n.south == r.n_south
        q = selected.loc[selected.route.eq(r.route) & selected.endpoint.eq(r.endpoint)
                         & selected.qualifies]
        months = month_number(q.selected_date)
        estimate = months[q.region_tag.eq('south')].mean() - months[q.region_tag.eq('north')].mean()
        assert np.isclose(estimate, r.estimate, atol=1e-8)
    contrasts.to_csv(output / 'conditional_spatial_intervals.csv', index=False)

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8.5,
                         'axes.titlesize': 9, 'axes.labelsize': 8.5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.unicode_minus': True,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(176 / 25.4, 190 / 25.4), layout='constrained')
    gs = fig.add_gridspec(4, 2, height_ratios=[.55, .82, .20, 1.15])
    a, b = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
    sample = fig.add_subplot(gs[2, :])
    c, d = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])
    for route, color, marker in [('A', '#222222', 'o'), ('D', '#7a7a7a', '^')]:
        q = counts.loc[counts.route.eq(route)].set_index('endpoint').loc[dates]
        a.plot(xdates, q.north + q.south, color=color, marker=marker,
               ms=4, lw=1.1, label=f'Route {route}')
        for r in contrasts.loc[contrasts.route.eq(route)].itertuples():
            # Small horizontal offsets distinguish routes at the same endpoint;
            # c/d below retain exact monthly date coordinates without jitter.
            xpos = mdates.date2num(r.endpoint) + (-15 if route == 'A' else 15)
            b.errorbar(xpos, r.estimate,
                       yerr=[[r.estimate - r.low], [r.high - r.estimate]],
                       fmt=marker, color=color, capsize=2, ms=4, lw=.85)
    a.axhline(27, color='.6', ls=':', lw=.8)
    a.set(ylabel='Cells with a date', ylim=(-1, 29))
    a.set_title('(a) Cells with a qualifying date', loc='left', pad=6)
    a.legend(frameon=False, ncol=2, fontsize=8, loc='lower right')
    b.axhline(0, color='.5', ls=':', lw=.85)
    b.set(ylabel='South minus north (months)', ylim=(-23, 23))
    b.set_title('(b) Mean date contrast: 3° spatial-block 95% intervals', loc='left', pad=6)
    for value, text, va in [(2, 'South later (+)', 'bottom'),
                            (-2, 'South earlier (−)', 'top')]:
        b.text(.02, value, text, transform=b.get_yaxis_transform(),
               fontsize=8.5, va=va, color='#222222')
    for ax in [a, b, sample]:
        ax.set_xlim(xdates[0] - 80, xdates[-1] + 80)
    for ax in [a, b]:
        ax.set_xticks(xdates, [x.strftime('%b\n%Y') for x in dates])
        ax.tick_params(length=3)
    b.set_xlabel('Observed record endpoint', labelpad=4)

    sample.set_ylim(0, 1)
    sample.axis('off')
    sample.text(0, 1.02, 'Reportable cells at each endpoint (N/S)',
                transform=sample.transAxes, fontsize=8, va='bottom')
    for route, y, color in [('A', .67, '#222222'), ('D', .18, '#666666')]:
        for xpos, date in zip(xdates, dates):
            r = counts.loc[counts.route.eq(route) & counts.endpoint.eq(date)].iloc[0]
            sample.text(xpos, y, f'{route}: {r.north}/{r.south}',
                        ha='center', va='center', fontsize=8, color=color)

    statuses, all_pairs, point_groups = [], [], []
    limits = np.array([2014.5, 2023.5]) * 12
    area = 25
    for ax, route, panel in [(c, 'A', 'c'), (d, 'D', 'd')]:
        q = selected.loc[selected.route.eq(route) & selected.qualifies]
        old = q.loc[q.endpoint.eq('2023-12-01')]
        new = q.loc[q.endpoint.eq('2024-08-01')]
        pairs = old.merge(new, on=['grid_id', 'region_tag'], suffixes=('_old', '_new'))
        pairs['old_month'] = month_number(pairs.selected_date_old)
        pairs['new_month'] = month_number(pairs.selected_date_new)
        pairs['revision_months'] = pairs.new_month - pairs.old_month
        pairs['route'] = route
        pairs['within_six_months'] = pairs.revision_months.abs().le(6)
        all_pairs.append(pairs[['route', 'grid_id', 'region_tag', 'selected_date_old',
                                'selected_date_new', 'revision_months', 'within_six_months']])
        stable = int(pairs.within_six_months.sum())
        lost = sorted(set(old.grid_id) - set(new.grid_id))
        gained = sorted(set(new.grid_id) - set(old.grid_id))
        statuses.append(dict(route=route,old_count=len(old),new_count=len(new),
                             paired_count=len(pairs),within_six_months=stable,
                             lost_count=len(lost),lost_grid_ids=lost,
                             gained_count=len(gained),gained_grid_ids=gained))
        grouped = pairs.groupby(['region_tag', 'old_month', 'new_month']).size().reset_index(name='cells')
        grouped['route'] = route
        point_groups.append(grouped)
        ax.fill_between(limits, limits - 6, limits + 6, color='.91', zorder=0)
        ax.plot(limits, limits, color='.4', lw=.8, zorder=1)
        for region, color, marker in [('north', BLUE, 'o'), ('south', RED, '^')]:
            g = grouped.loc[grouped.region_tag.eq(region)]
            ax.scatter(g.old_month, g.new_month, facecolors='none', edgecolors=color,
                       marker=marker, s=area * g.cells, linewidths=.9, zorder=3)
        ax.set(xlim=limits, ylim=limits, xlabel='Date at Dec 2023 endpoint')
        ax.set_aspect('equal')
        ticks = np.arange(2015, 2024, 2)
        ax.set_xticks(ticks * 12, ticks.astype(str))
        ax.set_yticks(ticks * 12, ticks.astype(str))
        ax.set_title(f'({panel}) Route {route}: {len(pairs)} paired cells', loc='left', pad=6)
        ax.text(.97, .025, f'{stable}/{len(pairs)} within ±6 months',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
                bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=.9))
        if route == 'D':
            ax.text(.035, .79, f'{len(lost)} cells lost\ntheir date',
                    transform=ax.transAxes, ha='left', va='top', weight='bold', fontsize=9)
    c.set_ylabel('Date at Aug 2024 endpoint')
    region_handles = [Line2D([], [], ls='none', marker=marker, markerfacecolor='none',
                             markeredgecolor=color, markersize=5, label=label)
                      for marker, color, label in [('o', BLUE, 'North'), ('^', RED, 'South')]]
    regions = c.legend(handles=region_handles, frameon=False, fontsize=8,
                       loc='lower left', bbox_to_anchor=(0, .12), handletextpad=.4)
    c.add_artist(regions)
    sizes = [Line2D([], [], ls='none', marker='o', markerfacecolor='none',
                    markeredgecolor='.4', markersize=np.sqrt(area * n), label=str(n))
             for n in [1, 2, 4]]
    c.legend(handles=sizes, title='Cells at identical dates', title_fontsize=8,
             frameon=False, fontsize=8, ncol=3, loc='upper left',
             columnspacing=.6, handletextpad=.3)
    fig.get_layout_engine().set(h_pad=.055, w_pad=.055, hspace=.035, wspace=.06)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(output / ('Fig03.' + ext), dpi=400, facecolor='white')
    plt.close(fig)
    assert [(r['paired_count'], r['within_six_months'], r['lost_count'])
            for r in statuses] == [(24, 13, 0), (23, 13, 4)]
    pd.concat(all_pairs, ignore_index=True).to_csv(output / 'paired_dates.csv', index=False)
    pd.concat(point_groups, ignore_index=True).to_csv(output / 'point_multiplicity.csv', index=False)
    result = {'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              'route_status': statuses, 'coordinates': 'integer calendar months; no jitter',
              'marker_area': '25 square points per cell, aggregated by exact paired dates and region',
              'regional_intervals': 'archived values, conditional on selected candidate dates and reportable cells at each endpoint',
              'width_mm': 176, 'height_mm': 190,
              'figure_sha256': sha(output / 'Fig03.pdf')}
    (output / 'figure3_evidence.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['route_status']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.output)
