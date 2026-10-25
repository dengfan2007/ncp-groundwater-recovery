"""Render Figure 4 candidate selection examples and series audits."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/monitoring_revision/2026-10-18-r1/processing'
IDS = ['g_115.0_35.5', 'g_116.0_37.5']
BLUE, RED = '#286294', '#a64242'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_candidates(series, candidates, picks):
    rows = []
    for r in candidates.loc[candidates.grid_id.isin(IDS)].itertuples():
        g = series.loc[series.grid_id.eq(r.grid_id) & series.route.eq(r.route)].sort_values('date')
        valid = g.qualification_signal.notna()
        npre = int((valid & g.date.lt(r.date)).sum())
        npost = int((valid & g.date.ge(r.date)).sum())
        gain = r.post_slope - r.pre_slope
        tests = {'T': r.date >= pd.Timestamp('2015-01-01'),
                 'P': np.isfinite(r.probability) and r.probability >= .5,
                 'N': npre >= 18 and npost >= 18 and np.isfinite(r.pre_slope) and np.isfinite(r.post_slope),
                 'S': np.isfinite(r.post_slope) and r.post_slope > 0,
                 'G': np.isfinite(gain) and gain > .5}
        failed = ','.join(k for k, passed in tests.items() if not passed)
        pp = picks.loc[picks.grid_id.eq(r.grid_id) & picks.route.eq(r.route)
                       & picks.qualifies & picks.selected_date.eq(r.date)]
        chosen = '/'.join('H' if x == 'highest_probability' else 'E' for x in pp.rule)
        # Recompute the two straight-line qualification slopes from archived
        # qualification values, without invoking the inference pipeline.
        for side, stored in [(g.date.lt(r.date), r.pre_slope),
                             (g.date.ge(r.date), r.post_slope)]:
            ok = (valid & side).to_numpy()
            if ok.sum() >= 18 and np.isfinite(stored):
                x = np.flatnonzero(ok) / 12
                slope = np.polyfit(x, g.qualification_signal.to_numpy()[ok], 1)[0]
                assert np.isclose(slope, stored, atol=1e-9, rtol=0)
        rows.append(dict(grid_id=r.grid_id, region=r.region_tag, route=r.route,
                         date=r.date, cpPr=r.probability, pre_slope=r.pre_slope,
                         post_slope=r.post_slope, slope_gain=gain,
                         pre_observed=npre, post_observed=npost,
                         qualified=not failed, failed_checks=failed or '-', selection=chosen or '-',
                         monthly_cpOccPr_at_date=float(g.loc[g.date.eq(r.date), 'cp_probability'].iloc[0])))
    audit = pd.DataFrame(rows).sort_values(['grid_id', 'route', 'date'])
    # Reproduce H/E decisions using full precision, including the later-date tie rule.
    for gid in IDS:
        for route in ['A', 'D']:
            q = audit.loc[audit.grid_id.eq(gid) & audit.route.eq(route) & audit.qualified]
            for rule in ['highest_probability', 'earliest']:
                row = (q.sort_values(['cpPr', 'date']).iloc[-1] if rule == 'highest_probability'
                       else q.sort_values('date').iloc[0])
                archived = picks.loc[picks.grid_id.eq(gid) & picks.route.eq(route) & picks.rule.eq(rule)]
                assert len(archived) == 1 and archived.qualifies.iloc[0]
                assert row.date == archived.selected_date.iloc[0]
    return audit


def table_tex(audit):
    visible = audit.loc[audit.date.ge('2014-01-01')]
    rows = []
    for r in visible.itertuples():
        cell = ('S' if r.region == 'south' else 'N') + r.route
        status = 'Q' if r.qualified else r.failed_checks
        rows.append(f'{cell} & {r.date:%Y-%m} & {r.cpPr:.10f} & {r.pre_slope:.3f} & '
                    f'{r.post_slope:.3f} & {r.slope_gain:.3f} & {status} & {r.selection} ' + r'\\')
    return r'''
\clearpage
\section{Candidate audit for the Figure 4 examples}

The two cells are purposive, post hoc illustrations of different input representations,
not random or representative selections: the southern example exposes early and later
qualifying reconstructed candidates, and the northern example exposes a jump/plateau
and the mission gap. Both fits end in November 2023. The displayed interval begins in
January 2014; Table~\ref{tab:fig4audit} lists every BEAST candidate in that interval.

BEAST returns candidate-associated probabilities (\texttt{trend.cpPr}) and monthly
change-occurrence probabilities (\texttt{trend.cpOccPr}) as separate fields. The 0.5
qualification threshold is applied to the former, not to individual months of the latter.
Candidate locations are extracted from the occurrence profile by BEAST's candidate
summary; a monthly peak below 0.5 does not imply failure of the candidate-level threshold.
The official field definitions are available at
\url{https://search.r-project.org/CRAN/refmans/Rbeast/html/beast.html}.

\begin{table}[htbp]
\centering\footnotesize
\caption{Candidate qualification and H/E selections for the Figure 4 examples.}
\label{tab:fig4audit}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{llrrrrll}
\toprule
Cell/route & Date & \texttt{cpPr} & Pre & Post & Gain & Check & Choice \\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\tnote{S: g\_115.0\_35.5 (south); N: g\_116.0\_37.5 (north); A/D denote the processing route.
Pre, Post and Gain are in cm/yr, calculated on the entire observed STL-deseasoned record
on each side, with at least 18 real observations per side. The same qualification record
is used for both routes, rather than qualifying D on its reconstructed signal.
Q: all checks pass. Failed checks are T (before January 2015), P (\texttt{cpPr}$<0.5$),
N (insufficient observed support or missing slope), S (Post$\leq0$), G (Gain$\leq0.5$).
H: highest \texttt{cpPr} among qualifying candidates, with exact ties resolved to the later
date; E: earliest qualifying candidate. Decisions use unrounded values.}
\end{table}

In southern route A, the June 2021 candidate has \texttt{cpPr}=0.9998750091,
although the maximum monthly \texttt{cpOccPr} is 0.4086250067. In northern route A,
the February 2016 candidate has high probability but a negative post-candidate slope
and therefore fails qualification. In northern route D, the qualifying May 2019 and
January 2021 candidates have equal stored probabilities of 1.0, so H is January 2021
under the later-date tie rule while E remains May 2019. Southern route D instead ranks
October 2020 (1.0) above April 2015 (0.9999999404), a distinction hidden by coarse rounding.

Figure 4 shows posterior mean trends only, conditional on the route input, endpoint and
BEAST settings; no trend uncertainty bands are plotted. The gray vertical shading denotes
the mission gap and is not an uncertainty interval. Figures S2--S3 show all route inputs,
not all-cell fitted trends or probability diagnostics.
'''


def render(output):
    output.mkdir(parents=True, exist_ok=False)
    files = [SOURCE / n for n in ['series.csv', 'candidates.csv', 'selections.csv']]
    s = pd.read_csv(files[0], parse_dates=['date'])
    c = pd.read_csv(files[1], parse_dates=['date'])
    picks = pd.read_csv(files[2], parse_dates=['selected_date'])
    audit = audit_candidates(s, c, picks)
    audit.to_csv(output / 'candidate_audit.csv', index=False, float_format='%.15g')
    visible = audit.loc[audit.date.ge('2014-01-01')]
    visible.to_csv(output / 'displayed_candidate_audit.csv', index=False, float_format='%.15g')
    (output / 'candidate_audit.tex').write_text(table_tex(audit), encoding='utf-8')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 9,
                         'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(176 / 25.4, 178 / 25.4), layout='constrained')
    outer = fig.add_gridspec(3, 2, height_ratios=[.20, 1, 1])
    legend = fig.add_subplot(outer[0, :]); legend.axis('off')
    handles = [Line2D([], [], ls='none', marker='.', color='.65', label='Observed GWA'),
               Line2D([], [], color=BLUE, lw=1.2, label='Route input'),
               Line2D([], [], color='.1', ls='--', lw=.95, label='BEAST mean trend'),
               Line2D([], [], color='.4', lw=.8, label='Monthly cpOccPr'),
               Line2D([], [], ls='none', marker='o', mfc='none', mec=RED, label='Qualified cpPr'),
               Line2D([], [], ls='none', marker='x', color='.45', label='Rejected cpPr')]
    legend.legend(handles=handles, ncol=3, loc='center', frameon=False, fontsize=8,
                  handlelength=2, columnspacing=1.3, handletextpad=.5, labelspacing=.5)
    maxima = {}
    for i, gid in enumerate(IDS):
        view = s.loc[s.grid_id.eq(gid) & s.date.ge('2014-01-01')]
        lo = view[['observed', 'input_signal', 'fitted_trend']].min().min() - 2
        hi = view[['observed', 'input_signal', 'fitted_trend']].max().max()
        hi = hi + .25 * (hi - lo)
        for j, route in enumerate(['A', 'D']):
            inner = outer[i + 1, j].subgridspec(2, 1, height_ratios=[2.8, 1], hspace=.04)
            ax = fig.add_subplot(inner[0]); prob = fig.add_subplot(inner[1], sharex=ax)
            g = view.loc[view.route.eq(route)].sort_values('date')
            ax.scatter(g.date, g.observed, s=5, color='.68', linewidths=0, zorder=2)
            ax.plot(g.date, g.input_signal, color=BLUE, lw=1.15, zorder=3)
            ax.plot(g.date, g.fitted_trend, color='.1', ls='--', lw=.95, zorder=4)
            for a in [ax, prob]:
                a.axvspan(pd.Timestamp('2017-07-01'), pd.Timestamp('2018-05-01'),
                          facecolor='.9', zorder=0)
                a.set_xlim(pd.Timestamp('2014-01-01'), pd.Timestamp('2024-01-01'))
            pp = picks.loc[picks.grid_id.eq(gid) & picks.route.eq(route) & picks.qualifies]
            h = pp.loc[pp.rule.eq('highest_probability'), 'selected_date'].iloc[0]
            e = pp.loc[pp.rule.eq('earliest'), 'selected_date'].iloc[0]
            for date in sorted(set([h, e])):
                ax.axvline(date, color='.25', ls='--' if date == h else '-.', lw=.75, zorder=1)
            # Full date labels in separate rows, with leaders to exact-date lines.
            for rule, date, level in [('H', h, .98), ('E', e, .85)]:
                xtext = date + pd.DateOffset(months=-10 if rule == 'H' else 8)
                ax.annotate(f'{rule}: {date:%Y-%m}', xy=(mdates.date2num(date), .70),
                            xycoords=ax.get_xaxis_transform(), xytext=(mdates.date2num(xtext), level),
                            textcoords=ax.get_xaxis_transform(), fontsize=8, ha='center', va='top',
                            bbox=dict(facecolor='white', edgecolor='none', pad=.5, alpha=.9),
                            arrowprops=dict(arrowstyle='-', color='.45', lw=.5), zorder=6)
            prob.plot(g.date, g.cp_probability, color='.4', lw=.8, zorder=2)
            q = visible.loc[visible.grid_id.eq(gid) & visible.route.eq(route)]
            for qualifies, marker, color in [(True, 'o', RED), (False, 'x', '.45')]:
                v = q.loc[q.qualified.eq(qualifies)]
                prob.scatter(v.date, v.cpPr, s=18, marker=marker,
                             **({'facecolors': 'none', 'edgecolors': color} if qualifies
                                else {'color': color}), linewidths=.9, zorder=4)
            prob.axhline(.5, color='.45', lw=.6, ls=':', zorder=1)
            prob.text(.02, .5, 'cpPr ≥ 0.5', transform=prob.get_yaxis_transform(),
                      fontsize=7, ha='left', va='bottom',
                      bbox=dict(facecolor='white', edgecolor='none', pad=.3, alpha=.85))
            prob.set(ylim=(-.04, 1.10), ylabel='Probability')
            prob.set_yticks([0, .5, 1], ['0', '.5', '1'])
            prob.xaxis.set_major_locator(mdates.YearLocator(3))
            prob.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.tick_params(labelbottom=False)
            ax.set(ylabel='GWA (cm)', ylim=(lo, hi))
            ax.set_title(f'({chr(97 + i * 2 + j)}) '+('South' if i == 0 else 'North')+
                         f' {gid} / {route}', loc='left', fontsize=8.5, pad=6)
            maxima[gid + '/' + route] = float(g.cp_probability.max())
    fig.get_layout_engine().set(h_pad=.06, w_pad=.06, hspace=.05, wspace=.06)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(output / ('Fig04.' + ext), dpi=400, facecolor='white')
    plt.close(fig)
    evidence = {'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
                'core_source_sha256': sha(ROOT/'analysis/ncp_monitoring_core.py'),
                'example_ids': IDS, 'selection_basis': 'Purposive post hoc illustrations; not representative sampling',
                'fit_endpoint': '2023-11-01', 'display_start': '2014-01-01',
                'qualification_field': 'trend.cpPr', 'monthly_profile_field': 'trend.cpOccPr',
                'maximum_monthly_profile': maxima, 'displayed_candidates': len(visible),
                'full_precision_ranking_reproduced': True, 'slopes_checked_against_archived_qualification_signal': True,
                'fit_uncertainty_bands': False, 'model_refits': 0,
                'field_documentation': 'https://search.r-project.org/CRAN/refmans/Rbeast/html/beast.html',
                'figure_sha256': sha(output/'Fig04.pdf')}
    (output/'figure4_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'candidates': len(visible), 'monthly_maxima': maxima, 'refits': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    render(parser.parse_args().output)
