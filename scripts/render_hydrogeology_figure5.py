"""Clarify Figure 5 using frozen simulations; apply fixed rules to one held-out set.

No simulations, BEAST fits, or threshold searches are run. Output must be new.
"""
from pathlib import Path
from statistics import NormalDist
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'outputs/monitoring_revision/2026-10-18-r1'
ROUTES = [('A', '#222222', 'o'), ('D', '#777777', '^')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k, n):
    z = NormalDist().inv_cdf(.975)
    p = k / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return center-half, center+half


def validation_comparison(truth, candidates, original):
    for field in ['noise_seed', 'beast_seed', 'ensemble_seed']:
        assert set(truth.loc[truth.split.eq('calibration'), field]).isdisjoint(
            set(truth.loc[truth.split.eq('validation'), field]))
    t = truth.loc[truth.split.eq('validation') & truth.scenario.isin(
        ['no_recovery', 'temporary_gain'])].copy()
    assert t.task_id.is_unique
    assert t.groupby(['scenario', 'regime']).size().eq(500).all()
    summaries, paired = [], []
    for route, _, _ in ROUTES:
        both = t[['task_id', 'scenario', 'regime', 'replicate']].copy()
        for label, probability, gain in [('original', .5, .5), ('strictest', .99, 2.)]:
            q = candidates.loc[candidates.route.eq(route) & candidates.split.eq('validation')
                               & candidates.date.ge('2015-01-01')
                               & candidates.probability.ge(probability)
                               & candidates.post_slope.gt(0)
                               & (candidates.post_slope-candidates.pre_slope).gt(gain)]
            chosen = q.sort_values(['task_id', 'probability', 'date'],
                                   ascending=[True, False, False]).drop_duplicates('task_id')
            chosen = chosen[['task_id', 'date']].rename(columns={'date': 'date_'+label})
            both = both.merge(chosen, on='task_id', how='left', validate='one_to_one')
            for (scenario, regime), group in both.groupby(['scenario', 'regime']):
                n = len(group)
                k = int(group['date_'+label].notna().sum())
                low, high = wilson(k, n)
                if label == 'original':
                    old = original.loc[original.route.eq(route) & original.scenario.eq(scenario)
                                       & original.regime.eq(regime)].iloc[0]
                    assert n == old.n and k == old.qualified
                    assert np.allclose([k/n, low, high], [old.fraction, old.low, old.high],
                                       atol=1e-12, rtol=0)
                summaries.append(dict(route=route, setting=label, split='validation',
                                      scenario=scenario, regime=regime, n=n, reported=k,
                                      fraction=k/n, low=low, high=high,
                                      probability_threshold=probability, gain_threshold=gain))
        both['route'] = route
        assert not (both.date_strictest.notna() & both.date_original.isna()).any()
        paired.append(both)
    return pd.DataFrame(summaries), pd.concat(paired, ignore_index=True)


def render(output):
    output.mkdir(parents=True, exist_ok=False)
    cand_path = EVIDENCE / 'calibration/candidates.csv'
    precomp_comparison = ROOT / 'outputs/current_figure_checks/figure5_evidence/same_validation_comparison.csv'
    precomp_paired = ROOT / 'outputs/current_figure_checks/figure5_evidence/paired_validation_reports.csv'
    if cand_path.is_file():
        paths = [EVIDENCE / x for x in ['calibration/truth.csv', 'calibration/candidates.csv',
                 'inference/original_rule_validation.csv', 'inference/two_stage_summary.csv',
                 'inference/two_stage_trials.csv', 'inference/rules.json']]
        truth, candidates, original, summary, trials = [pd.read_csv(p) for p in paths[:5]]
        comparison, paired = validation_comparison(truth, candidates, original)
    else:
        paths = [EVIDENCE / 'calibration/truth.csv', precomp_comparison, precomp_paired,
                 EVIDENCE / 'inference/original_rule_validation.csv',
                 EVIDENCE / 'inference/two_stage_summary.csv',
                 EVIDENCE / 'inference/two_stage_trials.csv', EVIDENCE / 'inference/rules.json']
        truth, comparison, paired, original, summary, trials = [pd.read_csv(p) for p in paths[:6]]
        for field in ['noise_seed', 'beast_seed', 'ensemble_seed']:
            assert set(truth.loc[truth.split.eq('calibration'), field]).isdisjoint(
                set(truth.loc[truth.split.eq('validation'), field]))
        for r in comparison.loc[comparison.setting.eq('original')].itertuples():
            old = original.loc[original.route.eq(r.route) & original.scenario.eq(r.scenario)
                               & original.regime.eq(r.regime)].iloc[0]
            assert r.n == old.n and r.reported == old.qualified
            assert np.allclose([r.fraction, r.low, r.high], [old.fraction, old.low, old.high],
                               atol=1e-12, rtol=0)
        assert not (paired.date_strictest.notna() & paired.date_original.isna()).any()
    comparison.to_csv(output/'same_validation_comparison.csv', index=False)
    paired.to_csv(output/'paired_validation_reports.csv', index=False)
    fixed = trials.loc[trials.setting.eq('original') & trials.rule.eq('highest_probability')
                       & trials.regime.eq('median') & trials.onset.eq('2018-01-01')
                       & trials.weak_slope.eq(.6)]
    onset = summary.loc[summary.setting.eq('original') & summary.rule.eq('highest_probability')
                        & summary.regime.eq('median') & summary.weak_slope.eq(.6)].copy()
    onset.to_csv(output/'onset_hit_settings.csv', index=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8,
                         'axes.titlesize': 8.5, 'axes.labelsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 2, figsize=(176/25.4, 174/25.4), layout='constrained')
    a, b, c, d = axs.flat
    dates = pd.date_range('2014-01-01', '2024-08-01', freq='MS')
    t = (dates.year-2003)+(dates.month-1)/12
    for weak, linestyle in [(.3, ':'), (.6, '-'), (1.2, '--')]:
        curve = -1.5*t+(1.5+weak)*np.maximum(t-15, 0)+(4-weak)*np.maximum(t-18, 0)
        a.plot(dates, curve, color='.15', ls=linestyle, lw=1.1,
               label=f'+{weak:g} cm/yr')
    a.axvspan(pd.Timestamp('2017-07-01'), pd.Timestamp('2018-05-01'), color='.92', zorder=0)
    a.axvline(pd.Timestamp('2018-01-01'), color='.4', ls='--', lw=.8)
    a.axvline(pd.Timestamp('2021-01-01'), color='.4', ls=':', lw=.8)
    a.text(pd.Timestamp('2018-03-01'), -11, 'First reversal\n2018-01', fontsize=7.5,
           va='top', bbox=dict(facecolor='white', edgecolor='none', pad=.4, alpha=.9))
    a.text(pd.Timestamp('2021-03-01'), -5, 'Acceleration\n2021-01', fontsize=7.5,
           va='top', bbox=dict(facecolor='white', edgecolor='none', pad=.4, alpha=.9))
    a.set_title('(a) Two-stage truths\n36 months between stages', loc='left', pad=6)
    a.set_ylabel('True storage trend (cm)')
    a.legend(title='Weak recovery slope', title_fontsize=7.5, fontsize=7.5,
             frameon=False, loc='upper left', handlelength=2)
    a.xaxis.set_major_locator(mdates.YearLocator(4))
    a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    nulls = [(s, r) for s in ['no_recovery', 'temporary_gain'] for r in ['low', 'median', 'high']]
    for j, (route, color, marker) in enumerate(ROUTES):
        for k, (label, filled, text) in enumerate([('original', True, 'original'),
                                                 ('strictest', False, 'strictest')]):
            g = comparison.loc[comparison.route.eq(route) & comparison.setting.eq(label)]
            g = g.set_index(['scenario', 'regime']).reindex(nulls)
            x = np.arange(6)+(j*2+k-1.5)*.13
            b.errorbar(x, g.fraction, yerr=[g.fraction-g.low, g.high-g.fraction],
                       fmt=marker, color=color, mfc=color if filled else 'white',
                       ms=3.5, capsize=1.5, lw=.8,
                       label=f'{route}: '+('orig .50/.5' if filled else 'strict .99/2'))
    b.axhline(.05, color='.5', ls=':', lw=.8)
    b.text(5.6, .055, '5%', fontsize=7.5, color='.4', va='bottom', ha='right')
    b.axvline(2.5, color='.85', lw=.6)
    b.set_xticks(range(6), ['L', 'M\nDecline', 'H', 'L', 'M\nTemp. gain', 'H'])
    b.set(xlabel='Null trajectory and residual level', ylabel='Fraction of 500 realizations',
          ylim=(0, .62))
    b.set_title('(b) False reports\nBoth rules: same held-out validation set', loc='left', pad=6)
    b.legend(title='cpPr / gain (cm/yr)', title_fontsize=7.5,
             frameon=False, fontsize=7.5, ncol=2, loc='upper left',
             handlelength=1, columnspacing=.6, handletextpad=.3)

    boxes, labels, colors, counts, error_rows = [], [], [], [], []
    for route, color, _ in ROUTES:
        for mask, text in [('complete', 'full'), ('observed', 'gaps')]:
            g = fixed.loc[fixed.route.eq(route) & fixed['mask'].eq(mask)]
            assert len(g) == 200 and g.task_id.is_unique
            q = g.error_months.dropna().to_numpy()
            assert g.qualifies.sum() == len(q)
            boxes.append(q); colors.append(color)
            labels.append(f'{route}\n{text}\nn={len(q)}')
            q1, med, q3 = np.percentile(q, [25, 50, 75])
            iqr = q3-q1
            low = q[q >= q1-1.5*iqr].min(); high = q[q <= q3+1.5*iqr].max()
            counts.append(dict(route=route, mask=mask, total=200, dated=len(q),
                               q1=q1, median=med, q3=q3, whisker_low=low, whisker_high=high,
                               outliers=int(((q < low) | (q > high)).sum())))
            error_rows.extend(dict(route=route, mask=mask, task_id=r.task_id,
                                   error_months=r.error_months) for r in g.itertuples()
                              if pd.notna(r.error_months))
    bp = c.boxplot(boxes, patch_artist=True, showfliers=True, widths=.55, whis=1.5,
                   medianprops=dict(color='.1', lw=1),
                   flierprops=dict(marker='.', markersize=2, alpha=.6))
    for box, color in zip(bp['boxes'], colors):
        box.set(facecolor=color, alpha=.3)
    c.axhline(0, color='.5', ls=':', lw=.8)
    c.axhline(36, color='.5', ls='--', lw=.8)
    c.set_xticks(range(1, 5), labels)
    c.set_xlim(.5, 6.2)
    for level, text in [(0, 'First\nreversal'), (36, 'Later\nacceleration')]:
        c.text(4.6, level, text, ha='left', va='center', fontsize=7.5, color='.3',
               bbox=dict(facecolor='white', edgecolor='none', pad=.7))
    c.set_ylabel('Selected date minus first reversal (months)')
    c.set_title('(c) Conditional date errors\n2018; +0.6 cm/yr; median residuals', loc='left', pad=6)
    pd.DataFrame(counts).to_csv(output/'conditional_box_statistics.csv', index=False)
    pd.DataFrame(error_rows).to_csv(output/'conditional_date_errors.csv', index=False)

    d.axvspan(2019.65, 2020.4, facecolor='.94', zorder=0)
    for route, color, _ in ROUTES:
        for mask, linestyle, marker, text in [('complete', '-', 'o', 'full'),
                                               ('observed', '--', '^', 'gaps')]:
            g = onset.loc[onset.route.eq(route) & onset['mask'].eq(mask)].sort_values('onset')
            assert len(g) == 4 and g.n.eq(200).all()
            years = pd.to_datetime(g.onset).dt.year.to_numpy()
            rate = g.onset_hit_rate.to_numpy()
            lower = (g.onset_hit_rate-g.onset_low).to_numpy()
            upper = (g.onset_high-g.onset_hit_rate).to_numpy()
            long = years < 2020; short = years == 2020
            d.errorbar(years[long], rate[long], yerr=[lower[long], upper[long]],
                       color=color, ls=linestyle, marker=marker, ms=3, capsize=2, lw=1,
                       label=route+': '+text)
            d.errorbar(years[short], rate[short], yerr=[lower[short], upper[short]],
                       color=color, ls='none', marker=marker, mfc='white', mec=color,
                       ms=4, capsize=2, lw=.9)
    d.set(xlabel='Year of first reversal', ylabel='Fraction of all 200 realizations',
          ylim=(-.03, 1.03), xlim=(2015.7, 2020.4), xticks=[2016, 2018, 2019, 2020])
    d.set_title('(d) Onset hit (±12 months)\nWeak +0.6 cm/yr; median residuals', loc='left', pad=6)
    d.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper left',
             handlelength=1.5, columnspacing=.7, handletextpad=.4)
    d.text(.97, .64, 'Short follow-up\n20 mo after acceleration',
           transform=d.transAxes, fontsize=7.5, ha='right', va='top',
           bbox=dict(facecolor='white', edgecolor='none', pad=.8, alpha=.9))
    fig.get_layout_engine().set(h_pad=.075, w_pad=.055, hspace=.06, wspace=.06)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(output/('Fig05.'+ext), dpi=400, facecolor='white')
    plt.close(fig)
    maxima = comparison.loc[comparison.setting.eq('strictest')].groupby('route').fraction.max().to_dict()
    assert maxima == {'A': .158, 'D': .334}
    metadata = {'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                'panel_b_split': 'same validation tasks for both fixed rules',
                'thresholds': {'original': [.5, .5], 'strictest': [.99, 2.]},
                'threshold_retuning': False, 'strictest_rule_status': 'fixed-grid diagnostic; not a feasible calibrated rule',
                'independence_checked': ['noise_seed', 'beast_seed', 'ensemble_seed'],
                'original_validation_counts_and_wilson_intervals_reproduced': True,
                'strictest_heldout_maximum_false_report_rate': maxima,
                'panel_c_dated_counts': counts,
                'boxplot': 'Q1-Q3; median; whiskers to observed extrema within 1.5 IQR; all outliers retained',
                'panels_c_d_settings': {'weak_slope_cm_per_yr': .6, 'residual_regime': 'median',
                                      'rule': 'original/highest_probability'},
                '2020_follow_up': '20 calendar months after acceleration; open symbols, no connecting segment',
                'new_simulations': 0, 'model_refits': 0, 'figure_sha256': sha(output/'Fig05.pdf')}
    (output/'figure5_evidence.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'strictest_validation_max': maxima, 'box_counts': [x['dated'] for x in counts]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    render(parser.parse_args().output)
