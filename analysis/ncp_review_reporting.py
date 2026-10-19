"""Summarize paired stage hits from existing trials; perform no model fits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/monitoring_revision/2026-10-18-r1/inference/two_stage_trials.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(SOURCE)
    data = data.loc[data.setting.eq('original') & data.rule.eq('highest_probability') & data.regime.eq('median')]
    rows = []
    for keys, group in data.groupby(['route', 'onset', 'weak_slope', 'mask'], sort=True):
        route, onset, weak, mask = keys
        first = group.onset_hit.to_numpy(dtype=bool)
        later = group.acceleration_hit.to_numpy(dtype=bool)
        delta = later.astype(float) - first.astype(float)
        # Resampling preserves each realization's paired stage outcomes.
        key = '|'.join(map(str, keys)).encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
        draws = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(axis=1)
        low, high = np.quantile(draws, [.025, .975])
        rows.append(dict(route=route, onset=onset, weak_slope=weak, mask=mask, n=len(group),
                         first_hit=first.mean(), acceleration_hit=later.mean(),
                         paired_difference=delta.mean(), low=low, high=high))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'paired_stage_hits.csv', index=False)
    observed = summary.loc[summary['mask'].eq('observed')]
    lines = [r'\begin{table}[htbp]', r'\centering\small', r'\renewcommand{\arraystretch}{1.1}',
             r'\caption{Paired first-reversal and acceleration hits on the observed calendar}',
             r'\label{tab:paired-stage-hits}', r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Route & First year & Weak slope & First (\%) & Accel. (\%) & Difference [95\%] \\', r'\midrule']
    for row in observed.itertuples():
        lines.append(f'{row.route} & {row.onset[:4]} & {row.weak_slope:.1f} & {100*row.first_hit:.1f} & '
                     f'{100*row.acceleration_hit:.1f} & {100*row.paired_difference:.1f} '
                     f'[{100*row.low:.1f}, {100*row.high:.1f}] '+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\par\smallskip\raggedright\footnotesize Original highest-probability rule; median residual level; 200 realizations per row. Weak slope is in cm/yr. Hits are within 12 months of the respective stage and include all realizations. Differences are acceleration minus first-reversal hits in percentage points; percentile intervals use 2,000 paired realization resamples. The 2020 trajectories have 20 calendar months after acceleration. Complete-calendar counterparts are in the numerical evidence.',
              r'\end{table}']
    (out / 'paired_stage_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    manifest = dict(model_fits_run=0, source=str(SOURCE.relative_to(ROOT)),
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    bootstrap_replicates=2000, rows=len(summary))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(observed.to_string(index=False))


if __name__ == '__main__':
    main()
