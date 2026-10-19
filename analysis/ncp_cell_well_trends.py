"""Descriptive cell-level trend correspondence between fixed wells and storage.

Added after the October 18 declaration as a descriptive comparator for the
revision; it fits no change-point or decomposition model. For each of the 18
common cells it estimates 2019--August 2024 OLS slopes (intercept, calendar
time, annual and semiannual harmonics) for fixed-cohort well anomalies and for
same-mask Noah GWA, TWSA and SMSA. Cross-cell agreement is summarized with a
cell-resampling percentile interval; it is not a significance test.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ncp_monitoring_core import design, seed, json_write, sha, ROOT
from ncp_monitoring_diagnostics import model_panels, well_cells

START, END, MIN_MONTHS = '2019-01-01', '2024-08-01', 24
COMPONENTS = ['gwa_cm', 'twsa_cm', 'smsa_cm']

def slope(values, dates):
    v = np.asarray(values, float); ok = np.isfinite(v)
    if ok.sum() < MIN_MONTHS: return np.nan, int(ok.sum())
    X = design(dates)
    return float(np.linalg.lstsq(X[ok], v[ok], rcond=None)[0][1]), int(ok.sum())

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output-dir', required=True); a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=False)
    data = next(d for d in model_panels() if d.model.iloc[0] == 'shared_noah')
    cells, common = well_cells(); dates = pd.date_range(START, END, freq='MS')
    meta = data[['grid_id', 'region_tag']].drop_duplicates('grid_id').set_index('grid_id')
    rows = []
    for aq in ['unconfined', 'confined']:
        well = cells.loc[cells.aquifer_group.eq(aq)].pivot(index='date', columns='grid_id', values='well').reindex(index=dates)
        for gid in common:
            sat = data.loc[data.grid_id.eq(gid)].set_index('date').reindex(dates)
            w = well[gid].to_numpy() if gid in well else np.full(len(dates), np.nan)
            ws, wn = slope(w, dates); row = dict(aquifer=aq, grid_id=gid, region_tag=meta.loc[gid, 'region_tag'], well_m_per_yr=ws, well_months=wn)
            for c in COMPONENTS:
                # Satellite slopes use months where both GRACE and wells are observed.
                s, n = slope(np.where(np.isfinite(w), sat[c].to_numpy(), np.nan), dates)
                row[c.replace('_cm', '_cm_per_yr')] = s; row[c.replace('_cm', '_months')] = n
            rows.append(row)
    cell = pd.DataFrame(rows); cell.to_csv(out / 'cell_well_trends.csv', index=False)
    summary = []
    for aq, g in cell.groupby('aquifer'):
        g = g.dropna(subset=['well_m_per_yr']); rng = np.random.default_rng(seed('cell_well_trends', aq))
        draws = rng.integers(len(g), size=(5000, len(g)))
        for c in ['gwa_cm_per_yr', 'twsa_cm_per_yr']:
            x, y = g[c].to_numpy(), g.well_m_per_yr.to_numpy()
            boot = np.array([np.corrcoef(x[i], y[i])[0, 1] if np.std(x[i]) > 0 and np.std(y[i]) > 0 else np.nan for i in draws])
            lo, hi = np.nanquantile(boot, [.025, .975])
            summary.append(dict(aquifer=aq, component=c, n_cells=len(g), pearson=float(np.corrcoef(x, y)[0, 1]),
                                spearman=float(pd.Series(x).corr(pd.Series(y), method='spearman')), low=float(lo), high=float(hi),
                                same_sign_cells=int((np.sign(x) == np.sign(y)).sum()),
                                north_well_mean=float(g.loc[g.region_tag.eq('north'), 'well_m_per_yr'].mean()),
                                south_well_mean=float(g.loc[g.region_tag.eq('south'), 'well_m_per_yr'].mean()),
                                north_component_mean=float(g.loc[g.region_tag.eq('north'), c].mean()),
                                south_component_mean=float(g.loc[g.region_tag.eq('south'), c].mean())))
    pd.DataFrame(summary).to_csv(out / 'cell_well_trend_summary.csv', index=False)
    json_write(out / 'manifest.json', dict(descriptive_only=True, declared_before_fitting=False, model_fits_run=0,
               window=[START, END], min_months=MIN_MONTHS, land_model='shared_noah', cells=len(common),
               script_sha256=sha(__file__), well_input_sha256=sha(ROOT / 'data/external/ncp_long2025_well_2005_2024.csv')))

if __name__ == '__main__':
    main()
