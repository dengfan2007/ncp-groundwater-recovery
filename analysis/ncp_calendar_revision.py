"""Calendar-preserving revision; write new evidence without overwriting historical runs.

Run with the documented Anaconda interpreter and --output-dir outputs/runs/<label>.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

import ncp_recovery_analysis as core
import ncp_revision_analyses as revision
from ncp_trajectory_postprocess import build_grid_metadata
from ncp_well_utils import standardize_well_data

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def spatial_effects(df, block_size=2.0, n_boot=5000):
    df = df.loc[df.is_recovery_relevant & df.onset_date.notna()].copy()
    df["onset_month"] = df.onset_date.dt.year * 12 + df.onset_date.dt.month - 1
    df["block"] = revision.assign_spatial_blocks(df, block_size)
    blocks = sorted(df.block.unique())
    rng = np.random.default_rng(SEED)
    draws = rng.integers(max(len(blocks), 1), size=(n_boot, len(blocks)))
    rows = []
    for col in ["onset_month", "post_trend_cm_per_year"]:
        sums, counts = [], []
        for block in blocks:
            b = df.loc[df.block == block]
            sums.append([b.loc[b.region_tag == r, col].sum() for r in ["north", "south"]])
            counts.append([sum(b.region_tag == r) for r in ["north", "south"]])
        sums, counts = np.asarray(sums).reshape(-1, 2), np.asarray(counts).reshape(-1, 2)
        sc, ss = counts[draws].sum(axis=1), sums[draws].sum(axis=1)
        ok = (sc > 0).all(axis=1)
        differences = (ss[ok] / sc[ok])[:, 1] - (ss[ok] / sc[ok])[:, 0]
        ci = np.quantile(differences, [.025, .975]) if len(differences) else [np.nan, np.nan]
        rows.append(dict(metric=col, estimate=df.loc[df.region_tag == "south", col].mean() - df.loc[df.region_tag == "north", col].mean(), ci_lower=ci[0], ci_upper=ci[1], block_degrees=block_size, n_blocks=len(blocks), n_north=sum(df.region_tag == "north"), n_south=sum(df.region_tag == "south"), valid_replicates=len(differences), requested_replicates=n_boot))
    return pd.DataFrame(rows)


def summarize(breaks, variant):
    valid = breaks.loc[breaks.is_recovery_relevant].copy()
    effects = spatial_effects(breaks)
    months = valid.onset_date.dt.year * 12 + valid.onset_date.dt.month - 1
    def monthstr(x):
        if pd.isna(x):
            return None
        x = int(np.rint(x))
        return f"{x // 12:04d}-{x % 12 + 1:02d}"
    return dict(variant=variant, n_grids=len(breaks), n_reportable=len(valid), n_beast=sum(breaks.selected_method == "beast"), n_segmented=sum(breaks.selected_method == "segmented_regression"), median_onset=monthstr(months.median()), q25_onset=monthstr(months.quantile(.25)), q75_onset=monthstr(months.quantile(.75)), median_posttrend=valid.post_trend_cm_per_year.median(), onset_south_minus_north=effects.iloc[0].estimate, trend_south_minus_north=effects.iloc[1].estimate)


def calendar_detrend(y):
    t = np.arange(len(y), dtype=float)
    design = np.column_stack([np.ones(len(t)), t, *[f(2*np.pi*k*t/12) for k in (1, 2) for f in (np.sin, np.cos)]])
    mask = np.isfinite(y)
    out = np.full(len(y), np.nan)
    if mask.sum() >= 12:
        out[mask] = y[mask] - design[mask] @ np.linalg.lstsq(design[mask], y[mask], rcond=None)[0]
    return out


def slope_ci(y, n_boot=1000):
    """Fixed calendar design; resample contiguous residual blocks, not raw trends."""
    t = np.arange(len(y), dtype=float) / 12
    ok = np.isfinite(y)
    slope, intercept, _, _ = stats.theilslopes(y[ok], t[ok])
    fitted = intercept + slope*t
    residual = y - fitted
    residual -= np.nanmean(residual)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(n_boot):
        idx = revision.moving_block_indices(len(y), 12, rng)
        b = fitted + residual[idx]
        valid = ok & np.isfinite(b)
        if valid.sum() >= 18:
            boot.append(stats.theilslopes(b[valid], t[valid])[0])
    low, high = np.quantile(boot, [.025, .975])
    return slope, low, high


def validation(panel, metadata, wells, out):
    series = revision.build_validation_series(panel, metadata, wells)
    # Use the full calendar, including months without a GRACE observation.
    frames, metrics = [], []
    for region in ["all", "north", "south"]:
        sub = series.loc[series.region_tag == region].set_index("date").sort_index()
        if sub.empty:
            continue
        sub = sub.reindex(pd.date_range(sub.index.min(), sub.index.max(), freq="MS"))
        sub.index.name = "date"
        sub["region_tag"] = region
        frames.append(sub.reset_index())
        for period, sel in [("full", sub), ("2019+", sub.loc["2019-01-01":])]:
            a, b = sel.gwa_cm.to_numpy(), sel.well_anomaly.to_numpy()
            ok = np.isfinite(a) & np.isfinite(b)
            da, db = np.diff(a), np.diff(b)
            dok = np.isfinite(da) & np.isfinite(db)
            ac, bc = calendar_detrend(a), calendar_detrend(b)
            cok = np.isfinite(ac) & np.isfinite(bc)
            rng = np.random.default_rng(SEED)
            cors = []
            for _ in range(2000):
                ix = revision.moving_block_indices(len(a), 12, rng)
                cors.append(revision.pearson_or_nan(a[ix], b[ix]))
            lo, hi = np.nanquantile(cors, [.025, .975])
            row = dict(region=region, period=period, calendar_months=len(sel), paired_months=int(ok.sum()), pearson=stats.pearsonr(a[ok], b[ok]).statistic, pearson_low=lo, pearson_high=hi, spearman=stats.spearmanr(a[ok], b[ok]).statistic, detrended_deseasoned_r=stats.pearsonr(ac[cok], bc[cok]).statistic, first_difference_r=stats.pearsonr(da[dok], db[dok]).statistic, consecutive_pairs=int(dok.sum()))
            ad, bd = np.diff(ac), np.diff(bc)
            ddok = np.isfinite(ad) & np.isfinite(bd)
            row["deseasoned_first_difference_r"] = stats.pearsonr(ad[ddok], bd[ddok]).statistic
            if period == "2019+":
                for name, values in [("remote", np.where(ok, a, np.nan)), ("well", np.where(ok, b, np.nan))]:
                    est, low, high = slope_ci(values)
                    row.update({f"{name}_slope": est, f"{name}_slope_low": low, f"{name}_slope_high": high})
            metrics.append(row)
    pd.concat(frames).to_csv(out / "validation_series.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "validation_metrics.csv", index=False)
    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    panel = core.load_panel(ROOT / "data/raw/grid_monthly_panel.csv")
    meta = build_grid_metadata(panel)
    summaries, break_frames = [], []
    variants = [("calendar_stl", panel, "stl"), ("observed_harmonic", panel, "harmonic")]
    for name in ["contemporaneous", "lag03"]:
        adjusted, coef = revision.build_adjusted_panel(panel, name)
        coef.rename("coefficient").to_csv(out / f"water_{name}_coefficients.csv")
        variants.append((f"water_{name}", adjusted, "stl"))
    variants.append(("through_2020", panel.loc[panel.date < "2021-01-01"], "stl"))
    # Separate correction of the time axis from release of climate-covariate filtering.
    legacy = pd.read_csv(ROOT / "outputs/main/grid_counterfactual.csv", parse_dates=["date"])
    variants.append(("legacy_support_calendar", legacy, "stl"))
    for variant, data, season in variants:
        print(f"Fitting {variant}", flush=True)
        breaks = core.detect_breakpoints(data, seasonal_method=season).merge(meta, on="grid_id")
        breaks["onset_date"] = pd.to_datetime(breaks.onset_date)
        breaks.to_csv(out / f"breakpoints_{variant}.csv", index=False)
        summaries.append(summarize(breaks, variant))
        break_frames.append(breaks.assign(variant=variant))
        print(summaries[-1], flush=True)
    base = break_frames[0].copy()
    segmented = base.copy()
    segmented["onset_date"] = pd.to_datetime(segmented.segmented_onset_date)
    segmented["post_trend_cm_per_year"] = segmented.segmented_post_trend_cm_per_year
    segmented["pre_trend_cm_per_year"] = segmented.segmented_pre_trend_cm_per_year
    segmented["is_recovery_relevant"] = [core.is_recovery_relevant_break(r.onset_date, r.pre_trend_cm_per_year, r.post_trend_cm_per_year) and r.segmented_model_type == "single_break" for r in segmented.itertuples()]
    segmented["selected_method"] = np.where(segmented.is_recovery_relevant, "segmented_regression", "no_break_detected")
    segmented.to_csv(out / "breakpoints_segmented_only.csv", index=False)
    summaries.append(summarize(segmented, "segmented_only"))
    spatial_effects(segmented).to_csv(out / "segmented_spatial_effects.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "sensitivity_summary.csv", index=False)
    spatial = pd.concat([spatial_effects(base, size) for size in [1., 2., 3.]], ignore_index=True)
    spatial.to_csv(out / "spatial_effects.csv", index=False)
    cluster = base.loc[base.onset_date >= "2018-01-01"]
    pd.concat([spatial_effects(cluster, size) for size in [1., 2., 3.]], ignore_index=True).to_csv(out / "late_cluster_spatial_effects.csv", index=False)
    jack = []
    for gid in base.grid_id:
        jack.append(spatial_effects(base.loc[base.grid_id != gid], n_boot=100).assign(omitted_grid=gid))
    pd.concat(jack).to_csv(out / "leave_one_grid_out.csv", index=False)
    coverage = []
    for threshold in [.6, .7, .8]:
        ids = panel.loc[panel.coverage_ratio >= threshold, "grid_id"].unique()
        coverage.append(spatial_effects(base.loc[base.grid_id.isin(ids)]).assign(coverage_threshold=threshold))
    pd.concat(coverage).to_csv(out / "coverage_effects.csv", index=False)
    quality = panel.groupby("grid_id").agg(calendar_months=("date", "size"), observed_gwa=("gwa_cm", "count"))
    quality["missing_fraction"] = 1 - quality.observed_gwa / quality.calendar_months
    quality.to_csv(out / "calendar_qc.csv")
    missing = panel.groupby("date").gwa_cm.count().eq(0)
    pd.DataFrame({"date": missing.index, "all_grids_missing": missing.to_numpy()}).to_csv(out / "monthly_availability.csv", index=False)
    # Candidate-method comparison uses only recovery-relevant candidates on both sides.
    comparisons = []
    for _, row in base.iterrows():
        seg = pd.to_datetime(row.segmented_onset_date)
        beast = pd.to_datetime(row.beast_onset_date)
        if core.is_recovery_relevant_break(seg, row.segmented_pre_trend_cm_per_year, row.segmented_post_trend_cm_per_year) and row.segmented_model_type == "single_break" and core.is_recovery_relevant_break(beast, row.beast_pre_trend_cm_per_year, row.beast_post_trend_cm_per_year) and row.beast_breakpoint_confidence >= .5:
            delta = (beast.year-seg.year)*12 + beast.month-seg.month
            comparisons.append(dict(grid_id=row.grid_id, segmented=seg, beast=beast, month_difference=delta))
    pd.DataFrame(comparisons).to_csv(out / "method_comparison.csv", index=False)
    comparable_ids = [r["grid_id"] for r in comparisons]
    pd.concat([spatial_effects(d.loc[d.grid_id.isin(comparable_ids)]).assign(method=name) for name,d in [("beast",base),("segmented",segmented)]]).to_csv(out / "paired_method_spatial_effects.csv", index=False)
    # Water fraction remains an area fraction; it is never subtracted as storage.
    water = panel.assign(period=np.where(panel.date < "2019-01-01", "pre", "post"))
    water = water.loc[water.date >= "2015-01-01"].groupby(["grid_id", "period"]).water_frac.mean().unstack()
    water["water_change"] = water.post-water.pre
    linked = base.merge(water.water_change, on="grid_id").dropna(subset=["post_trend_cm_per_year", "water_change"])
    rho, p = stats.spearmanr(linked.post_trend_cm_per_year, linked.water_change)
    linked.to_csv(out / "water_association.csv", index=False)
    print("Validating wells", flush=True)
    raw_wells = pd.read_csv(ROOT / "data/external/ncp_long2025_well_2005_2024.csv")
    raw_wells = raw_wells.loc[raw_wells.aquifer_group.eq("unconfined") & raw_wells.grid_id.isin(meta.grid_id)]
    wells = standardize_well_data(raw_wells, dataset_label="public shallow wells", value_kind="level", value_col="gw_level")
    metrics = validation(panel, meta, wells, out)
    # Stable network: at least 80% coverage separately before and after January 2019.
    wells["period"] = np.where(wells.date < "2019-01-01", "pre", "post")
    counts = wells.groupby(["well_id", "period"]).date.nunique().unstack(fill_value=0)
    pre_months = len(pd.date_range(wells.date.min(), "2018-12-01", freq="MS"))
    post_months = len(pd.date_range("2019-01-01", wells.date.max(), freq="MS"))
    stable_ids = counts.index[(counts.pre >= .8*pre_months) & (counts.post >= .8*post_months)]
    full_period_stable = len(stable_ids)
    # The network changes sharply in 2018; do not fabricate a balanced 2005--2024 cohort.
    # Audit a declared shorter window with coverage in both its 2018 and 2019+ parts.
    recent_raw = raw_wells.loc[raw_wells.date.ge("2018-01-01") & raw_wells.date.le(panel.date.max().strftime("%Y-%m-%d"))].copy()
    recent_raw["period"] = np.where(recent_raw.date < "2019-01-01", "pre", "post")
    recent_counts = recent_raw.groupby(["well_id", "period"]).date.nunique().unstack(fill_value=0)
    recent_post_months = len(pd.date_range("2019-01-01", panel.date.max(), freq="MS"))
    stable_ids = recent_counts.index[(recent_counts.pre >= .8*12) & (recent_counts.post >= .8*recent_post_months)]
    stable = standardize_well_data(recent_raw.loc[recent_raw.well_id.isin(stable_ids)], dataset_label="stable 2018+ shallow wells", value_kind="level", value_col="gw_level")
    stable_out = out / "stable_wells"
    stable_out.mkdir()
    validation(panel, meta, stable, stable_out)
    matched_out = out / "matched_stable_wells"
    matched_out.mkdir()
    validation(panel.loc[panel.grid_id.isin(stable.grid_id.unique())], meta, stable, matched_out)
    counts.to_csv(out / "well_period_coverage.csv")
    recent_counts.to_csv(out / "well_recent_coverage.csv")
    inputs = [ROOT / p for p in ["data/raw/grid_monthly_panel.csv", "data/external/ncp_long2025_well_2005_2024.csv", "outputs/main/grid_counterfactual.csv"]]
    manifest = dict(seed=SEED, python=sys.version, packages={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "statsmodels", "Rbeast", "matplotlib"]}, input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, git_base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / "analysis/ncp_recovery_analysis.py", ROOT / "analysis/ncp_revision_analyses.py", ROOT / "analysis/ncp_well_utils.py"]}, water_spearman=float(rho), water_p=float(p), shallow_wells=int(wells.well_id.nunique()), stable_wells=int(len(stable_ids)), stable_grids=int(stable.grid_id.nunique()), method_comparable=len(comparisons), method_mean_absolute_month_difference=float(np.mean([abs(c["month_difference"]) for c in comparisons])))
    manifest["full_period_80pct_cohort_wells"] = int(full_period_stable)
    # Duplicate distribution metadata can disagree with the imported module on this host.
    manifest["distribution_metadata_versions"] = manifest["packages"].copy()
    manifest["packages"] = {name: getattr(importlib.import_module(name), "__version__", importlib.metadata.version(name)) for name in manifest["packages"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
