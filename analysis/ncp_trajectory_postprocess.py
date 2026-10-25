from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ncp_well_utils import standardize_well_data


GRID_BBOX = (112.0, 34.5, 120.0, 40.6)
GRID_STEP = 1.0
NORTH_LAT_THRESHOLD = 37.0
WATER_PRE_START = pd.Timestamp("2015-01-01")
WATER_PRE_END = pd.Timestamp("2018-12-31")
WATER_POST_START = pd.Timestamp("2019-01-01")
LEGACY_END = pd.Timestamp("2016-12-31")
EARLY_REBOUND_START = pd.Timestamp("2018-10-01")
EARLY_REBOUND_END = pd.Timestamp("2019-08-31")
MAIN_REBOUND_START = pd.Timestamp("2019-09-01")
MAIN_REBOUND_END = pd.Timestamp("2020-12-31")
EVENT_START = pd.Timestamp("2019-01-01")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess NCP groundwater outputs into trajectory-focused contribution files.")
    parser.add_argument("--panel", required=True, help="Path to grid_monthly_panel.csv")
    parser.add_argument("--counterfactual", required=True, help="Path to grid_counterfactual.csv")
    parser.add_argument("--breakpoints", required=True, help="Path to grid_breakpoints.csv")
    parser.add_argument("--case", help="Path to hutuo_case_buffers.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for derived outputs")
    parser.add_argument("--trajectory-post-threshold", type=float, default=1.0)
    parser.add_argument("--well-data", help="Optional public well dataset CSV")
    parser.add_argument("--well-id-col", default="well_id")
    parser.add_argument("--well-date-col", default="date")
    parser.add_argument("--well-value-col", default="gw_level")
    parser.add_argument("--well-value-kind", choices=["anomaly", "level"], default="level")
    parser.add_argument("--well-grid-col", default="grid_id")
    parser.add_argument("--well-lon-col", default="lon")
    parser.add_argument("--well-lat-col", default="lat")
    parser.add_argument(
        "--well-query",
        help="Optional pandas query string applied to the raw well table before standardization.",
    )
    return parser.parse_args()


def parse_grid_id(grid_id: str) -> tuple[float | None, float | None]:
    match = re.fullmatch(r"g_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)", str(grid_id))
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def coord_to_grid_id(lon: float, lat: float) -> str | None:
    if any(pd.isna(v) for v in [lon, lat]):
        return None
    west, south, east, north = GRID_BBOX
    if lon < west or lon >= east or lat < south or lat >= north:
        return None
    lon_base = west + math.floor((lon - west) / GRID_STEP) * GRID_STEP
    lat_base = south + math.floor((lat - south) / GRID_STEP) * GRID_STEP
    return f"g_{lon_base:.1f}_{lat_base:.1f}"


def load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def build_grid_metadata(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ["grid_id", "grid_lon", "grid_lat", "region_tag"] if col in panel.columns]
    metadata = panel[cols].drop_duplicates("grid_id").copy()
    if "grid_lon" not in metadata.columns or "grid_lat" not in metadata.columns:
        parsed = metadata["grid_id"].apply(parse_grid_id)
        metadata["grid_lon"] = parsed.apply(lambda x: None if x[0] is None else x[0] + 0.5)
        metadata["grid_lat"] = parsed.apply(lambda x: None if x[1] is None else x[1] + 0.5)
    if "region_tag" not in metadata.columns:
        metadata["region_tag"] = np.where(metadata["grid_lat"] >= NORTH_LAT_THRESHOLD, "north", "south")
    return metadata


def build_water_change(counterfactual: pd.DataFrame) -> pd.DataFrame:
    counterfactual["date"] = pd.to_datetime(counterfactual["date"])
    pre = (
        counterfactual.loc[(counterfactual["date"] >= WATER_PRE_START) & (counterfactual["date"] <= WATER_PRE_END)]
        .groupby("grid_id", as_index=False)["water_frac"]
        .mean()
        .rename(columns={"water_frac": "water_frac_pre_2015_2018"})
    )
    post = (
        counterfactual.loc[counterfactual["date"] >= WATER_POST_START]
        .groupby("grid_id", as_index=False)["water_frac"]
        .mean()
        .rename(columns={"water_frac": "water_frac_post_2019_2024"})
    )
    out = pre.merge(post, on="grid_id", how="outer")
    out["water_change_2019_2024_minus_2015_2018"] = out["water_frac_post_2019_2024"] - out["water_frac_pre_2015_2018"]
    return out


def classify_trajectory(onset_date: pd.Timestamp | None, post_trend: float, threshold: float) -> str:
    if pd.notna(onset_date):
        if onset_date <= LEGACY_END and pd.notna(post_trend) and post_trend <= threshold:
            return "legacy_slowdown"
        if EARLY_REBOUND_START <= onset_date <= EARLY_REBOUND_END and pd.notna(post_trend) and post_trend > threshold:
            return "early_rebound"
        if MAIN_REBOUND_START <= onset_date <= MAIN_REBOUND_END and pd.notna(post_trend) and post_trend > threshold:
            return "main_rebound"
    return "stalled_or_declining"


def build_trajectory_types(
    breakpoints: pd.DataFrame,
    counterfactual: pd.DataFrame,
    metadata: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    water_change = build_water_change(counterfactual)
    traj = breakpoints.merge(metadata, on="grid_id", how="left").merge(water_change, on="grid_id", how="left")
    traj["onset_date"] = pd.to_datetime(traj["onset_date"], errors="coerce")
    traj["trajectory_type"] = traj.apply(
        lambda row: classify_trajectory(row["onset_date"], row["post_trend_cm_per_year"], threshold),
        axis=1,
    )
    return traj.sort_values(["trajectory_type", "region_tag", "grid_id"])


def build_trajectory_waterlink_summary(trajectory_df: pd.DataFrame, case_df: pd.DataFrame | None) -> pd.DataFrame:
    summary = (
        trajectory_df.groupby(["trajectory_type", "region_tag"], as_index=False)
        .agg(
            grid_count=("grid_id", "nunique"),
            mean_pre_trend_cm_per_year=("pre_trend_cm_per_year", "mean"),
            mean_post_trend_cm_per_year=("post_trend_cm_per_year", "mean"),
            mean_water_change=("water_change_2019_2024_minus_2015_2018", "mean"),
        )
        .assign(summary_level="trajectory_region")
    )
    overall = (
        trajectory_df.groupby("trajectory_type", as_index=False)
        .agg(
            grid_count=("grid_id", "nunique"),
            mean_pre_trend_cm_per_year=("pre_trend_cm_per_year", "mean"),
            mean_post_trend_cm_per_year=("post_trend_cm_per_year", "mean"),
            mean_water_change=("water_change_2019_2024_minus_2015_2018", "mean"),
        )
        .assign(summary_level="trajectory_all", region_tag="all")
    )
    frames = [overall, summary]
    if case_df is not None and not case_df.empty:
        case_summary = (
            case_df.groupby(["buffer_km", "case_grid_id"], as_index=False)
            .agg(
                mean_buffer_water_frac=("water_frac", "mean"),
                mean_buffer_water_persistence=("water_persistence", "mean"),
                mean_case_grid_residual_gwa=("case_grid_residual_gwa", "mean"),
            )
            .merge(
                trajectory_df[["grid_id", "trajectory_type", "region_tag"]],
                left_on="case_grid_id",
                right_on="grid_id",
                how="left",
            )
            .rename(columns={"trajectory_type": "case_grid_trajectory_type"})
            .assign(summary_level="case_buffer")
        )
        frames.append(case_summary)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_wells(args: argparse.Namespace, metadata: pd.DataFrame) -> pd.DataFrame | None:
    if not args.well_data:
        return None
    raw_wells = pd.read_csv(args.well_data)
    if args.well_query:
        try:
            raw_wells = raw_wells.query(args.well_query).copy()
        except Exception as exc:  # pragma: no cover - defensive CLI error
            raise ValueError(f"Failed to apply --well-query `{args.well_query}`: {exc}") from exc
    wells = standardize_well_data(
        raw_wells,
        dataset_label=f"well data `{args.well_data}`",
        value_kind=args.well_value_kind,
        well_id_col=args.well_id_col,
        date_col=args.well_date_col,
        value_col=args.well_value_col,
        grid_col=args.well_grid_col,
        lon_col=args.well_lon_col,
        lat_col=args.well_lat_col,
        coord_to_grid_id=coord_to_grid_id,
    )
    wells = (
        wells.merge(metadata[["grid_id", "region_tag"]], on="grid_id", how="left")
        .sort_values(["grid_id", "well_id", "date"])
    )
    return wells


def slope_per_year(df: pd.DataFrame, value_col: str) -> float:
    clean = df.dropna(subset=[value_col]).sort_values("date")
    if len(clean) < 3:
        return float("nan")
    x = np.arange(len(clean), dtype=float)
    res = stats.linregress(x, clean[value_col].to_numpy(dtype=float))
    return float(res.slope * 12.0)


def theilsen_slope_per_year(df: pd.DataFrame, value_col: str) -> float:
    clean = df.dropna(subset=[value_col]).sort_values("date")
    if len(clean) < 3:
        return float("nan")
    x = np.arange(len(clean), dtype=float)
    slope, _, _, _ = stats.theilslopes(clean[value_col].to_numpy(dtype=float), x, 0.95)
    return float(slope * 12.0)


def kendall_tau_with_time(df: pd.DataFrame, value_col: str) -> float:
    clean = df.dropna(subset=[value_col]).sort_values("date")
    if len(clean) < 3:
        return float("nan")
    x = np.arange(len(clean), dtype=float)
    res = stats.kendalltau(x, clean[value_col].to_numpy(dtype=float), nan_policy="omit")
    return float(res.statistic) if np.isfinite(res.statistic) else float("nan")


def best_break_date(df: pd.DataFrame, value_col: str) -> str | None:
    clean = df.dropna(subset=[value_col]).sort_values("date").reset_index(drop=True)
    if len(clean) < 48:
        return None
    x = np.arange(len(clean), dtype=float)
    y = clean[value_col].to_numpy(dtype=float)
    candidate_idx = [
        idx
        for idx, date in enumerate(clean["date"])
        if pd.Timestamp("2015-01-01") <= date <= pd.Timestamp("2023-09-01")
    ]
    if len(candidate_idx) < 24:
        return None
    best_idx = None
    best_rss = None
    for idx in candidate_idx:
        if idx < 18 or (len(clean) - idx) < 18:
            continue
        left = slice(0, idx)
        right = slice(idx, len(clean))
        left_fit = stats.linregress(x[left], y[left])
        right_fit = stats.linregress(x[right], y[right])
        left_resid = y[left] - (left_fit.intercept + left_fit.slope * x[left])
        right_resid = y[right] - (right_fit.intercept + right_fit.slope * x[right])
        rss = float(np.sum(left_resid**2) + np.sum(right_resid**2))
        if best_rss is None or rss < best_rss:
            best_rss = rss
            best_idx = idx
    if best_idx is None:
        return None
    return pd.Timestamp(clean["date"].iloc[best_idx]).strftime("%Y-%m-%d")


def build_well_validation_summary(wells: pd.DataFrame | None, counterfactual: pd.DataFrame, trajectory_df: pd.DataFrame) -> pd.DataFrame:
    if wells is None or wells.empty:
        return pd.DataFrame([{"validation_level": "status", "status": "skipped", "reason": "no_well_data"}])

    counterfactual["date"] = pd.to_datetime(counterfactual["date"])
    remote_rows = [counterfactual.groupby("date", as_index=False)["gwa_cm"].mean().assign(region_tag="all")]
    for region in ["north", "south"]:
        remote_rows.append(counterfactual.loc[counterfactual["region_tag"] == region].groupby("date", as_index=False)["gwa_cm"].mean().assign(region_tag=region))
    remote = pd.concat(remote_rows, ignore_index=True)

    well_grid = (
        wells.groupby(["grid_id", "date", "region_tag"], as_index=False)["well_anomaly"]
        .mean()
        .sort_values(["grid_id", "date"])
    )
    well_rows = [well_grid.groupby("date", as_index=False)["well_anomaly"].mean().assign(region_tag="all")]
    for region in ["north", "south"]:
        well_rows.append(
            well_grid.loc[well_grid["region_tag"] == region]
            .groupby("date", as_index=False)["well_anomaly"]
            .mean()
            .assign(region_tag=region)
        )
    well_region = pd.concat(well_rows, ignore_index=True)

    region_summary = []
    for region in ["all", "north", "south"]:
        merged = remote.loc[remote["region_tag"] == region].merge(well_region.loc[well_region["region_tag"] == region], on=["date", "region_tag"], how="inner")
        corr = np.corrcoef(merged["gwa_cm"], merged["well_anomaly"])[0, 1] if len(merged) >= 3 else np.nan
        post = merged.loc[merged["date"] >= EVENT_START].copy()
        corr_post = np.corrcoef(post["gwa_cm"], post["well_anomaly"])[0, 1] if len(post) >= 3 else np.nan
        region_summary.append(
            {
                "validation_level": "region_timeseries",
                "region_tag": region,
                "overlap_months": len(merged),
                "overlap_start_date": merged["date"].min().strftime("%Y-%m-%d") if not merged.empty else None,
                "overlap_end_date": merged["date"].max().strftime("%Y-%m-%d") if not merged.empty else None,
                "pearson_r": float(corr) if np.isfinite(corr) else np.nan,
                "pearson_r_post_2019": float(corr_post) if np.isfinite(corr_post) else np.nan,
                "remote_pre_2019_trend_cm_per_year": slope_per_year(merged.loc[merged["date"] < EVENT_START], "gwa_cm"),
                "remote_post_2019_trend_cm_per_year": slope_per_year(merged.loc[merged["date"] >= EVENT_START], "gwa_cm"),
                "remote_post_2019_theilsen_cm_per_year": theilsen_slope_per_year(merged.loc[merged["date"] >= EVENT_START], "gwa_cm"),
                "remote_post_2019_kendall_tau": kendall_tau_with_time(merged.loc[merged["date"] >= EVENT_START], "gwa_cm"),
                "well_pre_2019_trend_units_per_year": slope_per_year(merged.loc[merged["date"] < EVENT_START], "well_anomaly"),
                "well_post_2019_trend_units_per_year": slope_per_year(merged.loc[merged["date"] >= EVENT_START], "well_anomaly"),
                "well_post_2019_theilsen_units_per_year": theilsen_slope_per_year(merged.loc[merged["date"] >= EVENT_START], "well_anomaly"),
                "well_post_2019_kendall_tau": kendall_tau_with_time(merged.loc[merged["date"] >= EVENT_START], "well_anomaly"),
                "remote_best_break_date": best_break_date(merged, "gwa_cm"),
                "well_best_break_date": best_break_date(merged, "well_anomaly"),
            }
        )

    coverage = (
        wells.merge(trajectory_df[["grid_id", "trajectory_type"]], on="grid_id", how="left")
        .dropna(subset=["trajectory_type"])
        .groupby("trajectory_type", as_index=False)
        .agg(
            unique_wells=("well_id", "nunique"),
            covered_months=("date", "nunique"),
            mean_well_anomaly=("well_anomaly", "mean"),
            coverage_start_date=("date", "min"),
            coverage_end_date=("date", "max"),
        )
        .assign(validation_level="trajectory_coverage")
    )
    if not coverage.empty:
        coverage["coverage_start_date"] = pd.to_datetime(coverage["coverage_start_date"]).dt.strftime("%Y-%m-%d")
        coverage["coverage_end_date"] = pd.to_datetime(coverage["coverage_end_date"]).dt.strftime("%Y-%m-%d")
    return pd.concat([pd.DataFrame(region_summary), coverage], ignore_index=True, sort=False)


def write_figure_spec(output_path: Path, trajectory_df: pd.DataFrame, well_summary: pd.DataFrame) -> None:
    counts = trajectory_df["trajectory_type"].value_counts().to_dict()
    has_well_rows = not well_summary.empty and "status" not in well_summary.columns
    lines = [
        "# Paper Figure Spec",
        "",
        "1. Study area and 1-degree grid distribution map.",
        "2. Regional average GWA time series and key policy milestones.",
        "3. Spatial distribution of recovery trajectory types.",
        "4. Mean recovery trajectories by trajectory type.",
        "5. Comparison of surface-water changes by trajectory type.",
        "6. Hutuo River case study: buffer zone and representative grid trajectory types.",
        "7. Buffer water persistence and representative grid residual/GWA series.",
        "8. In-situ monitoring wells versus remote sensing regional average comparison." if has_well_rows else "8. Well comparison figure omitted; no active well validation results available.",
        "",
        f"- Grid count: {trajectory_df['grid_id'].nunique()}",
        f"- Trajectory counts: {json.dumps(counts, ensure_ascii=True)}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(Path(args.panel))
    counterfactual = pd.read_csv(args.counterfactual)
    breakpoints = pd.read_csv(args.breakpoints)
    case_df = pd.read_csv(args.case) if args.case else None

    metadata = build_grid_metadata(panel)
    if "region_tag" not in counterfactual.columns:
        counterfactual = counterfactual.merge(
            metadata[["grid_id", "region_tag"]],
            on="grid_id",
            how="left",
        )
    trajectory_df = build_trajectory_types(breakpoints, counterfactual, metadata, args.trajectory_post_threshold)
    trajectory_df["onset_date"] = pd.to_datetime(trajectory_df["onset_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    waterlink_df = build_trajectory_waterlink_summary(trajectory_df, case_df)
    wells = load_wells(args, metadata)
    well_summary = build_well_validation_summary(wells, counterfactual, trajectory_df)

    trajectory_df.to_csv(output_dir / "grid_trajectory_types.csv", index=False)
    waterlink_df.to_csv(output_dir / "trajectory_waterlink_summary.csv", index=False)
    well_summary.to_csv(output_dir / "well_validation_summary.csv", index=False)

    threshold_summary = []
    for threshold in [0.5, 1.0, 1.5]:
        threshold_df = build_trajectory_types(breakpoints, counterfactual, metadata, threshold)
        counts = threshold_df["trajectory_type"].value_counts().to_dict()
        threshold_summary.append(
            {
                "post_threshold_cm_per_year": threshold,
                "legacy_slowdown": counts.get("legacy_slowdown", 0),
                "early_rebound": counts.get("early_rebound", 0),
                "main_rebound": counts.get("main_rebound", 0),
                "stalled_or_declining": counts.get("stalled_or_declining", 0),
            }
        )
    pd.DataFrame(threshold_summary).to_csv(output_dir / "trajectory_threshold_sensitivity.csv", index=False)
    write_figure_spec(output_dir / "paper_figure_spec.md", trajectory_df, well_summary)


if __name__ == "__main__":
    main()
