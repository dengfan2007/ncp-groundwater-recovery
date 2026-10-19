from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ncp_well_utils import standardize_well_data


GRID_STEP = 1.0
HUTUO_LINE = [
    (113.55, 38.53),
    (113.84, 38.49),
    (114.15, 38.44),
    (114.48, 38.39),
    (114.82, 38.33),
    (115.12, 38.27),
    (115.30, 38.22),
]


DATA_SOURCE_ROWS = {
    "zh": [
        ["GRACE Mascon RL06.3Mv04", "TWSA", "2002-03-31 to 2024-09-30", "区域陆地总水储量异常"],
        ["GLDAS-2.1 NOAH 3-hourly", "SMSA/SWEA/CWA", "2000-01-01 to present", "土壤水、雪水和冠层水分量"],
        ["GPM IMERG Monthly V07", "precip_mm", "1998-01-01 to present", "降水背景"],
        ["MOD16A2GF V6.1", "et_mm", "2000-01-01 to present", "蒸散发背景"],
        ["JRC Monthly Water History", "water_frac", "1984-2021", "月尺度地表水历史"],
        ["Landsat Collection 2 L2", "water_frac", "2022-2024", "月尺度地表水补续"],
    ],
    "en": [
        ["GRACE Mascon RL06.3Mv04", "TWSA", "2002-03-31 to 2024-09-30", "Regional terrestrial water-storage anomaly"],
        ["GLDAS-2.1 NOAH 3-hourly", "SMSA/SWEA/CWA", "2000-01-01 to present", "Soil-moisture, snow-water, and canopy-water components"],
        ["GPM IMERG Monthly V07", "precip_mm", "1998-01-01 to present", "Precipitation background"],
        ["MOD16A2GF V6.1", "et_mm", "2000-01-01 to present", "Evapotranspiration background"],
        ["JRC Monthly Water History", "water_frac", "1984-2021", "Monthly surface-water history"],
        ["Landsat Collection 2 L2", "water_frac", "2022-2024", "Monthly surface-water continuation"],
    ],
}


TABLE_COLUMNS = {
    "zh": ["data_source", "variable", "time_range", "role_in_study"],
    "en": ["data_source", "variable", "time_range", "role_in_study"],
}


CHECKLISTS = {
    "zh": """# 绘图脚本清单

## 图件

- `fig01_grid_map.csv`
- `fig02_region_gwa_timeseries.csv`
- `fig03_trajectory_map.csv`
- `fig04_trajectory_mean_curves.csv`
- `fig05_recovery_metrics.csv`
- `fig06_surface_water_by_trajectory.csv`
- `fig06_case_grid_map.csv`
- `fig07_case_buffer_timeseries.csv`
- `fig08_validation_timeseries.csv`

## 表格

- `tab01_data_sources.csv`
- `tab02_trajectory_rules_counts.csv`
- `tab03_trajectory_stats.csv`
- `tab04_validation_stats.csv`
""",
    "en": """# Plotting Checklist

## Figures

- `fig01_grid_map.csv`
- `fig02_region_gwa_timeseries.csv`
- `fig03_trajectory_map.csv`
- `fig04_trajectory_mean_curves.csv`
- `fig05_recovery_metrics.csv`
- `fig06_surface_water_by_trajectory.csv`
- `fig06_case_grid_map.csv`
- `fig07_case_buffer_timeseries.csv`
- `fig08_validation_timeseries.csv`

## Tables

- `tab01_data_sources.csv`
- `tab02_trajectory_rules_counts.csv`
- `tab03_trajectory_stats.csv`
- `tab04_validation_stats.csv`
- `context_hrb_supply_use.csv`
""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare figure/table-ready CSV files for the NCP manuscript.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing downloaded raw CSV data.")
    parser.add_argument("--main-dir", required=True, help="Directory containing main analysis outputs.")
    parser.add_argument("--postprocess-dir", required=True, help="Directory containing postprocess outputs.")
    parser.add_argument("--revision-dir", required=True, help="Directory containing revision-support outputs.")
    parser.add_argument("--external-dir", required=True, help="Directory containing external validation data.")
    parser.add_argument("--output-dir", required=True, help="Output directory for figure/table datasets.")
    parser.add_argument("--well-data", help="Optional override path for the public well-proxy CSV.")
    parser.add_argument("--well-value-col", default="gw_level")
    parser.add_argument("--well-value-kind", choices=["anomaly", "level"], default="level")
    parser.add_argument("--well-query", help="Optional pandas query string applied to the raw well table.")
    parser.add_argument("--language", choices=["zh", "en"], default="zh", help="Output language.")
    return parser.parse_args()


def parse_grid_id(grid_id: str) -> tuple[float, float]:
    _, lon_text, lat_text = grid_id.split("_")
    return float(lon_text), float(lat_text)


def grid_bounds(grid_id: str) -> tuple[float, float, float, float]:
    west, south = parse_grid_id(grid_id)
    east = west + GRID_STEP
    north = south + GRID_STEP
    return west, east, south, north


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_fig01_grid_map(grid_df: pd.DataFrame, case_df: pd.DataFrame | None) -> pd.DataFrame:
    case_grids = set(case_df["case_grid_id"].dropna().unique()) if case_df is not None else set()
    rows = []
    for _, row in grid_df.iterrows():
        west, east, south, north = grid_bounds(row["grid_id"])
        rows.append(
            {
                "grid_id": row["grid_id"],
                "west": west,
                "east": east,
                "south": south,
                "north": north,
                "grid_lon": row["grid_lon"],
                "grid_lat": row["grid_lat"],
                "region_tag": row["region_tag"],
                "coverage_ratio": row["coverage_ratio"],
                "is_case_grid": int(row["grid_id"] in case_grids),
            }
        )
    return pd.DataFrame(rows)


def build_fig01_case_geometry() -> tuple[pd.DataFrame, pd.DataFrame]:
    line_df = pd.DataFrame(
        [{"order_id": idx + 1, "lon": lon, "lat": lat} for idx, (lon, lat) in enumerate(HUTUO_LINE)]
    )
    buffer_df = pd.DataFrame(
        [
            {"buffer_km": "0_5", "inner_m": 0, "outer_m": 5000},
            {"buffer_km": "5_10", "inner_m": 5000, "outer_m": 10000},
            {"buffer_km": "10_20", "inner_m": 10000, "outer_m": 20000},
            {"buffer_km": "20_30", "inner_m": 20000, "outer_m": 30000},
        ]
    )
    return line_df, buffer_df


def build_fig02_region_timeseries(panel_df: pd.DataFrame, grid_df: pd.DataFrame) -> pd.DataFrame:
    panel = panel_df.copy()
    if "region_tag" not in panel.columns:
        panel = panel.merge(grid_df[["grid_id", "region_tag"]], on="grid_id", how="left")
    frames = [
        panel.groupby("date", as_index=False)["gwa_cm"].mean().assign(region_tag="all"),
        panel.loc[panel["region_tag"] == "north"].groupby("date", as_index=False)["gwa_cm"].mean().assign(region_tag="north"),
        panel.loc[panel["region_tag"] == "south"].groupby("date", as_index=False)["gwa_cm"].mean().assign(region_tag="south"),
    ]
    return pd.concat(frames, ignore_index=True).sort_values(["region_tag", "date"])


def build_fig03_trajectory_map(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in trajectory_df.iterrows():
        west, east, south, north = grid_bounds(row["grid_id"])
        rows.append(
            {
                "grid_id": row["grid_id"],
                "west": west,
                "east": east,
                "south": south,
                "north": north,
                "grid_lon": row["grid_lon"],
                "grid_lat": row["grid_lat"],
                "region_tag": row["region_tag"],
                "trajectory_type": row["trajectory_type"],
                "onset_date": row["onset_date"],
                "post_trend_cm_per_year": row["post_trend_cm_per_year"],
            }
        )
    return pd.DataFrame(rows)


def build_fig04_mean_curves(counterfactual_df: pd.DataFrame, trajectory_df: pd.DataFrame) -> pd.DataFrame:
    merged = counterfactual_df.merge(
        trajectory_df[["grid_id", "trajectory_type", "region_tag"]],
        on="grid_id",
        how="left",
    )
    return (
        merged.groupby(["date", "trajectory_type"], as_index=False)["gwa_cm"]
        .mean()
        .rename(columns={"gwa_cm": "mean_gwa_cm"})
        .sort_values(["trajectory_type", "date"])
    )


def build_fig05_recovery_metrics(trajectory_df: pd.DataFrame, spatial_effect_df: pd.DataFrame) -> pd.DataFrame:
    df = trajectory_df.copy()
    df["onset_date"] = pd.to_datetime(df["onset_date"], errors="coerce")
    df = df.loc[
        df["is_recovery_relevant"].fillna(False)
        & df["onset_date"].notna()
        & (df["onset_date"] >= pd.Timestamp("2018-01-01"))
    ].copy()
    df["onset_month_index"] = df["onset_date"].dt.year * 12 + df["onset_date"].dt.month
    grid_rows = []
    for _, row in df.iterrows():
        grid_rows.append(
            {
                "row_type": "grid",
                "metric": "onset_month_index",
                "region_tag": row["region_tag"],
                "value": row["onset_month_index"],
                "ci_lower": None,
                "ci_upper": None,
            }
        )
        grid_rows.append(
            {
                "row_type": "grid",
                "metric": "post_trend_cm_per_year",
                "region_tag": row["region_tag"],
                "value": row["post_trend_cm_per_year"],
                "ci_lower": None,
                "ci_upper": None,
            }
        )

    effect_rows = []
    for _, row in spatial_effect_df.iterrows():
        effect_rows.append(
            {
                "row_type": "effect",
                "metric": row["metric"],
                "region_tag": "all",
                "value": row["estimate"],
                "ci_lower": row["ci_lower"],
                "ci_upper": row["ci_upper"],
            }
        )
    return pd.DataFrame(grid_rows + effect_rows)


def build_fig06_surface_water(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "grid_id",
        "region_tag",
        "trajectory_type",
        "water_frac_pre_2015_2018",
        "water_frac_post_2019_2024",
        "water_change_2019_2024_minus_2015_2018",
    ]
    return trajectory_df[cols].sort_values(["trajectory_type", "region_tag", "grid_id"])


def build_fig07_case_timeseries(case_df: pd.DataFrame, trajectory_df: pd.DataFrame) -> pd.DataFrame:
    merged = case_df.merge(
        trajectory_df[["grid_id", "trajectory_type"]],
        left_on="case_grid_id",
        right_on="grid_id",
        how="left",
    ).rename(columns={"trajectory_type": "case_grid_trajectory_type"})
    return merged.sort_values(["buffer_km", "date"])


def build_fig08_validation_timeseries(panel_df: pd.DataFrame, grid_df: pd.DataFrame, wells_df: pd.DataFrame) -> pd.DataFrame:
    panel = panel_df.copy()
    if "region_tag" not in panel.columns:
        panel = panel.merge(grid_df[["grid_id", "region_tag"]], on="grid_id", how="left")
    panel["date"] = pd.to_datetime(panel["date"])

    remote_frames = [
        panel.groupby("date", as_index=False)["gwa_cm"].mean().assign(region_tag="all"),
        panel.loc[panel["region_tag"] == "north"].groupby("date", as_index=False)["gwa_cm"].mean().assign(region_tag="north"),
        panel.loc[panel["region_tag"] == "south"].groupby("date", as_index=False)["gwa_cm"].mean().assign(region_tag="south"),
    ]
    remote = pd.concat(remote_frames, ignore_index=True)

    wells = wells_df.merge(grid_df[["grid_id", "region_tag"]], on="grid_id", how="left")
    wells = (
        wells.groupby(["grid_id", "date", "region_tag"], as_index=False)["well_anomaly"]
        .mean()
        .sort_values(["grid_id", "date"])
    )
    well_frames = [
        wells.groupby("date", as_index=False)["well_anomaly"].mean().assign(region_tag="all"),
        wells.loc[wells["region_tag"] == "north"].groupby("date", as_index=False)["well_anomaly"].mean().assign(region_tag="north"),
        wells.loc[wells["region_tag"] == "south"].groupby("date", as_index=False)["well_anomaly"].mean().assign(region_tag="south"),
    ]
    well_region = pd.concat(well_frames, ignore_index=True)

    return remote.merge(well_region, on=["date", "region_tag"], how="inner").sort_values(["region_tag", "date"])


def build_fig08_validation_coverage(grid_df: pd.DataFrame, wells_df: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        wells_df.groupby("grid_id", as_index=False)
        .agg(
            unique_wells=("well_id", "nunique"),
            covered_months=("date", lambda x: pd.to_datetime(x).dt.to_period("M").nunique()),
        )
    )
    grid_map = build_fig01_grid_map(grid_df, case_df=None)
    return (
        grid_map.merge(coverage, on="grid_id", how="left")
        .assign(
            unique_wells=lambda df: df["unique_wells"].fillna(0).astype(int),
            covered_months=lambda df: df["covered_months"].fillna(0).astype(int),
        )
        .sort_values(["region_tag", "grid_id"])
    )


def build_tbl01_data_sources_dynamic(well_path: Path, well_query: str | None, language: str) -> pd.DataFrame:
    rows = DATA_SOURCE_ROWS[language].copy()
    if "long2025" in well_path.name.lower():
        well_source = "GroundwaterNCP_open"
        well_range = "2005-2024"
        if language == "en":
            well_role = "Public shallow monitoring wells"
        else:
            well_role = "公开浅层监测井验证"
    else:
        well_source = "ReconstructedTWS 01_OK.npy"
        well_range = "2005-2018"
        well_role = "Public observation-based enhanced validation" if language == "en" else "公开观测增强验证"
    rows.append([well_source, "well_proxy", well_range, well_role])
    out = pd.DataFrame(rows, columns=TABLE_COLUMNS[language])
    if language == "en":
        out.columns = ["Source", "Variable", "Period", "Role"]
    return out


def build_tbl02_trajectory_rules(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    counts = trajectory_df["trajectory_type"].value_counts().to_dict()
    total = max(1, int(trajectory_df["grid_id"].nunique()))
    rows = [
        ["Legacy slowdown", "<= 2016-12-31", "<= 1.0", counts.get("legacy_slowdown", 0)],
        ["Early rebound", "2018-10-01 to 2019-08-31", "> 1.0", counts.get("early_rebound", 0)],
        ["Main rebound", "2019-09-01 to 2020-12-31", "> 1.0", counts.get("main_rebound", 0)],
        ["Stalled or declining", "Residual class", "all remaining", counts.get("stalled_or_declining", 0)],
    ]
    out = pd.DataFrame(rows, columns=["Trajectory type", "Onset window", "Post-break trend (cm/yr)", "n"])
    out["Share (%)"] = (out["n"] / total * 100.0).round(1)
    return out


def build_tbl03_trajectory_stats(waterlink_df: pd.DataFrame) -> pd.DataFrame:
    df = waterlink_df.loc[waterlink_df["summary_level"] == "trajectory_all"].copy()
    df = df[
        [
            "trajectory_type",
            "grid_count",
            "mean_pre_trend_cm_per_year",
            "mean_post_trend_cm_per_year",
            "mean_water_change",
        ]
    ].rename(
        columns={
            "trajectory_type": "Trajectory type",
            "grid_count": "n",
            "mean_pre_trend_cm_per_year": "Pre-break trend (cm/yr)",
            "mean_post_trend_cm_per_year": "Post-break trend (cm/yr)",
            "mean_water_change": "Mean Delta W",
        }
    )
    mapping = {
        "early_rebound": "Early rebound",
        "main_rebound": "Main rebound",
        "legacy_slowdown": "Legacy slowdown",
        "stalled_or_declining": "Stalled or declining",
    }
    df["Trajectory type"] = df["Trajectory type"].map(mapping)
    return df


def format_estimate_interval(
    estimate: float | int | None,
    lower: float | int | None,
    upper: float | int | None,
    decimals: int = 2,
) -> str:
    if pd.isna(estimate):
        return ""
    value = f"{float(estimate):.{decimals}f}"
    if pd.isna(lower) or pd.isna(upper):
        return value
    return f"{value} [{float(lower):.{decimals}f}, {float(upper):.{decimals}f}]"


def build_tbl04_validation_stats(
    well_validation_df: pd.DataFrame,
    validation_uncertainty_df: pd.DataFrame,
) -> pd.DataFrame:
    region_df = well_validation_df.loc[well_validation_df["validation_level"] == "region_timeseries"].copy()
    merged = region_df.merge(validation_uncertainty_df, on="region_tag", how="left")
    rows = []
    for _, row in merged.iterrows():
        remote_point = pd.to_datetime(row["remote_best_break_date"], errors="coerce")
        well_point = pd.to_datetime(row["well_best_break_date"], errors="coerce")
        rows.append(
            {
                "Region": {"all": "All NCP", "north": "North", "south": "South"}[row["region_tag"]],
                "Overlap": int(row["overlap_months"]),
                "Pearson r": format_estimate_interval(
                    row["pearson_r"],
                    row["pearson_all_ci_lower"],
                    row["pearson_all_ci_upper"],
                ),
                "Pearson r (2019+)": format_estimate_interval(
                    row["pearson_r_post_2019"],
                    row["pearson_post2019_ci_lower"],
                    row["pearson_post2019_ci_upper"],
                ),
                "Remote slope (2019+)": f"{float(row['remote_post_2019_theilsen_cm_per_year']):.2f}",
                "Well slope (2019+)": f"{float(row['well_post_2019_theilsen_units_per_year']):.2f}",
                "Best remote break": remote_point.strftime("%Y-%m") if pd.notna(remote_point) else "",
                "Best well break": well_point.strftime("%Y-%m") if pd.notna(well_point) else "",
            }
        )
    return pd.DataFrame(rows)


def build_context_hrb_supply_use(external_dir: Path) -> pd.DataFrame | None:
    workbook = external_dir / "GroundwaterNCP_open" / "WaterSupplyUse_HRB_2005_2023.xlsx"
    if not workbook.exists():
        return None
    raw = pd.read_excel(workbook)
    data = raw.iloc[1:].copy()
    data.columns = [
        "year",
        "renewable_water_resources_km3",
        "total_supply_km3",
        "surface_water_supply_km3",
        "unconfined_groundwater_supply_km3",
        "confined_groundwater_supply_km3",
        "yellow_river_diversion_km3",
        "yangtze_diversion_km3",
        "reclaimed_water_km3",
        "total_use_km3",
        "agricultural_use_km3",
        "industrial_use_km3",
        "municipal_use_km3",
        "environmental_use_km3",
    ]
    data = data.apply(pd.to_numeric, errors="coerce")
    return data.dropna(subset=["year"]).reset_index(drop=True)


def write_checklist(path: Path, language: str) -> None:
    path.write_text(CHECKLISTS[language], encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    main_dir = Path(args.main_dir)
    post_dir = Path(args.postprocess_dir)
    revision_dir = Path(args.revision_dir)
    ext_dir = Path(args.external_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    grid_df = pd.read_csv(raw_dir / "ncp_grid_definition.csv")
    panel_df = pd.read_csv(raw_dir / "grid_monthly_panel.csv")
    counterfactual_df = pd.read_csv(main_dir / "grid_counterfactual.csv")
    trajectory_df = pd.read_csv(post_dir / "grid_trajectory_types.csv")
    waterlink_df = pd.read_csv(post_dir / "trajectory_waterlink_summary.csv")
    well_validation_df = pd.read_csv(post_dir / "well_validation_summary.csv")
    spatial_effect_df = pd.read_csv(revision_dir / "spatial_effect_summary.csv")
    validation_uncertainty_df = pd.read_csv(revision_dir / "validation_uncertainty_summary.csv")
    case_df = pd.read_csv(main_dir / "hutuo_case_buffers.csv")
    well_path = Path(args.well_data) if args.well_data else ext_dir / "ncp_obsgrid_wellproxy_2005_2018.csv"
    raw_wells = pd.read_csv(well_path)
    if args.well_query:
        raw_wells = raw_wells.query(args.well_query).copy()
    wells_df = standardize_well_data(
        raw_wells,
        dataset_label=f"well proxy `{well_path}`",
        value_kind=args.well_value_kind,
        value_col=args.well_value_col,
    )

    fig01_grid = build_fig01_grid_map(grid_df, case_df)
    fig01_line, fig01_buffers = build_fig01_case_geometry()
    fig02 = build_fig02_region_timeseries(panel_df, grid_df)
    fig03 = build_fig03_trajectory_map(trajectory_df)
    fig04 = build_fig04_mean_curves(counterfactual_df, trajectory_df)
    fig05 = build_fig05_recovery_metrics(trajectory_df, spatial_effect_df)
    fig06 = build_fig06_surface_water(trajectory_df)
    fig06_case_grid = fig03.loc[fig03["grid_id"].isin(case_df["case_grid_id"].dropna().unique())].copy()
    fig07 = build_fig07_case_timeseries(case_df, trajectory_df)
    fig08 = build_fig08_validation_timeseries(panel_df, grid_df, wells_df)
    fig08_cov = build_fig08_validation_coverage(grid_df, wells_df)

    tbl01 = build_tbl01_data_sources_dynamic(well_path, args.well_query, args.language)
    tbl02 = build_tbl02_trajectory_rules(trajectory_df)
    tbl03 = build_tbl03_trajectory_stats(waterlink_df)
    tbl04 = build_tbl04_validation_stats(well_validation_df, validation_uncertainty_df)
    context_hrb = build_context_hrb_supply_use(ext_dir)

    fig01_grid.to_csv(out_dir / "fig01_grid_map.csv", index=False)
    fig01_line.to_csv(out_dir / "fig01_hutuo_line_vertices.csv", index=False)
    fig01_buffers.to_csv(out_dir / "fig01_case_buffers.csv", index=False)
    fig02.to_csv(out_dir / "fig02_region_gwa_timeseries.csv", index=False)
    fig03.to_csv(out_dir / "fig03_trajectory_map.csv", index=False)
    fig04.to_csv(out_dir / "fig04_trajectory_mean_curves.csv", index=False)
    fig05.to_csv(out_dir / "fig05_recovery_metrics.csv", index=False)
    fig06.to_csv(out_dir / "fig06_surface_water_by_trajectory.csv", index=False)
    fig06_case_grid.to_csv(out_dir / "fig06_case_grid_map.csv", index=False)
    fig07.to_csv(out_dir / "fig07_case_buffer_timeseries.csv", index=False)
    fig08.to_csv(out_dir / "fig08_validation_timeseries.csv", index=False)
    fig08_cov.to_csv(out_dir / "fig08_validation_coverage.csv", index=False)

    tbl01.to_csv(out_dir / "tab01_data_sources.csv", index=False)
    tbl02.to_csv(out_dir / "tab02_trajectory_rules_counts.csv", index=False)
    tbl03.to_csv(out_dir / "tab03_trajectory_stats.csv", index=False)
    tbl04.to_csv(out_dir / "tab04_validation_stats.csv", index=False)
    if context_hrb is not None:
        context_hrb.to_csv(out_dir / "context_hrb_supply_use.csv", index=False)

    write_checklist(out_dir / "plotting_script_checklist.md", args.language)


if __name__ == "__main__":
    main()
