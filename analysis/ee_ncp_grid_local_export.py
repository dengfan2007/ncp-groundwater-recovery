from __future__ import annotations

import argparse
from pathlib import Path

import ee
import pandas as pd

from ee_local import initialize_ee
from ee_ncp_common import (
    build_ncp_context,
    build_rect_grid,
    download_feature_collection_csv,
    download_feature_collection_dataframe,
    mask_landsat_l2,
    masked_empty,
    merge_feature_collections,
    month_starts,
    print_json,
    safe_mean,
    safe_sum,
    summarize_feature_collection,
)


CONFIG = {
    "start_date": "2003-01-01",
    "end_date_exclusive": "2024-10-01",
    "baseline_start": "2003-01-01",
    "baseline_end_exclusive": "2015-01-01",
    "stats_scale_meters": 10000,
    "max_elevation_meters": 200,
    "min_coverage_ratio": 0.60,
    "grid_step_degree": 1.0,
    "bbox": [112.0, 34.5, 120.0, 40.6],
    "province_names": ["Beijing Shi", "Tianjin Shi", "Hebei Sheng", "Henan Sheng", "Shandong Sheng"],
}

PANEL_SELECTORS = [
    "grid_id",
    "grid_lon",
    "grid_lat",
    "region_tag",
    "coverage_ratio",
    "date",
    "year",
    "month",
    "gwa_cm",
    "twsa_cm",
    "smsa_cm",
    "swea_cm",
    "cwa_cm",
    "precip_mm",
    "et_mm",
    "water_frac",
]
GRID_SELECTORS = ["grid_id", "grid_lon", "grid_lat", "coverage_ratio", "region_tag"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the NCP grid monthly panel directly to local CSV files.")
    parser.add_argument("--project", help="Earth Engine Cloud project.")
    parser.add_argument("--auth-mode", help="Optional Earth Engine auth mode, e.g. localhost.")
    parser.add_argument("--force-auth", action="store_true", help="Force re-authentication.")
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Local directory for downloaded CSV files.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print collection metadata only; do not download files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload all monthly chunks even if local chunk files already exist.",
    )
    return parser.parse_args()


def monthly_water_fraction(month_start: ee.Date, bbox_geom: ee.Geometry) -> ee.Image:
    jrc_monthly = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
    year = ee.Number.parse(month_start.format("Y"))
    month = ee.Number.parse(month_start.format("M"))
    jrc_filtered = jrc_monthly.filter(ee.Filter.eq("year", year)).filter(ee.Filter.eq("month", month))

    def _landsat_monthly() -> ee.Image:
        month_end = month_start.advance(1, "month")
        l8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(month_start, month_end)
            .filterBounds(bbox_geom)
            .map(mask_landsat_l2)
        )
        l9 = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterDate(month_start, month_end)
            .filterBounds(bbox_geom)
            .map(mask_landsat_l2)
        )
        monthly = l8.merge(l9)
        return ee.Image(
            ee.Algorithms.If(
                monthly.size().gt(0),
                monthly.map(_landsat_water_mask).mean().rename("water_frac"),
                masked_empty("water_frac"),
            )
        )

    return ee.Image(
        ee.Algorithms.If(
            month_start.millis().lt(ee.Date("2022-01-01").millis()),
            ee.Image(
                ee.Algorithms.If(
                    jrc_filtered.size().gt(0),
                    ee.Image(jrc_filtered.first()).select("water").eq(2).rename("water_frac"),
                    masked_empty("water_frac"),
                )
            ),
            _landsat_monthly(),
        )
    )


def _landsat_water_mask(image: ee.Image) -> ee.Image:
    green = image.select("SR_B3")
    red = image.select("SR_B4")
    nir = image.select("SR_B5")
    swir1 = image.select("SR_B6")
    mndwi = green.subtract(swir1).divide(green.add(swir1))
    ndvi = nir.subtract(red).divide(nir.add(red))
    return mndwi.gt(0).And(mndwi.gt(ndvi)).rename("water_frac")


def build_grid_and_monthly_final() -> tuple[ee.FeatureCollection, ee.ImageCollection]:
    context = build_ncp_context(
        bbox=CONFIG["bbox"],
        province_names=CONFIG["province_names"],
        max_elevation_meters=CONFIG["max_elevation_meters"],
    )
    bbox_geom = context["bbox"]
    lowland_mask = context["lowland_mask"]

    grid = build_rect_grid(
        bbox=CONFIG["bbox"],
        step_degree=CONFIG["grid_step_degree"],
        mask=lowland_mask,
        min_coverage_ratio=CONFIG["min_coverage_ratio"],
        include_centroid_fields=True,
        include_region_tag=True,
    )

    grace = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/MASCON_CRI")
    gldas = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H")
    imerg = ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07")
    mod16 = ee.ImageCollection("MODIS/061/MOD16A2GF")

    def _build_raw_month(month_start: ee.Date) -> ee.Image:
        month_start = ee.Date(month_start)
        month_end = month_start.advance(1, "month")
        days_in_month = month_end.difference(month_start, "day")
        month_number = ee.Number.parse(month_start.format("M"))
        year_number = ee.Number.parse(month_start.format("Y"))

        tws = safe_mean(grace, month_start, month_end, "lwe_thickness", "tws_cm")
        gldas_monthly = gldas.filterDate(month_start, month_end).mean()
        sms = (
            gldas_monthly.select(
                [
                    "SoilMoi0_10cm_inst",
                    "SoilMoi10_40cm_inst",
                    "SoilMoi40_100cm_inst",
                    "SoilMoi100_200cm_inst",
                ]
            )
            .reduce(ee.Reducer.sum())
            .divide(10)
            .rename("sms_cm")
        )
        swe = gldas_monthly.select("SWE_inst").divide(10).rename("swe_cm")
        cwa = gldas_monthly.select("CanopInt_inst").divide(10).rename("cwa_cm")
        precip_rate = safe_mean(imerg, month_start, month_end, "precipitation", "precip_rate_mmhr")
        precip = precip_rate.multiply(24).multiply(days_in_month).rename("precip_mm")
        et = safe_sum(mod16, month_start, month_end, "ET", "et_mm", 0.1)
        water_frac = monthly_water_fraction(month_start, bbox_geom)

        return (
            ee.Image.cat([tws, sms, swe, cwa, precip, et, water_frac])
            .updateMask(lowland_mask)
            .clip(bbox_geom)
            .set("system:time_start", month_start.millis())
            .set("date", month_start.format("YYYY-MM-dd"))
            .set("month", month_number)
            .set("year", year_number)
        )

    months = month_starts(CONFIG["start_date"], CONFIG["end_date_exclusive"])
    monthly_raw = ee.ImageCollection.fromImages(months.map(_build_raw_month))
    baseline = monthly_raw.filterDate(CONFIG["baseline_start"], CONFIG["baseline_end_exclusive"])
    baseline_means = ee.Image.cat(
        [
            baseline.select("tws_cm").mean().rename("tws_base"),
            baseline.select("sms_cm").mean().rename("sms_base"),
            baseline.select("swe_cm").mean().rename("swe_base"),
            baseline.select("cwa_cm").mean().rename("cwa_base"),
        ]
    )

    def _to_final(image: ee.Image) -> ee.Image:
        twsa = image.select("tws_cm").subtract(baseline_means.select("tws_base")).rename("twsa_cm")
        smsa = image.select("sms_cm").subtract(baseline_means.select("sms_base")).rename("smsa_cm")
        swea = image.select("swe_cm").subtract(baseline_means.select("swe_base")).rename("swea_cm")
        cwa = image.select("cwa_cm").subtract(baseline_means.select("cwa_base")).rename("cwa_cm")
        gwa = twsa.subtract(smsa).subtract(swea).subtract(cwa).rename("gwa_cm")
        return (
            ee.Image.cat(
                [
                    gwa,
                    twsa,
                    smsa,
                    swea,
                    cwa,
                    image.select("precip_mm"),
                    image.select("et_mm"),
                    image.select("water_frac"),
                ]
            )
            .set("system:time_start", image.get("system:time_start"))
            .set("date", image.get("date"))
            .set("month", image.get("month"))
            .set("year", image.get("year"))
        )

    monthly_final = monthly_raw.map(_to_final)

    return grid, monthly_final


def build_monthly_grid_features(
    monthly_final: ee.ImageCollection,
    grid: ee.FeatureCollection,
    month_start_text: str,
) -> ee.FeatureCollection:
    month_start = ee.Date(month_start_text)
    date_text = month_start.format("YYYY-MM-dd")
    monthly_match = monthly_final.filter(ee.Filter.eq("date", date_text))
    empty_image = ee.Image.cat(
        [
            masked_empty("gwa_cm"),
            masked_empty("twsa_cm"),
            masked_empty("smsa_cm"),
            masked_empty("swea_cm"),
            masked_empty("cwa_cm"),
            masked_empty("precip_mm"),
            masked_empty("et_mm"),
            masked_empty("water_frac"),
        ]
    )
    image = ee.Image(
        ee.Algorithms.If(
            monthly_match.size().gt(0),
            monthly_match.first(),
            empty_image.set("date", date_text),
        )
    )
    return image.reduceRegions(
        collection=grid,
        reducer=ee.Reducer.mean(),
        scale=CONFIG["stats_scale_meters"],
        tileScale=4,
    ).map(
        lambda feature: feature.set(
            {
                "date": month_start.format("YYYY-MM-dd"),
                "year": ee.Number.parse(month_start.format("Y")),
                "month": ee.Number.parse(month_start.format("M")),
                "system_time_start": month_start.millis(),
            }
        )
    )


def main() -> None:
    args = parse_args()
    project = initialize_ee(project=args.project, auth_mode=args.auth_mode, force_auth=args.force_auth)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid, monthly_final = build_grid_and_monthly_final()

    if args.describe:
        sample_month = CONFIG["start_date"]
        monthly_panel_sample = build_monthly_grid_features(monthly_final, grid, sample_month)
        print_json(
            {
                "project": project,
                "grid": summarize_feature_collection(grid, GRID_SELECTORS),
                "monthly_panel_sample_month": sample_month,
                "monthly_panel": summarize_feature_collection(monthly_panel_sample, PANEL_SELECTORS[:8]),
            }
        )
        return

    grid_result = download_feature_collection_csv(
        collection=grid,
        selectors=GRID_SELECTORS,
        output_path=output_dir / "ncp_grid_definition.csv",
        filename="ncp_grid_definition",
    )
    month_index = pd.date_range(
        CONFIG["start_date"],
        CONFIG["end_date_exclusive"],
        freq="MS",
        inclusive="left",
    )
    chunk_dir = output_dir / "_tmp_grid_monthly_panel"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    downloaded_months = 0
    reused_months = 0
    first_url = None
    for month_start in month_index:
        month_tag = month_start.strftime("%Y_%m")
        chunk_path = chunk_dir / f"grid_monthly_panel_{month_tag}.csv"
        if chunk_path.exists() and not args.overwrite:
            reused_months += 1
            continue
        month_text = month_start.strftime("%Y-%m-%d")
        month_fc = build_monthly_grid_features(monthly_final, grid, month_text)
        month_df, url = download_feature_collection_dataframe(
            collection=month_fc,
            selectors=PANEL_SELECTORS,
            filename=f"grid_monthly_panel_{month_tag}",
        )
        if first_url is None:
            first_url = url
        month_df.to_csv(chunk_path, index=False)
        downloaded_months += 1

    panel_path = output_dir / "grid_monthly_panel.csv"
    ordered_chunk_paths = [chunk_dir / f"grid_monthly_panel_{month.strftime('%Y_%m')}.csv" for month in month_index]
    frames = [pd.read_csv(path) for path in ordered_chunk_paths if path.exists()]
    pd.concat(frames, ignore_index=True).to_csv(panel_path, index=False)
    panel_result = {
        "output_path": str(panel_path),
        "month_count": int(len(month_index)),
        "row_count": int(sum(len(frame) for frame in frames)),
        "sample_download_url": first_url,
        "chunk_dir": str(chunk_dir),
        "downloaded_months": downloaded_months,
        "reused_months": reused_months,
    }
    print_json({"project": project, "grid_download": grid_result, "panel_download": panel_result})


if __name__ == "__main__":
    main()
