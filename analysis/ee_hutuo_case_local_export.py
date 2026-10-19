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
    summarize_feature_collection,
)


CONFIG = {
    "start_date": "2017-01-01",
    "end_date_exclusive": "2024-10-01",
    "stats_scale_meters": 300,
    "grid_step_degree": 1.0,
    "min_coverage_ratio": 0.60,
    "bbox": [112.0, 34.5, 120.0, 40.6],
    "case_aoi": [113.25, 37.65, 115.45, 38.75],
    "province_names": ["Beijing Shi", "Tianjin Shi", "Hebei Sheng", "Henan Sheng", "Shandong Sheng"],
    "max_elevation_meters": 200,
}

CASE_SELECTORS = [
    "buffer_km",
    "inner_m",
    "outer_m",
    "case_grid_id",
    "water_persistence",
    "date",
    "year",
    "month",
    "water_frac",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Hutuo replenishment case table directly to a local CSV file.")
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


def build_case_context() -> tuple[ee.FeatureCollection, ee.FeatureCollection, ee.ImageCollection]:
    context = build_ncp_context(
        bbox=CONFIG["bbox"],
        province_names=CONFIG["province_names"],
        max_elevation_meters=CONFIG["max_elevation_meters"],
    )
    lowland_mask = context["lowland_mask"]
    grid = build_rect_grid(
        bbox=CONFIG["bbox"],
        step_degree=CONFIG["grid_step_degree"],
        mask=lowland_mask,
        min_coverage_ratio=CONFIG["min_coverage_ratio"],
        include_centroid_fields=False,
        include_region_tag=False,
    )
    case_aoi = ee.Geometry.Rectangle(CONFIG["case_aoi"], proj="EPSG:4326", geodesic=False)

    hutuo_line = ee.Geometry.LineString(
        [
            [113.55, 38.53],
            [113.84, 38.49],
            [114.15, 38.44],
            [114.48, 38.39],
            [114.82, 38.33],
            [115.12, 38.27],
            [115.30, 38.22],
        ],
        proj="EPSG:4326",
        geodesic=False,
    )

    buffer_specs = [
        {"label": "0_5", "inner": 0, "outer": 5000},
        {"label": "5_10", "inner": 5000, "outer": 10000},
        {"label": "10_20", "inner": 10000, "outer": 20000},
        {"label": "20_30", "inner": 20000, "outer": 30000},
    ]

    def _build_buffer(spec: dict[str, int | str]) -> ee.Feature:
        outer = hutuo_line.buffer(spec["outer"], 50)
        ring = ee.Geometry(
            ee.Algorithms.If(
                spec["inner"] == 0,
                outer,
                outer.difference(hutuo_line.buffer(spec["inner"], 50), 50),
            )
        )

        def _set_overlap(feature: ee.Feature) -> ee.Feature:
            overlap = feature.geometry().intersection(ring, 1).area(1)
            return feature.set({"ring_overlap": overlap})

        dominant_grid = ee.Feature(grid.map(_set_overlap).sort("ring_overlap", False).first())
        return ee.Feature(
            ring,
            {
                "buffer_km": spec["label"],
                "inner_m": spec["inner"],
                "outer_m": spec["outer"],
                "case_grid_id": dominant_grid.get("grid_id"),
            },
        )

    buffers = ee.FeatureCollection([_build_buffer(spec) for spec in buffer_specs])

    def _build_monthly_water(month_start: ee.Date) -> ee.Image:
        month_start = ee.Date(month_start)
        month_end = month_start.advance(1, "month")
        l8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(month_start, month_end)
            .filterBounds(case_aoi)
            .map(mask_landsat_l2)
        )
        l9 = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterDate(month_start, month_end)
            .filterBounds(case_aoi)
            .map(mask_landsat_l2)
        )
        monthly = l8.merge(l9)
        return ee.Image(
            ee.Algorithms.If(
                monthly.size().gt(0),
                monthly.map(_landsat_water_mask).mean().rename("water"),
                masked_empty("water"),
            )
        ).clip(case_aoi)

    months = month_starts(CONFIG["start_date"], CONFIG["end_date_exclusive"])
    monthly_water = ee.ImageCollection.fromImages(
        months.map(
            lambda month_start: _build_monthly_water(ee.Date(month_start))
            .set("system:time_start", ee.Date(month_start).millis())
            .set("date", ee.Date(month_start).format("YYYY-MM-dd"))
        )
    )

    water_persistence_image = monthly_water.mean().rename("water_persistence")

    def _set_persistence(feature: ee.Feature) -> ee.Feature:
        persistence = water_persistence_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feature.geometry(),
            scale=CONFIG["stats_scale_meters"],
            bestEffort=True,
            maxPixels=1e13,
        ).get("water_persistence")
        return feature.set({"water_persistence": persistence})

    buffers_with_persistence = buffers.map(_set_persistence)

    def _buffer_month_features(month_start: ee.Date) -> ee.FeatureCollection:
        month_start = ee.Date(month_start)
        image = ee.Image(monthly_water.filterDate(month_start, month_start.advance(1, "month")).first())
        return image.reduceRegions(
            collection=buffers_with_persistence,
            reducer=ee.Reducer.mean(),
            scale=CONFIG["stats_scale_meters"],
            tileScale=4,
        ).map(
            lambda feature: feature.set(
                {
                    "date": month_start.format("YYYY-MM-dd"),
                    "year": ee.Number.parse(month_start.format("Y")),
                    "month": ee.Number.parse(month_start.format("M")),
                    "water_frac": feature.get("mean"),
                }
            ).select(CASE_SELECTORS)
        )

    return buffers_with_persistence, grid, monthly_water


def build_case_month_features(
    buffers_with_persistence: ee.FeatureCollection,
    monthly_water: ee.ImageCollection,
    month_start_text: str,
) -> ee.FeatureCollection:
    month_start = ee.Date(month_start_text)
    date_text = month_start.format("YYYY-MM-dd")
    monthly_match = monthly_water.filter(ee.Filter.eq("date", date_text))
    image = ee.Image(
        ee.Algorithms.If(
            monthly_match.size().gt(0),
            monthly_match.first(),
            masked_empty("water"),
        )
    )
    return image.reduceRegions(
        collection=buffers_with_persistence,
        reducer=ee.Reducer.mean(),
        scale=CONFIG["stats_scale_meters"],
        tileScale=4,
    ).map(
        lambda feature: feature.set(
            {
                "date": month_start.format("YYYY-MM-dd"),
                "year": ee.Number.parse(month_start.format("Y")),
                "month": ee.Number.parse(month_start.format("M")),
                "water_frac": feature.get("mean"),
            }
        ).select(CASE_SELECTORS)
    )


def _landsat_water_mask(image: ee.Image) -> ee.Image:
    green = image.select("SR_B3")
    red = image.select("SR_B4")
    nir = image.select("SR_B5")
    swir1 = image.select("SR_B6")
    mndwi = green.subtract(swir1).divide(green.add(swir1))
    ndvi = nir.subtract(red).divide(nir.add(red))
    return mndwi.gt(0).And(mndwi.gt(ndvi)).rename("water")


def main() -> None:
    args = parse_args()
    project = initialize_ee(project=args.project, auth_mode=args.auth_mode, force_auth=args.force_auth)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buffers_with_persistence, _, monthly_water = build_case_context()
    if args.describe:
        sample_month = CONFIG["start_date"]
        case_sample = build_case_month_features(buffers_with_persistence, monthly_water, sample_month)
        print_json(
            {
                "project": project,
                "case_table_sample_month": sample_month,
                "case_table": summarize_feature_collection(case_sample, CASE_SELECTORS),
            }
        )
        return

    output_path = output_dir / "hutuo_case_buffers_raw.csv"
    month_index = pd.date_range(
        CONFIG["start_date"],
        CONFIG["end_date_exclusive"],
        freq="MS",
        inclusive="left",
    )
    chunk_dir = output_dir / "_tmp_hutuo_case_buffers_raw"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    downloaded_months = 0
    reused_months = 0
    first_url = None
    for month_start in month_index:
        month_tag = month_start.strftime("%Y_%m")
        chunk_path = chunk_dir / f"hutuo_case_buffers_raw_{month_tag}.csv"
        if chunk_path.exists() and not args.overwrite:
            reused_months += 1
            continue
        month_text = month_start.strftime("%Y-%m-%d")
        month_fc = build_case_month_features(buffers_with_persistence, monthly_water, month_text)
        month_df, url = download_feature_collection_dataframe(
            collection=month_fc,
            selectors=CASE_SELECTORS,
            filename=f"hutuo_case_buffers_raw_{month_tag}",
        )
        if first_url is None:
            first_url = url
        month_df.to_csv(chunk_path, index=False)
        downloaded_months += 1

    ordered_chunk_paths = [chunk_dir / f"hutuo_case_buffers_raw_{month.strftime('%Y_%m')}.csv" for month in month_index]
    frames = [pd.read_csv(path) for path in ordered_chunk_paths if path.exists()]
    pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
    result = {
        "output_path": str(output_path),
        "month_count": int(len(month_index)),
        "row_count": int(sum(len(frame) for frame in frames)),
        "sample_download_url": first_url,
        "chunk_dir": str(chunk_dir),
        "downloaded_months": downloaded_months,
        "reused_months": reused_months,
    }
    print_json({"project": project, "case_download": result})


if __name__ == "__main__":
    main()
