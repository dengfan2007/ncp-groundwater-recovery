from __future__ import annotations

import argparse
from pathlib import Path

import ee
import pandas as pd

from ee_local import initialize_ee
from ee_ncp_common import (
    build_ncp_context,
    build_rect_grid,
    download_feature_collection_dataframe,
    mask_landsat_l2,
    month_starts,
)


CONFIG = {
    "start_date": "2019-01-01",
    "end_date_exclusive": "2022-01-01",
    "bbox": [112.0, 34.5, 120.0, 40.6],
    "province_names": ["Beijing Shi", "Tianjin Shi", "Hebei Sheng", "Henan Sheng", "Shandong Sheng"],
    "max_elevation_meters": 200,
    "grid_step_degree": 1.0,
    "min_coverage_ratio": 0.60,
    "stats_scale_meters": 10000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download JRC/Landsat overlap-period water fractions for the NCP grid.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def landsat_water_month(month_start: ee.Date, bbox_geom: ee.Geometry) -> ee.Image:
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

    def _mask(image: ee.Image) -> ee.Image:
        green = image.select("SR_B3")
        red = image.select("SR_B4")
        nir = image.select("SR_B5")
        swir1 = image.select("SR_B6")
        mndwi = green.subtract(swir1).divide(green.add(swir1))
        ndvi = nir.subtract(red).divide(nir.add(red))
        return mndwi.gt(0).And(mndwi.gt(ndvi)).rename("landsat_water_frac")

    return ee.Image(
        ee.Algorithms.If(
            monthly.size().gt(0),
            monthly.map(_mask).mean().rename("landsat_water_frac"),
            ee.Image.constant(0).rename("landsat_water_frac").updateMask(ee.Image.constant(0)),
        )
    )


def build_overlap_context() -> tuple[ee.FeatureCollection, ee.ImageCollection]:
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
    jrc = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")

    def _month_fc(month_start):
        month_start = ee.Date(month_start)
        year = ee.Number.parse(month_start.format("Y"))
        month = ee.Number.parse(month_start.format("M"))
        jrc_month = jrc.filter(ee.Filter.eq("year", year)).filter(ee.Filter.eq("month", month))
        jrc_image = ee.Image(
            ee.Algorithms.If(
                jrc_month.size().gt(0),
                ee.Image(jrc_month.first()).select("water").eq(2).rename("jrc_water_frac"),
                ee.Image.constant(0).rename("jrc_water_frac").updateMask(ee.Image.constant(0)),
            )
        )
        landsat_image = landsat_water_month(month_start, bbox_geom)
        image = ee.Image.cat([jrc_image, landsat_image]).updateMask(lowland_mask).clip(bbox_geom)
        return image.reduceRegions(
            collection=grid,
            reducer=ee.Reducer.mean(),
            scale=CONFIG["stats_scale_meters"],
            tileScale=4,
        ).map(
            lambda feature: feature.set(
                {
                    "date": month_start.format("YYYY-MM-dd"),
                    "year": year,
                    "month": month,
                }
            )
        )

    months = month_starts(CONFIG["start_date"], CONFIG["end_date_exclusive"])
    monthly_fc = ee.FeatureCollection(
        ee.List(months.map(_month_fc)).iterate(
            lambda item, acc: ee.FeatureCollection(acc).merge(ee.FeatureCollection(item)),
            ee.FeatureCollection([]),
        )
    )
    return grid, monthly_fc


def build_overlap_month_fc(grid: ee.FeatureCollection, month_start_text: str) -> ee.FeatureCollection:
    context = build_ncp_context(
        bbox=CONFIG["bbox"],
        province_names=CONFIG["province_names"],
        max_elevation_meters=CONFIG["max_elevation_meters"],
    )
    bbox_geom = context["bbox"]
    lowland_mask = context["lowland_mask"]
    jrc = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory")
    month_start = ee.Date(month_start_text)
    year = ee.Number.parse(month_start.format("Y"))
    month = ee.Number.parse(month_start.format("M"))
    jrc_month = jrc.filter(ee.Filter.eq("year", year)).filter(ee.Filter.eq("month", month))
    jrc_image = ee.Image(
        ee.Algorithms.If(
            jrc_month.size().gt(0),
            ee.Image(jrc_month.first()).select("water").eq(2).rename("jrc_water_frac"),
            ee.Image.constant(0).rename("jrc_water_frac").updateMask(ee.Image.constant(0)),
        )
    )
    landsat_image = landsat_water_month(month_start, bbox_geom)
    image = ee.Image.cat([jrc_image, landsat_image]).updateMask(lowland_mask).clip(bbox_geom)
    return image.reduceRegions(
        collection=grid,
        reducer=ee.Reducer.mean(),
        scale=CONFIG["stats_scale_meters"],
        tileScale=4,
    ).map(
        lambda feature: feature.set(
            {
                "date": month_start.format("YYYY-MM-dd"),
                "year": year,
                "month": month,
            }
        )
    )


def main() -> None:
    args = parse_args()
    initialize_ee(project=args.project, auth_mode=None, force_auth=False)
    context = build_ncp_context(
        bbox=CONFIG["bbox"],
        province_names=CONFIG["province_names"],
        max_elevation_meters=CONFIG["max_elevation_meters"],
    )
    grid = build_rect_grid(
        bbox=CONFIG["bbox"],
        step_degree=CONFIG["grid_step_degree"],
        mask=context["lowland_mask"],
        min_coverage_ratio=CONFIG["min_coverage_ratio"],
        include_centroid_fields=True,
        include_region_tag=True,
    )
    month_index = pd.date_range(CONFIG["start_date"], CONFIG["end_date_exclusive"], freq="MS", inclusive="left")
    frames = []
    for month_start in month_index:
        month_text = month_start.strftime("%Y-%m-%d")
        month_fc = build_overlap_month_fc(grid, month_text)
        month_df, _ = download_feature_collection_dataframe(
            collection=month_fc,
            selectors=[
                "grid_id",
                "grid_lon",
                "grid_lat",
                "region_tag",
                "coverage_ratio",
                "date",
                "year",
                "month",
                "jrc_water_frac",
                "landsat_water_frac",
            ],
            filename=f"water_overlap_{month_start.strftime('%Y_%m')}",
        )
        frames.append(month_df)
    df = pd.concat(frames, ignore_index=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
