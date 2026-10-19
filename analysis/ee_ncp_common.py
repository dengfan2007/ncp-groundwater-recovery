from __future__ import annotations

import json
from io import BytesIO
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import ee
import pandas as pd


def month_starts(start_date: str, end_date_exclusive: str) -> ee.List:
    start = ee.Date(start_date)
    end = ee.Date(end_date_exclusive)
    month_count = end.difference(start, "month")

    def _advance(n: ee.Number) -> ee.Date:
        return start.advance(n, "month")

    return ee.List.sequence(0, month_count.subtract(1)).map(_advance)


def masked_empty(name: str) -> ee.Image:
    return ee.Image.constant(0).rename(name).updateMask(ee.Image.constant(0))


def safe_mean(collection: ee.ImageCollection, start: ee.Date, end: ee.Date, band_name: str, out_name: str) -> ee.Image:
    filtered = collection.filterDate(start, end).select(band_name)
    return ee.Image(
        ee.Algorithms.If(
            filtered.size().gt(0),
            filtered.mean().rename(out_name),
            masked_empty(out_name),
        )
    )


def safe_sum(
    collection: ee.ImageCollection,
    start: ee.Date,
    end: ee.Date,
    band_name: str,
    out_name: str,
    scale_factor: float,
) -> ee.Image:
    filtered = collection.filterDate(start, end).select(band_name)
    return ee.Image(
        ee.Algorithms.If(
            filtered.size().gt(0),
            filtered.sum().multiply(scale_factor).rename(out_name),
            masked_empty(out_name),
        )
    )


def mask_landsat_l2(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 0).eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
    return image.addBands(optical, overwrite=True).updateMask(mask)


def merge_feature_collections(collections: ee.List) -> ee.FeatureCollection:
    def _merge(item: ee.ComputedObject, acc: ee.FeatureCollection) -> ee.FeatureCollection:
        return ee.FeatureCollection(acc).merge(ee.FeatureCollection(item))

    return ee.FeatureCollection(collections.iterate(_merge, ee.FeatureCollection([])))


def build_ncp_context(
    bbox: list[float],
    province_names: list[str],
    max_elevation_meters: float,
) -> dict[str, ee.ComputedObject]:
    bbox_geom = ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)
    provinces = (
        ee.FeatureCollection("FAO/GAUL/2015/level1")
        .filter(ee.Filter.eq("ADM0_NAME", "China"))
        .filter(ee.Filter.inList("ADM1_NAME", province_names))
    )
    province_geom = provinces.geometry().dissolve()
    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(bbox_geom)
    lowland_mask = elevation.lte(max_elevation_meters).selfMask().clip(province_geom)
    return {
        "bbox": bbox_geom,
        "provinces": provinces,
        "province_geom": province_geom,
        "lowland_mask": lowland_mask,
    }


def build_rect_grid(
    bbox: list[float],
    step_degree: float,
    mask: ee.Image,
    min_coverage_ratio: float,
    include_centroid_fields: bool = True,
    include_region_tag: bool = False,
) -> ee.FeatureCollection:
    lon_list = ee.List.sequence(bbox[0], bbox[2] - step_degree, step_degree)
    lat_list = ee.List.sequence(bbox[1], bbox[3] - step_degree, step_degree)

    def _per_lon(lon: ee.Number) -> ee.List:
        lon = ee.Number(lon)

        def _per_lat(lat: ee.Number) -> ee.Feature:
            lat = ee.Number(lat)
            rect = ee.Geometry.Rectangle(
                [lon, lat, lon.add(step_degree), lat.add(step_degree)],
                proj="EPSG:4326",
                geodesic=False,
            )
            overlap_dict = ee.Dictionary(
                ee.Image.pixelArea().updateMask(mask).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=rect,
                    scale=1000,
                    bestEffort=True,
                    maxPixels=1e13,
                )
            )
            overlap = ee.Number(
                ee.Algorithms.If(overlap_dict.contains("area"), overlap_dict.get("area"), 0)
            )
            properties: dict[str, ee.ComputedObject] = {
                "grid_id": ee.String("g_").cat(lon.format("%.1f")).cat("_").cat(lat.format("%.1f")),
                "coverage_ratio": overlap.divide(rect.area(1)),
            }

            if include_centroid_fields:
                center = rect.centroid(1)
                center_lon = center.coordinates().get(0)
                center_lat = ee.Number(center.coordinates().get(1))
                properties["grid_lon"] = center_lon
                properties["grid_lat"] = center_lat
                if include_region_tag:
                    properties["region_tag"] = ee.String(
                        ee.Algorithms.If(center_lat.gte(37.0), "north", "south")
                    )

            return ee.Feature(rect, properties)

        return lat_list.map(_per_lat)

    nested = lon_list.map(_per_lon)
    return ee.FeatureCollection(ee.List(nested).flatten()).filter(
        ee.Filter.gte("coverage_ratio", min_coverage_ratio)
    )


def feature_collection_download_url(
    collection: ee.FeatureCollection,
    selectors: list[str],
    filename: str,
) -> str:
    return str(
        collection.getDownloadURL(
            **{
                "filetype": "CSV",
                "selectors": selectors,
                "filename": filename,
            }
        )
    )


def download_url_to_path(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_url_with_retry(url) as response, output_path.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    return output_path


def download_feature_collection_csv(
    collection: ee.FeatureCollection,
    selectors: list[str],
    output_path: Path,
    filename: str | None = None,
) -> dict[str, Any]:
    output_name = filename or output_path.stem
    url = feature_collection_download_url(collection, selectors, output_name)
    download_url_to_path(url, output_path)
    return {"output_path": str(output_path), "download_url": url}


def download_feature_collection_dataframe(
    collection: ee.FeatureCollection,
    selectors: list[str],
    filename: str,
) -> tuple[pd.DataFrame, str]:
    url = feature_collection_download_url(collection, selectors, filename)
    with _open_url_with_retry(url) as response:
        data = response.read()
    return pd.read_csv(BytesIO(data)), url


def _open_url_with_retry(url: str, attempts: int = 5, timeout: int = 180):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(min(30, 2 ** attempt))
    if last_error is None:
        raise RuntimeError("Failed to open URL for unknown reasons.")
    raise last_error


def summarize_feature_collection(
    collection: ee.FeatureCollection,
    sample_selectors: list[str] | None = None,
) -> dict[str, Any]:
    sample_feature = ee.Feature(collection.first())
    if sample_selectors:
        sample_feature = ee.Feature(sample_feature.select(sample_selectors))
    summary = {
        "feature_count": collection.size().getInfo(),
        "sample_feature": sample_feature.toDictionary().getInfo(),
    }
    return summary


def print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))
