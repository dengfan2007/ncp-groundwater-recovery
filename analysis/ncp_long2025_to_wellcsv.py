from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


GRID_BBOX = (112.0, 34.5, 120.0, 40.6)
GRID_STEP = 1.0
DATE_COL_PATTERN = re.compile(r"^\d{4}-\d{2}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the GroundwaterNCP_open workbook into a standardized well CSV."
    )
    parser.add_argument("--excel", required=True, help="Path to GroundwaterDepth.xlsx.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def coord_to_grid_id(lon: float, lat: float) -> str | None:
    if any(pd.isna(v) for v in [lon, lat]):
        return None
    west, south, east, north = GRID_BBOX
    if lon < west or lon >= east or lat < south or lat >= north:
        return None
    lon_base = west + np.floor((lon - west) / GRID_STEP) * GRID_STEP
    lat_base = south + np.floor((lat - south) / GRID_STEP) * GRID_STEP
    return f"g_{lon_base:.1f}_{lat_base:.1f}"


def aquifer_group_from_type(well_type: float) -> str:
    if pd.isna(well_type):
        return "unknown"
    if 10 < well_type < 20:
        return "unconfined"
    if 20 < well_type < 30:
        return "confined"
    return "other"


def aquifer_layer_from_type(well_type: float) -> float | None:
    if pd.isna(well_type):
        return None
    if 10 < well_type < 30:
        return float(int(well_type) % 10)
    return None


def main() -> None:
    args = parse_args()
    excel_path = Path(args.excel)
    output_path = Path(args.output)

    wide = pd.read_excel(excel_path)
    date_cols = [column for column in wide.columns if DATE_COL_PATTERN.match(str(column))]
    if not date_cols:
        raise ValueError(f"No monthly date columns like YYYY-MM were found in {excel_path}")

    id_cols = ["ID", "City", "Type", "Lon", "Lat"]
    missing = [column for column in id_cols if column not in wide.columns]
    if missing:
        raise ValueError(f"Missing required columns in {excel_path}: {missing}")

    long_df = wide.melt(id_vars=id_cols, value_vars=date_cols, var_name="date", value_name="gw_depth_m")
    long_df["date"] = pd.to_datetime(long_df["date"] + "-01", errors="coerce")
    long_df["grid_id"] = long_df.apply(lambda row: coord_to_grid_id(row["Lon"], row["Lat"]), axis=1)
    long_df["well_id"] = long_df["ID"].apply(lambda value: f"long2025_{int(value)}" if pd.notna(value) else None)
    long_df["aquifer_group"] = long_df["Type"].apply(aquifer_group_from_type)
    long_df["aquifer_layer"] = long_df["Type"].apply(aquifer_layer_from_type)
    long_df["source_dataset"] = "GroundwaterNCP_open_2005_2024"
    # The public workbook stores depth to groundwater: larger depth means weaker groundwater conditions.
    # Flip the sign into a recovery-consistent direction for the downstream anomaly workflow.
    long_df["gw_level"] = -long_df["gw_depth_m"]

    out = (
        long_df.dropna(subset=["well_id", "date", "gw_depth_m", "Lon", "Lat"])
        .rename(
            columns={
                "ID": "source_well_id",
                "City": "city",
                "Type": "type",
                "Lon": "lon",
                "Lat": "lat",
            }
        )[
            [
                "well_id",
                "source_well_id",
                "city",
                "type",
                "aquifer_group",
                "aquifer_layer",
                "lon",
                "lat",
                "grid_id",
                "date",
                "gw_depth_m",
                "gw_level",
                "source_dataset",
            ]
        ]
        .sort_values(["well_id", "date"])
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
