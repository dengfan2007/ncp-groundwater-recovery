from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


START_DATE = "2005-01-01"
LAT_START = 34.5
LAT_END = 40.5
LON_START = 113.0
LON_END = 119.5
STEP = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the public NCP observation-derived groundwater grid (OK interpolation) into a 1-degree proxy well CSV."
    )
    parser.add_argument("--npy", required=True, help="Path to 01_OK.npy or equivalent public observation grid.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def grid_id_from_coord(lon: float, lat: float) -> str:
    lon_base = 112.0 + np.floor((lon - 112.0) / 1.0) * 1.0
    lat_base = 34.5 + np.floor((lat - 34.5) / 1.0) * 1.0
    return f"g_{lon_base:.1f}_{lat_base:.1f}"


def main() -> None:
    args = parse_args()
    arr = np.load(args.npy)
    if arr.shape != (168, 24, 26):
        raise ValueError(f"Unexpected observation grid shape: {arr.shape}")

    dates = pd.date_range(START_DATE, periods=arr.shape[0], freq="MS")
    lat_values = np.arange(LAT_START, LAT_END, STEP)
    lon_values = np.arange(LON_START, LON_END, STEP)

    rows = []
    for t_idx, date in enumerate(dates):
        frame = arr[t_idx]
        for lat_idx, lat in enumerate(lat_values):
            for lon_idx, lon in enumerate(lon_values):
                value = frame[lat_idx, lon_idx]
                if np.isnan(value):
                    continue
                rows.append(
                    {
                        "well_id": f"proxy_{grid_id_from_coord(lon, lat)}",
                        "grid_id": grid_id_from_coord(lon, lat),
                        "date": date.strftime("%Y-%m-%d"),
                        "gw_level": float(value),
                        "source_dataset": "ReconstructedTWS_OK_2005_2018",
                    }
                )

    df = pd.DataFrame(rows)
    out = (
        df.groupby(["well_id", "grid_id", "date", "source_dataset"], as_index=False)["gw_level"]
        .mean()
        .sort_values(["grid_id", "date"])
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
