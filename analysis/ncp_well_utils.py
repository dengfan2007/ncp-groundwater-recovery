from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def _missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [column for column in required if column not in df.columns]


def standardize_well_data(
    wells: pd.DataFrame,
    *,
    dataset_label: str,
    value_kind: str,
    well_id_col: str = "well_id",
    date_col: str = "date",
    value_col: str = "gw_level",
    grid_col: str = "grid_id",
    lon_col: str = "lon",
    lat_col: str = "lat",
    coord_to_grid_id: Callable[[float, float], str | None] | None = None,
) -> pd.DataFrame:
    required = [well_id_col, date_col, value_col]
    missing = _missing_columns(wells, required)
    if missing:
        raise ValueError(
            f"{dataset_label} is missing required columns {missing}. "
            f"Available columns: {sorted(wells.columns.tolist())}"
        )

    standardized = wells.rename(
        columns={
            well_id_col: "well_id",
            date_col: "date",
            value_col: "well_value",
        }
    ).copy()
    standardized["date"] = pd.to_datetime(standardized["date"]).dt.to_period("M").dt.to_timestamp()

    if grid_col in standardized.columns:
        standardized["grid_id"] = standardized[grid_col]
    else:
        coord_missing = _missing_columns(standardized, [lon_col, lat_col])
        if coord_missing or coord_to_grid_id is None:
            raise ValueError(
                f"{dataset_label} must provide either `{grid_col}` or both `{lon_col}` and `{lat_col}`. "
                f"Available columns: {sorted(standardized.columns.tolist())}"
            )
        standardized["grid_id"] = standardized.apply(
            lambda row: coord_to_grid_id(row[lon_col], row[lat_col]),
            axis=1,
        )

    standardized = standardized.dropna(subset=["well_id", "grid_id", "date", "well_value"]).copy()
    if standardized.empty:
        return standardized.assign(well_anomaly=pd.Series(dtype=float))

    if value_kind == "level":
        standardized["well_anomaly"] = standardized.groupby("well_id")["well_value"].transform(
            lambda series: series - series.mean()
        )
    else:
        standardized["well_anomaly"] = standardized["well_value"]

    return (
        standardized.groupby(["well_id", "grid_id", "date"], as_index=False)["well_anomaly"]
        .mean()
        .sort_values(["grid_id", "well_id", "date"])
    )
