from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPECTED_START = pd.Timestamp("2003-01-01")
EXPECTED_END = pd.Timestamp("2024-09-01")
TRANSITION_BEFORE = pd.Timestamp("2021-12-01")
TRANSITION_AFTER = pd.Timestamp("2022-01-01")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality-control the exported NCP groundwater datasets.")
    parser.add_argument("--panel", required=True, help="Path to grid_monthly_panel.csv")
    parser.add_argument("--grid", help="Path to ncp_grid_definition.csv")
    parser.add_argument("--case", help="Path to hutuo_case_buffers_raw.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for QC outputs")
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def expected_months() -> pd.DatetimeIndex:
    return pd.date_range(EXPECTED_START, EXPECTED_END, freq="MS")


def load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    ensure_columns(
        panel,
        [
            "grid_id",
            "date",
            "gwa_cm",
            "twsa_cm",
            "smsa_cm",
            "swea_cm",
            "cwa_cm",
            "precip_mm",
            "et_mm",
            "water_frac",
        ],
        "panel",
    )
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["grid_id", "date"]).reset_index(drop=True)
    return panel


def panel_qc(panel: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    expected = expected_months()
    actual = pd.DatetimeIndex(sorted(panel["date"].drop_duplicates()))
    missing_global = expected.difference(actual)

    rows = []
    for grid_id, group in panel.groupby("grid_id"):
        dates = pd.DatetimeIndex(group["date"])
        missing = expected.difference(dates)
        group_sorted = group.sort_values("date").copy()
        group_sorted["month_gap"] = group_sorted["date"].diff().dt.days.div(30.0)
        max_gap = float(group_sorted["month_gap"].fillna(1).max())
        rows.append(
            {
                "grid_id": grid_id,
                "row_count": int(len(group)),
                "missing_month_count": int(len(missing)),
                "max_month_gap": max_gap,
                "gwa_missing_ratio": float(group["gwa_cm"].isna().mean()),
                "water_missing_ratio": float(group["water_frac"].isna().mean()),
            }
        )

    grid_missing_df = pd.DataFrame(rows).sort_values(["missing_month_count", "water_missing_ratio"], ascending=[False, False])
    summary = {
        "expected_month_count": int(len(expected)),
        "actual_global_month_count": int(len(actual)),
        "global_missing_month_count": int(len(missing_global)),
        "global_missing_months": [d.strftime("%Y-%m-%d") for d in missing_global],
        "grid_count": int(panel["grid_id"].nunique()),
        "grid_with_missing_months": int((grid_missing_df["missing_month_count"] > 0).sum()),
        "grid_with_gap_gt_1_5_months": int((grid_missing_df["max_month_gap"] > 1.5).sum()),
        "gwa_missing_ratio_mean": float(grid_missing_df["gwa_missing_ratio"].mean()),
        "water_missing_ratio_mean": float(grid_missing_df["water_missing_ratio"].mean()),
    }
    return summary, grid_missing_df


def water_transition_qc(panel: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    pivot = (
        panel.loc[panel["date"].isin([TRANSITION_BEFORE, TRANSITION_AFTER])]
        .pivot_table(index="grid_id", columns="date", values="water_frac", aggfunc="mean")
        .rename(columns={TRANSITION_BEFORE: "water_2021_12", TRANSITION_AFTER: "water_2022_01"})
        .reset_index()
    )

    if pivot.empty:
        summary = {
            "transition_grid_count": 0,
            "water_jump_mean": None,
            "water_jump_abs_mean": None,
            "water_jump_abs_gt_0_20_count": 0,
        }
        return summary, pivot

    pivot["water_jump"] = pivot["water_2022_01"] - pivot["water_2021_12"]
    summary = {
        "transition_grid_count": int(len(pivot)),
        "water_jump_mean": float(pivot["water_jump"].mean()),
        "water_jump_abs_mean": float(pivot["water_jump"].abs().mean()),
        "water_jump_abs_gt_0_20_count": int((pivot["water_jump"].abs() > 0.20).sum()),
    }
    return summary, pivot.sort_values("water_jump", ascending=False)


def grid_qc(path: Path | None) -> tuple[dict[str, object], pd.DataFrame | None]:
    if path is None:
        return {"grid_definition_loaded": False}, None
    grid = pd.read_csv(path)
    ensure_columns(grid, ["grid_id", "coverage_ratio"], "grid definition")
    summary = {
        "grid_definition_loaded": True,
        "grid_definition_count": int(len(grid)),
        "coverage_ratio_min": float(grid["coverage_ratio"].min()),
        "coverage_ratio_mean": float(grid["coverage_ratio"].mean()),
        "coverage_ratio_lt_0_60_count": int((grid["coverage_ratio"] < 0.60).sum()),
    }
    return summary, grid.sort_values("coverage_ratio")


def case_qc(path: Path | None) -> tuple[dict[str, object], pd.DataFrame | None]:
    if path is None:
        return {"case_loaded": False}, None
    case = pd.read_csv(path)
    ensure_columns(
        case,
        ["buffer_km", "date", "water_frac", "water_persistence", "case_grid_id"],
        "case table",
    )
    case["date"] = pd.to_datetime(case["date"])
    expected_case = pd.date_range(case["date"].min(), case["date"].max(), freq="MS")

    rows = []
    for buffer_km, group in case.groupby("buffer_km"):
        missing = expected_case.difference(pd.DatetimeIndex(group["date"]))
        rows.append(
            {
                "buffer_km": buffer_km,
                "row_count": int(len(group)),
                "missing_month_count": int(len(missing)),
                "water_frac_missing_ratio": float(group["water_frac"].isna().mean()),
                "water_persistence_mean": float(group["water_persistence"].mean()),
                "case_grid_count": int(group["case_grid_id"].nunique()),
            }
        )

    buffer_df = pd.DataFrame(rows).sort_values("buffer_km")
    summary = {
        "case_loaded": True,
        "buffer_count": int(case["buffer_km"].nunique()),
        "buffer_with_missing_months": int((buffer_df["missing_month_count"] > 0).sum()),
        "case_unique_grids": int(case["case_grid_id"].nunique()),
    }
    return summary, buffer_df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(Path(args.panel))
    panel_summary, grid_missing_df = panel_qc(panel)
    transition_summary, transition_df = water_transition_qc(panel)
    grid_summary, grid_df = grid_qc(Path(args.grid) if args.grid else None)
    case_summary, case_df = case_qc(Path(args.case) if args.case else None)

    summary = {
        "panel": panel_summary,
        "water_transition": transition_summary,
        "grid_definition": grid_summary,
        "case": case_summary,
    }

    (output_dir / "qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    grid_missing_df.to_csv(output_dir / "qc_grid_missingness.csv", index=False)
    transition_df.to_csv(output_dir / "qc_water_transition.csv", index=False)
    if grid_df is not None:
        grid_df.to_csv(output_dir / "qc_grid_definition.csv", index=False)
    if case_df is not None:
        case_df.to_csv(output_dir / "qc_case_buffers.csv", index=False)


if __name__ == "__main__":
    main()
