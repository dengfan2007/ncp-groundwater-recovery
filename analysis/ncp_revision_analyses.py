from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import ncp_recovery_analysis as core
from ncp_well_utils import standardize_well_data


EARLY_START = pd.Timestamp("2018-10-01")
EARLY_END = pd.Timestamp("2019-08-31")
MAIN_START = pd.Timestamp("2019-09-01")
MAIN_END = pd.Timestamp("2020-12-31")
LEGACY_END = pd.Timestamp("2016-12-31")
EVENT_START = pd.Timestamp("2019-01-01")
REPORTING_RECOVERY_START = pd.Timestamp("2018-01-01")
SPATIAL_BLOCK_DEGREES = 2.0
SPATIAL_BOOTSTRAP_ITER = 2000
VALIDATION_BLOCK_MONTHS = 12
VALIDATION_BOOTSTRAP_ITER = 1000
GRID_WEST = 112.0
GRID_SOUTH = 34.5
MIN_BREAK_SEGMENT_MONTHS = 18
BASELINE_VARIANTS = {
    "baseline_2003_2014": ("2003-01-01", "2014-12-01"),
    "baseline_2004_2013": ("2004-01-01", "2013-12-01"),
    "baseline_2005_2014": ("2005-01-01", "2014-12-01"),
}
COVERAGE_THRESHOLDS = [0.60, 0.70, 0.80]
BREAKPOINT_RULE_VARIANTS = {
    "relaxed": {"beast_prob": 0.40, "bic_delta": 0.0, "min_segment_months": 18},
    "baseline": {"beast_prob": 0.50, "bic_delta": 2.0, "min_segment_months": 18},
    "strict": {"beast_prob": 0.60, "bic_delta": 4.0, "min_segment_months": 18},
}
SPATIAL_BLOCK_VARIANTS = [1.0, 2.0, 3.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate revision-support analyses for the NCP manuscript.")
    parser.add_argument("--postprocess-dir", required=True)
    parser.add_argument("--main-dir", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--beast-main-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--water-overlap", help="Optional path to precomputed JRC-Landsat overlap table")
    parser.add_argument("--well-data")
    parser.add_argument("--well-query")
    return parser.parse_args()


def classify(onset: pd.Timestamp | None, post_trend: float, threshold: float, scheme: str = "baseline") -> str:
    if pd.isna(onset):
        return "stalled_or_declining"

    if scheme == "minus3":
        early_start, early_end = pd.Timestamp("2018-07-01"), pd.Timestamp("2019-05-31")
        main_start, main_end = pd.Timestamp("2019-06-01"), pd.Timestamp("2020-09-30")
    elif scheme == "plus3":
        early_start, early_end = pd.Timestamp("2019-01-01"), pd.Timestamp("2019-11-30")
        main_start, main_end = pd.Timestamp("2019-12-01"), pd.Timestamp("2021-03-31")
    else:
        early_start, early_end = EARLY_START, EARLY_END
        main_start, main_end = MAIN_START, MAIN_END

    if onset <= LEGACY_END and post_trend <= threshold:
        return "legacy_slowdown"
    if early_start <= onset <= early_end and post_trend > threshold:
        return "early_rebound"
    if main_start <= onset <= main_end and post_trend > threshold:
        return "main_rebound"
    return "stalled_or_declining"


def month_index(series: pd.Series) -> pd.Series:
    return series.dt.year * 12 + series.dt.month


def month_index_to_timestamp(value: float | int | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    month_idx = int(round(float(value)))
    year = month_idx // 12
    month = month_idx % 12
    if month == 0:
        year -= 1
        month = 12
    return pd.Timestamp(year=year, month=month, day=1).strftime("%Y-%m")


def quantile_ci(values: list[float], low: float = 0.025, high: float = 0.975) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return float(np.quantile(arr, low)), float(np.quantile(arr, high))


def interquartile_window(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def zscore_or_nan(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    if int(mask.sum()) < 3:
        return np.full(arr.shape, np.nan, dtype=float)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std == 0:
        return np.full(arr.shape, np.nan, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    out[mask] = (arr[mask] - mean) / std
    return out


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    a_z = zscore_or_nan(a)
    b_z = zscore_or_nan(b)
    mask = np.isfinite(a_z) & np.isfinite(b_z)
    if int(mask.sum()) < 3:
        return float("nan")
    a_sign = np.sign(a_z[mask])
    b_sign = np.sign(b_z[mask])
    return float(np.mean(a_sign == b_sign))


def reportable_subset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["onset_date"] = pd.to_datetime(out["onset_date"], errors="coerce")
    return out.loc[
        out["is_recovery_relevant"].fillna(False)
        & out["onset_date"].notna()
        & (out["onset_date"] >= REPORTING_RECOVERY_START)
        & out["post_trend_cm_per_year"].notna()
    ].copy()


def summarize_recovery_variant(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    reportable = reportable_subset(df)
    rows: list[dict[str, object]] = []
    rows.append({"variant": variant, "metric": "grid_count", "value": float(df["grid_id"].nunique())})
    rows.append({"variant": variant, "metric": "reportable_count", "value": float(len(reportable))})
    if reportable.empty:
        return pd.DataFrame(rows)

    reportable["onset_month_index"] = month_index(reportable["onset_date"])
    rows.extend(
        [
            {"variant": variant, "metric": "median_onset_all", "value": month_index_to_timestamp(reportable["onset_month_index"].median())},
            {"variant": variant, "metric": "median_posttrend_all", "value": float(reportable["post_trend_cm_per_year"].median())},
        ]
    )
    for region in ["north", "south"]:
        sub = reportable.loc[reportable["region_tag"] == region]
        if sub.empty:
            continue
        rows.extend(
            [
                {"variant": variant, "metric": f"{region}_count", "value": float(len(sub))},
                {"variant": variant, "metric": f"{region}_median_onset", "value": month_index_to_timestamp(sub["onset_month_index"].median())},
                {"variant": variant, "metric": f"{region}_median_posttrend", "value": float(sub["post_trend_cm_per_year"].median())},
            ]
        )

    north = reportable.loc[reportable["region_tag"] == "north"]
    south = reportable.loc[reportable["region_tag"] == "south"]
    if not north.empty and not south.empty:
        rows.extend(
            [
                {
                    "variant": variant,
                    "metric": "south_minus_north_mean_onset_month_diff",
                    "value": float(south["onset_month_index"].mean() - north["onset_month_index"].mean()),
                },
                {
                    "variant": variant,
                    "metric": "south_minus_north_mean_posttrend_cm_per_year",
                    "value": float(south["post_trend_cm_per_year"].mean() - north["post_trend_cm_per_year"].mean()),
                },
            ]
        )
    return pd.DataFrame(rows)


def temporarily_override_breakpoint_settings(
    *,
    beast_prob: float | None = None,
    bic_delta: float | None = None,
    min_segment_months: int | None = None,
):
    class _Override:
        def __enter__(self_inner):
            self_inner.original = {
                "BEAST_PROB_THRESHOLD": core.BEAST_PROB_THRESHOLD,
                "SEGMENTED_BIC_DELTA_THRESHOLD": core.SEGMENTED_BIC_DELTA_THRESHOLD,
                "MIN_SEGMENT_MONTHS": core.MIN_SEGMENT_MONTHS,
            }
            if beast_prob is not None:
                core.BEAST_PROB_THRESHOLD = float(beast_prob)
            if bic_delta is not None:
                core.SEGMENTED_BIC_DELTA_THRESHOLD = float(bic_delta)
            if min_segment_months is not None:
                core.MIN_SEGMENT_MONTHS = int(min_segment_months)
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            core.BEAST_PROB_THRESHOLD = self_inner.original["BEAST_PROB_THRESHOLD"]
            core.SEGMENTED_BIC_DELTA_THRESHOLD = self_inner.original["SEGMENTED_BIC_DELTA_THRESHOLD"]
            core.MIN_SEGMENT_MONTHS = self_inner.original["MIN_SEGMENT_MONTHS"]

    return _Override()


def detect_breakpoints_with_settings(
    panel: pd.DataFrame,
    *,
    beast_prob: float | None = None,
    bic_delta: float | None = None,
    min_segment_months: int | None = None,
) -> pd.DataFrame:
    with temporarily_override_breakpoint_settings(
        beast_prob=beast_prob,
        bic_delta=bic_delta,
        min_segment_months=min_segment_months,
    ):
        return core.detect_breakpoints(panel)


def rebuild_proxy_from_components(panel: pd.DataFrame, baseline_start: str, baseline_end: str) -> pd.DataFrame:
    rebased = panel.copy()
    start = pd.Timestamp(baseline_start)
    end = pd.Timestamp(baseline_end)
    baseline_mask = (rebased["date"] >= start) & (rebased["date"] <= end)
    component_cols = ["twsa_cm", "smsa_cm", "swea_cm", "cwa_cm"]
    for col in component_cols:
        shift = rebased.loc[baseline_mask].groupby("grid_id")[col].mean()
        rebased[col] = rebased[col] - rebased["grid_id"].map(shift)
    rebased["gwa_cm"] = (
        rebased["twsa_cm"]
        - rebased["smsa_cm"]
        - rebased["swea_cm"]
        - rebased["cwa_cm"]
    )
    return rebased

def permutation_mean_diff(a: np.ndarray, b: np.ndarray, n_iter: int = 20000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    observed = float(np.nanmean(a) - np.nanmean(b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_iter):
        rng.shuffle(pooled)
        diff = np.nanmean(pooled[:n_a]) - np.nanmean(pooled[n_a:])
        if abs(diff) >= abs(observed):
            count += 1
    return {"observed_mean_diff": observed, "permutation_pvalue": (count + 1) / (n_iter + 1)}


def method_breakpoint_views(
    main_breakpoints: pd.DataFrame,
    beast_breakpoints: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if {"segmented_onset_date", "segmented_post_trend_cm_per_year"}.issubset(main_breakpoints.columns):
        segmented = main_breakpoints[
            ["grid_id", "segmented_onset_date", "segmented_post_trend_cm_per_year"]
        ].rename(
            columns={
                "segmented_onset_date": "onset_date",
                "segmented_post_trend_cm_per_year": "post_trend_cm_per_year",
            }
        )
    else:
        segmented = main_breakpoints[["grid_id", "onset_date", "post_trend_cm_per_year"]].copy()

    if beast_breakpoints is not None:
        beast = beast_breakpoints[["grid_id", "onset_date", "post_trend_cm_per_year", "breakpoint_confidence"]].copy()
    else:
        required = {"beast_onset_date", "beast_post_trend_cm_per_year", "beast_breakpoint_confidence"}
        if not required.issubset(main_breakpoints.columns):
            raise ValueError(
                "Main breakpoint table does not contain BEAST detail columns. "
                "Provide `--beast-main-dir` or regenerate `grid_breakpoints.csv` with the updated analysis script."
            )
        beast = main_breakpoints[
            ["grid_id", "beast_onset_date", "beast_post_trend_cm_per_year", "beast_breakpoint_confidence"]
        ].rename(
            columns={
                "beast_onset_date": "onset_date",
                "beast_post_trend_cm_per_year": "post_trend_cm_per_year",
                "beast_breakpoint_confidence": "breakpoint_confidence",
            }
        )
    return segmented, beast


def build_method_comparison(seg_df: pd.DataFrame, beast_df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = seg_df[["grid_id", "onset_date", "post_trend_cm_per_year"]].copy()
    base["onset_date"] = pd.to_datetime(base["onset_date"], errors="coerce")
    base["trajectory_type_segmented"] = base.apply(
        lambda row: classify(row["onset_date"], row["post_trend_cm_per_year"], threshold, "baseline"),
        axis=1,
    )

    beast = beast_df[["grid_id", "onset_date", "post_trend_cm_per_year", "breakpoint_confidence"]].copy()
    beast["onset_date"] = pd.to_datetime(beast["onset_date"], errors="coerce")
    beast["trajectory_type_beast"] = beast.apply(
        lambda row: classify(row["onset_date"], row["post_trend_cm_per_year"], threshold, "baseline"),
        axis=1,
    )

    merged = base.merge(beast, on="grid_id", how="inner", suffixes=("_seg", "_beast"))
    month_diff = (
        (merged["onset_date_beast"].dt.year - merged["onset_date_seg"].dt.year) * 12
        + (merged["onset_date_beast"].dt.month - merged["onset_date_seg"].dt.month)
    )
    merged["onset_month_diff_beast_minus_segmented"] = month_diff
    merged["trajectory_match"] = merged["trajectory_type_segmented"] == merged["trajectory_type_beast"]
    comparable = merged.dropna(subset=["onset_month_diff_beast_minus_segmented"]).copy()

    summary = pd.DataFrame(
        [
            {"metric": "grid_count", "value": int(len(merged))},
            {"metric": "comparable_break_count", "value": int(len(comparable))},
            {"metric": "mean_abs_onset_month_diff", "value": float(np.nanmean(np.abs(comparable["onset_month_diff_beast_minus_segmented"])))},
            {"metric": "share_within_6_months", "value": float((np.abs(comparable["onset_month_diff_beast_minus_segmented"]) <= 6).mean()) if not comparable.empty else np.nan},
            {"metric": "trajectory_match_share", "value": float(merged["trajectory_match"].mean())},
        ]
    )
    return merged, summary


def build_window_sensitivity(trajectory_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = trajectory_df[["grid_id", "onset_date", "post_trend_cm_per_year", "trajectory_type"]].copy()
    df["onset_date"] = pd.to_datetime(df["onset_date"], errors="coerce")
    rows = []
    for scheme in ["baseline", "minus3", "plus3"]:
        scheme_labels = df.apply(
            lambda row: classify(row["onset_date"], row["post_trend_cm_per_year"], threshold, scheme),
            axis=1,
        )
        counts = scheme_labels.value_counts().to_dict()
        rows.append(
            {
                "scheme": scheme,
                "early_rebound": counts.get("early_rebound", 0),
                "main_rebound": counts.get("main_rebound", 0),
                "legacy_slowdown": counts.get("legacy_slowdown", 0),
                "stalled_or_declining": counts.get("stalled_or_declining", 0),
                "agreement_with_baseline": float((scheme_labels == df["trajectory_type"]).mean()) if scheme != "baseline" else 1.0,
            }
        )
    return pd.DataFrame(rows)


def fixed_margin_exact_test(contingency: pd.DataFrame) -> float:
    table = contingency.to_numpy(dtype=int)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError("fixed_margin_exact_test currently supports r x 2 contingency tables only.")

    row_sums = table.sum(axis=1).astype(int)
    north_total = int(table[:, 0].sum())
    total = int(table.sum())
    observed_first_col = table[:, 0].astype(int)
    observed_prob = float(
        np.prod([math.comb(int(row_sum), int(x)) for row_sum, x in zip(row_sums, observed_first_col)])
        / math.comb(total, north_total)
    )
    tolerance = 1e-18

    def recurse(row_idx: int, remaining: int, chosen: list[int]) -> float:
        if row_idx == len(row_sums) - 1:
            last = remaining
            if 0 <= last <= row_sums[row_idx]:
                prob = float(
                    np.prod(
                        [math.comb(int(row_sum), int(x)) for row_sum, x in zip(row_sums, chosen + [last])]
                    )
                    / math.comb(total, north_total)
                )
                return prob if prob <= observed_prob + tolerance else 0.0
            return 0.0

        remaining_rows = int(row_sums[row_idx + 1 :].sum())
        lower = max(0, remaining - remaining_rows)
        upper = min(int(row_sums[row_idx]), remaining)
        pvalue = 0.0
        for value in range(lower, upper + 1):
            pvalue += recurse(row_idx + 1, remaining - value, chosen + [value])
        return pvalue

    return recurse(0, north_total, [])


def build_region_type_stats(trajectory_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    contingency = pd.crosstab(trajectory_df["trajectory_type"], trajectory_df["region_tag"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    exact_p = fixed_margin_exact_test(contingency)
    summary = pd.DataFrame(
        [
            {"metric": "chi2", "value": float(chi2)},
            {"metric": "asymptotic_pvalue", "value": float(p)},
            {"metric": "dof", "value": int(dof)},
            {"metric": "fixed_margin_exact_pvalue", "value": float(exact_p)},
            {"metric": "min_expected_count", "value": float(np.min(expected))},
            {"metric": "expected_count_cells_lt5", "value": int((expected < 5).sum())},
        ]
    )
    return contingency.reset_index(), summary


def build_water_stats(trajectory_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = []
    for traj in ["early_rebound", "main_rebound", "legacy_slowdown", "stalled_or_declining"]:
        vals = trajectory_df.loc[trajectory_df["trajectory_type"] == traj, "water_change_2019_2024_minus_2015_2018"].dropna().to_numpy()
        if len(vals) > 0:
            groups.append(vals)
    kw = stats.kruskal(*groups)
    spearman = stats.spearmanr(
        trajectory_df["post_trend_cm_per_year"],
        trajectory_df["water_change_2019_2024_minus_2015_2018"],
        nan_policy="omit",
    )
    rebound = trajectory_df.loc[
        trajectory_df["trajectory_type"].isin(["early_rebound", "main_rebound"]),
        "water_change_2019_2024_minus_2015_2018",
    ].dropna().to_numpy()
    stalled = trajectory_df.loc[
        trajectory_df["trajectory_type"] == "stalled_or_declining",
        "water_change_2019_2024_minus_2015_2018",
    ].dropna().to_numpy()
    perm = permutation_mean_diff(rebound, stalled)

    summary = pd.DataFrame(
        [
            {"metric": "kruskal_statistic", "value": float(kw.statistic)},
            {"metric": "kruskal_pvalue", "value": float(kw.pvalue)},
            {"metric": "spearman_rho_posttrend_vs_waterchange", "value": float(spearman.statistic)},
            {"metric": "spearman_pvalue", "value": float(spearman.pvalue)},
            {"metric": "rebound_vs_stalled_mean_diff", "value": perm["observed_mean_diff"]},
            {"metric": "rebound_vs_stalled_permutation_pvalue", "value": perm["permutation_pvalue"]},
        ]
    )
    detail = trajectory_df[
        [
            "grid_id",
            "region_tag",
            "trajectory_type",
            "pre_trend_cm_per_year",
            "post_trend_cm_per_year",
            "water_change_2019_2024_minus_2015_2018",
        ]
    ].copy()
    return detail, summary


def build_continuous_region_tests(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    df = trajectory_df.copy()
    df["onset_date"] = pd.to_datetime(df["onset_date"], errors="coerce")
    rows: list[dict[str, float | str]] = []

    onset = df.dropna(subset=["onset_date"]).copy()
    if not onset.empty:
        onset["onset_month_index"] = month_index(onset["onset_date"])
        north = onset.loc[onset["region_tag"] == "north", "onset_month_index"].to_numpy(dtype=float)
        south = onset.loc[onset["region_tag"] == "south", "onset_month_index"].to_numpy(dtype=float)
        if len(north) and len(south):
            mw = stats.mannwhitneyu(north, south, alternative="two-sided")
            perm = permutation_mean_diff(south, north)
            rows.extend(
                [
                    {"metric": "north_median_onset_month_index", "value": float(np.median(north))},
                    {"metric": "south_median_onset_month_index", "value": float(np.median(south))},
                    {"metric": "onset_mannwhitney_pvalue", "value": float(mw.pvalue)},
                    {"metric": "south_minus_north_mean_onset_month_diff", "value": perm["observed_mean_diff"]},
                    {"metric": "onset_permutation_pvalue", "value": perm["permutation_pvalue"]},
                ]
            )

    trend = df.dropna(subset=["post_trend_cm_per_year"]).copy()
    north_trend = trend.loc[trend["region_tag"] == "north", "post_trend_cm_per_year"].to_numpy(dtype=float)
    south_trend = trend.loc[trend["region_tag"] == "south", "post_trend_cm_per_year"].to_numpy(dtype=float)
    if len(north_trend) and len(south_trend):
        mw = stats.mannwhitneyu(north_trend, south_trend, alternative="two-sided")
        perm = permutation_mean_diff(south_trend, north_trend)
        rows.extend(
            [
                {"metric": "north_median_post_trend_cm_per_year", "value": float(np.median(north_trend))},
                {"metric": "south_median_post_trend_cm_per_year", "value": float(np.median(south_trend))},
                {"metric": "posttrend_mannwhitney_pvalue", "value": float(mw.pvalue)},
                {"metric": "south_minus_north_mean_posttrend_diff", "value": perm["observed_mean_diff"]},
                {"metric": "posttrend_permutation_pvalue", "value": perm["permutation_pvalue"]},
            ]
        )

    return pd.DataFrame(rows)


def assign_spatial_blocks(df: pd.DataFrame, block_degrees: float = SPATIAL_BLOCK_DEGREES) -> pd.Series:
    block_x = np.floor((df["grid_lon"] - GRID_WEST) / block_degrees).astype(int)
    block_y = np.floor((df["grid_lat"] - GRID_SOUTH) / block_degrees).astype(int)
    return block_x.astype(str) + "_" + block_y.astype(str)


def build_breakpoint_uncertainty_summary(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    df = trajectory_df.copy()
    df["onset_date"] = pd.to_datetime(df["onset_date"], errors="coerce")
    df = df.loc[
        df["is_recovery_relevant"].fillna(False)
        & df["onset_date"].notna()
        & (df["onset_date"] >= REPORTING_RECOVERY_START)
    ].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "n_grids",
                "p10_onset",
                "p25_onset",
                "median_onset",
                "p75_onset",
                "p90_onset",
                "mean_post_trend_cm_per_year",
                "median_post_trend_cm_per_year",
            ]
        )

    df["onset_month_index"] = month_index(df["onset_date"])
    rows: list[dict[str, object]] = []
    for group_name, group_df in [
        ("all", df),
        ("north", df.loc[df["region_tag"] == "north"]),
        ("south", df.loc[df["region_tag"] == "south"]),
    ]:
        if group_df.empty:
            continue
        onset_values = group_df["onset_month_index"].to_numpy(dtype=float)
        rows.append(
            {
                "group": group_name,
                "n_grids": int(len(group_df)),
                "p10_onset": month_index_to_timestamp(np.quantile(onset_values, 0.10)),
                "p25_onset": month_index_to_timestamp(np.quantile(onset_values, 0.25)),
                "median_onset": month_index_to_timestamp(np.quantile(onset_values, 0.50)),
                "p75_onset": month_index_to_timestamp(np.quantile(onset_values, 0.75)),
                "p90_onset": month_index_to_timestamp(np.quantile(onset_values, 0.90)),
                "mean_post_trend_cm_per_year": float(group_df["post_trend_cm_per_year"].mean()),
                "median_post_trend_cm_per_year": float(group_df["post_trend_cm_per_year"].median()),
            }
        )
    return pd.DataFrame(rows)


def build_spatial_effect_summary(
    trajectory_df: pd.DataFrame,
    *,
    block_degrees: float = SPATIAL_BLOCK_DEGREES,
    bootstrap_iter: int = SPATIAL_BOOTSTRAP_ITER,
) -> pd.DataFrame:
    df = trajectory_df.copy()
    df["onset_date"] = pd.to_datetime(df["onset_date"], errors="coerce")
    df = df.loc[
        df["is_recovery_relevant"].fillna(False)
        & df["onset_date"].notna()
        & (df["onset_date"] >= REPORTING_RECOVERY_START)
        & df["post_trend_cm_per_year"].notna()
    ].copy()
    if df.empty:
        return pd.DataFrame(
            columns=["metric", "estimate", "ci_lower", "ci_upper", "north_count", "south_count", "bootstrap_n"]
        )

    df["onset_month_index"] = month_index(df["onset_date"])
    df["block_id"] = assign_spatial_blocks(df, block_degrees=block_degrees)
    north = df.loc[df["region_tag"] == "north"].copy()
    south = df.loc[df["region_tag"] == "south"].copy()
    blocks = sorted(df["block_id"].dropna().unique().tolist())
    rng = np.random.default_rng(42)
    onset_boot: list[float] = []
    post_boot: list[float] = []

    for _ in range(bootstrap_iter):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        sample = pd.concat(
            [df.loc[df["block_id"] == block_id].copy() for block_id in sampled_blocks],
            ignore_index=True,
        )
        north_sample = sample.loc[sample["region_tag"] == "north"]
        south_sample = sample.loc[sample["region_tag"] == "south"]
        if north_sample.empty or south_sample.empty:
            continue
        onset_boot.append(
            float(south_sample["onset_month_index"].mean() - north_sample["onset_month_index"].mean())
        )
        post_boot.append(
            float(south_sample["post_trend_cm_per_year"].mean() - north_sample["post_trend_cm_per_year"].mean())
        )

    onset_ci = quantile_ci(onset_boot)
    post_ci = quantile_ci(post_boot)
    return pd.DataFrame(
        [
            {
                "metric": "south_minus_north_mean_onset_month_diff",
                "estimate": float(south["onset_month_index"].mean() - north["onset_month_index"].mean()),
                "ci_lower": onset_ci[0],
                "ci_upper": onset_ci[1],
                "north_count": int(len(north)),
                "south_count": int(len(south)),
                "bootstrap_n": int(len(onset_boot)),
                "block_degrees": float(block_degrees),
            },
            {
                "metric": "south_minus_north_mean_posttrend_cm_per_year",
                "estimate": float(south["post_trend_cm_per_year"].mean() - north["post_trend_cm_per_year"].mean()),
                "ci_lower": post_ci[0],
                "ci_upper": post_ci[1],
                "north_count": int(len(north)),
                "south_count": int(len(south)),
                "bootstrap_n": int(len(post_boot)),
                "block_degrees": float(block_degrees),
            },
        ]
    )


def merge_breakpoint_meta(breakpoints: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["grid_id", "onset_date", "post_trend_cm_per_year", "is_recovery_relevant", "selected_method"]
    available = [col for col in cols if col in breakpoints.columns]
    merged = meta.merge(breakpoints[available], on="grid_id", how="left")
    merged["onset_date"] = pd.to_datetime(merged["onset_date"], errors="coerce")
    return merged


def build_baseline_sensitivity(panel: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variant, (start, end) in BASELINE_VARIANTS.items():
        rebased_panel = rebuild_proxy_from_components(panel, start, end)
        breakpoints = detect_breakpoints_with_settings(rebased_panel)
        merged = merge_breakpoint_meta(breakpoints, meta)
        frames.append(summarize_recovery_variant(merged, variant))
    return pd.concat(frames, ignore_index=True)


def build_coverage_threshold_sensitivity(trajectory_df: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    coverage = panel[["grid_id", "coverage_ratio"]].drop_duplicates("grid_id")
    merged = trajectory_df.merge(coverage, on="grid_id", how="left")
    frames: list[pd.DataFrame] = []
    for threshold in COVERAGE_THRESHOLDS:
        subset = merged.loc[merged["coverage_ratio"] >= threshold].copy()
        frames.append(summarize_recovery_variant(subset, f"coverage_ge_{threshold:.2f}"))
    return pd.concat(frames, ignore_index=True)


def build_breakpoint_rule_sensitivity(panel: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for variant, params in BREAKPOINT_RULE_VARIANTS.items():
        breakpoints = detect_breakpoints_with_settings(
            panel,
            beast_prob=params["beast_prob"],
            bic_delta=params["bic_delta"],
            min_segment_months=params["min_segment_months"],
        )
        merged = merge_breakpoint_meta(breakpoints, meta)
        frames.append(summarize_recovery_variant(merged, variant))
    return pd.concat(frames, ignore_index=True)


def build_spatial_block_sensitivity(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for block_size in SPATIAL_BLOCK_VARIANTS:
        summary = build_spatial_effect_summary(
            trajectory_df,
            block_degrees=block_size,
            bootstrap_iter=max(1000, SPATIAL_BOOTSTRAP_ITER // 2),
        ).copy()
        summary["variant"] = f"block_{block_size:.1f}deg"
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def build_proxy_robustness(
    panel: pd.DataFrame,
    baseline_traj: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_cols = ["grid_id", "grid_lon", "grid_lat", "region_tag"]
    meta = baseline_traj[meta_cols].drop_duplicates("grid_id")
    frames = [summarize_recovery_variant(baseline_traj, "baseline_proxy")]
    detail_rows = []

    for variant in ["contemporaneous", "lag03"]:
        adjusted_panel, _ = build_adjusted_panel(panel, variant)
        adjusted_breakpoints = detect_breakpoints_with_settings(adjusted_panel)
        adjusted_breakpoints["onset_date"] = pd.to_datetime(adjusted_breakpoints["onset_date"], errors="coerce")
        adjusted_breakpoints["trajectory_type"] = adjusted_breakpoints.apply(
            lambda row: classify(row["onset_date"], row["post_trend_cm_per_year"], threshold),
            axis=1,
        )
        merged = merge_breakpoint_meta(adjusted_breakpoints, meta)
        merged["variant"] = variant
        detail_rows.append(merged)
        frames.append(summarize_recovery_variant(merged, f"water_adjusted_{variant}"))

    return (
        pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame(),
        pd.concat(frames, ignore_index=True),
    )


def build_validation_series(panel: pd.DataFrame, trajectory_df: pd.DataFrame, wells: pd.DataFrame) -> pd.DataFrame:
    metadata = trajectory_df[["grid_id", "region_tag"]].drop_duplicates("grid_id")
    remote = panel.copy()
    if "region_tag" not in remote.columns:
        remote = remote.merge(metadata, on="grid_id", how="left")
    remote_frames = [
        remote.groupby("date", as_index=False)["gwa_cm"].mean().assign(region_tag="all"),
        remote.loc[remote["region_tag"] == "north"].groupby("date", as_index=False)["gwa_cm"].mean().assign(region_tag="north"),
        remote.loc[remote["region_tag"] == "south"].groupby("date", as_index=False)["gwa_cm"].mean().assign(region_tag="south"),
    ]
    remote_region = pd.concat(remote_frames, ignore_index=True)

    well_grid = (
        wells.merge(metadata, on="grid_id", how="left")
        .groupby(["grid_id", "date", "region_tag"], as_index=False)["well_anomaly"]
        .mean()
        .sort_values(["grid_id", "date"])
    )
    well_frames = [
        well_grid.groupby("date", as_index=False)["well_anomaly"].mean().assign(region_tag="all"),
        well_grid.loc[well_grid["region_tag"] == "north"].groupby("date", as_index=False)["well_anomaly"].mean().assign(region_tag="north"),
        well_grid.loc[well_grid["region_tag"] == "south"].groupby("date", as_index=False)["well_anomaly"].mean().assign(region_tag="south"),
    ]
    well_region = pd.concat(well_frames, ignore_index=True)
    return remote_region.merge(well_region, on=["date", "region_tag"], how="inner").sort_values(["region_tag", "date"])


def moving_block_indices(n_obs: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    starts = np.arange(max(1, n_obs - block_len + 1))
    chosen: list[np.ndarray] = []
    while sum(len(block) for block in chosen) < n_obs:
        start = int(rng.choice(starts))
        stop = min(start + block_len, n_obs)
        chosen.append(np.arange(start, stop))
    return np.concatenate(chosen)[:n_obs]


def pearson_or_nan(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def theilsen_from_array(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    clean = values[mask]
    if len(clean) < 3:
        return float("nan")
    x = np.arange(len(clean), dtype=float)
    slope, _, _, _ = stats.theilslopes(clean.astype(float), x, 0.95)
    return float(slope * 12.0)


def best_break_index(values: np.ndarray, idx_min: int | None = None, idx_max: int | None = None) -> int | None:
    if len(values) < MIN_BREAK_SEGMENT_MONTHS * 2:
        return None
    x = np.arange(len(values), dtype=float)
    best_idx = None
    best_bic = None
    if not np.all(np.isfinite(values)):
        return None
    null_fit = stats.linregress(x, values)
    null_resid = values - (null_fit.intercept + null_fit.slope * x)
    bic_null = len(values) * np.log(max(float(np.sum(null_resid**2)) / len(values), 1e-12)) + 2 * np.log(len(values))

    start_idx = max(MIN_BREAK_SEGMENT_MONTHS, idx_min if idx_min is not None else MIN_BREAK_SEGMENT_MONTHS)
    end_idx = min(len(values) - MIN_BREAK_SEGMENT_MONTHS, idx_max if idx_max is not None else len(values) - MIN_BREAK_SEGMENT_MONTHS)
    for idx in range(start_idx, end_idx + 1):
        left = slice(0, idx)
        right = slice(idx, len(values))
        left_fit = stats.linregress(x[left], values[left])
        right_fit = stats.linregress(x[right], values[right])
        left_resid = values[left] - (left_fit.intercept + left_fit.slope * x[left])
        right_resid = values[right] - (right_fit.intercept + right_fit.slope * x[right])
        rss = float(np.sum(left_resid**2) + np.sum(right_resid**2))
        bic_single = len(values) * np.log(max(rss / len(values), 1e-12)) + 5 * np.log(len(values))
        if best_bic is None or bic_single < best_bic:
            best_bic = bic_single
            best_idx = idx
    if best_bic is None or bic_null - best_bic <= 2.0:
        return None
    return best_idx


def load_well_data(path: str | None, query: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    raw = pd.read_csv(path)
    if query:
        raw = raw.query(query).copy()
    return standardize_well_data(
        raw,
        dataset_label=f"revision well data `{path}`",
        value_kind="level",
        value_col="gw_level",
    )


def build_validation_uncertainty_summary(
    panel: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    wells: pd.DataFrame | None,
) -> pd.DataFrame:
    if wells is None or wells.empty:
        return pd.DataFrame(
            columns=[
                "region_tag",
                "pearson_all_estimate",
                "pearson_all_ci_lower",
                "pearson_all_ci_upper",
                "pearson_post2019_estimate",
                "pearson_post2019_ci_lower",
                "pearson_post2019_ci_upper",
                "remote_slope_estimate",
                "remote_slope_ci_lower",
                "remote_slope_ci_upper",
                "well_slope_estimate",
                "well_slope_ci_lower",
                "well_slope_ci_upper",
                "remote_break_estimate",
                "remote_break_window_start",
                "remote_break_window_end",
                "well_break_estimate",
                "well_break_window_start",
                "well_break_window_end",
            ]
        )

    merged = build_validation_series(panel, trajectory_df, wells)
    rng = np.random.default_rng(42)
    rows: list[dict[str, object]] = []

    for region in ["all", "north", "south"]:
        sub = merged.loc[merged["region_tag"] == region].copy().sort_values("date")
        if sub.empty:
            continue
        gwa = sub["gwa_cm"].to_numpy(dtype=float)
        well = sub["well_anomaly"].to_numpy(dtype=float)
        post_mask = (sub["date"] >= EVENT_START).to_numpy()
        gwa_post = gwa[post_mask]
        well_post = well[post_mask]
        date_values = sub["date"].to_numpy(dtype="datetime64[ns]")
        candidate_start_idx = int(np.searchsorted(date_values, np.datetime64("2015-01-01"), side="left"))
        candidate_end_idx = int(np.searchsorted(date_values, np.datetime64("2023-09-01"), side="right") - 1)

        pearson_all_boot: list[float] = []
        pearson_post_boot: list[float] = []
        remote_slope_boot: list[float] = []
        well_slope_boot: list[float] = []
        remote_break_boot: list[float] = []
        well_break_boot: list[float] = []

        for _ in range(VALIDATION_BOOTSTRAP_ITER):
            full_idx = moving_block_indices(len(sub), VALIDATION_BLOCK_MONTHS, rng)
            post_idx = moving_block_indices(len(gwa_post), VALIDATION_BLOCK_MONTHS, rng)
            gwa_full_boot = gwa[full_idx]
            well_full_boot = well[full_idx]
            gwa_post_boot = gwa_post[post_idx]
            well_post_boot = well_post[post_idx]

            pearson_all_boot.append(pearson_or_nan(gwa_full_boot, well_full_boot))
            pearson_post_boot.append(pearson_or_nan(gwa_post_boot, well_post_boot))
            remote_slope_boot.append(theilsen_from_array(gwa_post_boot))
            well_slope_boot.append(theilsen_from_array(well_post_boot))

            remote_idx = best_break_index(gwa_full_boot, candidate_start_idx, candidate_end_idx)
            well_idx = best_break_index(well_full_boot, candidate_start_idx, candidate_end_idx)
            if remote_idx is not None:
                remote_break_boot.append(float(remote_idx))
            if well_idx is not None:
                well_break_boot.append(float(well_idx))

        pearson_all_ci = quantile_ci(pearson_all_boot)
        pearson_post_ci = quantile_ci(pearson_post_boot)
        remote_slope_ci = quantile_ci(remote_slope_boot)
        well_slope_ci = quantile_ci(well_slope_boot)
        remote_break_iqr = interquartile_window(remote_break_boot)
        well_break_iqr = interquartile_window(well_break_boot)

        remote_break_idx = best_break_index(gwa, candidate_start_idx, candidate_end_idx)
        well_break_idx = best_break_index(well, candidate_start_idx, candidate_end_idx)
        rows.append(
            {
                "region_tag": region,
                "pearson_all_estimate": pearson_or_nan(gwa, well),
                "pearson_all_ci_lower": pearson_all_ci[0],
                "pearson_all_ci_upper": pearson_all_ci[1],
                "pearson_post2019_estimate": pearson_or_nan(gwa_post, well_post),
                "pearson_post2019_ci_lower": pearson_post_ci[0],
                "pearson_post2019_ci_upper": pearson_post_ci[1],
                "remote_slope_estimate": theilsen_from_array(gwa_post),
                "remote_slope_ci_lower": remote_slope_ci[0],
                "remote_slope_ci_upper": remote_slope_ci[1],
                "well_slope_estimate": theilsen_from_array(well_post),
                "well_slope_ci_lower": well_slope_ci[0],
                "well_slope_ci_upper": well_slope_ci[1],
                "remote_break_estimate": month_index_to_timestamp(month_index(sub.iloc[[remote_break_idx]]["date"]).iloc[0])
                if remote_break_idx is not None
                else None,
                "remote_break_window_start": month_index_to_timestamp(
                    month_index(sub.iloc[[0]]["date"]).iloc[0] + remote_break_iqr[0]
                )
                if remote_break_iqr[0] is not None
                else None,
                "remote_break_window_end": month_index_to_timestamp(
                    month_index(sub.iloc[[0]]["date"]).iloc[0] + remote_break_iqr[1]
                )
                if remote_break_iqr[1] is not None
                else None,
                "well_break_estimate": month_index_to_timestamp(month_index(sub.iloc[[well_break_idx]]["date"]).iloc[0])
                if well_break_idx is not None
                else None,
                "well_break_window_start": month_index_to_timestamp(
                    month_index(sub.iloc[[0]]["date"]).iloc[0] + well_break_iqr[0]
                )
                if well_break_iqr[0] is not None
                else None,
                "well_break_window_end": month_index_to_timestamp(
                    month_index(sub.iloc[[0]]["date"]).iloc[0] + well_break_iqr[1]
                )
                if well_break_iqr[1] is not None
                else None,
            }
        )
    return pd.DataFrame(rows)


def build_validation_metric_robustness(
    panel: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    wells: pd.DataFrame | None,
) -> pd.DataFrame:
    if wells is None or wells.empty:
        return pd.DataFrame(
            columns=[
                "region_tag",
                "spearman_all_estimate",
                "spearman_all_ci_lower",
                "spearman_all_ci_upper",
                "spearman_post2019_estimate",
                "spearman_post2019_ci_lower",
                "spearman_post2019_ci_upper",
                "sign_agreement_all_estimate",
                "sign_agreement_all_ci_lower",
                "sign_agreement_all_ci_upper",
                "sign_agreement_post2019_estimate",
                "sign_agreement_post2019_ci_lower",
                "sign_agreement_post2019_ci_upper",
            ]
        )

    merged = build_validation_series(panel, trajectory_df, wells)
    rng = np.random.default_rng(314)
    rows: list[dict[str, object]] = []
    for region in ["all", "north", "south"]:
        sub = merged.loc[merged["region_tag"] == region].copy().sort_values("date")
        if sub.empty:
            continue
        gwa = sub["gwa_cm"].to_numpy(dtype=float)
        well = sub["well_anomaly"].to_numpy(dtype=float)
        post_mask = (sub["date"] >= EVENT_START).to_numpy()
        gwa_post = gwa[post_mask]
        well_post = well[post_mask]

        spearman_all_boot: list[float] = []
        spearman_post_boot: list[float] = []
        sign_all_boot: list[float] = []
        sign_post_boot: list[float] = []
        for _ in range(VALIDATION_BOOTSTRAP_ITER):
            full_idx = moving_block_indices(len(sub), VALIDATION_BLOCK_MONTHS, rng)
            post_idx = moving_block_indices(len(gwa_post), VALIDATION_BLOCK_MONTHS, rng)
            gwa_full_boot = gwa[full_idx]
            well_full_boot = well[full_idx]
            gwa_post_boot = gwa_post[post_idx]
            well_post_boot = well_post[post_idx]
            spearman_all_boot.append(float(stats.spearmanr(gwa_full_boot, well_full_boot, nan_policy="omit").statistic))
            spearman_post_boot.append(float(stats.spearmanr(gwa_post_boot, well_post_boot, nan_policy="omit").statistic))
            sign_all_boot.append(sign_agreement(gwa_full_boot, well_full_boot))
            sign_post_boot.append(sign_agreement(gwa_post_boot, well_post_boot))

        rows.append(
            {
                "region_tag": region,
                "spearman_all_estimate": float(stats.spearmanr(gwa, well, nan_policy="omit").statistic),
                "spearman_all_ci_lower": quantile_ci(spearman_all_boot)[0],
                "spearman_all_ci_upper": quantile_ci(spearman_all_boot)[1],
                "spearman_post2019_estimate": float(stats.spearmanr(gwa_post, well_post, nan_policy="omit").statistic),
                "spearman_post2019_ci_lower": quantile_ci(spearman_post_boot)[0],
                "spearman_post2019_ci_upper": quantile_ci(spearman_post_boot)[1],
                "sign_agreement_all_estimate": sign_agreement(gwa, well),
                "sign_agreement_all_ci_lower": quantile_ci(sign_all_boot)[0],
                "sign_agreement_all_ci_upper": quantile_ci(sign_all_boot)[1],
                "sign_agreement_post2019_estimate": sign_agreement(gwa_post, well_post),
                "sign_agreement_post2019_ci_lower": quantile_ci(sign_post_boot)[0],
                "sign_agreement_post2019_ci_upper": quantile_ci(sign_post_boot)[1],
            }
        )
    return pd.DataFrame(rows)


def build_coarse_recovery_summary(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    df = trajectory_df.copy()
    df["coarse_group"] = np.where(
        df["trajectory_type"].isin(["early_rebound", "main_rebound"]),
        "rebound",
        "non_rebound",
    )
    contingency = pd.crosstab(df["coarse_group"], df["region_tag"])
    if contingency.shape == (2, 2):
        odds_ratio, fisher_p = stats.fisher_exact(contingency.to_numpy(dtype=int))
    else:
        odds_ratio, fisher_p = np.nan, np.nan
    rows = [
        {"metric": "north_rebound_count", "value": float(contingency.get("north", pd.Series()).get("rebound", 0))},
        {"metric": "south_rebound_count", "value": float(contingency.get("south", pd.Series()).get("rebound", 0))},
        {"metric": "north_non_rebound_count", "value": float(contingency.get("north", pd.Series()).get("non_rebound", 0))},
        {"metric": "south_non_rebound_count", "value": float(contingency.get("south", pd.Series()).get("non_rebound", 0))},
        {"metric": "fisher_odds_ratio", "value": float(odds_ratio) if np.isfinite(odds_ratio) else np.nan},
        {"metric": "fisher_pvalue", "value": float(fisher_p) if np.isfinite(fisher_p) else np.nan},
    ]
    return pd.DataFrame(rows)


def build_validation_scope_note(well_summary: pd.DataFrame) -> str:
    region_rows = well_summary.loc[well_summary["validation_level"] == "region_timeseries"].copy()
    if region_rows.empty:
        return "# Validation Scope Note\n\nNo usable external well-validation rows were generated.\n"

    overlap = int(region_rows["overlap_months"].max())
    start_date = pd.to_datetime(region_rows["overlap_start_date"], errors="coerce").min()
    end_date = pd.to_datetime(region_rows["overlap_end_date"], errors="coerce").max()
    covers_post_2019 = pd.notna(end_date) and end_date >= EVENT_START

    lines = [
        "# Validation Scope Note",
        "",
        f"- External well validation overlap: {start_date.strftime('%Y-%m') if pd.notna(start_date) else 'NA'} to {end_date.strftime('%Y-%m') if pd.notna(end_date) else 'NA'}",
        f"- Maximum overlapping months: {overlap}",
    ]
    if covers_post_2019:
        lines.extend(
            [
                "- The public well data now cover the post-2019 period and can be used to check regional trend direction and the approximate recovery window.",
                "- Even with this extension, the well data primarily support region- and subregion-scale recovery timing, not grid-wise confirmation of the four exploratory trajectory classes.",
            ]
        )
    else:
        lines.extend(
            [
                "- The public well validation still ends before 2019 and therefore only supports pre-recovery consistency checks.",
                "- It cannot directly confirm the post-2019 recovery trajectories.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_water_overlap_stats(overlap_df: pd.DataFrame | None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if overlap_df is None or overlap_df.empty:
        return None, None
    overlap = overlap_df.dropna(subset=["jrc_water_frac", "landsat_water_frac"]).copy()
    corr = np.corrcoef(overlap["jrc_water_frac"], overlap["landsat_water_frac"])[0, 1] if len(overlap) >= 3 else np.nan
    overlap["diff"] = overlap["landsat_water_frac"] - overlap["jrc_water_frac"]
    grid_summary = (
        overlap.groupby("grid_id", as_index=False)
        .agg(
            overlap_months=("date", "nunique"),
            mean_jrc=("jrc_water_frac", "mean"),
            mean_landsat=("landsat_water_frac", "mean"),
            mean_abs_diff=("diff", lambda x: float(np.mean(np.abs(x)))),
        )
    )
    summary = pd.DataFrame(
        [
            {"metric": "overlap_rows", "value": int(len(overlap))},
            {"metric": "overlap_grid_count", "value": int(overlap["grid_id"].nunique())},
            {"metric": "jrc_landsat_pearson_r", "value": float(corr) if np.isfinite(corr) else np.nan},
            {"metric": "mean_abs_diff", "value": float(np.mean(np.abs(overlap["diff"])))},
            {"metric": "share_abs_diff_gt_0.10", "value": float((np.abs(overlap["diff"]) > 0.10).mean())},
        ]
    )
    return grid_summary, summary


def build_adjusted_panel(panel: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.Series]:
    adjusted = panel.copy()
    adjusted["month"] = adjusted["date"].dt.month
    term_cols = ["water_frac"]
    if variant == "lag03":
        grouped = adjusted.groupby("grid_id", group_keys=False)
        for lag in range(1, 4):
            col = f"water_frac_lag{lag}"
            adjusted[col] = grouped["water_frac"].shift(lag)
            term_cols.append(col)

    centered_term_cols: list[str] = []
    for term in term_cols:
        centered_col = f"{term}_centered"
        adjusted[centered_col] = adjusted[term] - adjusted.groupby("grid_id")[term].transform("mean")
        centered_term_cols.append(centered_col)

    fit_df = adjusted.dropna(subset=["gwa_cm", *centered_term_cols]).copy()
    formula = "gwa_cm ~ " + " + ".join(centered_term_cols) + " + C(grid_id) + C(month)"
    model = smf.ols(formula, data=fit_df).fit()

    water_component = pd.Series(0.0, index=adjusted.index)
    for term, centered_col in zip(term_cols, centered_term_cols):
        water_component += model.params.get(centered_col, 0.0) * adjusted[centered_col].fillna(0.0)
    adjusted["gwa_cm"] = adjusted["gwa_cm"] - water_component
    return adjusted[["grid_id", "date", "gwa_cm"]].dropna(subset=["gwa_cm"]).copy(), model.params


def summarize_trajectory_comparison(baseline: pd.DataFrame, adjusted: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = baseline[["grid_id", "region_tag", "trajectory_type", "onset_date", "post_trend_cm_per_year", "water_change_2019_2024_minus_2015_2018"]].copy()
    comp = adjusted[["grid_id", "onset_date", "post_trend_cm_per_year", "trajectory_type"]].copy()
    base["onset_date"] = pd.to_datetime(base["onset_date"], errors="coerce")
    comp["onset_date"] = pd.to_datetime(comp["onset_date"], errors="coerce")

    merged = base.merge(comp, on="grid_id", how="inner", suffixes=("_baseline", "_adjusted"))
    merged["variant"] = variant
    month_diff = (
        (merged["onset_date_adjusted"].dt.year - merged["onset_date_baseline"].dt.year) * 12
        + (merged["onset_date_adjusted"].dt.month - merged["onset_date_baseline"].dt.month)
    )
    merged["onset_month_diff_adjusted_minus_baseline"] = month_diff
    merged["trajectory_match"] = merged["trajectory_type_baseline"] == merged["trajectory_type_adjusted"]

    counts = adjusted["trajectory_type"].value_counts().to_dict()
    spearman = stats.spearmanr(
        merged["post_trend_cm_per_year_adjusted"],
        merged["water_change_2019_2024_minus_2015_2018"],
        nan_policy="omit",
    )
    summary_rows = [
        {"variant": variant, "metric": "mean_abs_onset_month_diff", "value": float(np.nanmean(np.abs(month_diff)))},
        {"variant": variant, "metric": "trajectory_match_share", "value": float(merged["trajectory_match"].mean())},
        {"variant": variant, "metric": "spearman_rho_adjusted_posttrend_vs_waterchange", "value": float(spearman.statistic)},
        {"variant": variant, "metric": "spearman_pvalue_adjusted_posttrend_vs_waterchange", "value": float(spearman.pvalue)},
        {"variant": variant, "metric": "early_rebound_count", "value": float(counts.get("early_rebound", 0))},
        {"variant": variant, "metric": "main_rebound_count", "value": float(counts.get("main_rebound", 0))},
        {"variant": variant, "metric": "legacy_slowdown_count", "value": float(counts.get("legacy_slowdown", 0))},
        {"variant": variant, "metric": "stalled_or_declining_count", "value": float(counts.get("stalled_or_declining", 0))},
    ]
    return merged, pd.DataFrame(summary_rows)


def build_surfacewater_sensitivity(panel: pd.DataFrame, baseline_traj: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_cols = [
        "grid_id",
        "grid_lon",
        "grid_lat",
        "region_tag",
        "water_frac_pre_2015_2018",
        "water_frac_post_2019_2024",
        "water_change_2019_2024_minus_2015_2018",
    ]
    meta = baseline_traj[meta_cols].drop_duplicates("grid_id")
    detail_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for variant in ["contemporaneous", "lag03"]:
        adjusted_panel, params = build_adjusted_panel(panel, variant)
        adjusted_breakpoints = detect_breakpoints_with_settings(adjusted_panel)
        adjusted_breakpoints["onset_date"] = pd.to_datetime(adjusted_breakpoints["onset_date"], errors="coerce")
        adjusted_breakpoints["trajectory_type"] = adjusted_breakpoints.apply(
            lambda row: classify(row["onset_date"], row["post_trend_cm_per_year"], threshold),
            axis=1,
        )
        adjusted_traj = meta.merge(adjusted_breakpoints, on="grid_id", how="left")
        comparison_detail, comparison_summary = summarize_trajectory_comparison(baseline_traj, adjusted_traj, variant)
        detail_frames.append(comparison_detail)
        summary_frames.append(comparison_summary)
        for key, value in params.items():
            if "water_frac" in key:
                summary_frames.append(
                    pd.DataFrame(
                        [
                            {
                                "variant": variant,
                                "metric": f"coef_{key}",
                                "value": float(value),
                            }
                        ]
                    )
                )

    return pd.concat(detail_frames, ignore_index=True), pd.concat(summary_frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    post_dir = Path(args.postprocess_dir)
    main_dir = Path(args.main_dir)
    panel_path = Path(args.panel)
    beast_dir = Path(args.beast_main_dir) if args.beast_main_dir else None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = pd.read_csv(post_dir / "grid_trajectory_types.csv")
    panel = pd.read_csv(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    analysis_panel = pd.read_csv(main_dir / "grid_counterfactual.csv")
    analysis_panel["date"] = pd.to_datetime(analysis_panel["date"])
    main_breakpoints = pd.read_csv(main_dir / "grid_breakpoints.csv")
    beast_breakpoints = (
        pd.read_csv(beast_dir / "grid_breakpoints.csv")
        if beast_dir is not None and (beast_dir / "grid_breakpoints.csv").exists()
        else None
    )
    well_summary = pd.read_csv(post_dir / "well_validation_summary.csv")
    overlap_df = pd.read_csv(args.water_overlap) if args.water_overlap and Path(args.water_overlap).exists() else None
    wells = load_well_data(args.well_data, args.well_query)

    threshold = 1.0
    meta = traj[["grid_id", "grid_lon", "grid_lat", "region_tag"]].drop_duplicates("grid_id")
    seg, beast = method_breakpoint_views(main_breakpoints, beast_breakpoints)
    method_detail, method_summary = build_method_comparison(seg, beast, threshold)
    window_sensitivity = build_window_sensitivity(traj, threshold)
    region_table, region_summary = build_region_type_stats(traj)
    water_detail, water_summary = build_water_stats(traj)
    continuous_region = build_continuous_region_tests(traj)
    breakpoint_uncertainty = build_breakpoint_uncertainty_summary(traj)
    spatial_effect = build_spatial_effect_summary(traj)
    validation_uncertainty = build_validation_uncertainty_summary(panel, traj, wells)
    validation_metric_robustness = build_validation_metric_robustness(panel, traj, wells)
    coarse_recovery = build_coarse_recovery_summary(traj)
    sw_detail, sw_summary = build_surfacewater_sensitivity(panel, traj, threshold)
    overlap_grid, overlap_summary = build_water_overlap_stats(overlap_df)
    if overlap_summary is None and (out_dir / "water_overlap_stats.csv").exists():
        overlap_summary = pd.read_csv(out_dir / "water_overlap_stats.csv")
    if overlap_grid is None and (out_dir / "water_overlap_grid_summary.csv").exists():
        overlap_grid = pd.read_csv(out_dir / "water_overlap_grid_summary.csv")
    baseline_sensitivity = build_baseline_sensitivity(analysis_panel, meta)
    coverage_sensitivity = build_coverage_threshold_sensitivity(traj, panel)
    breakpoint_rule_sensitivity = build_breakpoint_rule_sensitivity(analysis_panel, meta)
    spatial_block_sensitivity = build_spatial_block_sensitivity(traj)
    proxy_detail, proxy_summary = build_proxy_robustness(panel, traj, threshold)

    method_detail.to_csv(out_dir / "trajectory_method_comparison.csv", index=False)
    method_summary.to_csv(out_dir / "trajectory_method_summary.csv", index=False)
    window_sensitivity.to_csv(out_dir / "trajectory_window_sensitivity.csv", index=False)
    region_table.to_csv(out_dir / "trajectory_region_contingency.csv", index=False)
    region_summary.to_csv(out_dir / "trajectory_region_stats.csv", index=False)
    water_detail.to_csv(out_dir / "water_coupling_detail.csv", index=False)
    water_summary.to_csv(out_dir / "water_coupling_stats.csv", index=False)
    continuous_region.to_csv(out_dir / "continuous_region_tests.csv", index=False)
    breakpoint_uncertainty.to_csv(out_dir / "breakpoint_uncertainty_summary.csv", index=False)
    spatial_effect.to_csv(out_dir / "spatial_effect_summary.csv", index=False)
    validation_uncertainty.to_csv(out_dir / "validation_uncertainty_summary.csv", index=False)
    validation_metric_robustness.to_csv(out_dir / "validation_metric_robustness.csv", index=False)
    coarse_recovery.to_csv(out_dir / "coarse_recovery_summary.csv", index=False)
    sw_detail.to_csv(out_dir / "surfacewater_sensitivity_detail.csv", index=False)
    sw_summary.to_csv(out_dir / "surfacewater_sensitivity_summary.csv", index=False)
    baseline_sensitivity.to_csv(out_dir / "baseline_sensitivity_summary.csv", index=False)
    coverage_sensitivity.to_csv(out_dir / "coverage_threshold_sensitivity.csv", index=False)
    breakpoint_rule_sensitivity.to_csv(out_dir / "breakpoint_rule_sensitivity.csv", index=False)
    spatial_block_sensitivity.to_csv(out_dir / "spatial_block_sensitivity.csv", index=False)
    proxy_detail.to_csv(out_dir / "proxy_robustness_detail.csv", index=False)
    proxy_summary.to_csv(out_dir / "proxy_robustness_summary.csv", index=False)
    if overlap_grid is not None:
        overlap_grid.to_csv(out_dir / "water_overlap_grid_summary.csv", index=False)
    if overlap_summary is not None:
        overlap_summary.to_csv(out_dir / "water_overlap_stats.csv", index=False)
    (out_dir / "validation_scope_note.md").write_text(
        build_validation_scope_note(well_summary),
        encoding="utf-8",
    )

    supplement_cols = [
        "grid_id",
        "grid_lon",
        "grid_lat",
        "region_tag",
        "onset_date",
        "selected_method",
        "selected_reason",
        "is_recovery_relevant",
        "segmented_model_type",
        "segmented_bic_delta",
        "beast_onset_ci_start",
        "beast_onset_ci_end",
        "post_trend_cm_per_year",
        "water_change_2019_2024_minus_2015_2018",
        "trajectory_type",
    ]
    traj[supplement_cols].to_csv(out_dir / "supplement_grid_attributes.csv", index=False)

    summary = {
        "method_agreement": method_summary.set_index("metric")["value"].to_dict(),
        "water_stats": water_summary.set_index("metric")["value"].to_dict(),
        "region_stats": region_summary.set_index("metric")["value"].to_dict(),
        "continuous_region": continuous_region.set_index("metric")["value"].to_dict() if not continuous_region.empty else {},
        "spatial_effect": spatial_effect.set_index("metric")["estimate"].to_dict() if not spatial_effect.empty else {},
        "coarse_recovery": coarse_recovery.set_index("metric")["value"].to_dict() if not coarse_recovery.empty else {},
    }
    if not breakpoint_uncertainty.empty:
        summary["breakpoint_uncertainty"] = breakpoint_uncertainty.to_dict(orient="records")
    if not validation_uncertainty.empty:
        summary["validation_uncertainty"] = validation_uncertainty.to_dict(orient="records")
    if not validation_metric_robustness.empty:
        summary["validation_metric_robustness"] = validation_metric_robustness.to_dict(orient="records")
    if not sw_summary.empty:
        for variant, group in sw_summary.groupby("variant"):
            summary[f"surfacewater_{variant}"] = group.set_index("metric")["value"].to_dict()
    if not baseline_sensitivity.empty:
        summary["baseline_sensitivity"] = baseline_sensitivity.to_dict(orient="records")
    if not coverage_sensitivity.empty:
        summary["coverage_threshold_sensitivity"] = coverage_sensitivity.to_dict(orient="records")
    if not breakpoint_rule_sensitivity.empty:
        summary["breakpoint_rule_sensitivity"] = breakpoint_rule_sensitivity.to_dict(orient="records")
    if not spatial_block_sensitivity.empty:
        summary["spatial_block_sensitivity"] = spatial_block_sensitivity.to_dict(orient="records")
    if not proxy_summary.empty:
        summary["proxy_robustness"] = proxy_summary.to_dict(orient="records")
    if overlap_summary is not None:
        summary["water_overlap"] = overlap_summary.set_index("metric")["value"].to_dict()
    (out_dir / "revision_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
