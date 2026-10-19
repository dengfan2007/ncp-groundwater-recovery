from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.tsa.seasonal import STL

try:
    import Rbeast as rb
except Exception as exc:  # pragma: no cover
    rb = None
    RBEAST_IMPORT_ERROR = exc
else:
    RBEAST_IMPORT_ERROR = None


EVENT_START = pd.Timestamp("2019-01-01")
RECOVERY_LABEL_MIN_DATE = pd.Timestamp("2015-01-01")
BEAST_SEED = 42
MIN_SEGMENT_MONTHS = 18
BEAST_PROB_THRESHOLD = 0.50
SEGMENTED_BIC_DELTA_THRESHOLD = 2.0
RECOVERY_SLOPE_MARGIN = 0.5


@dataclass
class BreakpointResult:
    grid_id: str
    onset_date: str | None
    onset_source: str
    selected_method: str
    selected_reason: str
    breakpoint_confidence: float | None
    pre_trend_cm_per_year: float | None
    post_trend_cm_per_year: float | None
    is_recovery_relevant: bool
    segmented_onset_date: str | None
    segmented_pre_trend_cm_per_year: float | None
    segmented_post_trend_cm_per_year: float | None
    segmented_model_type: str
    segmented_bic_delta: float | None
    beast_onset_date: str | None
    beast_onset_ci_start: str | None
    beast_onset_ci_end: str | None
    beast_breakpoint_confidence: float | None
    beast_pre_trend_cm_per_year: float | None
    beast_post_trend_cm_per_year: float | None


@dataclass
class SegmentedResult:
    candidate_onset_date: pd.Timestamp | None
    break_idx: int | None
    pre_trend_cm_per_year: float | None
    post_trend_cm_per_year: float | None
    model_type: str
    bic_delta: float | None


@dataclass
class BeastCandidate:
    onset_date: pd.Timestamp | None
    break_idx: int | None
    confidence: float | None
    ci_start: pd.Timestamp | None
    ci_end: pd.Timestamp | None
    pre_trend_cm_per_year: float | None
    post_trend_cm_per_year: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the North China Plain grid panel exported from GEE."
    )
    parser.add_argument("--panel", required=True, help="Path to grid_monthly_panel.csv")
    parser.add_argument("--case", help="Path to hutuo_case_buffers_raw.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument(
        "--treated-quantile",
        type=float,
        default=0.30,
        help="Upper/lower quantile share for treated and control groups.",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def require_rbeast() -> None:
    if rb is None:
        raise RuntimeError(
            "Rbeast is required for breakpoint analysis. "
            "Install it with `python -m pip install -r analysis/requirements.txt`."
        ) from RBEAST_IMPORT_ERROR


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
    panel["year"] = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month
    panel["date_str"] = panel["date"].dt.strftime("%Y-%m-%d")
    return panel


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    grouped = panel.groupby("grid_id", group_keys=False)
    for lag in range(1, 7):
        panel[f"precip_lag{lag}"] = grouped["precip_mm"].shift(lag)
    panel["smsa_lag1"] = grouped["smsa_cm"].shift(1)
    panel["et_lag1"] = grouped["et_mm"].shift(1)

    panel["precip_0_2"] = panel[["precip_mm", "precip_lag1", "precip_lag2"]].sum(axis=1, min_count=3)
    panel["precip_3_6"] = panel[
        ["precip_lag3", "precip_lag4", "precip_lag5", "precip_lag6"]
    ].sum(axis=1, min_count=4)
    panel["smsa_0_1"] = panel[["smsa_cm", "smsa_lag1"]].mean(axis=1)
    panel["et_0_1"] = panel[["et_mm", "et_lag1"]].mean(axis=1)
    return panel


def blocked_year_cv(train_df: pd.DataFrame, formula: str) -> dict[str, float]:
    years = sorted(train_df["year"].unique())
    fold_scores: list[float] = []
    for year in years:
        fit_df = train_df.loc[train_df["year"] != year].copy()
        val_df = train_df.loc[train_df["year"] == year].copy()
        if fit_df.empty or val_df.empty:
            continue
        model = smf.ols(formula, data=fit_df).fit()
        pred = model.predict(val_df)
        corr = np.corrcoef(pred, val_df["gwa_cm"])[0, 1]
        if np.isfinite(corr):
            fold_scores.append(float(corr))
    return {
        "cv_year_count": len(fold_scores),
        "cv_mean_correlation": float(np.mean(fold_scores)) if fold_scores else float("nan"),
    }


def fit_counterfactual(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    formula = (
        "gwa_cm ~ precip_0_2 + precip_3_6 + smsa_0_1 + et_0_1 + "
        "C(grid_id) + C(month)"
    )
    model_df = panel.dropna(
        subset=["gwa_cm", "precip_0_2", "precip_3_6", "smsa_0_1", "et_0_1"]
    ).copy()
    train_df = model_df.loc[model_df["date"] < EVENT_START].copy()
    model = smf.ols(formula, data=train_df).fit()

    model_df["climate_expected_gwa"] = model.predict(model_df)
    model_df["residual_gwa"] = model_df["gwa_cm"] - model_df["climate_expected_gwa"]

    diagnostics: dict[str, float | int | bool] = {
        "train_r2": float(model.rsquared),
        "train_adj_r2": float(model.rsquared_adj),
        "train_nobs": int(model.nobs),
        "beast_available": True,
        "beast_seed": BEAST_SEED,
        "beast_probability_threshold": BEAST_PROB_THRESHOLD,
        "segmented_bic_delta_threshold": SEGMENTED_BIC_DELTA_THRESHOLD,
        "minimum_segment_months": MIN_SEGMENT_MONTHS,
    }
    diagnostics.update(blocked_year_cv(train_df, formula))
    return model_df, diagnostics


def fit_trends_for_breakpoint(
    series_df: pd.DataFrame,
    break_idx: int | None,
) -> tuple[float | None, float | None]:
    if break_idx is None:
        return None, None
    if break_idx < MIN_SEGMENT_MONTHS or (len(series_df) - break_idx) < MIN_SEGMENT_MONTHS:
        return None, None

    x = np.arange(len(series_df), dtype=float)
    y = series_df["deseasoned_gwa"].to_numpy(dtype=float)
    left = slice(0, break_idx)
    right = slice(break_idx, len(series_df))
    left_valid = np.isfinite(y[left])
    right_valid = np.isfinite(y[right])
    if left_valid.sum() < MIN_SEGMENT_MONTHS or right_valid.sum() < MIN_SEGMENT_MONTHS:
        return None, None
    left_fit = stats.linregress(x[left][left_valid], y[left][left_valid])
    right_fit = stats.linregress(x[right][right_valid], y[right][right_valid])
    return float(left_fit.slope * 12.0), float(right_fit.slope * 12.0)


def breakpoint_index(series_df: pd.DataFrame, onset_date: pd.Timestamp | None) -> int | None:
    if onset_date is None:
        return None
    matches = np.flatnonzero(series_df["date"].to_numpy(dtype="datetime64[ns]") == np.datetime64(onset_date))
    return int(matches[0]) if matches.size else None


def bic_from_rss(rss: float, n_obs: int, n_params: int) -> float:
    safe_rss = max(float(rss), 1e-12)
    return float(n_obs * np.log(safe_rss / max(n_obs, 1)) + n_params * np.log(max(n_obs, 1)))


def null_linear_fit(series_df: pd.DataFrame) -> tuple[float | None, float]:
    x = np.arange(len(series_df), dtype=float)
    y = series_df["deseasoned_gwa"].to_numpy(dtype=float)
    valid = np.isfinite(y)
    x, y = x[valid], y[valid]
    fit = stats.linregress(x, y)
    residual = y - (fit.intercept + fit.slope * x)
    return float(fit.slope * 12.0), bic_from_rss(float(np.sum(residual**2)), len(y), 2)


def piecewise_linear_break(series_df: pd.DataFrame) -> SegmentedResult:
    x = np.arange(len(series_df), dtype=float)
    y = series_df["deseasoned_gwa"].to_numpy(dtype=float)
    if len(series_df) < MIN_SEGMENT_MONTHS * 2:
        return SegmentedResult(
            candidate_onset_date=None,
            break_idx=None,
            pre_trend_cm_per_year=None,
            post_trend_cm_per_year=None,
            model_type="insufficient_length",
            bic_delta=None,
        )

    _, bic_null = null_linear_fit(series_df)
    candidate_positions = range(MIN_SEGMENT_MONTHS, len(series_df) - MIN_SEGMENT_MONTHS + 1)
    best_bic = None
    best_result = SegmentedResult(
        candidate_onset_date=None,
        break_idx=None,
        pre_trend_cm_per_year=None,
        post_trend_cm_per_year=None,
        model_type="no_break_linear",
        bic_delta=None,
    )

    for idx in candidate_positions:
        left = slice(0, idx)
        right = slice(idx, len(series_df))
        lv, rv = np.isfinite(y[left]), np.isfinite(y[right])
        if lv.sum() < MIN_SEGMENT_MONTHS or rv.sum() < MIN_SEGMENT_MONTHS:
            continue
        # Candidate dates must be observed; interpolation supplies only the STL season.
        if not np.isfinite(y[idx]):
            continue
        lx, ly, rx, ry = x[left][lv], y[left][lv], x[right][rv], y[right][rv]
        left_fit = stats.linregress(lx, ly)
        right_fit = stats.linregress(rx, ry)
        left_resid = ly - (left_fit.intercept + left_fit.slope * lx)
        right_resid = ry - (right_fit.intercept + right_fit.slope * rx)
        bic_single = bic_from_rss(float(np.sum(left_resid**2) + np.sum(right_resid**2)), int(lv.sum() + rv.sum()), 5)
        if best_bic is None or bic_single < best_bic:
            best_bic = bic_single
            best_result = SegmentedResult(
                candidate_onset_date=pd.Timestamp(series_df["date"].iloc[idx]),
                break_idx=idx,
                pre_trend_cm_per_year=float(left_fit.slope * 12.0),
                post_trend_cm_per_year=float(right_fit.slope * 12.0),
                model_type="single_break",
                bic_delta=float(bic_null - bic_single),
            )

    if best_result.bic_delta is None or best_result.bic_delta <= SEGMENTED_BIC_DELTA_THRESHOLD:
        return SegmentedResult(
            candidate_onset_date=best_result.candidate_onset_date,
            break_idx=best_result.break_idx,
            pre_trend_cm_per_year=best_result.pre_trend_cm_per_year,
            post_trend_cm_per_year=best_result.post_trend_cm_per_year,
            model_type="no_break_linear",
            bic_delta=best_result.bic_delta,
        )
    return best_result


def _numeric_array(value: object) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value, dtype=float)
    else:
        arr = np.asarray([value], dtype=float)
    return arr[np.isfinite(arr)]


def fractional_year_to_timestamp(value: float) -> pd.Timestamp:
    year = int(np.floor(value))
    return pd.Timestamp(year=year, month=1, day=1) + pd.DateOffset(months=int(np.round((value - year) * 12)))


def maybe_date_from_cp(series_df: pd.DataFrame, cp_value: float) -> pd.Timestamp:
    if cp_value > 1900:
        return fractional_year_to_timestamp(cp_value)
    cp_index = int(np.clip(round(cp_value) - 1, 0, len(series_df) - 1))
    return pd.Timestamp(series_df["date"].iloc[cp_index])


def is_recovery_relevant_break(
    onset_date: pd.Timestamp | None,
    pre_trend_cm_per_year: float | None,
    post_trend_cm_per_year: float | None,
) -> bool:
    if onset_date is None or pd.isna(onset_date):
        return False
    if pre_trend_cm_per_year is None or post_trend_cm_per_year is None:
        return False
    if not np.isfinite(pre_trend_cm_per_year) or not np.isfinite(post_trend_cm_per_year):
        return False
    if onset_date < RECOVERY_LABEL_MIN_DATE:
        return False
    return bool(
        (post_trend_cm_per_year > 0.0)
        and (post_trend_cm_per_year > pre_trend_cm_per_year + RECOVERY_SLOPE_MARGIN)
    )


def beast_breakpoint(series_df: pd.DataFrame) -> BeastCandidate | None:
    require_rbeast()

    values = series_df["deseasoned_gwa"].to_numpy(dtype=float)
    try:
        result = rb.beast(
            values,
            start=int(series_df["date"].dt.year.iloc[0]) + (int(series_df["date"].dt.month.iloc[0]) - 1) / 12,
            deltat=1 / 12,
            period=1.0,
            season="none",
            tseg_minlength=MIN_SEGMENT_MONTHS,
            quiet=1,
            mcmc_seed=BEAST_SEED,
            print_param=False,
            print_progress=False,
            print_warning=False,
        )
    except Exception as exc:
        raise RuntimeError("BEAST failed on the calendar-indexed series") from exc

    trend = getattr(result, "trend", None)
    if trend is None:
        return None

    cp_values = _numeric_array(getattr(trend, "cp", None))
    cp_probs = _numeric_array(getattr(trend, "cpPr", None))
    cp_ci = getattr(trend, "cpCI", None)
    if cp_ci is None:
        cp_ci_array = np.empty((0, 2), dtype=float)
    else:
        cp_ci_array = np.asarray(cp_ci, dtype=float)
        if cp_ci_array.ndim == 1:
            cp_ci_array = cp_ci_array.reshape(-1, 2)

    candidate_rows: list[BeastCandidate] = []
    for idx, cp_value in enumerate(cp_values):
        cp_prob = float(cp_probs[min(idx, cp_probs.size - 1)]) if cp_probs.size else np.nan
        onset = maybe_date_from_cp(series_df, float(cp_value))
        break_idx = breakpoint_index(series_df, onset)
        pre_trend, post_trend = fit_trends_for_breakpoint(series_df, break_idx)
        if break_idx is None or pre_trend is None or post_trend is None:
            continue
        ci_start = None
        ci_end = None
        if idx < len(cp_ci_array):
            ci_pair = cp_ci_array[idx]
            if len(ci_pair) == 2 and np.all(np.isfinite(ci_pair)):
                ci_start = maybe_date_from_cp(series_df, float(min(ci_pair)))
                ci_end = maybe_date_from_cp(series_df, float(max(ci_pair)))
        candidate_rows.append(
            BeastCandidate(
                onset_date=onset,
                break_idx=break_idx,
                confidence=None if np.isnan(cp_prob) else cp_prob,
                ci_start=ci_start,
                ci_end=ci_end,
                pre_trend_cm_per_year=pre_trend,
                post_trend_cm_per_year=post_trend,
            )
        )

    if candidate_rows:
        recovery_candidates = [
            row
            for row in candidate_rows
            if is_recovery_relevant_break(
                row.onset_date,
                row.pre_trend_cm_per_year,
                row.post_trend_cm_per_year,
            )
        ]
        pool = recovery_candidates if recovery_candidates else candidate_rows
        return max(
            pool,
            key=lambda row: (
                -np.inf if row.confidence is None else row.confidence,
                -np.inf if row.onset_date is None else row.onset_date.toordinal(),
            ),
        )

    occ_probs = _numeric_array(getattr(trend, "cpOccPr", None))
    if occ_probs.size == len(series_df):
        best_idx = int(np.argmax(occ_probs))
        pre_trend, post_trend = fit_trends_for_breakpoint(series_df, best_idx)
        if pre_trend is not None and post_trend is not None:
            return BeastCandidate(
                onset_date=pd.Timestamp(series_df["date"].iloc[best_idx]),
                break_idx=best_idx,
                confidence=float(occ_probs[best_idx]),
                ci_start=None,
                ci_end=None,
                pre_trend_cm_per_year=pre_trend,
                post_trend_cm_per_year=post_trend,
            )
    return None


def calendar_series(group: pd.DataFrame) -> pd.DataFrame:
    """Keep missing months in place; never compress calendar time into row number."""
    group = group.sort_values("date").copy()
    if group["date"].duplicated().any():
        raise ValueError("Duplicate grid-month observations")
    dates = pd.date_range(group["date"].min(), group["date"].max(), freq="MS")
    return group.set_index("date").reindex(dates).rename_axis("date").reset_index()


def detect_breakpoints(panel: pd.DataFrame, seasonal_method: str = "stl") -> pd.DataFrame:
    require_rbeast()
    results: list[BreakpointResult] = []
    for grid_id, group in panel.groupby("grid_id"):
        group = calendar_series(group)
        if group["gwa_cm"].notna().sum() < 60:
            results.append(
                BreakpointResult(
                    grid_id=grid_id,
                    onset_date=None,
                    onset_source="insufficient_data",
                    selected_method="insufficient_data",
                    selected_reason="Fewer than 60 monthly observations were available.",
                    breakpoint_confidence=None,
                    pre_trend_cm_per_year=None,
                    post_trend_cm_per_year=None,
                    is_recovery_relevant=False,
                    segmented_onset_date=None,
                    segmented_pre_trend_cm_per_year=None,
                    segmented_post_trend_cm_per_year=None,
                    segmented_model_type="insufficient_data",
                    segmented_bic_delta=None,
                    beast_onset_date=None,
                    beast_onset_ci_start=None,
                    beast_onset_ci_end=None,
                    beast_breakpoint_confidence=None,
                    beast_pre_trend_cm_per_year=None,
                    beast_post_trend_cm_per_year=None,
                )
            )
            continue

        if seasonal_method == "stl":
            filled = group["gwa_cm"].interpolate(limit_direction="both")
            seasonal = STL(filled, period=12, robust=True).fit().seasonal
        elif seasonal_method == "harmonic":
            # Observed-only annual + semiannual harmonics, with a cubic trend nuisance.
            t = np.arange(len(group), dtype=float)
            u = (t - t.mean()) / max(t.std(), 1)
            trend = np.column_stack([u**k for k in range(4)])
            harmonics = np.column_stack([f(2 * np.pi * k * t / 12) for k in (1, 2) for f in (np.sin, np.cos)])
            design = np.column_stack([trend, harmonics])
            valid = group["gwa_cm"].notna().to_numpy()
            coef = np.linalg.lstsq(design[valid], group.loc[valid, "gwa_cm"], rcond=None)[0]
            seasonal = harmonics @ coef[4:]
        else:
            raise ValueError(f"Unknown seasonal method: {seasonal_method}")
        group["deseasoned_gwa"] = group["gwa_cm"] - seasonal

        segmented_result = piecewise_linear_break(group)
        beast_result = beast_breakpoint(group)
        beast_reportable = (
            beast_result is not None
            and beast_result.confidence is not None
            and beast_result.confidence >= BEAST_PROB_THRESHOLD
            and is_recovery_relevant_break(
                beast_result.onset_date,
                beast_result.pre_trend_cm_per_year,
                beast_result.post_trend_cm_per_year,
            )
        )
        segmented_reportable = (
            segmented_result.model_type == "single_break"
            and is_recovery_relevant_break(
                segmented_result.candidate_onset_date,
                segmented_result.pre_trend_cm_per_year,
                segmented_result.post_trend_cm_per_year,
            )
        )

        chosen_date = None
        chosen_conf = None
        chosen_pre_trend = None
        chosen_post_trend = None
        chosen_source = "no_break_detected"
        selected_reason = (
            "No recovery-type breakpoint passed the BEAST probability threshold or the segmented BIC threshold."
        )

        if beast_reportable:
            chosen_date = beast_result.onset_date
            chosen_conf = beast_result.confidence
            chosen_pre_trend = beast_result.pre_trend_cm_per_year
            chosen_post_trend = beast_result.post_trend_cm_per_year
            chosen_source = "beast"
            selected_reason = f"Selected the top BEAST trend breakpoint with cpPr={beast_result.confidence:.3f}."
        elif segmented_reportable:
            chosen_date = segmented_result.candidate_onset_date
            chosen_pre_trend = segmented_result.pre_trend_cm_per_year
            chosen_post_trend = segmented_result.post_trend_cm_per_year
            chosen_source = "segmented_regression"
            selected_reason = (
                f"Segmented single-break model improved BIC over the no-break model by "
                f"{segmented_result.bic_delta:.2f}."
            )
        elif beast_result is not None and beast_result.confidence is not None:
            selected_reason = (
                f"BEAST detected a candidate breakpoint with cpPr={beast_result.confidence:.3f}, "
                f"but it was not both high-confidence and recovery-relevant; segmented BIC support was also insufficient."
            )
        elif segmented_result.bic_delta is not None:
            selected_reason = (
                f"The segmented single-break candidate improved BIC by {segmented_result.bic_delta:.2f}, "
                f"but it was not retained as a recovery-type breakpoint."
            )

        is_recovery_relevant = beast_reportable or segmented_reportable
        results.append(
            BreakpointResult(
                grid_id=grid_id,
                onset_date=chosen_date.strftime("%Y-%m-%d") if chosen_date is not None else None,
                onset_source=chosen_source,
                selected_method=chosen_source,
                selected_reason=selected_reason,
                breakpoint_confidence=chosen_conf,
                pre_trend_cm_per_year=chosen_pre_trend,
                post_trend_cm_per_year=chosen_post_trend,
                is_recovery_relevant=is_recovery_relevant,
                segmented_onset_date=segmented_result.candidate_onset_date.strftime("%Y-%m-%d")
                if segmented_result.candidate_onset_date is not None
                else None,
                segmented_pre_trend_cm_per_year=segmented_result.pre_trend_cm_per_year,
                segmented_post_trend_cm_per_year=segmented_result.post_trend_cm_per_year,
                segmented_model_type=segmented_result.model_type,
                segmented_bic_delta=segmented_result.bic_delta,
                beast_onset_date=beast_result.onset_date.strftime("%Y-%m-%d")
                if beast_result is not None and beast_result.onset_date is not None
                else None,
                beast_onset_ci_start=beast_result.ci_start.strftime("%Y-%m-%d")
                if beast_result is not None and beast_result.ci_start is not None
                else None,
                beast_onset_ci_end=beast_result.ci_end.strftime("%Y-%m-%d")
                if beast_result is not None and beast_result.ci_end is not None
                else None,
                beast_breakpoint_confidence=None if beast_result is None else beast_result.confidence,
                beast_pre_trend_cm_per_year=None if beast_result is None else beast_result.pre_trend_cm_per_year,
                beast_post_trend_cm_per_year=None if beast_result is None else beast_result.post_trend_cm_per_year,
            )
        )
    return pd.DataFrame([asdict(row) for row in results])


def build_treated_control(counterfactual_df: pd.DataFrame, quantile_share: float) -> pd.DataFrame:
    summary = (
        counterfactual_df.assign(period=np.where(counterfactual_df["date"] < EVENT_START, "pre", "post"))
        .loc[counterfactual_df["date"] >= pd.Timestamp("2015-01-01")]
        .groupby(["grid_id", "period"], as_index=False)["water_frac"]
        .mean()
        .pivot(index="grid_id", columns="period", values="water_frac")
        .reset_index()
    )
    summary["water_change"] = summary["post"] - summary["pre"]
    upper = summary["water_change"].quantile(1 - quantile_share)
    lower = summary["water_change"].quantile(quantile_share)
    summary["treated"] = (summary["water_change"] >= upper).astype(int)
    summary["control"] = (summary["water_change"] <= lower).astype(int)
    return summary


def run_event_study(
    counterfactual_df: pd.DataFrame,
    treated_control: pd.DataFrame,
    quantile_share: float,
) -> pd.DataFrame:
    sample = counterfactual_df.merge(
        treated_control[["grid_id", "treated", "control", "water_change"]],
        on="grid_id",
        how="left",
    )
    sample = sample.loc[(sample["treated"] == 1) | (sample["control"] == 1)].copy()
    sample["treated"] = sample["treated"].fillna(0).astype(int)
    sample["event_month"] = (
        (sample["date"].dt.year - EVENT_START.year) * 12
        + sample["date"].dt.month
        - EVENT_START.month
    )
    sample = sample.loc[(sample["event_month"] >= -24) & (sample["event_month"] <= 36)].copy()

    event_levels = [k for k in sorted(sample["event_month"].unique()) if k != -1]
    event_columns: list[str] = []
    for k in event_levels:
        col = f"event_m{k}".replace("-", "neg_")
        sample[col] = ((sample["event_month"] == k) & (sample["treated"] == 1)).astype(int)
        event_columns.append(col)

    formula = "residual_gwa ~ C(grid_id) + C(date_str)"
    if event_columns:
        formula += " + " + " + ".join(event_columns)

    model = smf.ols(formula, data=sample).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["grid_id"]},
    )

    rows = []
    treated_label = f"top{int(quantile_share * 100)}_vs_bottom{int(quantile_share * 100)}_water_change"
    for k in event_levels:
        col = f"event_m{k}".replace("-", "neg_")
        beta = model.params.get(col, np.nan)
        se = model.bse.get(col, np.nan)
        rows.append(
            {
                "event_month": int(k),
                "beta": float(beta),
                "ci_low": float(beta - 1.96 * se) if np.isfinite(se) else np.nan,
                "ci_high": float(beta + 1.96 * se) if np.isfinite(se) else np.nan,
                "treated_definition": treated_label,
            }
        )
    return pd.DataFrame(rows)


def merge_case(case_path: Path | None, counterfactual_df: pd.DataFrame) -> pd.DataFrame | None:
    if case_path is None:
        return None
    case_df = pd.read_csv(case_path)
    ensure_columns(
        case_df,
        ["buffer_km", "case_grid_id", "water_persistence", "date", "water_frac"],
        "case",
    )
    case_df["date"] = pd.to_datetime(case_df["date"])
    residual = counterfactual_df[["grid_id", "date", "residual_gwa"]].copy()
    merged = case_df.merge(
        residual,
        left_on=["case_grid_id", "date"],
        right_on=["grid_id", "date"],
        how="left",
    ).rename(columns={"residual_gwa": "case_grid_residual_gwa"})
    return merged[
        [
            "buffer_km",
            "inner_m",
            "outer_m",
            "case_grid_id",
            "date",
            "water_frac",
            "water_persistence",
            "case_grid_residual_gwa",
        ]
    ].sort_values(["buffer_km", "date"])


def main() -> None:
    args = parse_args()
    require_rbeast()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = add_lag_features(load_panel(Path(args.panel)))
    counterfactual_df, diagnostics = fit_counterfactual(panel)
    # Recovery detection does not depend on availability of climate-model covariates.
    breakpoint_df = detect_breakpoints(panel)
    treated_control = build_treated_control(counterfactual_df, args.treated_quantile)
    event_study_df = run_event_study(counterfactual_df, treated_control, args.treated_quantile)
    case_df = merge_case(Path(args.case) if args.case else None, counterfactual_df)

    counterfactual_df[
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
            "climate_expected_gwa",
            "residual_gwa",
            "precip_0_2",
            "precip_3_6",
            "smsa_0_1",
            "et_0_1",
        ]
    ].to_csv(output_dir / "grid_counterfactual.csv", index=False)

    breakpoint_df.to_csv(output_dir / "grid_breakpoints.csv", index=False)
    event_study_df.to_csv(output_dir / "did_eventstudy.csv", index=False)
    if case_df is not None:
        case_df.to_csv(output_dir / "hutuo_case_buffers.csv", index=False)

    diagnostics.update(
        {
            "grid_count": int(counterfactual_df["grid_id"].nunique()),
            "date_count": int(counterfactual_df["date"].nunique()),
            "treated_grid_count": int(treated_control["treated"].sum()),
            "control_grid_count": int(treated_control["control"].sum()),
        }
    )
    (output_dir / "model_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
