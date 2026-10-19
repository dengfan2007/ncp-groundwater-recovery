from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import ncp_recovery_analysis as core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run robustness experiments for NCP groundwater recovery analysis.")
    parser.add_argument("--panel", required=True, help="Path to grid_monthly_panel.csv")
    parser.add_argument("--case", help="Path to hutuo_case_buffers_raw.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for robustness outputs")
    return parser.parse_args()


def fit_time_trend_baseline(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    model_df = panel.dropna(subset=["gwa_cm"]).copy()
    model_df["time_index"] = model_df.groupby("grid_id").cumcount()
    train_df = model_df.loc[model_df["date"] < core.EVENT_START].copy()
    formula = "gwa_cm ~ time_index + C(grid_id) + C(month)"
    model = smf.ols(formula, data=train_df).fit()
    model_df["time_trend_expected_gwa"] = model.predict(model_df)
    model_df["time_trend_residual_gwa"] = model_df["gwa_cm"] - model_df["time_trend_expected_gwa"]
    diagnostics = {
        "trend_baseline_r2": float(model.rsquared),
        "trend_baseline_adj_r2": float(model.rsquared_adj),
        "trend_baseline_nobs": float(model.nobs),
    }
    return model_df, diagnostics


def quantile_event_runs(counterfactual_df: pd.DataFrame, quantiles: list[float]) -> tuple[pd.DataFrame, dict[str, object]]:
    event_tables = []
    direction_summary = []
    for quantile in quantiles:
      treated_control = core.build_treated_control(counterfactual_df, quantile)
      event_df = core.run_event_study(counterfactual_df, treated_control, quantile)
      event_df["quantile"] = quantile
      event_tables.append(event_df)

      post_df = event_df.loc[(event_df["event_month"] >= 0) & (event_df["event_month"] <= 24)]
      direction_summary.append(
          {
              "quantile": quantile,
              "post_0_24_mean_beta": float(post_df["beta"].mean()) if not post_df.empty else np.nan,
              "positive_share": float((post_df["beta"] > 0).mean()) if not post_df.empty else np.nan,
          }
      )

    merged = pd.concat(event_tables, ignore_index=True) if event_tables else pd.DataFrame()
    return merged, {"quantile_direction_summary": direction_summary}


def breakpoint_agreement(breakpoints: pd.DataFrame) -> dict[str, float]:
    comparable = breakpoints.dropna(subset=["onset_date", "segmented_onset_date"]).copy()
    if comparable.empty:
        return {
            "breakpoint_comparable_count": 0,
            "breakpoint_mean_abs_month_diff": float("nan"),
            "breakpoint_share_within_6_months": float("nan"),
        }
    chosen = pd.to_datetime(comparable["onset_date"])
    segmented = pd.to_datetime(comparable["segmented_onset_date"])
    month_diff = (chosen.dt.year - segmented.dt.year) * 12 + (chosen.dt.month - segmented.dt.month)
    return {
        "breakpoint_comparable_count": int(len(comparable)),
        "breakpoint_mean_abs_month_diff": float(np.mean(np.abs(month_diff))),
        "breakpoint_share_within_6_months": float((np.abs(month_diff) <= 6).mean()),
    }


def main() -> None:
    args = parse_args()
    core.require_rbeast()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = core.add_lag_features(core.load_panel(Path(args.panel)))
    counterfactual_df, climate_diag = core.fit_counterfactual(panel)
    trend_df, trend_diag = fit_time_trend_baseline(panel)
    breakpoints = core.detect_breakpoints(counterfactual_df)
    event_df, quantile_diag = quantile_event_runs(counterfactual_df, [0.25, 0.30, 0.40])
    case_df = core.merge_case(Path(args.case) if args.case else None, counterfactual_df)

    climate_vs_trend = {
        "climate_model_cv_mean_correlation": climate_diag.get("cv_mean_correlation"),
        "climate_model_train_r2": climate_diag.get("train_r2"),
        "trend_baseline_train_r2": trend_diag.get("trend_baseline_r2"),
    }
    climate_vs_trend["climate_minus_trend_train_r2"] = (
        float(climate_diag["train_r2"]) - float(trend_diag["trend_baseline_r2"])
    )

    event_df.to_csv(output_dir / "robustness_event_study.csv", index=False)
    breakpoints.to_csv(output_dir / "robustness_breakpoints.csv", index=False)
    if case_df is not None:
        case_df.to_csv(output_dir / "robustness_case_buffers.csv", index=False)
    trend_df[
        ["grid_id", "date", "gwa_cm", "time_trend_expected_gwa", "time_trend_residual_gwa"]
    ].to_csv(output_dir / "trend_baseline_counterfactual.csv", index=False)

    summary = {
        "climate_vs_trend": climate_vs_trend,
        "breakpoint_agreement": breakpoint_agreement(breakpoints),
    }
    summary.update(quantile_diag)

    (output_dir / "robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
