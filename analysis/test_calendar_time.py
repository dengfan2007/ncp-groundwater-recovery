"""Regression tests for calendar gaps, slope units, and sparse contrasts."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import ncp_recovery_analysis as core
from ncp_calendar_revision import spatial_effects


class CalendarTests(unittest.TestCase):
    def test_gap_preserves_elapsed_time_and_annual_slope(self):
        dates = pd.date_range("2003-07-01", periods=180, freq="MS")
        t = np.arange(180)
        observed = np.ones(180, dtype=bool)
        observed[50:62] = False
        series = core.calendar_series(pd.DataFrame({"date": dates[observed], "gwa_cm": .25*t[observed]}))
        self.assertEqual(len(series), 180)
        self.assertTrue(series.gwa_cm.iloc[50:62].isna().all())
        series["deseasoned_gwa"] = series.gwa_cm
        pre, post = core.fit_trends_for_breakpoint(series, 90)
        self.assertAlmostEqual(pre, 3., places=10)
        self.assertAlmostEqual(post, 3., places=10)

    def test_segmented_break_and_slopes_survive_mission_gap(self):
        t = np.arange(240)
        y = np.where(t < 150, -.1*t, -15 + .3*(t-150)).astype(float)
        y[100:112] = np.nan
        dates = pd.date_range("2003-01-01", periods=240, freq="MS")
        result = core.piecewise_linear_break(pd.DataFrame({"date": dates, "deseasoned_gwa": y}))
        self.assertLessEqual(abs((result.candidate_onset_date-dates[150]).days), 31)
        self.assertAlmostEqual(result.pre_trend_cm_per_year, -1.2, places=8)
        self.assertAlmostEqual(result.post_trend_cm_per_year, 3.6, places=8)

    def test_beast_receives_fractional_start_and_missing_months(self):
        dates = pd.date_range("2003-07-01", periods=90, freq="MS")
        y = np.arange(90, dtype=float)
        y[35:45] = np.nan
        with patch.object(core.rb, "beast", return_value=object()) as call:
            self.assertIsNone(core.beast_breakpoint(pd.DataFrame({"date": dates, "deseasoned_gwa": y})))
        self.assertEqual(call.call_args.kwargs["start"], 2003.5)
        self.assertTrue(np.isnan(call.call_args.args[0][35:45]).all())

    def test_sparse_region_has_no_fabricated_interval(self):
        frame = pd.DataFrame({"is_recovery_relevant": [True], "onset_date": pd.to_datetime(["2019-07-01"]), "region_tag": ["south"], "grid_lon": [115.5], "grid_lat": [35.], "post_trend_cm_per_year": [2.]})
        result = spatial_effects(frame, n_boot=20)
        self.assertTrue(result.ci_lower.isna().all())
        self.assertTrue(result.estimate.isna().all())
        self.assertTrue(result.valid_replicates.eq(0).all())

    def test_date_rounding_rolls_over_new_year(self):
        self.assertEqual(core.fractional_year_to_timestamp(2019.999), pd.Timestamp("2020-01-01"))


if __name__ == "__main__":
    unittest.main()
