import unittest

import pandas as pd

from backend.data_ingestion import (
    latest_common_complete_month,
    validate_daily_source_coverage,
)
from backend.live_forecast import build_source_readiness_summary


class DailyCoverageValidationTests(unittest.TestCase):
    def setUp(self):
        self.expected = pd.date_range("2026-08-01", "2026-08-03", freq="D")

    def test_accepts_complete_unique_finite_daily_data(self):
        frame = pd.DataFrame({
            "date": self.expected,
            "rain_mm": [1.0, 2.0, 3.0],
        })
        validate_daily_source_coverage(
            frame, "test-source", self.expected, ["rain_mm"]
        )

    def test_rejects_missing_day(self):
        frame = pd.DataFrame({
            "date": self.expected[[0, 2]],
            "rain_mm": [1.0, 3.0],
        })
        with self.assertRaisesRegex(RuntimeError, "coverage is incomplete"):
            validate_daily_source_coverage(
                frame, "test-source", self.expected, ["rain_mm"]
            )

    def test_rejects_duplicate_day(self):
        frame = pd.DataFrame({
            "date": [self.expected[0], self.expected[0], self.expected[2]],
            "rain_mm": [1.0, 2.0, 3.0],
        })
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            validate_daily_source_coverage(
                frame, "test-source", self.expected, ["rain_mm"]
            )

    def test_rejects_missing_required_value(self):
        frame = pd.DataFrame({
            "date": self.expected,
            "rain_mm": [1.0, float("nan"), 3.0],
        })
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            validate_daily_source_coverage(
                frame, "test-source", self.expected, ["rain_mm"]
            )

    def test_common_month_uses_the_captured_source_dates(self):
        selected = latest_common_complete_month({
            "era5_land": "2026-10-31",
            "chirps": "2026-10-31",
        })
        self.assertEqual(selected, pd.Timestamp("2026-10-01"))

    def test_common_month_steps_back_when_latest_source_is_mid_month(self):
        selected = latest_common_complete_month({
            "era5_land": "2026-10-31",
            "chirps": "2026-10-29",
        })
        self.assertEqual(selected, pd.Timestamp("2026-09-01"))

    def test_rejects_missing_required_field(self):
        frame = pd.DataFrame({"date": self.expected})
        with self.assertRaisesRegex(RuntimeError, "missing required fields"):
            validate_daily_source_coverage(
                frame, "test-source", self.expected, ["rain_mm"]
            )


class SourceReadinessSummaryTests(unittest.TestCase):
    def setUp(self):
        self.source_dates = {
            "era5_land": "2026-10-26",
            "chirps": "2026-09-30",
        }

    def test_summary_uses_one_common_source_snapshot(self):
        summary = build_source_readiness_summary(
            self.source_dates,
            pd.Timestamp("2026-11-07T12:00:00Z").to_pydatetime(),
        )
        self.assertEqual(summary["latest_common_complete_month"], "2026-09")
        self.assertEqual(summary["input_cutoff"], "2026-09-30")
        self.assertEqual(summary["one_month_forecast_target_month"], "2026-10")
        self.assertTrue(summary["freshness_gate_passed"])
        self.assertTrue(summary["one_month_target_already_started"])
        self.assertFalse(summary["one_month_issue_time_gate_passed"])
        self.assertFalse(summary["source_publication_time_verified"])

    def test_summary_can_pass_before_target_month_starts(self):
        summary = build_source_readiness_summary(
            {"era5_land": "2026-09-30", "chirps": "2026-09-30"},
            pd.Timestamp("2026-09-30T21:59:00Z").to_pydatetime(),
        )
        self.assertTrue(summary["freshness_gate_passed"])
        self.assertFalse(summary["one_month_target_already_started"])
        self.assertTrue(summary["one_month_issue_time_gate_passed"])


    def test_local_target_month_boundary_uses_eswatini_time(self):
        source_dates = {"era5_land": "2026-09-30", "chirps": "2026-09-30"}

        before_local_month = build_source_readiness_summary(
            source_dates, pd.Timestamp("2026-09-30T21:59:00Z").to_pydatetime()
        )
        self.assertFalse(before_local_month["one_month_target_already_started"])
        self.assertTrue(before_local_month["one_month_issue_time_gate_passed"])

        at_local_month_start = build_source_readiness_summary(
            source_dates, pd.Timestamp("2026-09-30T22:00:00Z").to_pydatetime()
        )
        self.assertTrue(at_local_month_start["one_month_target_already_started"])
        self.assertFalse(at_local_month_start["one_month_issue_time_gate_passed"])

        after_local_month_start = build_source_readiness_summary(
            source_dates, pd.Timestamp("2026-09-30T22:30:00Z").to_pydatetime()
        )
        self.assertTrue(after_local_month_start["one_month_target_already_started"])
        self.assertFalse(after_local_month_start["one_month_issue_time_gate_passed"])

    def test_summary_fails_stale_input_gate(self):
        summary = build_source_readiness_summary(
            self.source_dates,
            pd.Timestamp("2026-11-20T12:00:00Z").to_pydatetime(),
        )
        self.assertFalse(summary["freshness_gate_passed"])
        self.assertFalse(summary["one_month_issue_time_gate_passed"])


if __name__ == "__main__":
    unittest.main()
