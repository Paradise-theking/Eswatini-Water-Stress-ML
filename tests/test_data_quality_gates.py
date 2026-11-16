import unittest

import pandas as pd

from backend.data_ingestion import (
    latest_common_complete_month,
    validate_daily_source_coverage,
)


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


if __name__ == "__main__":
    unittest.main()
