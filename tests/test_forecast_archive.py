import tempfile
import unittest
from pathlib import Path

from backend import forecast_archive


class ForecastArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_path = forecast_archive.DB_PATH
        forecast_archive.DB_PATH = Path(self.temp_dir.name) / "archive.sqlite3"

    def tearDown(self):
        forecast_archive.DB_PATH = self.original_path
        self.temp_dir.cleanup()

    def sample_forecast(self):
        return {
            "issue_timestamp": "2026-11-16T10:00:00+00:00",
            "retrieved_at": "2026-11-16T09:58:00+00:00",
            "observation_month": "2026-09",
            "forecast_month": "2026-10",
            "forecast_type": "test_forecast",
            "water_stress_index": -0.75,
            "category": "Moderate Water Stress",
            "provenance": {
                "model_sha256": "a" * 64,
                "input_fingerprint": "b" * 64,
            },
            "data_quality": {"status": "passed"},
        }

    def test_archive_writes_issue_and_returns_id(self):
        archive_id = forecast_archive.archive_forecast(self.sample_forecast())
        self.assertEqual(len(archive_id), 64)
        records = forecast_archive.list_forecast_issues()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["archive_id"], archive_id)
        self.assertEqual(records[0]["forecast_month"], "2026-10")

    def test_identical_retry_is_idempotent(self):
        first = forecast_archive.archive_forecast(self.sample_forecast())
        second = forecast_archive.archive_forecast(self.sample_forecast())
        self.assertEqual(first, second)
        self.assertEqual(len(forecast_archive.list_forecast_issues()), 1)

    def test_archive_is_append_only(self):
        archive_id = forecast_archive.archive_forecast(self.sample_forecast())
        import sqlite3
        with sqlite3.connect(forecast_archive.DB_PATH) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE forecast_issues SET category = ? WHERE archive_id = ?",
                    ("changed", archive_id),
                )
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "DELETE FROM forecast_issues WHERE archive_id = ?",
                    (archive_id,),
                )

    def test_missing_provenance_is_rejected(self):
        record = self.sample_forecast()
        record["provenance"] = {}
        with self.assertRaises(forecast_archive.ForecastArchiveError):
            forecast_archive.archive_forecast(record)

    def test_archive_list_limit_is_bounded(self):
        self.assertEqual(forecast_archive.list_forecast_issues(10000), [])


if __name__ == "__main__":
    unittest.main()
