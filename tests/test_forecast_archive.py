import unittest
from unittest.mock import patch

from google.api_core.exceptions import AlreadyExists

from backend import forecast_archive


class FakeDocument:
    def __init__(self, store, document_id):
        self.store = store
        self.document_id = document_id

    def create(self, payload):
        if self.document_id in self.store:
            raise AlreadyExists("Document already exists")
        self.store[self.document_id] = dict(payload)


class FakeQuery:
    def __init__(self, store):
        self.store = store
        self.maximum = 50

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, maximum):
        self.maximum = maximum
        return self

    def stream(self):
        ordered = sorted(
            self.store.items(),
            key=lambda item: item[1].get("issue_timestamp", ""),
            reverse=True,
        )
        return [
            type("Snapshot", (), {"to_dict": lambda self, value=value: dict(value)})()
            for _, value in ordered[:self.maximum]
        ]


class FakeCollection:
    def __init__(self, store):
        self.store = store

    def document(self, document_id):
        return FakeDocument(self.store, document_id)

    def order_by(self, *args, **kwargs):
        return FakeQuery(self.store)


class FakeClient:
    def __init__(self):
        self.store = {}

    def collection(self, name):
        self.collection_name = name
        return FakeCollection(self.store)


class ForecastArchiveTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.client_patch = patch.object(
            forecast_archive, "_client", return_value=self.client
        )
        self.client_patch.start()

    def tearDown(self):
        self.client_patch.stop()

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

    def test_archive_writes_issue_to_firebase_collection(self):
        archive_id = forecast_archive.archive_forecast(self.sample_forecast())
        self.assertEqual(len(archive_id), 64)
        self.assertIn(archive_id, self.client.store)
        self.assertEqual(
            self.client.store[archive_id]["forecast_month"], "2026-10"
        )
        self.assertEqual(
            self.client.collection_name, forecast_archive.COLLECTION_NAME
        )

    def test_identical_retry_is_idempotent(self):
        first = forecast_archive.archive_forecast(self.sample_forecast())
        second = forecast_archive.archive_forecast(self.sample_forecast())
        self.assertEqual(first, second)
        self.assertEqual(len(self.client.store), 1)

    def test_existing_issue_is_not_overwritten(self):
        record = self.sample_forecast()
        archive_id = forecast_archive.archive_forecast(record)
        changed = dict(record)
        changed["category"] = "Altered category"
        self.assertEqual(forecast_archive.archive_forecast(changed), archive_id)
        self.assertEqual(
            self.client.store[archive_id]["category"], "Moderate Water Stress"
        )

    def test_missing_provenance_is_rejected(self):
        record = self.sample_forecast()
        record["provenance"] = {}
        with self.assertRaises(forecast_archive.ForecastArchiveError):
            forecast_archive.archive_forecast(record)

    def test_archive_list_limit_is_bounded(self):
        self.assertEqual(forecast_archive.list_forecast_issues(10000), [])
        self.assertEqual(self.client.collection_name, forecast_archive.COLLECTION_NAME)


if __name__ == "__main__":
    unittest.main()
