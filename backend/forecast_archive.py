from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.exceptions import AlreadyExists, GoogleAPICallError


COLLECTION_NAME = "waterwatch_forecast_issues"


class ForecastArchiveError(RuntimeError):
    """Raised when the Firebase forecast archive is unavailable or rejects a write."""


def _firebase_app():
    """Initialise the Admin SDK from deployment credentials; never embed secrets."""
    try:
        try:
            return firebase_admin.get_app()
        except ValueError:
            raw_service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
            if raw_service_account:
                service_account = json.loads(raw_service_account)
                credential = credentials.Certificate(service_account)
            else:
                # Render or another trusted runtime should provide ADC via
                # GOOGLE_APPLICATION_CREDENTIALS / workload identity.
                credential = credentials.ApplicationDefault()

            project_id = (
                os.getenv("FIREBASE_PROJECT_ID")
                or os.getenv("GOOGLE_CLOUD_PROJECT")
                or "eswatiniwaterstress"
            )
            return firebase_admin.initialize_app(
                credential,
                {"projectId": project_id},
            )
    except Exception as exc:
        raise ForecastArchiveError(
            "Firebase Admin initialisation failed. Configure FIREBASE_SERVICE_ACCOUNT_JSON "
            "or GOOGLE_APPLICATION_CREDENTIALS and grant Firestore access."
        ) from exc


def _client():
    try:
        return firestore.client(app=_firebase_app())
    except Exception as exc:
        raise ForecastArchiveError(f"Could not connect to Cloud Firestore: {exc}") from exc


def initialize_archive() -> None:
    """Validate that the Firebase Admin client can be initialised."""
    _client()


def _record_identity(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    required = (
        "issue_timestamp", "retrieved_at", "observation_month",
        "forecast_month", "forecast_type", "water_stress_index",
        "category", "provenance", "data_quality",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise ForecastArchiveError(f"Forecast response is missing archive fields: {missing}")

    provenance = result["provenance"]
    model_sha256 = provenance.get("model_sha256")
    input_fingerprint = provenance.get("input_fingerprint")
    if not model_sha256 or not input_fingerprint:
        raise ForecastArchiveError("Model hash and input fingerprint are required for archiving.")

    identity = f"{result['issue_timestamp']}|{model_sha256}|{input_fingerprint}"
    archive_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    try:
        payload = json.loads(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ForecastArchiveError(f"Forecast payload is not valid JSON: {exc}") from exc

    payload.update({
        "archive_id": archive_id,
        "model_sha256": model_sha256,
        "input_fingerprint": input_fingerprint,
    })
    return archive_id, payload


def archive_forecast(result: dict[str, Any]) -> str:
    """Create one immutable Firestore issue document; identical retries are idempotent."""
    archive_id, payload = _record_identity(result)
    payload["archived_at"] = firestore.SERVER_TIMESTAMP
    document = _client().collection(COLLECTION_NAME).document(archive_id)

    try:
        document.create(payload)
    except AlreadyExists:
        # Idempotent retry: an existing issue with the same deterministic ID is retained.
        pass
    except GoogleAPICallError as exc:
        raise ForecastArchiveError(f"Firestore rejected the forecast archive write: {exc}") from exc
    except Exception as exc:
        raise ForecastArchiveError(f"Could not archive forecast in Firebase: {exc}") from exc

    return archive_id


def get_latest_forecast() -> dict[str, Any] | None:
    """Read the complete latest forecast from Firebase for stale/offline display."""
    try:
        query = (
            _client()
            .collection(COLLECTION_NAME)
            .order_by("issue_timestamp", direction=firestore.Query.DESCENDING)
            .limit(1)
        )
        snapshots = list(query.stream())
        if not snapshots:
            return None
        record = snapshots[0].to_dict() or {}
        archived_at = record.get("archived_at")
        if hasattr(archived_at, "isoformat"):
            record["archived_at"] = archived_at.isoformat()
        return record
    except Exception as exc:
        raise ForecastArchiveError(f"Could not read latest forecast from Firebase: {exc}") from exc


def list_forecast_issues(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent issue metadata from Firestore, newest first."""
    safe_limit = max(1, min(int(limit), 100))
    try:
        query = (
            _client()
            .collection(COLLECTION_NAME)
            .order_by("issue_timestamp", direction=firestore.Query.DESCENDING)
            .limit(safe_limit)
        )
        records = []
        for snapshot in query.stream():
            record = snapshot.to_dict() or {}
            archived_at = record.get("archived_at")
            if hasattr(archived_at, "isoformat"):
                record["archived_at"] = archived_at.isoformat()
            # Keep the public history endpoint compact; full response remains in Firestore.
            record.pop("latest_indicators", None)
            record.pop("model_features", None)
            records.append(record)
        return records
    except Exception as exc:
        raise ForecastArchiveError(f"Could not read forecast archive from Firebase: {exc}") from exc
