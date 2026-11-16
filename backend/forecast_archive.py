from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "runtime" / "waterwatch_forecast_archive.sqlite3"
DB_PATH = Path(os.getenv("WATERWATCH_ARCHIVE_DB_PATH", str(DEFAULT_DB_PATH))).expanduser()


class ForecastArchiveError(RuntimeError):
    """Raised when a forecast cannot be safely recorded in the archive."""


def _connect() -> sqlite3.Connection:
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(DB_PATH, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
    except Exception as exc:
        raise ForecastArchiveError(f"Could not open forecast archive at {DB_PATH}: {exc}") from exc


def initialize_archive() -> None:
    """Create an append-only SQLite archive. Mount DB_PATH on persistent storage in deployment."""
    try:
        with _connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_issues (
                    archive_id TEXT PRIMARY KEY,
                    issue_timestamp TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,
                    observation_month TEXT NOT NULL,
                    forecast_month TEXT NOT NULL,
                    forecast_type TEXT NOT NULL,
                    water_stress_index REAL NOT NULL,
                    category TEXT NOT NULL,
                    model_sha256 TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    archived_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_forecast_issues_target
                ON forecast_issues(forecast_month, issue_timestamp)
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS forecast_issues_no_update
                BEFORE UPDATE ON forecast_issues
                BEGIN
                    SELECT RAISE(ABORT, 'forecast archive is append-only');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS forecast_issues_no_delete
                BEFORE DELETE ON forecast_issues
                BEGIN
                    SELECT RAISE(ABORT, 'forecast archive is append-only');
                END
                """
            )
    except ForecastArchiveError:
        raise
    except Exception as exc:
        raise ForecastArchiveError(f"Could not initialize forecast archive: {exc}") from exc


def archive_forecast(result: dict[str, Any]) -> str:
    """Append a forecast issue once; identical retries are idempotent."""
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
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)

    try:
        initialize_archive()
        with _connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO forecast_issues (
                    archive_id, issue_timestamp, retrieved_at,
                    observation_month, forecast_month, forecast_type,
                    water_stress_index, category, model_sha256,
                    input_fingerprint, response_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    archive_id,
                    result["issue_timestamp"],
                    result["retrieved_at"],
                    result["observation_month"],
                    result["forecast_month"],
                    result["forecast_type"],
                    float(result["water_stress_index"]),
                    result["category"],
                    model_sha256,
                    input_fingerprint,
                    payload,
                ),
            )
        return archive_id
    except ForecastArchiveError:
        raise
    except Exception as exc:
        raise ForecastArchiveError(f"Could not append forecast issue: {exc}") from exc


def list_forecast_issues(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent archived issues, newest first; omit the full feature payload."""
    safe_limit = max(1, min(int(limit), 100))
    try:
        initialize_archive()
        with _connect() as connection:
            rows = connection.execute(
                """
                SELECT archive_id, issue_timestamp, retrieved_at,
                       observation_month, forecast_month, forecast_type,
                       water_stress_index, category, model_sha256,
                       input_fingerprint, archived_at
                FROM forecast_issues
                ORDER BY issue_timestamp DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    except ForecastArchiveError:
        raise
    except Exception as exc:
        raise ForecastArchiveError(f"Could not read forecast archive: {exc}") from exc
