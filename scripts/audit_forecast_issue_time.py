from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFRESH_ROOT = ROOT / "outputs" / "expanded" / "shadow_evaluation_data"
SHADOW_FORECASTS = (
    ROOT
    / "outputs"
    / "expanded"
    / "drought_onset_model_preparation"
    / "prospective_shadow"
    / "shadow_forecasts.csv"
)


def parse_utc(value: str, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must include a timezone: {value}")
    return timestamp.tz_convert("UTC")


def audit_refresh_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    refreshed_at = parse_utc(manifest["refresh_run_utc"], "refresh_run_utc")
    common_date = pd.Timestamp(manifest["common_complete_data_through"])
    row: dict[str, Any] = {
        "manifest_path": str(path),
        "refresh_run_utc": refreshed_at.isoformat(),
        "common_complete_data_through": common_date.strftime("%Y-%m-%d"),
        "common_data_age_days_at_refresh": (
            refreshed_at.date() - common_date.date()
        ).days,
        "audit_passed": bool(manifest.get("audit_passed", False)),
        "source_freshness": {},
        "output_files": manifest.get("outputs", {}),
        "publication_latency_verified": False,
        "interpretation": (
            "Age of latest dated observation at retrieval; not the source's "
            "first-publication time or revision history."
        ),
    }
    source_dates = manifest.get("source_latest_dates_at_refresh")
    if not isinstance(source_dates, dict) or not source_dates:
        raise ValueError(
            f"Missing source_latest_dates_at_refresh in {path}"
        )
    for source, value in source_dates.items():
        latest_date = pd.Timestamp(value)
        row["source_freshness"][source] = {
            "latest_observation_date": latest_date.strftime("%Y-%m-%d"),
            "observation_age_days_at_refresh": (
                refreshed_at.date() - latest_date.date()
            ).days,
        }
    return row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_snapshot_files(
    earlier: dict[str, Any],
    later: dict[str, Any],
    output_key: str,
    date_column: str,
    label: str,
) -> dict[str, Any]:
    earlier_path = Path(earlier["output_files"][output_key])
    later_path = Path(later["output_files"][output_key])
    if not earlier_path.is_file() or not later_path.is_file():
        raise FileNotFoundError(
            f"Cannot compare {label} snapshots: "
            f"{earlier_path} or {later_path} is missing."
        )
    old = pd.read_csv(earlier_path, parse_dates=[date_column]).set_index(
        date_column
    )
    new = pd.read_csv(later_path, parse_dates=[date_column]).set_index(
        date_column
    )
    shared_dates = old.index.intersection(new.index)
    shared_columns = [
        column
        for column in old.columns.intersection(new.columns)
        if pd.api.types.is_numeric_dtype(old[column])
        and pd.api.types.is_numeric_dtype(new[column])
    ]
    if not len(shared_dates) or not shared_columns:
        raise ValueError(f"No overlapping numeric {label} data to compare.")
    old_values = old.loc[shared_dates, shared_columns].to_numpy(dtype=float)
    new_values = new.loc[shared_dates, shared_columns].to_numpy(dtype=float)
    both_missing = np.isnan(old_values) & np.isnan(new_values)
    changed = (~both_missing) & (
        ~np.isclose(old_values, new_values, rtol=0, atol=1e-10, equal_nan=True)
    )
    finite_delta = np.abs(new_values - old_values)
    max_abs_difference = (
        float(np.nanmax(finite_delta))
        if np.isfinite(finite_delta).any()
        else 0.0
    )
    return {
        "dataset": label,
        "earlier_file_sha256": sha256_file(earlier_path),
        "later_file_sha256": sha256_file(later_path),
        "shared_rows": len(shared_dates),
        "shared_numeric_columns": len(shared_columns),
        "changed_cells_at_tolerance_1e-10": int(changed.sum()),
        "maximum_absolute_numeric_difference": max_abs_difference,
    }


def compare_snapshot_pair(
    earlier: dict[str, Any],
    later: dict[str, Any],
) -> dict[str, Any]:
    return {
        "earlier_refresh_utc": earlier["refresh_run_utc"],
        "later_refresh_utc": later["refresh_run_utc"],
        "earlier_cutoff": earlier["common_complete_data_through"],
        "later_cutoff": later["common_complete_data_through"],
        "comparisons": [
            compare_snapshot_files(
                earlier, later, "chirps_daily", "date", "CHIRPS daily"
            ),
            compare_snapshot_files(
                earlier,
                later,
                "era5_land_daily",
                "date",
                "ERA5-Land daily",
            ),
            compare_snapshot_files(
                earlier,
                later,
                "monthly_swba",
                "month_date",
                "Monthly SWBA inputs and labels",
            ),
        ],
    }


def audit_existing_forecasts(path: Path) -> list[dict[str, Any]]:
    forecasts = pd.read_csv(path, dtype=str)
    required = (
        "forecast_created_at_utc",
        "data_refreshed_at_utc",
        "data_available_through",
        "forecast_origin",
        "target_period",
        "model_version",
    )
    missing = [column for column in required if column not in forecasts.columns]
    if missing:
        raise ValueError(f"Shadow log is missing required columns: {missing}")

    records: list[dict[str, Any]] = []
    for row in forecasts.to_dict(orient="records"):
        target_start = pd.Period(row["target_period"], freq="M").to_timestamp(
            ).tz_localize("UTC")
        issue_time = parse_utc(
            row["forecast_created_at_utc"], "forecast_created_at_utc"
        )
        refreshed_at = parse_utc(
            row["data_refreshed_at_utc"], "data_refreshed_at_utc"
        )
        records.append(
            {
                "model_version": row["model_version"],
                "forecast_origin": row["forecast_origin"],
                "target_period": row["target_period"],
                "data_available_through": row["data_available_through"],
                "forecast_created_at_utc": issue_time.isoformat(),
                "data_refreshed_at_utc": refreshed_at.isoformat(),
                "days_from_target_start_to_issue": (
                    issue_time - target_start
                ).total_seconds()
                / 86400,
                "days_from_target_start_to_refresh": (
                    refreshed_at - target_start
                ).total_seconds()
                / 86400,
                "issued_before_target_start": issue_time < target_start,
                "snapshot_available_before_target_start": (
                    refreshed_at < target_start
                ),
                "timely_for_declared_target": (
                    issue_time < target_start and refreshed_at < target_start
                ),
                "alert_issued": row.get("alert_issued", ""),
            }
        )
    return records


def main() -> None:
    manifest_paths = sorted(REFRESH_ROOT.glob("refresh_through_*/refresh_manifest.json"))
    if not manifest_paths:
        raise FileNotFoundError(
            f"No refresh manifests found under {REFRESH_ROOT}"
        )

    refresh_rows = [audit_refresh_manifest(path) for path in manifest_paths]
    refresh_rows.sort(key=lambda row: row["refresh_run_utc"])
    forecast_rows = (
        audit_existing_forecasts(SHADOW_FORECASTS)
        if SHADOW_FORECASTS.is_file()
        else []
    )
    distinct_cutoffs = len(
        {
            row["common_complete_data_through"]
            for row in refresh_rows
        }
    )
    snapshot_comparisons = [
        compare_snapshot_pair(earlier, later)
        for earlier, later in zip(refresh_rows, refresh_rows[1:])
    ]

    run_time = datetime.now(timezone.utc)
    output_dir = REFRESH_ROOT / (
        "issue_time_audit_" + run_time.strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "created_at_utc": run_time.isoformat(),
        "refresh_manifest_count": len(refresh_rows),
        "historical_forecasts_audited": len(forecast_rows),
        "distinct_common_data_cutoffs": distinct_cutoffs,
        "retrieval_freshness_samples_available": len(refresh_rows) > 1,
        "publication_latency_distribution_available": False,
        "snapshot_revision_comparisons": snapshot_comparisons,
        "refresh_snapshots": refresh_rows,
        "existing_shadow_forecasts": forecast_rows,
        "conclusion": (
            "Snapshots measure retrieval-time freshness only. Repeated "
            "snapshots with distinct source coverage can begin to describe "
            "refresh cadence, but they do not prove first-publication time or "
            "quantify revisions without source-vintage metadata."
        ),
    }
    (output_dir / "issue_time_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"\nAudit saved to: {output_dir}")


if __name__ == "__main__":
    main()
