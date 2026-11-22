from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.data_ingestion import (
    fetch_latest_monthly_data,
    initialize_earth_engine,
    latest_available_dates,
    latest_common_complete_month,
)
from backend.feature_engineering import (
    MODEL_FEATURES,
    build_latest_model_input,
    forecast_month_from_observation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "water_stress_ridge.joblib"
MAX_INPUT_AGE_DAYS = int(os.getenv("WATERWATCH_MAX_INPUT_AGE_DAYS", "45"))
ESWATINI_TZ = timezone(timedelta(hours=2))


class ForecastReadinessError(RuntimeError):
    """Raised when current data cannot support a defensible advance forecast."""


def classify_water_stress(prediction: float) -> tuple[str, str]:
    """Research communication categories; not operator-approved warning thresholds."""
    if prediction <= -2.0:
        return "Extreme Water Stress", "Exceptionally dry conditions are forecast relative to the historical reference."
    if prediction <= -1.5:
        return "Severe Water Stress", "Significantly drier-than-normal conditions are forecast."
    if prediction <= -1.0:
        return "High Water Stress", "Dry conditions are forecast, with elevated hydroclimatic stress."
    if prediction <= -0.5:
        return "Moderate Water Stress", "Slightly drier-than-normal conditions are forecast."
    if prediction < 0.5:
        return "Near Normal", "Conditions are forecast close to the historical monthly norm."
    if prediction < 1.0:
        return "Low Water Stress", "Wetter-than-normal conditions are forecast."
    if prediction <= 2.0:
        return "Very Low Water Stress", "Substantially wetter-than-normal conditions are forecast."
    return "Exceptionally Wet", "Exceptionally wet conditions are forecast relative to the historical reference."


_cached_live_forecast: dict | None = None
_cached_input_fingerprint: str | None = None


def _utc_timestamp(value: datetime | None = None) -> pd.Timestamp:
    stamp = pd.Timestamp(value or datetime.now(timezone.utc))
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _target_month_start_utc(month: pd.Timestamp) -> pd.Timestamp:
    """Convert the local Eswatini start of a target month to UTC."""
    local_start = month.to_pydatetime().replace(tzinfo=ESWATINI_TZ)
    return _utc_timestamp(local_start)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_fingerprint(monthly: pd.DataFrame, model_sha256: str) -> str:
    values = pd.util.hash_pandas_object(monthly, index=True).values.tobytes()
    return hashlib.sha256(values + model_sha256.encode("ascii")).hexdigest()


def build_source_readiness_summary(
    source_dates: dict[str, str],
    retrieved_at: datetime | None = None,
) -> dict:
    """Summarise freshness and target timing from one captured source-date snapshot."""
    observed_at = _utc_timestamp(retrieved_at)
    try:
        observation_month = latest_common_complete_month(source_dates)
        cutoff = observation_month + pd.offsets.MonthEnd(1)
        cutoff_utc = _utc_timestamp(
            cutoff.to_pydatetime().replace(tzinfo=timezone.utc)
        )
        age_days = (observed_at - cutoff_utc).total_seconds() / 86400
        if age_days < 0:
            raise ForecastReadinessError(
                "Common observation cutoff is later than the diagnostic retrieval time."
            )

        target_month = forecast_month_from_observation(observation_month)
        target_start = _target_month_start_utc(target_month)
        freshness_passed = age_days <= MAX_INPUT_AGE_DAYS
        target_started = target_start <= observed_at

        source_ages = {}
        for source, date in source_dates.items():
            source_age = (
                observed_at
                - _utc_timestamp(
                    pd.Timestamp(date).to_pydatetime().replace(tzinfo=timezone.utc)
                )
            ).total_seconds() / 86400
            if source_age < 0:
                raise ForecastReadinessError(
                    f"{source} reports an observation date later than diagnostic retrieval time."
                )
            source_ages[source] = round(source_age, 2)
        return {
            "retrieved_at": observed_at.isoformat(),
            "source_dates": source_dates,
            "source_observation_age_days": source_ages,
            "latest_common_complete_month": observation_month.strftime("%Y-%m"),
            "input_cutoff": cutoff.strftime("%Y-%m-%d"),
            "input_age_days": round(age_days, 2),
            "maximum_input_age_days": MAX_INPUT_AGE_DAYS,
            "freshness_gate_passed": freshness_passed,
            "one_month_forecast_target_month": target_month.strftime("%Y-%m"),
            "one_month_target_already_started": target_started,
            "one_month_issue_time_gate_passed": freshness_passed and not target_started,
            "source_publication_time_verified": False,
        }
    except ForecastReadinessError:
        raise
    except Exception as exc:
        raise ForecastReadinessError(
            f"Could not summarise source-date readiness: {exc}"
        ) from exc


def generate_live_forecast() -> dict:
    """
    Generate the existing one-month-ahead Ridge forecast only when the
    common observation cutoff is recent, the four-month feature window is
    contiguous and complete, and the target month has not started.

    This remains a research forecast, not an authorised water-supply warning.
    """
    global _cached_live_forecast, _cached_input_fingerprint

    source_dates = latest_available_dates()
    observation_month = latest_common_complete_month(source_dates)
    monthly = fetch_latest_monthly_data(observation_month).sort_values("month_date").reset_index(drop=True)
    retrieved_at = _utc_timestamp()

    expected_months = pd.date_range(end=observation_month, periods=4, freq="MS")
    actual_months = pd.to_datetime(monthly["month_date"]).dt.to_period("M").dt.to_timestamp()
    if len(monthly) != 4 or list(actual_months) != list(expected_months):
        raise ForecastReadinessError(
            "The four consecutive monthly rows required for model features are unavailable."
        )

    numeric = monthly.select_dtypes(include=[np.number])
    if monthly.isna().any().any() or (not numeric.empty and not np.isfinite(numeric.to_numpy(dtype=float)).all()):
        missing = monthly.columns[monthly.isna().any()].tolist()
        raise ForecastReadinessError(
            f"Monthly inputs contain missing or non-finite values. Missing columns: {missing}"
        )

    cutoff = observation_month + pd.offsets.MonthEnd(1)
    cutoff_utc = _utc_timestamp(cutoff.to_pydatetime().replace(tzinfo=timezone.utc))
    age_days = (retrieved_at - cutoff_utc).total_seconds() / 86400
    if age_days < 0:
        raise ForecastReadinessError("Observation cutoff is later than the retrieval timestamp.")
    if age_days > MAX_INPUT_AGE_DAYS:
        raise ForecastReadinessError(
            f"Latest complete observation month is {age_days:.1f} days old; "
            f"configured maximum is {MAX_INPUT_AGE_DAYS} days."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model_sha256 = _sha256_file(MODEL_PATH)
    fingerprint = _input_fingerprint(monthly, model_sha256)

    model_input, model_observation_month = build_latest_model_input(monthly)
    if model_observation_month.to_period("M") != observation_month.to_period("M"):
        raise ForecastReadinessError("Model feature origin does not match the verified observation cutoff.")

    forecast_month = forecast_month_from_observation(model_observation_month)
    target_start = _target_month_start_utc(forecast_month)
    preissue_time = _utc_timestamp()
    if target_start <= preissue_time:
        raise ForecastReadinessError(
            "The target month has already started; this result would not be a genuine advance forecast."
        )

    if _cached_live_forecast is not None and fingerprint == _cached_input_fingerprint:
        served_at = _utc_timestamp()
        cached = dict(_cached_live_forecast)
        cached["served_at"] = served_at.isoformat()
        cached["cache_status"] = "reused_identical_inputs"
        return cached

    model = joblib.load(MODEL_PATH)
    prediction = float(model.predict(model_input[MODEL_FEATURES])[0])
    if not np.isfinite(prediction):
        raise ForecastReadinessError("Model returned a non-finite prediction.")

    # Timestamp the issue after data retrieval and inference, then verify
    # that the target period is still wholly in the future.
    issue_time = _utc_timestamp()
    target_start = _target_month_start_utc(forecast_month)
    if target_start <= issue_time:
        raise ForecastReadinessError(
            "The target month has already started; this result would not be a genuine advance forecast."
        )

    category, description = classify_water_stress(prediction)
    latest = monthly.iloc[-1]
    result = {
        "forecast_type": "one_month_ahead_hydroclimatic_swba_regression",
        "issue_timestamp": issue_time.isoformat(),
        "retrieved_at": retrieved_at.isoformat(),
        "served_at": issue_time.isoformat(),
        "observation_month": model_observation_month.strftime("%Y-%m"),
        "input_cutoff": cutoff.strftime("%Y-%m-%d"),
        "forecast_month": forecast_month.strftime("%Y-%m"),
        "lead_time_months": 1,
        "water_stress_index": prediction,
        "category": category,
        "description": description,
        "warning_authority": "research_only_not_operator_approved",
        "data_quality": {
            "status": "passed",
            "source_dates": source_dates,
            "common_complete_month": observation_month.strftime("%Y-%m"),
            "input_age_days": round(age_days, 2),
            "maximum_input_age_days": MAX_INPUT_AGE_DAYS,
            "monthly_rows": int(len(monthly)),
            "required_months_contiguous": True,
            "missing_values": 0,
        },
        "provenance": {
            "sources": {
                "chirps": "UCSB-CHG/CHIRPS/DAILY",
                "era5_land": "ECMWF/ERA5_LAND/DAILY_AGGR",
            },
            "source_vintage_note": (
                "Current Earth Engine collection snapshot. Reported image dates are not "
                "provider publication timestamps; source revision history is not archived "
                "by this adapter."
            ),
            "model": "Ridge Regression",
            "model_artifact": MODEL_PATH.name,
            "model_sha256": model_sha256,
            "input_fingerprint": fingerprint,
            "feature_count": len(MODEL_FEATURES),
            "features": MODEL_FEATURES,
        },
        "latest_indicators": {
            key: float(latest[key])
            for key in (
                "precipitation_mm", "pet_mm", "temperature_max_c",
                "soil_moisture_layer1", "soil_moisture_layer2",
                "runoff_mm", "solar_radiation",
            )
        },
        # Preserve the existing frontend contract.
        "model_features": {
            feature: float(model_input.iloc[0][feature])
            for feature in MODEL_FEATURES
        },
    }
    _cached_live_forecast = result
    _cached_input_fingerprint = fingerprint
    return result


def clear_live_forecast_cache() -> None:
    global _cached_live_forecast, _cached_input_fingerprint
    _cached_live_forecast = None
    _cached_input_fingerprint = None


if __name__ == "__main__":
    initialize_earth_engine()
    print(generate_live_forecast())
