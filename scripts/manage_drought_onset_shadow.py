from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_nested_drought_onset_backtest as backtest


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_INPUT = backtest.INPUT_FILE
HISTORICAL_AUDIT = backtest.AUDIT_REPORT
BACKTEST_DIR = backtest.OUTPUT_DIR
REFRESH_ROOT = ROOT / "outputs" / "expanded" / "shadow_evaluation_data"
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "expanded"
    / "drought_onset_model_preparation"
    / "prospective_shadow"
)
FORECAST_LOG = OUTPUT_DIR / "shadow_forecasts.csv"
SCORE_LOG = OUTPUT_DIR / "shadow_scores.csv"

DATE_COLUMN = backtest.DATE_COLUMN
SWBA_COLUMN = backtest.SWBA_COLUMN
TRAINING_CUTOFF = pd.Period("2015-12", freq="M")
VALIDATION_START = 2016
VALIDATION_END = 2020
LOCKED_FEATURE_SET_NAME = "hydro_core"
LOCKED_FEATURE_SET = backtest.FEATURE_SETS[LOCKED_FEATURE_SET_NAME]
LOCKED_C = 1.0
LOCKED_CLASS_WEIGHT: str | None = None
LOCKED_THRESHOLD = 0.25697561208197944
LOCKED_TRAINING_SAMPLES = 349
LOCKED_TRAINING_ONSETS = 27
LOCKED_VALIDATION_SAMPLES = 52
LOCKED_VALIDATION_ONSETS = 2
LOCKED_VALIDATION_FALSE_WARNING_EPISODES = 2
LOCKED_VALIDATION_FALSE_WARNING_MONTHS = 2
LOCK_VERSION = "mnjoli-onset-shadow-cv-fold-2021-2025-v1"

FORECAST_COLUMNS = (
    "forecast_id",
    "model_version",
    "forecast_created_at_utc",
    "data_refreshed_at_utc",
    "data_available_through",
    "forecast_origin",
    "target_period",
    "origin_swba",
    "predicted_probability",
    "frozen_threshold",
    "threshold_exceeded_research_only",
    "alert_issued",
    "evaluation_status",
    "training_cutoff",
    "training_samples",
    "training_onsets",
    "validation_period",
    "validation_samples",
    "validation_onsets",
    "validation_false_warning_episodes",
    "validation_false_warning_months",
    "feature_set",
    "features_json",
    "historical_input_sha256",
    "refreshed_input_sha256",
    "historical_audit_sha256",
    "refreshed_audit_sha256",
    "backtest_selection_sha256",
    "backtest_source_sha256",
    "shadow_source_sha256",
    "forecast_origin_values_json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit/provenance file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_refreshed_snapshot(
    input_file: Path,
    audit_file: Path,
    manifest_file: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], str]:
    audit = read_json(audit_file)
    manifest = read_json(manifest_file)
    if not audit.get("all_checks_passed"):
        raise ValueError(f"Refreshed data audit failed: {audit_file}")
    input_hash = sha256_file(input_file)
    if input_hash != audit.get("input_sha256"):
        raise ValueError("Refreshed input hash does not match its audit report.")
    if not manifest.get("audit_passed"):
        raise ValueError("Refresh manifest does not confirm a passing audit.")
    if manifest.get("audit_input_sha256") != input_hash:
        raise ValueError("Refresh manifest hash does not match refreshed input.")

    data = pd.read_csv(input_file, parse_dates=[DATE_COLUMN])
    if data[DATE_COLUMN].isna().any():
        raise ValueError("Refreshed data contains invalid monthly dates.")
    data = data.sort_values(DATE_COLUMN).reset_index(drop=True)
    periods = data[DATE_COLUMN].dt.to_period("M")
    if periods.duplicated().any():
        raise ValueError("Refreshed data contains duplicate monthly periods.")
    expected = pd.period_range(periods.min(), periods.max(), freq="M")
    if not pd.PeriodIndex(periods, freq="M").equals(expected):
        raise ValueError("Refreshed data does not contain consecutive months.")

    audited_end = pd.Period(audit["dataset"]["end_month"], freq="M")
    available_through = pd.Period(
        manifest["common_complete_data_through"][:7], freq="M"
    )
    if periods.max() != audited_end or audited_end != available_through:
        raise ValueError(
            "Refreshed input end month, audit end month, and manifest "
            "complete-coverage month do not agree."
        )
    return data, audit, manifest, input_hash


def validate_frozen_backtest() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data, samples, audit = backtest.load_and_gate_data()
    selection_file = BACKTEST_DIR / "selected_models_and_thresholds.csv"
    metadata_file = BACKTEST_DIR / "run_metadata.json"
    selection = pd.read_csv(selection_file)
    metadata = read_json(metadata_file)
    chosen = selection.loc[selection["test_period"] == "2021-2025"]
    if len(chosen) != 1:
        raise ValueError("Expected one frozen 2021-2025 backtest selection.")
    row = chosen.iloc[0]
    checks = {
        "training_cutoff": str(row["training_cutoff"]) == str(TRAINING_CUTOFF),
        "validation_period": str(row["validation_period"]) == "2016-2020",
        "selected_feature_set": (
            row["selected_feature_set"] == LOCKED_FEATURE_SET_NAME
        ),
        "selected_features": (
            row["selected_features"] == ", ".join(LOCKED_FEATURE_SET)
        ),
        "C": float(row["C"]) == LOCKED_C,
        "class_weight": row["class_weight"] == "none",
        "validation_threshold": (
            np.isclose(
                float(row["validation_threshold"]),
                LOCKED_THRESHOLD,
                rtol=0,
                atol=1e-12,
            )
        ),
        "training_samples": int(row["training_samples"])
        == LOCKED_TRAINING_SAMPLES,
        "training_onsets": int(row["training_onsets"])
        == LOCKED_TRAINING_ONSETS,
        "validation_samples": int(row["validation_samples"])
        == LOCKED_VALIDATION_SAMPLES,
        "validation_onsets": int(row["validation_onsets"])
        == LOCKED_VALIDATION_ONSETS,
        "validation_false_warning_episodes": int(
            row["validation_false_warning_episodes"]
        )
        == LOCKED_VALIDATION_FALSE_WARNING_EPISODES,
        "validation_false_warning_months": int(
            row["validation_false_warning_months"]
        )
        == LOCKED_VALIDATION_FALSE_WARNING_MONTHS,
        "historical_input_hash": (
            metadata["input_sha256"] == audit["input_sha256"]
        ),
        "feature_definition": (
            metadata["feature_sets"][LOCKED_FEATURE_SET_NAME]
            == list(LOCKED_FEATURE_SET)
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Frozen model artifacts differ from the documented lock: "
            + ", ".join(failed)
        )
    return data, samples, audit


def append_csv(path: Path, row: dict[str, Any], columns: tuple[str, ...]) -> None:
    frame = pd.DataFrame([row], columns=columns)
    if path.exists():
        existing_columns = tuple(pd.read_csv(path, nrows=0).columns)
        if existing_columns != columns:
            raise ValueError(f"Existing log has an unexpected schema: {path}")
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def validate_forecast_issue_timing(
    target_period: pd.Period,
    refreshed_at: str,
    issue_time: datetime,
) -> None:
    target_start = target_period.to_timestamp().tz_localize("UTC")
    refresh_time = pd.Timestamp(refreshed_at)
    if refresh_time.tzinfo is None:
        raise ValueError("Refresh timestamp must include a timezone.")
    if refresh_time.tz_convert("UTC") >= target_start:
        raise ValueError(
            f"Input snapshot became available after target {target_period} "
            "began; refusing a retrospective forecast."
        )
    issue_timestamp = pd.Timestamp(issue_time)
    if issue_timestamp.tzinfo is None:
        raise ValueError("Forecast issue timestamp must include a timezone.")
    if issue_timestamp.tz_convert("UTC") >= target_start:
        raise ValueError(
            f"Target {target_period} has already begun; refusing a "
            "retrospective forecast."
        )


def make_forecast(
    input_file: Path,
    audit_file: Path,
    manifest_file: Path,
) -> dict[str, Any]:
    historical, samples, historical_audit = validate_frozen_backtest()
    refreshed, refreshed_audit, manifest, refreshed_hash = (
        load_refreshed_snapshot(input_file, audit_file, manifest_file)
    )
    origin_period = refreshed[DATE_COLUMN].dt.to_period("M").max()
    target_period = origin_period + 1
    available_through = pd.Period(
        manifest["common_complete_data_through"][:7], freq="M"
    )
    if origin_period != available_through:
        raise ValueError(
            "The latest monthly row is not the latest complete month in "
            "the refresh manifest."
        )
    issue_time = datetime.now(timezone.utc)
    validate_forecast_issue_timing(
        target_period,
        manifest["refresh_run_utc"],
        issue_time,
    )
    if (refreshed[DATE_COLUMN].dt.to_period("M") == target_period).any():
        raise ValueError(
            f"Target {target_period} is already in the input; refusing to "
            "create a post-outcome forecast."
        )
    if FORECAST_LOG.is_file():
        existing = pd.read_csv(FORECAST_LOG)
        if existing["target_period"].astype(str).eq(str(target_period)).any():
            raise ValueError(
                f"A shadow forecast for {target_period} already exists; "
                "forecasts are append-only."
            )

    indexed = refreshed.assign(
        period=refreshed[DATE_COLUMN].dt.to_period("M")
    ).set_index("period")
    origin = indexed.loc[origin_period]
    if pd.isna(origin[SWBA_COLUMN]):
        raise ValueError(f"Origin SWBA is missing for {origin_period}.")
    if float(origin[SWBA_COLUMN]) <= backtest.THRESHOLD:
        raise ValueError(
            f"Origin {origin_period} is already in drought; an onset warning "
            "is not eligible under the frozen definition."
        )

    sample: dict[str, Any] = {
        "target_period": target_period,
        "target_month": target_period.to_timestamp(),
        "forecast_origin": origin_period.to_timestamp(),
        "onset_label": 0,
        "origin_swba": float(origin[SWBA_COLUMN]),
        "origin_month": origin_period.month,
    }
    for feature in backtest.MODEL_HYDRO_FEATURES:
        if feature not in indexed.columns:
            raise ValueError(f"Forecast input is missing {feature}.")
        sample[feature] = origin[feature]
    sample["origin_month_sin"] = np.sin(
        2 * np.pi * sample["origin_month"] / 12
    )
    sample["origin_month_cos"] = np.cos(
        2 * np.pi * sample["origin_month"] / 12
    )
    sample_frame = pd.DataFrame([sample])
    columns = backtest.model_columns(LOCKED_FEATURE_SET)

    training = samples.loc[
        samples["target_period"] <= TRAINING_CUTOFF
    ].copy()
    if len(training) != LOCKED_TRAINING_SAMPLES or int(
        training["onset_label"].sum()
    ) != LOCKED_TRAINING_ONSETS:
        raise ValueError("Frozen training cohort no longer matches its lock.")
    training_x = backtest.transform_features(
        training,
        historical,
        TRAINING_CUTOFF,
    )
    forecast_x = backtest.transform_features(
        sample_frame,
        historical,
        TRAINING_CUTOFF,
    )
    model = backtest.make_model(LOCKED_C, LOCKED_CLASS_WEIGHT)
    model.fit(training_x[columns], training_x["onset_label"])
    probability = float(model.predict_proba(forecast_x[columns])[:, 1][0])
    feature_values = {
        column: (
            None
            if pd.isna(forecast_x.iloc[0][column])
            else float(forecast_x.iloc[0][column])
        )
        for column in columns
    }
    created_at = issue_time.isoformat()
    selection_file = BACKTEST_DIR / "selected_models_and_thresholds.csv"
    backtest_source = Path(backtest.__file__)
    shadow_source = Path(__file__)
    lock_payload = {
        "lock_version": LOCK_VERSION,
        "historical_input_sha256": sha256_file(HISTORICAL_INPUT),
        "backtest_selection_sha256": sha256_file(selection_file),
        "backtest_source_sha256": sha256_file(backtest_source),
        "shadow_source_sha256": sha256_file(shadow_source),
        "training_cutoff": str(TRAINING_CUTOFF),
        "feature_set": LOCKED_FEATURE_SET_NAME,
        "features": list(LOCKED_FEATURE_SET),
        "C": LOCKED_C,
        "class_weight": "none",
        "threshold": LOCKED_THRESHOLD,
    }
    model_version = hashlib.sha256(
        json.dumps(lock_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    forecast_id = hashlib.sha256(
        f"{model_version}|{target_period}|{refreshed_hash}".encode("utf-8")
    ).hexdigest()[:20]
    return {
        "forecast_id": forecast_id,
        "model_version": f"{LOCK_VERSION}-{model_version}",
        "forecast_created_at_utc": created_at,
        "data_refreshed_at_utc": manifest["refresh_run_utc"],
        "data_available_through": str(available_through),
        "forecast_origin": str(origin_period),
        "target_period": str(target_period),
        "origin_swba": float(origin[SWBA_COLUMN]),
        "predicted_probability": probability,
        "frozen_threshold": LOCKED_THRESHOLD,
        "threshold_exceeded_research_only": int(
            probability >= LOCKED_THRESHOLD
        ),
        "alert_issued": False,
        "evaluation_status": "prospective_shadow_non_alerting",
        "training_cutoff": str(TRAINING_CUTOFF),
        "training_samples": len(training),
        "training_onsets": int(training["onset_label"].sum()),
        "validation_period": f"{VALIDATION_START}-{VALIDATION_END}",
        "validation_samples": LOCKED_VALIDATION_SAMPLES,
        "validation_onsets": LOCKED_VALIDATION_ONSETS,
        "validation_false_warning_episodes": (
            LOCKED_VALIDATION_FALSE_WARNING_EPISODES
        ),
        "validation_false_warning_months": (
            LOCKED_VALIDATION_FALSE_WARNING_MONTHS
        ),
        "feature_set": LOCKED_FEATURE_SET_NAME,
        "features_json": json.dumps(list(LOCKED_FEATURE_SET)),
        "historical_input_sha256": sha256_file(HISTORICAL_INPUT),
        "refreshed_input_sha256": refreshed_hash,
        "historical_audit_sha256": historical_audit["input_sha256"],
        "refreshed_audit_sha256": refreshed_audit["input_sha256"],
        "backtest_selection_sha256": sha256_file(selection_file),
        "backtest_source_sha256": sha256_file(backtest_source),
        "shadow_source_sha256": sha256_file(shadow_source),
        "forecast_origin_values_json": json.dumps(feature_values, sort_keys=True),
    }


def score_pending_forecasts(
    input_file: Path,
    audit_file: Path,
    manifest_file: Path,
) -> int:
    if not FORECAST_LOG.is_file():
        raise FileNotFoundError(f"No shadow forecasts found at {FORECAST_LOG}")
    refreshed, audit, manifest, _ = load_refreshed_snapshot(
        input_file, audit_file, manifest_file
    )
    forecasts = pd.read_csv(FORECAST_LOG, dtype={"target_period": str})
    if forecasts["alert_issued"].astype(str).str.lower().eq("true").any():
        raise ValueError("Shadow log unexpectedly contains an issued alert.")
    already_scored: set[str] = set()
    if SCORE_LOG.exists():
        already_scored = set(
            pd.read_csv(SCORE_LOG, dtype={"target_period": str})[
                "target_period"
            ].astype(str)
        )

    indexed = refreshed.assign(
        period=refreshed[DATE_COLUMN].dt.to_period("M")
    ).set_index("period")
    latest_complete = pd.Period(
        manifest["common_complete_data_through"][:7], freq="M"
    )
    new_scores = 0
    for _, forecast in forecasts.iterrows():
        target_period = pd.Period(forecast["target_period"], freq="M")
        if str(target_period) in already_scored:
            continue
        if target_period > latest_complete or target_period not in indexed.index:
            continue
        origin_period = target_period - 1
        if origin_period not in indexed.index:
            raise ValueError(
                f"Forecast origin {origin_period} is absent from refreshed data."
            )
        origin_swba = indexed.loc[origin_period, SWBA_COLUMN]
        target_swba = indexed.loc[target_period, SWBA_COLUMN]
        if pd.isna(origin_swba) or pd.isna(target_swba):
            raise ValueError(
                f"Cannot score {target_period}: required SWBA value is missing."
            )
        if float(origin_swba) <= backtest.THRESHOLD:
            raise ValueError(
                f"Stored forecast origin {origin_period} violates the "
                "non-drought eligibility definition."
            )
        observed_onset = int(float(target_swba) <= backtest.THRESHOLD)
        predicted_warning = int(
            float(forecast["threshold_exceeded_research_only"]) == 1
        )
        score_row = {
            "forecast_id": forecast["forecast_id"],
            "model_version": forecast["model_version"],
            "forecast_created_at_utc": forecast["forecast_created_at_utc"],
            "score_created_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_period": str(target_period),
            "forecast_origin": str(origin_period),
            "data_available_through": str(latest_complete),
            "predicted_probability": float(
                forecast["predicted_probability"]
            ),
            "frozen_threshold": float(forecast["frozen_threshold"]),
            "predicted_warning_research_only": predicted_warning,
            "alert_issued": False,
            "observed_origin_swba": float(origin_swba),
            "observed_target_swba": float(target_swba),
            "observed_onset": observed_onset,
            "hit": int(predicted_warning == 1 and observed_onset == 1),
            "false_warning": int(
                predicted_warning == 1 and observed_onset == 0
            ),
            "miss": int(predicted_warning == 0 and observed_onset == 1),
            "refreshed_input_sha256": sha256_file(input_file),
            "refreshed_audit_sha256": audit["input_sha256"],
        }
        append_csv(
            SCORE_LOG,
            score_row,
            tuple(score_row.keys()),
        )
        already_scored.add(str(target_period))
        new_scores += 1
    return new_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create non-alerting drought-onset shadow forecasts and score "
            "them only after their target month is fully observed."
        )
    )
    parser.add_argument(
        "mode",
        choices=("forecast", "score"),
        help="Create one frozen forecast or score forecasts whose targets are complete.",
    )
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--audit-file", type=Path)
    parser.add_argument("--manifest-file", type=Path)
    return parser.parse_args()


def latest_passing_refresh() -> tuple[Path, Path, Path]:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for manifest_file in REFRESH_ROOT.glob(
        "refresh_through_*/refresh_manifest.json"
    ):
        manifest = read_json(manifest_file)
        if not manifest.get("audit_passed"):
            continue
        refresh_time = pd.Timestamp(manifest["refresh_run_utc"])
        if refresh_time.tzinfo is None:
            raise ValueError(
                f"Refresh manifest timestamp must include a timezone: "
                f"{manifest_file}"
            )
        candidates.append((refresh_time, manifest_file))
    if not candidates:
        raise FileNotFoundError(
            f"No passing refresh snapshots found under {REFRESH_ROOT}"
        )
    _, manifest_file = max(candidates, key=lambda candidate: candidate[0])
    refresh_dir = manifest_file.parent
    return (
        refresh_dir / "mnjoli_water_stress_monthly_updated_swba.csv",
        refresh_dir / "data_audit" / "audit_report.json",
        manifest_file,
    )


def main() -> None:
    args = parse_args()
    if args.input_file and args.audit_file and args.manifest_file:
        input_file = args.input_file
        audit_file = args.audit_file
        manifest_file = args.manifest_file
    else:
        latest_input, latest_audit, latest_manifest = latest_passing_refresh()
        input_file = args.input_file or latest_input
        audit_file = args.audit_file or latest_audit
        manifest_file = args.manifest_file or latest_manifest
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "forecast":
        row = make_forecast(
            input_file,
            audit_file,
            manifest_file,
        )
        append_csv(FORECAST_LOG, row, FORECAST_COLUMNS)
        print(
            "Archived non-alerting drought-onset shadow forecast: "
            f"{row['forecast_origin']} -> {row['target_period']}"
        )
        print(f"Probability: {row['predicted_probability']:.6f}")
        print(f"Frozen threshold: {row['frozen_threshold']:.6f}")
        print(
            "Research threshold exceeded: "
            f"{bool(row['threshold_exceeded_research_only'])}"
        )
        print("Operational alert issued: no")
        print(f"Forecast log: {FORECAST_LOG}")
    else:
        count = score_pending_forecasts(
            input_file,
            audit_file,
            manifest_file,
        )
        if count:
            print(f"Scored {count} completed target month(s).")
        else:
            print("No unscored forecast target is fully observed yet.")
        print(f"Score log: {SCORE_LOG}")


if __name__ == "__main__":
    main()
