from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "expanded"
    / "mnjoli_water_stress_monthly_1981_2025_swba.csv"
)
AUDIT_FILE = (
    ROOT
    / "outputs"
    / "expanded"
    / "drought_onset_model_preparation"
    / "data_audit"
    / "audit_report.json"
)
OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "expanded"
    / "drought_onset_model_preparation"
)

DATE_COLUMN = "month_date"
SWBA_COLUMN = "swba_3month"
THRESHOLD = -1.0
HORIZONS = (1, 2, 3)
OUTER_FOLDS = (
    (2006, 2010),
    (2011, 2015),
    (2016, 2020),
    (2021, 2025),
)
HYDRO_FEATURES = (
    "precipitation_mm",
    "pet_mm",
    "volumetric_soil_water_layer_1",
    "water_balance_3month",
)
METHODS = ("training_prevalence", "seasonal_baseline", "current_state", "hydro_core")
SEASONAL_PRIOR_MONTHS = 12.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data() -> tuple[pd.DataFrame, dict[str, Any], str]:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Required monthly input is missing: {INPUT_FILE}")
    if not AUDIT_FILE.is_file():
        raise FileNotFoundError(
            f"Run the drought-onset data audit first: {AUDIT_FILE}"
        )
    audit = json.loads(AUDIT_FILE.read_text(encoding="utf-8"))
    if not audit.get("all_checks_passed"):
        raise ValueError(f"The source data audit did not pass: {AUDIT_FILE}")

    input_hash = sha256_file(INPUT_FILE)
    if input_hash != audit.get("input_sha256"):
        raise ValueError(
            "The monthly input differs from the audited source. "
            "Re-run scripts/audit_drought_onset_data.py before screening."
        )

    data = pd.read_csv(INPUT_FILE, parse_dates=[DATE_COLUMN])
    required = [DATE_COLUMN, SWBA_COLUMN, *HYDRO_FEATURES]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")
    if data[DATE_COLUMN].isna().any():
        raise ValueError("Invalid monthly dates were found.")
    data = data.sort_values(DATE_COLUMN).reset_index(drop=True)
    periods = pd.PeriodIndex(data[DATE_COLUMN].dt.to_period("M"), freq="M")
    expected = pd.period_range(periods.min(), periods.max(), freq="M")
    if periods.has_duplicates or not periods.equals(expected):
        raise ValueError("The monthly data must have unique, consecutive months.")
    if str(periods.min()) != audit["dataset"]["start_month"]:
        raise ValueError("Input start month does not match the passing audit.")
    if str(periods.max()) != audit["dataset"]["end_month"]:
        raise ValueError("Input end month does not match the passing audit.")

    for column in [SWBA_COLUMN, *HYDRO_FEATURES]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["period"] = periods
    data["calendar_month"] = data[DATE_COLUMN].dt.month
    return data.set_index("period", drop=False), audit, input_hash


def build_samples(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    swba = data[SWBA_COLUMN]
    onset_by_period: dict[pd.Period, bool] = {}
    for period in data.index:
        current = swba.loc[period]
        previous = swba.get(period - 1, np.nan)
        onset_by_period[period] = bool(
            pd.notna(current)
            and pd.notna(previous)
            and current <= THRESHOLD
            and previous > THRESHOLD
        )

    records: list[dict[str, Any]] = []
    for origin_period in data.index:
        origin = data.loc[origin_period]
        if pd.isna(origin[SWBA_COLUMN]) or origin[SWBA_COLUMN] <= THRESHOLD:
            continue
        target_periods = [
            origin_period + offset for offset in range(1, horizon + 1)
        ]
        if any(period not in data.index for period in target_periods):
            continue

        record: dict[str, Any] = {
            "forecast_origin": origin_period,
            "origin_month": origin_period.month,
            "onset_within_horizon": int(
                any(onset_by_period[period] for period in target_periods)
            ),
            "onset_months_within_horizon": [
                str(period)
                for period in target_periods
                if onset_by_period[period]
            ],
            "origin_swba": float(origin[SWBA_COLUMN]),
        }
        for feature in HYDRO_FEATURES:
            record[feature] = origin[feature]
        records.append(record)

    samples = pd.DataFrame(records)
    if samples.empty:
        raise ValueError(f"No eligible samples for the {horizon}-month horizon.")
    return samples


def add_hydro_anomalies(
    samples: pd.DataFrame,
    data: pd.DataFrame,
    reference_cutoff: pd.Period,
) -> pd.DataFrame:
    reference = data.loc[data.index <= reference_cutoff]
    transformed = samples.copy()
    for feature in HYDRO_FEATURES:
        grouped = reference.groupby("calendar_month")[feature]
        means = grouped.mean()
        stds = grouped.std(ddof=0)
        month = transformed["origin_month"]
        row_means = month.map(means)
        row_stds = month.map(stds)
        if (
            row_means.isna().any()
            or row_stds.isna().any()
            or (row_stds <= 0).any()
        ):
            raise ValueError(
                f"Invalid training-only calendar-month statistics for {feature}."
            )
        transformed[f"{feature}_anom"] = (
            transformed[feature].to_numpy() - row_means.to_numpy()
        ) / row_stds.to_numpy()
    return transformed


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def seasonal_probabilities(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    prevalence: float,
) -> np.ndarray:
    counts = training.groupby("origin_month")["onset_within_horizon"].agg(
        ["sum", "count"]
    )
    smoothed = (counts["sum"] + SEASONAL_PRIOR_MONTHS * prevalence) / (
        counts["count"] + SEASONAL_PRIOR_MONTHS
    )
    return evaluation["origin_month"].map(smoothed).fillna(prevalence).to_numpy()


def safe_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, probabilities))


def score_fold(
    data: pd.DataFrame,
    samples: pd.DataFrame,
    horizon: int,
    test_start: int,
    test_end: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    first_test_period = pd.Period(f"{test_start}-01", freq="M")
    last_test_period = pd.Period(f"{test_end}-12", freq="M")
    training_cutoff = first_test_period - horizon - 1
    test_origin_end = last_test_period - horizon
    training = samples.loc[
        samples["forecast_origin"] <= training_cutoff
    ].copy()
    evaluation = samples.loc[
        samples["forecast_origin"].map(
            lambda period: test_start <= period.year <= test_end
        )
        & (samples["forecast_origin"] <= test_origin_end)
    ].copy()
    if training.empty or evaluation.empty:
        raise ValueError(
            f"Empty split for horizon {horizon}, fold {test_start}-{test_end}."
        )
    if training["onset_within_horizon"].nunique() != 2:
        raise ValueError(
            f"Training split lacks both classes for horizon {horizon}, "
            f"fold {test_start}-{test_end}."
        )

    prevalence = float(training["onset_within_horizon"].mean())
    evaluation_labels = evaluation["onset_within_horizon"].to_numpy(dtype=int)
    model_predictions: dict[str, np.ndarray] = {
        "training_prevalence": np.full(len(evaluation), prevalence),
        "seasonal_baseline": seasonal_probabilities(
            training, evaluation, prevalence
        ),
    }

    state_model = make_model()
    state_model.fit(training[["origin_swba"]], training["onset_within_horizon"])
    model_predictions["current_state"] = state_model.predict_proba(
        evaluation[["origin_swba"]]
    )[:, 1]

    hydro_training = add_hydro_anomalies(
        training, data, training_cutoff
    )
    hydro_evaluation = add_hydro_anomalies(
        evaluation, data, training_cutoff
    )
    hydro_columns = [f"{feature}_anom" for feature in HYDRO_FEATURES]
    hydro_model = make_model()
    hydro_model.fit(
        hydro_training[hydro_columns],
        hydro_training["onset_within_horizon"],
    )
    model_predictions["hydro_core"] = hydro_model.predict_proba(
        hydro_evaluation[hydro_columns]
    )[:, 1]

    event_months = sorted(
        {
            event_month
            for value in evaluation["onset_months_within_horizon"]
            for event_month in value
        }
    )
    fold_id = f"{test_start}-{test_end}"
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for method in METHODS:
        probabilities = model_predictions[method]
        brier = float(brier_score_loss(evaluation_labels, probabilities))
        baseline_brier = float(
            brier_score_loss(
                evaluation_labels, np.full(len(evaluation), prevalence)
            )
        )
        fold_rows.append(
            {
                "horizon_months": horizon,
                "test_period": fold_id,
                "training_cutoff": str(training_cutoff),
                "test_origin_end": str(test_origin_end),
                "method": method,
                "training_origins": len(training),
                "training_positive_windows": int(
                    training["onset_within_horizon"].sum()
                ),
                "test_origins": len(evaluation),
                "positive_test_windows": int(evaluation_labels.sum()),
                "unique_onset_months_in_test_windows": len(event_months),
                "brier_score": brier,
                "prevalence_baseline_brier": baseline_brier,
                "brier_skill_vs_prevalence": (
                    1 - brier / baseline_brier if baseline_brier > 0 else np.nan
                ),
                "average_precision": safe_average_precision(
                    evaluation_labels, probabilities
                ),
                "threshold_or_warning_episodes_evaluated": False,
            }
        )
        for sample, probability in zip(
            evaluation.itertuples(index=False), probabilities
        ):
            prediction_rows.append(
                {
                    "horizon_months": horizon,
                    "test_period": fold_id,
                    "method": method,
                    "forecast_origin": str(sample.forecast_origin),
                    "onset_within_horizon": int(sample.onset_within_horizon),
                    "predicted_probability": float(probability),
                    "training_prevalence": prevalence,
                }
            )
    return fold_rows, prediction_rows


def summarize(
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (horizon, method), group in predictions.groupby(
        ["horizon_months", "method"], sort=True
    ):
        labels = group["onset_within_horizon"].to_numpy(dtype=int)
        probabilities = group["predicted_probability"].to_numpy(dtype=float)
        baselines = group["training_prevalence"].to_numpy(dtype=float)
        brier = float(brier_score_loss(labels, probabilities))
        baseline_brier = float(brier_score_loss(labels, baselines))
        fold_group = fold_metrics.loc[
            (fold_metrics["horizon_months"] == horizon)
            & (fold_metrics["method"] == method)
        ]
        rows.append(
            {
                "horizon_months": int(horizon),
                "method": method,
                "test_origins": len(group),
                "positive_test_windows": int(labels.sum()),
                "folds_with_positive_brier_skill": int(
                    (fold_group["brier_skill_vs_prevalence"] > 0).sum()
                ),
                "folds_evaluated": len(fold_group),
                "pooled_brier_score": brier,
                "pooled_prevalence_baseline_brier": baseline_brier,
                "pooled_brier_skill_vs_prevalence": (
                    1 - brier / baseline_brier if baseline_brier > 0 else np.nan
                ),
                "pooled_average_precision": safe_average_precision(
                    labels, probabilities
                ),
                "threshold_or_warning_episodes_evaluated": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data, audit, input_hash = load_data()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = OUTPUT_ROOT / f"feasibility_screen_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        samples = build_samples(data, horizon)
        for test_start, test_end in OUTER_FOLDS:
            metrics, predictions = score_fold(
                data, samples, horizon, test_start, test_end
            )
            fold_rows.extend(metrics)
            prediction_rows.extend(predictions)

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize(fold_metrics, predictions)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "outer_predictions.csv", index=False)
    summary.to_csv(output_dir / "pooled_summary.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT_FILE),
        "input_sha256": input_hash,
        "audit_file": str(AUDIT_FILE),
        "audit_passed": bool(audit["all_checks_passed"]),
        "source_period": {
            "start_month": audit["dataset"]["start_month"],
            "end_month": audit["dataset"]["end_month"],
        },
        "outcome": (
            "At least one SWBA <= -1 onset during the next 1, 2, or 3 "
            "months; forecast origins must have SWBA > -1."
        ),
        "methods": list(METHODS),
        "feature_sets": {
            "current_state": ["origin_swba"],
            "hydro_core": list(HYDRO_FEATURES),
            "seasonal_baseline": (
                "Origin-calendar-month onset-window prevalence smoothed "
                f"toward training prevalence with {SEASONAL_PRIOR_MONTHS:g} "
                "prior months."
            ),
        },
        "evaluation": (
            "Chronological 2006-2025 folds previously examined during "
            "development; this is a retrospective feasibility screen, not "
            "independent validation. Training outcome windows end before "
            "each test block. Test origins are restricted so their complete "
            "outcome windows end within each block."
        ),
        "limitations": [
            "No warning threshold or false-warning episode rate is evaluated.",
            "Adjacent forecast windows overlap and are statistically dependent.",
            "Historical predictor values are not proof of issue-time availability.",
            "The final test fold excludes origins whose outcome window extends beyond 2025.",
        ],
        "output_files": [
            "fold_metrics.csv",
            "outer_predictions.csv",
            "pooled_summary.csv",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Feasibility screen written to: {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
