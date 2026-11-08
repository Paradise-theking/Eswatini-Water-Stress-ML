from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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
AUDIT_REPORT = (
    ROOT
    / "outputs"
    / "expanded"
    / "drought_onset_model_preparation"
    / "data_audit"
    / "audit_report.json"
)
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "expanded"
    / "drought_onset_model_preparation"
    / "nested_chronological_backtest_decorrelated_features_1month"
)

DATE_COLUMN = "month_date"
SWBA_COLUMN = "swba_3month"
THRESHOLD = -1.0
INNER_VALIDATION_START = 1991
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
ADDITIONAL_HYDRO_FEATURES = (
    "water_balance_6month",
    "volumetric_soil_water_layer_2",
    "temperature_max_c",
    "surface_solar_radiation_downwards_sum",
)
MODEL_HYDRO_FEATURES = HYDRO_FEATURES + ADDITIONAL_HYDRO_FEATURES
FEATURE_SETS = {
    "current_state": ("origin_swba",),
    "hydro_core": HYDRO_FEATURES,
    "state_plus_hydro": (
        "origin_swba",
        "precipitation_mm",
        "pet_mm",
        "volumetric_soil_water_layer_1",
    ),
    "state_plus_season": (
        "origin_swba",
        "origin_month_sin",
        "origin_month_cos",
    ),
    "hydro_memory": (
        "precipitation_mm",
        "pet_mm",
        "water_balance_3month",
        "water_balance_6month",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
    ),
    "atmospheric_demand": (
        "pet_mm",
        "temperature_max_c",
        "surface_solar_radiation_downwards_sum",
    ),
    "antecedent_stress": (
        "origin_swba",
        "water_balance_6month",
        "volumetric_soil_water_layer_2",
        "pet_mm",
        "temperature_max_c",
    ),
}
REGULARIZATION_STRENGTHS = (0.1, 1.0, 10.0)
CLASS_WEIGHTS = (None, "balanced")
CALENDAR_MONTH_PRIOR_STRENGTH = 12.0
MAX_FALSE_WARNING_EPISODES_PER_YEAR = 0.5
MAX_FALSE_WARNING_MONTHS_PER_YEAR = 3.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_gate_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not AUDIT_REPORT.is_file():
        raise FileNotFoundError(
            f"Run the data audit before model evaluation: {AUDIT_REPORT}"
        )
    audit = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
    if not audit.get("all_checks_passed"):
        raise ValueError("The drought-onset data audit did not pass.")

    input_hash = sha256_file(INPUT_FILE)
    if input_hash != audit.get("input_sha256"):
        raise ValueError(
            "The audited input data has changed. Re-run "
            "scripts/audit_drought_onset_data.py before modeling."
        )

    data = pd.read_csv(INPUT_FILE, parse_dates=[DATE_COLUMN])
    required = [
        DATE_COLUMN,
        SWBA_COLUMN,
        *MODEL_HYDRO_FEATURES,
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Required input columns are missing: {missing}")
    if data[DATE_COLUMN].isna().any():
        raise ValueError("Invalid monthly dates were found.")

    data = data.sort_values(DATE_COLUMN).reset_index(drop=True)
    periods = data[DATE_COLUMN].dt.to_period("M")
    if periods.duplicated().any():
        raise ValueError("Duplicate monthly periods were found.")
    expected = pd.period_range(periods.min(), periods.max(), freq="M")
    if not pd.PeriodIndex(periods, freq="M").equals(expected):
        raise ValueError("The monthly input contains missing periods.")

    samples = build_samples(data)
    expected_samples = int(
        audit["eligible_forecast_samples"]["rows_by_lead"]["1"]
    )
    if len(samples) != expected_samples:
        raise ValueError(
            f"Eligible sample count {len(samples)} does not match the "
            f"audited count {expected_samples}."
        )
    if int(samples["onset_label"].sum()) != int(
        audit["onsets"]["onset_events"]
    ):
        raise ValueError("Onset labels do not match the audited event count.")
    return data, samples, audit


def build_samples(data: pd.DataFrame) -> pd.DataFrame:
    indexed = data.copy()
    indexed["period"] = indexed[DATE_COLUMN].dt.to_period("M")
    indexed = indexed.set_index("period")
    swba = indexed[SWBA_COLUMN]
    eligible_rows: list[dict[str, Any]] = []

    for period, target in indexed.iterrows():
        previous_swba = swba.get(period - 1, np.nan)
        if pd.isna(target[SWBA_COLUMN]) or pd.isna(previous_swba):
            continue
        is_onset = (
            target[SWBA_COLUMN] <= THRESHOLD
            and previous_swba > THRESHOLD
        )
        origin_period = period - 1
        origin = indexed.loc[origin_period]
        if origin[SWBA_COLUMN] <= THRESHOLD:
            continue

        row: dict[str, Any] = {
            "target_period": period,
            "target_month": period.to_timestamp(),
            "forecast_origin": origin_period.to_timestamp(),
            "onset_label": int(is_onset),
            "origin_swba": float(origin[SWBA_COLUMN]),
            "origin_month": origin_period.month,
        }
        for feature in HYDRO_FEATURES:
            row[feature] = origin[feature]
        for feature in ADDITIONAL_HYDRO_FEATURES:
            row[feature] = origin[feature]
        eligible_rows.append(row)

    samples = pd.DataFrame(eligible_rows)
    samples = samples.dropna(
        subset=["origin_swba", *HYDRO_FEATURES]
    ).reset_index(drop=True)
    samples["origin_month_sin"] = np.sin(
        2 * np.pi * samples["origin_month"] / 12
    )
    samples["origin_month_cos"] = np.cos(
        2 * np.pi * samples["origin_month"] / 12
    )
    return samples


def transform_features(
    samples: pd.DataFrame,
    data: pd.DataFrame,
    reference_cutoff: pd.Period,
) -> pd.DataFrame:
    reference = data.loc[
        data[DATE_COLUMN].dt.to_period("M") <= reference_cutoff
    ].copy()
    if reference.empty:
        raise ValueError(f"No historical predictors through {reference_cutoff}.")
    reference["calendar_month"] = reference[DATE_COLUMN].dt.month
    transformed = samples.copy()

    for feature in MODEL_HYDRO_FEATURES:
        stats = reference.groupby("calendar_month")[feature].agg(
            mean="mean",
            std=lambda values: values.std(ddof=0),
        )
        months = transformed["origin_month"]
        means = months.map(stats["mean"])
        stds = months.map(stats["std"])
        if means.isna().any() or stds.isna().any() or (stds <= 0).any():
            raise ValueError(
                f"Invalid historical calendar-month statistics for {feature}."
            )
        transformed[f"{feature}_anom"] = (
            transformed[feature].to_numpy() - means.to_numpy()
        ) / stds.to_numpy()
    return transformed


def model_columns(feature_set: tuple[str, ...]) -> list[str]:
    return [
        f"{name}_anom" if name in MODEL_HYDRO_FEATURES else name
        for name in feature_set
    ]


def make_model(c_value: float, class_weight: str | None) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def select_recall_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_periods: pd.Series,
    evaluation_years: int,
) -> dict[str, float | int]:
    candidates = np.unique(
        np.concatenate(([0.0], probabilities, [1.000001]))
    )
    results = []
    period_values = pd.PeriodIndex(target_periods, freq="M")
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        false_warning_months = int(
            np.sum((predictions == 1) & (labels == 0))
        )
        warning_frame = pd.DataFrame(
            {
                "target_period": period_values,
                "target_month": period_values.to_timestamp(),
                "actual_onset": labels,
                "predicted_onset": predictions,
            }
        )
        episode_metrics = episode_counts(warning_frame)
        false_episodes = int(episode_metrics["false_warning_episodes"])
        false_episodes_per_year = (
            false_episodes / evaluation_years
            if evaluation_years
            else float("inf")
        )
        false_months_per_year = (
            false_warning_months / evaluation_years
            if evaluation_years
            else float("inf")
        )
        if (
            false_episodes_per_year
            > MAX_FALSE_WARNING_EPISODES_PER_YEAR
            or false_months_per_year
            > MAX_FALSE_WARNING_MONTHS_PER_YEAR
        ):
            continue
        results.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "precision": float(
                    precision_score(labels, predictions, zero_division=0)
                ),
                "recall": float(
                    recall_score(labels, predictions, zero_division=0)
                ),
                "false_warning_episodes": false_episodes,
                "false_warning_months": false_warning_months,
                "false_warning_episodes_per_year": float(
                    false_episodes_per_year
                ),
                "false_warning_months_per_year": float(
                    false_months_per_year
                ),
                "predicted_warnings": int(predictions.sum()),
            }
        )
    if not results:
        raise ValueError(
            "No threshold met the configured false-warning episode cap."
        )
    return max(
        results,
        key=lambda row: (
            row["recall"],
            row["precision"],
            -row["predicted_warnings"],
            row["threshold"],
        ),
    )


def safe_average_precision(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    if labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, probabilities))


def probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
) -> dict[str, float]:
    brier = float(brier_score_loss(labels, probabilities))
    baseline_brier = float(
        brier_score_loss(labels, baseline_probabilities)
    )
    return {
        "average_precision": safe_average_precision(labels, probabilities),
        "brier_score": brier,
        "baseline_brier_score": baseline_brier,
        "brier_skill_score": (
            1.0 - brier / baseline_brier
            if baseline_brier > 0
            else float("nan")
        ),
    }


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    baseline_probabilities: np.ndarray,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "samples": int(len(labels)),
        "observed_onsets": int(labels.sum()),
        "onset_prevalence": float(labels.mean()),
        "predicted_warnings": int(predictions.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_warning_months": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        **probability_metrics(labels, probabilities, baseline_probabilities),
    }


def inner_splits(
    samples: pd.DataFrame,
    outer_validation_start: int,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, pd.Period]]:
    splits = []
    for validation_start in range(
        INNER_VALIDATION_START, outer_validation_start, 5
    ):
        validation_end = validation_start + 4
        validation_first = pd.Period(f"{validation_start}-01", freq="M")
        cutoff = validation_first - 1
        training = samples.loc[samples["target_period"] <= cutoff].copy()
        validation = samples.loc[
            samples["target_period"].dt.year.between(
                validation_start, validation_end
            )
        ].copy()
        if training.empty or validation.empty:
            continue
        if training["onset_label"].nunique() != 2:
            continue
        if validation["onset_label"].nunique() != 2:
            continue
        splits.append((validation_start, training, validation, cutoff))
    if not splits:
        raise ValueError(
            f"No usable inner chronological folds before {outer_validation_start}."
        )
    return splits


def select_candidate(
    data: pd.DataFrame,
    samples: pd.DataFrame,
    outer_validation_start: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splits = inner_splits(samples, outer_validation_start)
    rows = []
    for feature_name, feature_set in FEATURE_SETS.items():
        columns = model_columns(feature_set)
        for c_value in REGULARIZATION_STRENGTHS:
            for class_weight in CLASS_WEIGHTS:
                fold_scores = []
                for validation_start, training, validation, cutoff in splits:
                    train_x = transform_features(training, data, cutoff)
                    validation_x = transform_features(validation, data, cutoff)
                    model = make_model(c_value, class_weight)
                    model.fit(train_x[columns], train_x["onset_label"])
                    probabilities = model.predict_proba(
                        validation_x[columns]
                    )[:, 1]
                    prevalence = float(training["onset_label"].mean())
                    metrics = probability_metrics(
                        validation_x["onset_label"].to_numpy(),
                        probabilities,
                        np.full(len(validation_x), prevalence),
                    )
                    fold_scores.append(
                        {
                            "inner_validation_period": (
                                f"{validation_start}-{validation_start + 4}"
                            ),
                            "inner_training_samples": len(training),
                            "inner_training_onsets": int(
                                training["onset_label"].sum()
                            ),
                            "inner_validation_samples": len(validation),
                            "inner_validation_onsets": int(
                                validation["onset_label"].sum()
                            ),
                            **metrics,
                        }
                    )
                rows.append(
                    {
                        "feature_set": feature_name,
                        "features": ", ".join(feature_set),
                        "predictor_count": len(feature_set),
                        "C": c_value,
                        "class_weight": class_weight or "none",
                        "inner_fold_count": len(fold_scores),
                        "mean_inner_brier": float(
                            np.mean([score["brier_score"] for score in fold_scores])
                        ),
                        "mean_inner_average_precision": float(
                            np.nanmean(
                                [
                                    score["average_precision"]
                                    for score in fold_scores
                                ]
                            )
                        ),
                        "inner_fold_scores": fold_scores,
                    }
                )
    selected = min(
        rows,
        key=lambda row: (
            row["mean_inner_brier"],
            -row["mean_inner_average_precision"],
            row["predictor_count"],
            row["class_weight"] != "none",
            row["C"],
        ),
    )
    return selected, rows


def calendar_month_probabilities(
    training: pd.DataFrame,
    target: pd.DataFrame,
) -> np.ndarray:
    prevalence = float(training["onset_label"].mean())
    month_rates = training.assign(
        target_calendar_month=training["target_period"].map(
            lambda period: period.month
        )
    ).groupby("target_calendar_month")["onset_label"].agg(["sum", "count"])
    rates = (
        month_rates["sum"] + CALENDAR_MONTH_PRIOR_STRENGTH * prevalence
    ) / (month_rates["count"] + CALENDAR_MONTH_PRIOR_STRENGTH)
    months = target["target_period"].map(lambda period: period.month)
    return months.map(rates).fillna(prevalence).to_numpy(dtype=float)


def episode_counts(predictions: pd.DataFrame) -> dict[str, Any]:
    warned = predictions.loc[predictions["predicted_onset"] == 1].copy()
    if warned.empty:
        return {
            "warning_episodes": 0,
            "detected_onset_episodes": 0,
            "false_warning_episodes": 0,
            "false_warning_episodes_per_year": 0.0,
        }
    warned["target_period"] = pd.PeriodIndex(
        warned["target_period"], freq="M"
    )
    warned = warned.sort_values("target_period")
    groups = []
    current = [warned.iloc[0]]
    for _, row in warned.iloc[1:].iterrows():
        if row["target_period"] == current[-1]["target_period"] + 1:
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    groups.append(current)
    detected = sum(
        int(any(int(row["actual_onset"]) == 1 for row in group))
        for group in groups
    )
    false_alarms = len(groups) - detected
    years = predictions["target_month"].dt.year.nunique()
    return {
        "warning_episodes": len(groups),
        "detected_onset_episodes": detected,
        "false_warning_episodes": false_alarms,
        "false_warning_episodes_per_year": (
            float(false_alarms / years) if years else float("nan")
        ),
    }


def year_block_uncertainty(
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    years = np.sort(predictions["target_month"].dt.year.unique())
    grouped = {
        year: predictions.loc[
            predictions["target_month"].dt.year == year
        ]
        for year in years
    }
    metric_values: dict[str, list[float]] = {
        name: []
        for name in (
            "recall",
            "precision",
            "f1",
            "average_precision",
            "brier_score",
            "brier_skill_score",
        )
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        selected_years = rng.choice(years, size=len(years), replace=True)
        replicate = pd.concat(
            [grouped[int(year)] for year in selected_years],
            ignore_index=True,
        )
        labels = replicate["actual_onset"].to_numpy(dtype=int)
        probabilities = replicate["predicted_probability"].to_numpy(dtype=float)
        predictions_binary = replicate["predicted_onset"].to_numpy(dtype=int)
        metric_values["recall"].append(
            float(recall_score(labels, predictions_binary, zero_division=0))
        )
        metric_values["precision"].append(
            float(
                precision_score(labels, predictions_binary, zero_division=0)
            )
        )
        metric_values["f1"].append(
            float(f1_score(labels, predictions_binary, zero_division=0))
        )
        metric_values["average_precision"].append(
            safe_average_precision(labels, probabilities)
        )
        baseline = replicate["training_prevalence"].to_numpy(dtype=float)
        scores = probability_metrics(labels, probabilities, baseline)
        metric_values["brier_score"].append(scores["brier_score"])
        metric_values["brier_skill_score"].append(
            scores["brier_skill_score"]
        )

    return [
        {
            "metric": name,
            "bootstrap_unit": "calendar year",
            "replicates": BOOTSTRAP_REPLICATES,
            "valid_replicates": int(np.isfinite(values).sum()),
            "percentile_2_5": float(np.nanpercentile(values, 2.5)),
            "percentile_97_5": float(np.nanpercentile(values, 97.5)),
            "interpretation_note": (
                "Descriptive uncertainty only; few onset events and "
                "historical development folds limit inference."
            ),
        }
        for name, values in metric_values.items()
    ]


def main() -> None:
    data, samples, audit = load_and_gate_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inner_score_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    baseline_metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    for test_start, test_end in OUTER_FOLDS:
        validation_start = test_start - 5
        validation_end = test_start - 1
        validation_first = pd.Period(f"{validation_start}-01", freq="M")
        test_first = pd.Period(f"{test_start}-01", freq="M")
        training_cutoff = validation_first - 1
        training = samples.loc[
            samples["target_period"] <= training_cutoff
        ].copy()
        validation = samples.loc[
            samples["target_period"].dt.year.between(
                validation_start, validation_end
            )
        ].copy()
        test = samples.loc[
            samples["target_period"].dt.year.between(test_start, test_end)
        ].copy()
        if any(part.empty for part in (training, validation, test)):
            raise ValueError(
                f"Empty outer split for test period {test_start}-{test_end}."
            )
        if training["onset_label"].nunique() != 2:
            raise ValueError("Outer training data lacks both outcome classes.")
        if validation["onset_label"].sum() == 0:
            raise ValueError(
                f"Validation period {validation_start}-{validation_end} "
                "has no onsets for threshold selection."
            )

        selected, candidate_rows = select_candidate(
            data, training, validation_start
        )
        inner_score_rows.extend(
            {
                "outer_test_period": f"{test_start}-{test_end}",
                **{
                    key: value
                    for key, value in row.items()
                    if key != "inner_fold_scores"
                },
                **{
                    f"fold_{index + 1}_{key}": value
                    for index, score in enumerate(row["inner_fold_scores"])
                    for key, value in score.items()
                },
            }
            for row in candidate_rows
        )

        feature_set = FEATURE_SETS[selected["feature_set"]]
        columns = model_columns(feature_set)
        training_x = transform_features(training, data, training_cutoff)
        validation_x = transform_features(validation, data, training_cutoff)
        test_x = transform_features(test, data, training_cutoff)
        selected_model = make_model(
            float(selected["C"]),
            None
            if selected["class_weight"] == "none"
            else selected["class_weight"],
        )
        selected_model.fit(
            training_x[columns],
            training_x["onset_label"],
        )
        validation_probabilities = selected_model.predict_proba(
            validation_x[columns]
        )[:, 1]
        chosen_threshold = select_recall_threshold(
            validation_x["onset_label"].to_numpy(dtype=int),
            validation_probabilities,
            validation_x["target_period"],
            evaluation_years=validation_end - validation_start + 1,
        )
        test_probabilities = selected_model.predict_proba(test_x[columns])[:, 1]
        training_prevalence = float(training["onset_label"].mean())
        test_labels = test_x["onset_label"].to_numpy(dtype=int)
        test_baseline_probabilities = np.full(
            len(test_x), training_prevalence
        )
        model_metrics = classification_metrics(
            test_labels,
            test_probabilities,
            chosen_threshold["threshold"],
            test_baseline_probabilities,
        )
        fold_id = f"{test_start}-{test_end}"
        selection_record = {
            "test_period": fold_id,
            "validation_period": f"{validation_start}-{validation_end}",
            "training_cutoff": str(training_cutoff),
            "training_samples": len(training),
            "training_onsets": int(training["onset_label"].sum()),
            "validation_samples": len(validation),
            "validation_onsets": int(validation["onset_label"].sum()),
            "inner_fold_count": selected["inner_fold_count"],
            "selected_feature_set": selected["feature_set"],
            "selected_features": ", ".join(feature_set),
            "C": selected["C"],
            "class_weight": selected["class_weight"],
            "mean_inner_brier": selected["mean_inner_brier"],
            "mean_inner_average_precision": (
                selected["mean_inner_average_precision"]
            ),
            "validation_threshold": chosen_threshold["threshold"],
            "validation_f1_at_threshold": chosen_threshold["f1"],
            "validation_precision_at_threshold": chosen_threshold["precision"],
            "validation_recall_at_threshold": chosen_threshold["recall"],
            "validation_false_warning_episodes": (
                chosen_threshold["false_warning_episodes"]
            ),
            "validation_false_warning_episodes_per_year": (
                chosen_threshold["false_warning_episodes_per_year"]
            ),
            "validation_false_warning_months": (
                chosen_threshold["false_warning_months"]
            ),
            "validation_false_warning_months_per_year": (
                chosen_threshold["false_warning_months_per_year"]
            ),
            "validation_warning_months": (
            chosen_threshold["predicted_warnings"]
            ),
            "validation_predicted_warnings": (
                chosen_threshold["predicted_warnings"]
            ),
        }
        selected_rows.append(selection_record)
        fold_metric_rows.append(
            {
                "test_period": fold_id,
                "validation_period": f"{validation_start}-{validation_end}",
                "selected_feature_set": selected["feature_set"],
                "threshold_from_validation": chosen_threshold["threshold"],
                "training_prevalence": training_prevalence,
                "false_warning_months_per_year": (
                    model_metrics["false_warning_months"]
                    / (test_end - test_start + 1)
                ),
                **model_metrics,
            }
        )

        prevalence_val = np.full(len(validation), training_prevalence)
        prevalence_threshold = select_recall_threshold(
            validation["onset_label"].to_numpy(dtype=int),
            prevalence_val,
            validation["target_period"],
            evaluation_years=validation_end - validation_start + 1,
        )
        prevalence_test = np.full(len(test), training_prevalence)
        calendar_val = calendar_month_probabilities(training, validation)
        calendar_threshold = select_recall_threshold(
            validation["onset_label"].to_numpy(dtype=int),
            calendar_val,
            validation["target_period"],
            evaluation_years=validation_end - validation_start + 1,
        )
        calendar_test = calendar_month_probabilities(training, test)

        current_state_model = make_model(1.0, None)
        current_state_model.fit(
            training_x[["origin_swba"]], training_x["onset_label"]
        )
        current_state_val = current_state_model.predict_proba(
            validation_x[["origin_swba"]]
        )[:, 1]
        current_state_threshold = select_recall_threshold(
            validation_x["onset_label"].to_numpy(dtype=int),
            current_state_val,
            validation_x["target_period"],
            evaluation_years=validation_end - validation_start + 1,
        )
        current_state_test = current_state_model.predict_proba(
            test_x[["origin_swba"]]
        )[:, 1]

        baseline_specs = (
            (
                "training_prevalence",
                prevalence_test,
                prevalence_threshold["threshold"],
            ),
            (
                "calendar_month_smoothed",
                calendar_test,
                calendar_threshold["threshold"],
            ),
            (
                "current_state_logistic",
                current_state_test,
                current_state_threshold["threshold"],
            ),
        )
        baseline_values: dict[str, tuple[np.ndarray, float]] = {}
        for baseline_name, probabilities, threshold in baseline_specs:
            baseline_values[baseline_name] = (probabilities, threshold)
            baseline_metric_rows.append(
                {
                    "test_period": fold_id,
                    "baseline": baseline_name,
                    "threshold_from_validation": threshold,
                    "training_prevalence": training_prevalence,
                    "false_warning_months_per_year": (
                        int(
                            np.sum(
                                (probabilities >= threshold)
                                & (test_labels == 0)
                            )
                        )
                        / (test_end - test_start + 1)
                    ),
                    **classification_metrics(
                        test_labels,
                        probabilities,
                        threshold,
                        test_baseline_probabilities,
                    ),
                }
            )

        for index, (_, sample) in enumerate(test_x.iterrows()):
            prediction = {
                "test_period": fold_id,
                "target_period": str(sample["target_period"]),
                "target_month": sample["target_month"],
                "forecast_origin": sample["forecast_origin"],
                "actual_onset": int(sample["onset_label"]),
                "predicted_probability": float(test_probabilities[index]),
                "predicted_onset": int(
                    test_probabilities[index]
                    >= chosen_threshold["threshold"]
                ),
                "threshold": chosen_threshold["threshold"],
                "training_prevalence": training_prevalence,
                "selected_feature_set": selected["feature_set"],
            }
            for baseline_name, (baseline_probs, baseline_threshold) in (
                baseline_values.items()
            ):
                prediction[f"{baseline_name}_probability"] = float(
                    baseline_probs[index]
                )
                prediction[f"{baseline_name}_threshold"] = float(
                    baseline_threshold
                )
                prediction[f"{baseline_name}_prediction"] = int(
                    baseline_probs[index] >= baseline_threshold
                )
            prediction_rows.append(prediction)

        fold_prediction_frame = pd.DataFrame(
            [row for row in prediction_rows if row["test_period"] == fold_id]
        )
        episode_rows.append(
            {
                "test_period": fold_id,
                **episode_counts(fold_prediction_frame),
            }
        )

    predictions_frame = pd.DataFrame(prediction_rows)
    labels = predictions_frame["actual_onset"].to_numpy(dtype=int)
    probabilities = predictions_frame["predicted_probability"].to_numpy(
        dtype=float
    )
    baseline_probabilities = predictions_frame[
        "training_prevalence"
    ].to_numpy(dtype=float)
    pooled_specs = [
        ("selected_model", "predicted_probability", "predicted_onset"),
        (
            "training_prevalence",
            "training_prevalence_probability",
            "training_prevalence_prediction",
        ),
        (
            "calendar_month_smoothed",
            "calendar_month_smoothed_probability",
            "calendar_month_smoothed_prediction",
        ),
        (
            "current_state_logistic",
            "current_state_logistic_probability",
            "current_state_logistic_prediction",
        ),
    ]
    pooled_rows = []
    for model_name, probability_column, prediction_column in pooled_specs:
        model_probabilities = predictions_frame[
            probability_column
        ].to_numpy(dtype=float)
        binary_predictions = predictions_frame[
            prediction_column
        ].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(
            labels, binary_predictions, labels=[0, 1]
        ).ravel()
        model_predictions = predictions_frame.copy()
        model_predictions["predicted_onset"] = binary_predictions
        pooled_rows.append(
            {
                "model_or_baseline": model_name,
                "test_period": "pooled_2006-2025_development_backtests",
                "threshold_note": (
                    "Fold-specific thresholds selected on the preceding "
                    "validation period; pooled confusion metrics use those "
                    "frozen predictions."
                ),
                "samples": int(len(labels)),
                "observed_onsets": int(labels.sum()),
                "onset_prevalence": float(labels.mean()),
                "predicted_warnings": int(binary_predictions.sum()),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_warning_months": int(fp),
                "false_warning_months_per_year": float(fp / 20),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "precision": float(
                    precision_score(labels, binary_predictions, zero_division=0)
                ),
                "recall": float(
                    recall_score(labels, binary_predictions, zero_division=0)
                ),
                "f1": float(
                    f1_score(labels, binary_predictions, zero_division=0)
                ),
                **probability_metrics(
                    labels, model_probabilities, baseline_probabilities
                ),
                **episode_counts(model_predictions),
            }
        )

    uncertainty = year_block_uncertainty(predictions_frame)
    reliability = predictions_frame.copy()
    reliability["probability_bin"] = pd.qcut(
        reliability["predicted_probability"],
        q=min(5, reliability["predicted_probability"].nunique()),
        duplicates="drop",
    )
    reliability_frame = (
        reliability.groupby("probability_bin", observed=True)
        .agg(
            samples=("actual_onset", "size"),
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_onset_rate=("actual_onset", "mean"),
            onsets=("actual_onset", "sum"),
        )
        .reset_index()
    )
    reliability_frame["probability_bin"] = (
        reliability_frame["probability_bin"].astype(str)
    )

    pd.DataFrame(inner_score_rows).to_csv(
        OUTPUT_DIR / "inner_model_scores.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(
        OUTPUT_DIR / "selected_models_and_thresholds.csv", index=False
    )
    pd.DataFrame(fold_metric_rows).to_csv(
        OUTPUT_DIR / "outer_fold_metrics.csv", index=False
    )
    pd.DataFrame(baseline_metric_rows).to_csv(
        OUTPUT_DIR / "baseline_fold_metrics.csv", index=False
    )
    predictions_frame.to_csv(
        OUTPUT_DIR / "outer_test_predictions.csv", index=False
    )
    pd.DataFrame(episode_rows).to_csv(
        OUTPUT_DIR / "warning_episode_metrics.csv", index=False
    )
    pd.DataFrame(pooled_rows).to_csv(
        OUTPUT_DIR / "pooled_metrics.csv", index=False
    )
    pd.DataFrame(uncertainty).to_csv(
        OUTPUT_DIR / "year_block_bootstrap_intervals.csv", index=False
    )
    reliability_frame.to_csv(
        OUTPUT_DIR / "pooled_reliability_bins.csv", index=False
    )
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "input_file": str(INPUT_FILE),
                "input_sha256": sha256_file(INPUT_FILE),
                "audit_report": str(AUDIT_REPORT),
                "audit_input_sha256": audit["input_sha256"],
                "all_data_audit_checks_passed": audit["all_checks_passed"],
                "lead_months": 1,
                "drought_threshold": THRESHOLD,
                "outer_folds": [
                    f"{start}-{end}" for start, end in OUTER_FOLDS
                ],
                "feature_sets": {
                    name: list(features)
                    for name, features in FEATURE_SETS.items()
                },
                "missing_value_imputation": (
                    "Median imputation is fitted inside each model-training "
                    "fold before scaling; it supports sparse missing values "
                    "in added predictors."
                ),
                "inner_selection": (
                    "Mean Brier score across expanding, five-year "
                    "forward-chaining validation blocks; tie-break by average "
                    "precision, predictor count, unweighted model, then C."
                ),
                "threshold_selection": (
                    "Maximize recall on the separate outer validation block "
                    "subject to predeclared caps of 0.5 false-warning episodes "
                    "and 3 false-warning months per year; "
                    "tie-break by precision, fewer warnings, then higher "
                    "threshold."
                ),
                "maximum_false_warning_episodes_per_year": (
                    MAX_FALSE_WARNING_EPISODES_PER_YEAR
                ),
                "maximum_false_warning_months_per_year": (
                    MAX_FALSE_WARNING_MONTHS_PER_YEAR
                ),
                "outer_test_policy": (
                    "The model is fit on outer training only. Validation "
                    "labels select the threshold only. No validation or test "
                    "labels are used to refit model parameters."
                ),
                "calibration": (
                    "No probability calibration fitted; event counts in "
                    "validation folds are too small for reliable calibration."
                ),
                "calendar_month_baseline_prior_strength": (
                    CALENDAR_MONTH_PRIOR_STRENGTH
                ),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Selected models and frozen validation thresholds:")
    print(pd.DataFrame(selected_rows).to_string(index=False))
    print("\nOuter-fold model metrics:")
    print(pd.DataFrame(fold_metric_rows).to_string(index=False))
    print("\nPooled development-backtest metrics:")
    print(pd.DataFrame(pooled_rows).to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
