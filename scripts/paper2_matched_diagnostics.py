"""Matched-origin Paper 2 diagnostics with strictly forward validation blocks."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/processed/expanded/mnjoli_water_stress_monthly_1981_2025_swba.csv"
OUT = ROOT / "outputs/research/paper2_matched_diagnostics"
HORIZONS = [1, 3, 6]
N_FOLDS = 5
STATES = [0, 1, 2]
DIRECTIONS = [-1, 0, 1]


def prepare_data():
    data = (
        pd.read_csv(SRC, parse_dates=["month_date"])
        .sort_values("month_date")
        .set_index("month_date")
        .asfreq("MS")
    )
    swba = pd.to_numeric(data["swba_3month"], errors="coerce")

    state = pd.Series(np.nan, index=data.index, dtype=float)
    state.loc[swba.notna() & (swba > -0.5)] = 0
    state.loc[swba.notna() & (swba <= -0.5) & (swba > -1)] = 1
    state.loc[swba.notna() & (swba <= -1)] = 2

    feature_columns = [
        "precipitation_mm", "pet_mm", "water_balance_mm",
        "temperature_mean_c", "temperature_min_c", "temperature_max_c",
        "dewpoint_c", "surface_pressure", "u_component_of_wind_10m",
        "v_component_of_wind_10m", "surface_solar_radiation_downwards_sum",
        "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
        "runoff_sum", "surface_runoff_sum", "water_balance_3month",
        "water_balance_6month",
    ]
    feature_columns = [column for column in feature_columns if column in data.columns]
    if not feature_columns:
        raise RuntimeError("No expected hydroclimatic feature columns were found.")

    features = data[feature_columns].copy()
    for column in feature_columns:
        features[f"{column}_lag1"] = data[column].shift(1)
        features[f"{column}_lag3"] = data[column].shift(3)

    features["swba_origin"] = swba
    features["origin_state"] = state
    for lag in [1, 3, 6]:
        features[f"swba_lag{lag}"] = swba.shift(lag)
    features["month_sin"] = np.sin(2 * np.pi * data.index.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * data.index.month / 12)
    return data, swba, state, features


def make_folds(n_rows):
    """Return expanding-window folds: validation blocks never include earlier rows."""
    first_validation = int(n_rows * 0.60)
    boundaries = np.linspace(first_validation, n_rows, N_FOLDS + 1, dtype=int)
    folds = []
    for fold_index in range(N_FOLDS):
        validation_start = int(boundaries[fold_index])
        validation_end = int(boundaries[fold_index + 1])
        if validation_end <= validation_start:
            continue
        train_indices = np.arange(0, validation_start, dtype=int)
        validation_indices = np.arange(validation_start, validation_end, dtype=int)
        folds.append((train_indices, validation_indices))
    return folds


def make_model():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1,
                    max_iter=3000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def pooled_metrics(predictions):
    summaries = []
    for (task, horizon), group in predictions.groupby(["task", "horizon"]):
        actual = group["actual"].astype(int).to_numpy()
        predicted = group["predicted"].astype(int).to_numpy()
        result = {
            "task": task,
            "horizon": int(horizon),
            "n": int(len(group)),
        }
        if task in ("drought_state", "onset"):
            probability = group["probability"].astype(float).to_numpy()
            tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
            result.update(
                {
                    "events": int(actual.sum()),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tn": int(tn),
                    "precision": precision_score(actual, predicted, zero_division=0),
                    "recall": recall_score(actual, predicted, zero_division=0),
                    "f1": f1_score(actual, predicted, zero_division=0),
                    "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
                    "pr_auc": average_precision_score(actual, probability) if actual.sum() else np.nan,
                    "brier": brier_score_loss(actual, probability),
                }
            )
        else:
            labels = STATES if task == "multistate" else DIRECTIONS
            probability = np.vstack(group["probabilities"].map(json.loads).to_numpy())
            result.update(
                {
                    "accuracy": accuracy_score(actual, predicted),
                    "macro_f1": f1_score(
                        actual, predicted, labels=labels, average="macro", zero_division=0
                    ),
                    "log_loss": log_loss(actual, probability, labels=labels),
                    "brier": float(
                        np.mean(
                            np.sum(
                                (
                                    np.eye(len(labels))[
                                        [labels.index(int(value)) for value in actual]
                                    ]
                                    - probability
                                )
                                ** 2,
                                axis=1,
                            )
                        )
                    ),
                }
            )
        summaries.append(result)
    return pd.DataFrame(summaries)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, swba, state, features = prepare_data()
    prediction_rows = []
    fold_metric_rows = []
    fold_window_rows = []

    folds = make_folds(len(data))
    if not folds:
        raise RuntimeError("No non-empty forward validation folds could be constructed.")

    for horizon in HORIZONS:
        future_swba = swba.shift(-horizon)
        future_state = state.shift(-horizon)
        previous_swba = swba.shift(-(horizon - 1))

        onset = ((future_swba <= -1) & (previous_swba > -1)).astype(float)
        onset.loc[future_swba.isna() | previous_swba.isna()] = np.nan

        drought_state = (future_swba <= -1).astype(float)
        drought_state.loc[future_swba.isna()] = np.nan

        direction = np.sign(future_state - state)
        tasks = {
            "drought_state": drought_state,
            "onset": onset,
            "multistate": future_state,
            "direction": direction,
        }

        for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
            validation_start = int(validation_indices[0])
            validation_end = int(validation_indices[-1])
            # A training origin is admissible only when its full target horizon
            # ends strictly before the first validation target month.
            safe_training_indices = train_indices[
                train_indices + horizon < validation_start
            ]
            if len(validation_indices) == 0 or len(safe_training_indices) == 0:
                continue

            fold_window_rows.append(
                {
                    "fold": fold_number,
                    "horizon": horizon,
                    "train_origin_start": data.index[safe_training_indices[0]].date().isoformat(),
                    "train_origin_end": data.index[safe_training_indices[-1]].date().isoformat(),
                    "last_training_target": data.index[safe_training_indices[-1] + horizon].date().isoformat(),
                    "validation_origin_start": data.index[validation_start].date().isoformat(),
                    "validation_origin_end": data.index[validation_end].date().isoformat(),
                    "n_train_origins_before_label_filter": int(len(safe_training_indices)),
                    "n_validation_origins_before_label_filter": int(len(validation_indices)),
                }
            )

            for task_name, target in tasks.items():
                valid_target = target.notna()
                train_rows = safe_training_indices[
                    valid_target.iloc[safe_training_indices].to_numpy()
                ]
                validation_rows = validation_indices[
                    valid_target.iloc[validation_indices].to_numpy()
                ]
                if len(train_rows) == 0 or len(validation_rows) == 0:
                    continue

                y_train = target.iloc[train_rows].astype(int)
                y_validation = target.iloc[validation_rows].astype(int)
                if y_train.nunique() < 2:
                    continue

                estimator = make_model()
                estimator.fit(features.iloc[train_rows], y_train)
                raw_probabilities = estimator.predict_proba(features.iloc[validation_rows])
                observed_classes = estimator.named_steps["classifier"].classes_
                meta = {"task": task_name, "horizon": horizon, "fold": fold_number}

                if task_name in ("drought_state", "onset"):
                    positive_probability = (
                        raw_probabilities[:, list(observed_classes).index(1)]
                        if 1 in observed_classes
                        else np.zeros(len(validation_rows))
                    )
                    predictions = (positive_probability >= 0.5).astype(int)
                    for row_number, origin_index in enumerate(validation_rows):
                        prediction_rows.append(
                            {
                                **meta,
                                "origin": data.index[origin_index].date().isoformat(),
                                "target_date": data.index[origin_index + horizon].date().isoformat(),
                                "actual": int(y_validation.iloc[row_number]),
                                "predicted": int(predictions[row_number]),
                                "probability": float(positive_probability[row_number]),
                                "probabilities": "",
                            }
                        )
                    tn, fp, fn, tp = confusion_matrix(
                        y_validation, predictions, labels=[0, 1]
                    ).ravel()
                    fold_metric_rows.append(
                        {
                            **meta,
                            "n": int(len(validation_rows)),
                            "events": int(y_validation.sum()),
                            "tp": int(tp),
                            "fp": int(fp),
                            "fn": int(fn),
                            "tn": int(tn),
                            "precision": precision_score(y_validation, predictions, zero_division=0),
                            "recall": recall_score(y_validation, predictions, zero_division=0),
                            "f1": f1_score(y_validation, predictions, zero_division=0),
                            "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
                            "pr_auc": (
                                average_precision_score(y_validation, positive_probability)
                                if y_validation.sum()
                                else np.nan
                            ),
                            "brier": brier_score_loss(y_validation, positive_probability),
                        }
                    )
                else:
                    labels = STATES if task_name == "multistate" else DIRECTIONS
                    full_probabilities = np.zeros((len(validation_rows), len(labels)))
                    for class_column, class_value in enumerate(observed_classes):
                        full_probabilities[:, labels.index(int(class_value))] = raw_probabilities[
                            :, class_column
                        ]
                    predictions = np.asarray(labels)[full_probabilities.argmax(axis=1)]
                    for row_number, origin_index in enumerate(validation_rows):
                        prediction_rows.append(
                            {
                                **meta,
                                "origin": data.index[origin_index].date().isoformat(),
                                "target_date": data.index[origin_index + horizon].date().isoformat(),
                                "actual": int(y_validation.iloc[row_number]),
                                "predicted": int(predictions[row_number]),
                                "probability": np.nan,
                                "probabilities": json.dumps(
                                    full_probabilities[row_number].tolist()
                                ),
                            }
                        )
                    fold_metric_rows.append(
                        {
                            **meta,
                            "n": int(len(validation_rows)),
                            "accuracy": accuracy_score(y_validation, predictions),
                            "macro_f1": f1_score(
                                y_validation,
                                predictions,
                                labels=labels,
                                average="macro",
                                zero_division=0,
                            ),
                            "log_loss": log_loss(
                                y_validation, full_probabilities, labels=labels
                            ),
                            "brier": float(
                                np.mean(
                                    np.sum(
                                        (
                                            np.eye(len(labels))[
                                                [
                                                    labels.index(int(value))
                                                    for value in y_validation
                                                ]
                                            ]
                                            - full_probabilities
                                        )
                                        ** 2,
                                        axis=1,
                                    )
                                )
                            ),
                        }
                    )

    predictions_frame = pd.DataFrame(prediction_rows)
    fold_metrics_frame = pd.DataFrame(fold_metric_rows)
    if fold_metrics_frame.empty or predictions_frame.empty:
        (OUT / "diagnostic_error.txt").write_text(
            "No model evaluations were produced. Inspect feature coverage, target validity, "
            "fold boundaries, and class availability. No performance result is valid.\n"
        )
        raise RuntimeError(
            "No fold metrics generated; inspect diagnostic_error.txt and feature/target coverage."
        )

    # Hard guard: every validation origin must occur after the corresponding
    # training origins and no validation block may contain pre-validation dates.
    for row in fold_window_rows:
        if row["last_training_target"] >= row["validation_origin_start"]:
            raise RuntimeError("Temporal boundary violation detected in a fold window.")

    predictions_frame.to_csv(OUT / "matched_validation_predictions.csv", index=False)
    fold_metrics_frame.to_csv(OUT / "matched_fold_metrics.csv", index=False)
    pooled_metrics(predictions_frame).to_csv(OUT / "matched_metric_summary.csv", index=False)
    pd.DataFrame(fold_window_rows).to_csv(OUT / "matched_fold_windows.csv", index=False)

    metadata = {
        "validation_design": (
            "Expanding-window validation. First 60% of the chronological monthly timeline "
            "is initial training; remaining 40% is partitioned into five non-overlapping, "
            "contiguous validation blocks."
        ),
        "horizons_months": HORIZONS,
        "training_boundary": (
            "For horizon h, training origins are retained only when origin_index + h "
            "is strictly earlier than the first validation origin."
        ),
        "tasks": {
            "drought_state": "Future SWBA <= -1",
            "onset": "At t+h SWBA <= -1 and at t+h-1 SWBA > -1",
            "multistate": "Normal > -0.5; Watch (-1, -0.5]; Drought <= -1",
            "direction": "Sign of future severity-state index minus origin-state index",
        },
        "model": (
            "Fold-local median imputation, standardization, and balanced Logistic Regression "
            "(C=1, max_iter=3000)."
        ),
        "pooled_summary": "Metrics computed from concatenated non-overlapping out-of-fold predictions.",
        "final_test_used": False,
        "warning": (
            "Exploratory validation only. Onset counts are sparse. Class-weighted probabilities "
            "are not assumed calibrated. Current-origin-month predictors imply issuance after "
            "the origin month's observations are available."
        ),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
