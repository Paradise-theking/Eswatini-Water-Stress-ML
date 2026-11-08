from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
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
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "expanded"
    / "drought_onset_model_preparation"
    / "model_selection_validation"
)

DATE_COLUMN = "month_date"
SWBA_COLUMN = "swba_3month"
ONSET_THRESHOLD = -1.0
LEADS = (1, 2, 3)
TEST_PERIODS = ((2006, 2010), (2011, 2015), (2016, 2020), (2021, 2025))
REGULARIZATION_STRENGTHS = (0.1, 1.0, 10.0)
CLASS_WEIGHTS = (None, "balanced")

HYDRO_FEATURES = (
    "precipitation_mm",
    "pet_mm",
    "volumetric_soil_water_layer_1",
    "water_balance_3month",
)
FEATURE_SETS = {
    "origin_swba": ("origin_swba",),
    "precipitation": ("precipitation_mm",),
    "pet": ("pet_mm",),
    "soil_moisture": ("volumetric_soil_water_layer_1",),
    "water_balance": ("water_balance_3month",),
    "rainfall_pet_soil": (
        "precipitation_mm",
        "pet_mm",
        "volumetric_soil_water_layer_1",
    ),
    "rainfall_pet_balance": (
        "precipitation_mm",
        "pet_mm",
        "water_balance_3month",
    ),
    "four_hydro_features": HYDRO_FEATURES,
    "state_plus_hydro": ("origin_swba",) + HYDRO_FEATURES,
    "state_hydro_season": (
        "origin_swba",
        *HYDRO_FEATURES,
        "origin_month_sin",
        "origin_month_cos",
    ),
}


def load_monthly_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT_FILE, parse_dates=[DATE_COLUMN])
    data = data.sort_values(DATE_COLUMN).reset_index(drop=True)

    required = [DATE_COLUMN, SWBA_COLUMN, *HYDRO_FEATURES]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[DATE_COLUMN].isna().any():
        raise ValueError("Invalid monthly dates were found.")
    if data[DATE_COLUMN].duplicated().any():
        raise ValueError("Duplicate monthly dates were found.")

    periods = data[DATE_COLUMN].dt.to_period("M")
    expected_periods = pd.period_range(
        periods.min(),
        periods.max(),
        freq="M",
    )
    if not periods.reset_index(drop=True).equals(
        pd.Series(expected_periods, name=DATE_COLUMN)
    ):
        raise ValueError("The monthly dataset contains missing months.")

    return data


def build_samples(data: pd.DataFrame, lead: int) -> pd.DataFrame:
    targets = data[[DATE_COLUMN, SWBA_COLUMN]].copy()
    targets["previous_swba"] = targets[SWBA_COLUMN].shift(1)
    targets["onset_label"] = (
        (targets[SWBA_COLUMN] <= ONSET_THRESHOLD)
        & (targets["previous_swba"] > ONSET_THRESHOLD)
    ).astype(int)
    targets.loc[targets["previous_swba"].isna(), "onset_label"] = np.nan
    targets = targets.dropna(subset=[SWBA_COLUMN, "onset_label"]).copy()
    targets["forecast_origin"] = (
        targets[DATE_COLUMN] - pd.DateOffset(months=lead)
    )

    origin_columns = [DATE_COLUMN, SWBA_COLUMN, *HYDRO_FEATURES]
    origins = data[origin_columns].rename(
        columns={
            DATE_COLUMN: "forecast_origin",
            SWBA_COLUMN: "origin_swba",
        }
    )
    samples = targets.merge(
        origins,
        on="forecast_origin",
        how="inner",
        validate="one_to_one",
    )
    samples = samples.loc[
        samples["origin_swba"] > ONSET_THRESHOLD
    ].copy()
    samples["onset_label"] = samples["onset_label"].astype(int)
    samples["origin_month"] = samples["forecast_origin"].dt.month
    samples["origin_month_sin"] = np.sin(
        2 * np.pi * samples["origin_month"] / 12
    )
    samples["origin_month_cos"] = np.cos(
        2 * np.pi * samples["origin_month"] / 12
    )
    samples = samples.dropna(
        subset=["origin_swba", *HYDRO_FEATURES]
    ).reset_index(drop=True)
    return samples


def transform_features(
    samples: pd.DataFrame,
    reference_cutoff: pd.Timestamp,
    data: pd.DataFrame,
) -> pd.DataFrame:
    reference = data.loc[data[DATE_COLUMN] <= reference_cutoff].copy()
    if reference.empty:
        raise ValueError(
            f"No climatology reference data available by {reference_cutoff}."
        )
    reference["calendar_month"] = reference[DATE_COLUMN].dt.month
    result = samples.copy()

    for feature in HYDRO_FEATURES:
        monthly_stats = reference.groupby("calendar_month")[feature].agg(
            mean="mean",
            std=lambda values: values.std(ddof=0),
        )
        means = result["origin_month"].map(monthly_stats["mean"])
        stds = result["origin_month"].map(monthly_stats["std"])
        if means.isna().any() or stds.isna().any() or (stds <= 0).any():
            raise ValueError(
                f"Invalid calendar-month reference statistics for {feature}."
            )
        result[f"{feature}_anom"] = (
            result[feature].to_numpy() - means.to_numpy()
        ) / stds.to_numpy()

    return result


def model_columns(feature_set: tuple[str, ...]) -> list[str]:
    return [
        f"{feature}_anom" if feature in HYDRO_FEATURES else feature
        for feature in feature_set
    ]


def make_model(c_value: float, class_weight: str | None) -> Pipeline:
    return Pipeline(
        [
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


def select_f1_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    candidates = np.unique(
        np.concatenate(([0.0], probabilities, [1.000001]))
    )
    scored = []
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        scored.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "precision": float(
                    precision_score(labels, predictions, zero_division=0)
                ),
                "recall": float(
                    recall_score(labels, predictions, zero_division=0)
                ),
            }
        )
    return max(
        scored,
        key=lambda result: (
            result["f1"],
            result["precision"],
            result["threshold"],
        ),
    )


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    baseline_probability: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()
    brier = float(brier_score_loss(labels, probabilities))
    baseline_brier = float(
        brier_score_loss(
            labels,
            np.full(len(labels), baseline_probability),
        )
    )
    return {
        "samples": len(labels),
        "observed_onsets": int(labels.sum()),
        "predicted_warnings": int(predictions.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier_score": brier,
        "training_prevalence_brier": baseline_brier,
        "brier_skill_score": (
            1.0 - brier / baseline_brier
            if baseline_brier > 0
            else np.nan
        ),
    }


def main() -> None:
    data = load_monthly_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_grid: list[dict] = []
    selections: list[dict] = []
    fold_metrics: list[dict] = []
    prediction_rows: list[dict] = []

    for lead in LEADS:
        samples = build_samples(data, lead)
        for test_start, test_end in TEST_PERIODS:
            validation_start = test_start - 5
            validation_end = test_start - 1
            validation_first = pd.Timestamp(validation_start, 1, 1)
            test_first = pd.Timestamp(test_start, 1, 1)
            validation_cutoff = (
                validation_first - pd.DateOffset(months=lead)
            )
            test_cutoff = test_first - pd.DateOffset(months=lead)

            training = samples.loc[
                samples[DATE_COLUMN] <= validation_cutoff
            ].copy()
            validation = samples.loc[
                samples[DATE_COLUMN].dt.year.between(
                    validation_start,
                    validation_end,
                )
            ].copy()
            final_training = samples.loc[
                samples[DATE_COLUMN] <= test_cutoff
            ].copy()
            test = samples.loc[
                samples[DATE_COLUMN].dt.year.between(test_start, test_end)
            ].copy()

            for name, subset in (
                ("training", training),
                ("validation", validation),
                ("final training", final_training),
                ("test", test),
            ):
                if subset.empty:
                    raise ValueError(
                        f"Empty {name} set: lead={lead}, "
                        f"test={test_start}-{test_end}."
                    )
            if training["onset_label"].nunique() != 2:
                raise ValueError(
                    f"Validation training data lacks both labels: "
                    f"lead={lead}, test={test_start}-{test_end}."
                )
            if validation["onset_label"].sum() == 0:
                raise ValueError(
                    f"Validation period has no onsets: "
                    f"lead={lead}, test={test_start}-{test_end}."
                )
            if final_training["onset_label"].nunique() != 2:
                raise ValueError(
                    f"Final training data lacks both labels: "
                    f"lead={lead}, test={test_start}-{test_end}."
                )

            training_x = transform_features(
                training, validation_cutoff, data
            )
            validation_x = transform_features(
                validation, validation_cutoff, data
            )
            prevalence = float(training["onset_label"].mean())
            candidate_rows = []

            for feature_name, feature_set in FEATURE_SETS.items():
                columns = model_columns(feature_set)
                for c_value in REGULARIZATION_STRENGTHS:
                    for class_weight in CLASS_WEIGHTS:
                        model = make_model(c_value, class_weight)
                        model.fit(
                            training_x[columns],
                            training_x["onset_label"],
                        )
                        probabilities = model.predict_proba(
                            validation_x[columns]
                        )[:, 1]
                        threshold_result = select_f1_threshold(
                            validation_x["onset_label"].to_numpy(),
                            probabilities,
                        )
                        metrics = classification_metrics(
                            validation_x["onset_label"].to_numpy(),
                            probabilities,
                            threshold_result["threshold"],
                            prevalence,
                        )
                        row = {
                            "lead_months": lead,
                            "test_period": f"{test_start}-{test_end}",
                            "validation_period": (
                                f"{validation_start}-{validation_end}"
                            ),
                            "feature_set": feature_name,
                            "features": ", ".join(feature_set),
                            "predictor_count": len(feature_set),
                            "C": c_value,
                            "class_weight": class_weight or "none",
                            "training_prevalence": prevalence,
                            "validation_threshold": (
                                threshold_result["threshold"]
                            ),
                            "validation_threshold_f1": (
                                threshold_result["f1"]
                            ),
                            "validation_threshold_precision": (
                                threshold_result["precision"]
                            ),
                            "validation_threshold_recall": (
                                threshold_result["recall"]
                            ),
                            **metrics,
                        }
                        candidate_rows.append(row)
                        validation_grid.append(row)

            # Select by validation Brier score; use discrimination and
            # parsimony only to resolve exact or near-exact ties.
            best_brier = min(row["brier_score"] for row in candidate_rows)
            near_best = [
                row for row in candidate_rows
                if row["brier_score"] <= best_brier + 1e-12
            ]
            selected = max(
                near_best,
                key=lambda row: (
                    row["average_precision"],
                    -row["predictor_count"],
                    -row["C"],
                    row["class_weight"] == "none",
                ),
            )
            selections.append(selected)

            final_training_x = transform_features(
                final_training, test_cutoff, data
            )
            test_x = transform_features(test, test_cutoff, data)
            selected_features = FEATURE_SETS[selected["feature_set"]]
            columns = model_columns(selected_features)
            final_model = make_model(
                selected["C"],
                None if selected["class_weight"] == "none"
                else selected["class_weight"],
            )
            final_model.fit(
                final_training_x[columns],
                final_training_x["onset_label"],
            )
            probabilities = final_model.predict_proba(test_x[columns])[:, 1]
            threshold = selected["validation_threshold"]
            metrics = classification_metrics(
                test_x["onset_label"].to_numpy(),
                probabilities,
                threshold,
                float(final_training["onset_label"].mean()),
            )
            fold_row = {
                "lead_months": lead,
                "test_period": f"{test_start}-{test_end}",
                "validation_period": (
                    f"{validation_start}-{validation_end}"
                ),
                "selected_feature_set": selected["feature_set"],
                "features": ", ".join(selected_features),
                "C": selected["C"],
                "class_weight": selected["class_weight"],
                "training_prevalence": float(
                    final_training["onset_label"].mean()
                ),
                "validation_brier": selected["brier_score"],
                "threshold_from_validation": threshold,
                **metrics,
            }
            fold_metrics.append(fold_row)

            for index, (_, sample) in enumerate(test_x.iterrows()):
                prediction_rows.append(
                    {
                        "lead_months": lead,
                        "test_period": f"{test_start}-{test_end}",
                        "target_month": sample[DATE_COLUMN].strftime("%Y-%m"),
                        "forecast_origin": sample[
                            "forecast_origin"
                        ].strftime("%Y-%m"),
                        "actual_onset": int(sample["onset_label"]),
                        "predicted_probability": float(probabilities[index]),
                        "threshold_from_validation": threshold,
                        "predicted_onset": int(
                            probabilities[index] >= threshold
                        ),
                        "selected_feature_set": selected["feature_set"],
                    }
                )

    grid_frame = pd.DataFrame(validation_grid)
    selection_frame = pd.DataFrame(selections)
    fold_frame = pd.DataFrame(fold_metrics)
    predictions_frame = pd.DataFrame(prediction_rows)

    pooled_rows = []
    for lead, group in predictions_frame.groupby("lead_months"):
        lead_folds = fold_frame.loc[fold_frame["lead_months"] == lead]
        labels = group["actual_onset"].to_numpy()
        probabilities = group["predicted_probability"].to_numpy()
        predictions = group["predicted_onset"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).ravel()
        brier = float(brier_score_loss(labels, probabilities))
        training_prevalence = group["test_period"].map(
            lead_folds.set_index("test_period")["training_prevalence"]
        ).to_numpy(dtype=float)
        baseline_brier = float(
            brier_score_loss(labels, training_prevalence)
        )
        pooled_rows.append(
            {
                "lead_months": int(lead),
                "samples": len(group),
                "observed_onsets": int(labels.sum()),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "precision": float(
                    precision_score(labels, predictions, zero_division=0)
                ),
                "recall": float(
                    recall_score(labels, predictions, zero_division=0)
                ),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "average_precision": float(
                    average_precision_score(labels, probabilities)
                ),
                "brier_score": brier,
                "training_prevalence_brier": baseline_brier,
                "brier_skill_score": (
                    1.0 - brier / baseline_brier
                    if baseline_brier > 0
                    else np.nan
                ),
                "mean_fold_brier_skill_score": float(
                    lead_folds["brier_skill_score"].mean()
                ),
            }
        )

    grid_frame.to_csv(OUTPUT_DIR / "validation_model_grid.csv", index=False)
    selection_frame.to_csv(
        OUTPUT_DIR / "selected_models_by_fold.csv",
        index=False,
    )
    fold_frame.to_csv(OUTPUT_DIR / "heldout_fold_metrics.csv", index=False)
    predictions_frame.to_csv(
        OUTPUT_DIR / "heldout_predictions.csv",
        index=False,
    )
    pd.DataFrame(pooled_rows).to_csv(
        OUTPUT_DIR / "pooled_heldout_metrics.csv",
        index=False,
    )

    print("Selected models by chronological fold:")
    print(
        selection_frame[
            [
                "lead_months",
                "test_period",
                "feature_set",
                "C",
                "class_weight",
                "brier_score",
                "average_precision",
                "validation_threshold",
                "validation_threshold_f1",
            ]
        ].to_string(index=False)
    )
    print("\nPooled held-out results:")
    print(pd.DataFrame(pooled_rows).to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
