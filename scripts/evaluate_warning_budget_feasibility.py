from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import screen_drought_warning_feasibility as feasibility


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "expanded"
    / "drought_onset_model_preparation"
)
HORIZONS = feasibility.HORIZONS
OUTER_FOLDS = feasibility.OUTER_FOLDS
METHODS = feasibility.METHODS
MAX_FALSE_WARNING_EPISODES_PER_YEAR = 0.5
MAX_FALSE_WARNING_MONTHS_PER_YEAR = 3.0


def select_threshold(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    validation_years: int,
) -> dict[str, Any]:
    labels = validation["onset_within_horizon"].to_numpy(dtype=int)
    candidates = np.unique(
        np.concatenate(
            (
                np.array([0.0]),
                probabilities,
                np.array([np.nextafter(probabilities.max(), np.inf)]),
            )
        )
    )
    feasible: list[dict[str, Any]] = []
    for threshold in candidates:
        warned = probabilities >= threshold
        false_warning_months = int(np.sum(warned & (labels == 0)))
        warning_episode_count, false_warning_episodes = count_warning_episodes(
            validation, warned
        )
        episodes_per_year = false_warning_episodes / validation_years
        months_per_year = false_warning_months / validation_years
        if (
            episodes_per_year
            > MAX_FALSE_WARNING_EPISODES_PER_YEAR
            or months_per_year > MAX_FALSE_WARNING_MONTHS_PER_YEAR
        ):
            continue

        observed_events = unique_events(validation)
        detected_events = unique_events(validation.loc[warned])
        detected_count = len(observed_events & detected_events)
        feasible.append(
            {
                "threshold": float(threshold),
                "validation_predicted_warning_months": int(warned.sum()),
                "validation_false_warning_months": false_warning_months,
                "validation_warning_episodes": warning_episode_count,
                "validation_false_warning_episodes": false_warning_episodes,
                "validation_false_warning_episodes_per_year": episodes_per_year,
                "validation_false_warning_months_per_year": months_per_year,
                "validation_unique_onsets": len(observed_events),
                "validation_detected_unique_onsets": detected_count,
                "validation_event_recall": (
                    detected_count / len(observed_events)
                    if observed_events
                    else float("nan")
                ),
                "validation_constraints_met": True,
            }
        )

    if not feasible:
        raise ValueError("No threshold candidates were evaluated.")
    return max(
        feasible,
        key=lambda result: (
            result["validation_event_recall"],
            -result["validation_false_warning_months"],
            -result["validation_warning_episodes"],
            result["threshold"],
        ),
    )


def unique_events(samples: pd.DataFrame) -> set[str]:
    return {
        event_month
        for event_months in samples["onset_months_within_horizon"]
        for event_month in event_months
    }


def count_warning_episodes(
    samples: pd.DataFrame,
    warned: np.ndarray,
) -> tuple[int, int]:
    warning_periods = sorted(
        samples.loc[warned, "forecast_origin"].tolist()
    )
    if not warning_periods:
        return 0, 0

    episode_groups: list[list[pd.Period]] = []
    current = [warning_periods[0]]
    for period in warning_periods[1:]:
        if period == current[-1] + 1:
            current.append(period)
        else:
            episode_groups.append(current)
            current = [period]
    episode_groups.append(current)

    false_episodes = 0
    for episode in episode_groups:
        episode_samples = samples.loc[
            samples["forecast_origin"].isin(episode)
        ]
        if not episode_samples["onset_within_horizon"].astype(bool).any():
            false_episodes += 1
    return len(episode_groups), false_episodes


def predictions_for_fold(
    data: pd.DataFrame,
    samples: pd.DataFrame,
    horizon: int,
    test_start: int,
    test_end: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Period,
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    validation_start = test_start - 5
    validation_end = test_start - 1
    validation_first = pd.Period(f"{validation_start}-01", freq="M")
    training_cutoff = validation_first - horizon - 1
    validation_end_period = pd.Period(f"{validation_end}-12", freq="M")
    test_end_period = pd.Period(f"{test_end}-12", freq="M")

    training = samples.loc[
        samples["forecast_origin"] <= training_cutoff
    ].copy()
    validation = samples.loc[
        samples["forecast_origin"].map(
            lambda period: validation_start <= period.year <= validation_end
        )
        & (
            samples["forecast_origin"]
            <= validation_end_period - horizon
        )
    ].copy()
    test = samples.loc[
        samples["forecast_origin"].map(
            lambda period: test_start <= period.year <= test_end
        )
        & (samples["forecast_origin"] <= test_end_period - horizon)
    ].copy()
    if training.empty or validation.empty or test.empty:
        raise ValueError(
            f"Empty split for horizon {horizon}, fold {test_start}-{test_end}."
        )
    if training["onset_within_horizon"].nunique() != 2:
        raise ValueError(
            f"Training data lacks both classes for horizon {horizon}, "
            f"fold {test_start}-{test_end}."
        )

    prevalence = float(training["onset_within_horizon"].mean())
    prediction_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "training_prevalence": (
            np.full(len(validation), prevalence),
            np.full(len(test), prevalence),
        ),
        "seasonal_baseline": (
            feasibility.seasonal_probabilities(
                training, validation, prevalence
            ),
            feasibility.seasonal_probabilities(training, test, prevalence),
        ),
    }

    for method, columns in (
        ("current_state", ["origin_swba"]),
        (
            "hydro_core",
            [f"{feature}_anom" for feature in feasibility.HYDRO_FEATURES],
        ),
    ):
        model = feasibility.make_model()
        if method == "current_state":
            training_features = training
            validation_features = validation
            test_features = test
        else:
            training_features = feasibility.add_hydro_anomalies(
                training, data, training_cutoff
            )
            validation_features = feasibility.add_hydro_anomalies(
                validation, data, training_cutoff
            )
            test_features = feasibility.add_hydro_anomalies(
                test, data, training_cutoff
            )
        model.fit(
            training_features[columns],
            training_features["onset_within_horizon"],
        )
        prediction_sets[method] = (
            model.predict_proba(validation_features[columns])[:, 1],
            model.predict_proba(test_features[columns])[:, 1],
        )

    if set(prediction_sets) != set(METHODS):
        raise RuntimeError("The feasibility-screen methods changed unexpectedly.")
    return validation, test, training_cutoff, prediction_sets


def evaluate_test(
    samples: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    horizon: int,
    fold: str,
    method: str,
    validation_result: dict[str, Any],
    evaluation_years: int,
) -> dict[str, Any]:
    warned = probabilities >= threshold
    labels = samples["onset_within_horizon"].to_numpy(dtype=int)
    false_warning_months = int(np.sum(warned & (labels == 0)))
    warning_episodes, false_warning_episodes = count_warning_episodes(
        samples, warned
    )
    observed_events = unique_events(samples)
    detected_events = unique_events(samples.loc[warned])
    detected_count = len(observed_events & detected_events)
    brier = float(brier_score_loss(labels, probabilities))
    prevalence = float(samples.attrs["training_prevalence"])
    baseline_brier = float(
        brier_score_loss(labels, np.full(len(samples), prevalence))
    )
    return {
        "horizon_months": horizon,
        "test_period": fold,
        "method": method,
        "selected_threshold": threshold,
        **validation_result,
        "test_origins": len(samples),
        "test_positive_windows": int(labels.sum()),
        "test_unique_onsets": len(observed_events),
        "test_detected_unique_onsets": detected_count,
        "test_missed_unique_onsets": len(observed_events) - detected_count,
        "test_event_recall": (
            detected_count / len(observed_events)
            if observed_events
            else float("nan")
        ),
        "test_predicted_warning_months": int(warned.sum()),
        "test_false_warning_months": false_warning_months,
        "test_false_warning_months_per_year": (
            false_warning_months / evaluation_years
        ),
        "test_warning_episodes": warning_episodes,
        "test_false_warning_episodes": false_warning_episodes,
        "test_false_warning_episodes_per_year": (
            false_warning_episodes / evaluation_years
        ),
        "test_false_warning_limits_met": (
            false_warning_months / evaluation_years
            <= MAX_FALSE_WARNING_MONTHS_PER_YEAR
            and false_warning_episodes / evaluation_years
            <= MAX_FALSE_WARNING_EPISODES_PER_YEAR
        ),
        "test_brier_score": brier,
        "test_prevalence_baseline_brier": baseline_brier,
        "test_brier_skill_vs_prevalence": (
            1 - brier / baseline_brier if baseline_brier > 0 else np.nan
        ),
    }


def main() -> None:
    data, audit, input_hash = feasibility.load_data()
    output_dir = OUTPUT_ROOT / (
        "warning_budget_feasibility_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        samples = feasibility.build_samples(data, horizon)
        for test_start, test_end in OUTER_FOLDS:
            validation, test, training_cutoff, prediction_sets = (
                predictions_for_fold(
                    data, samples, horizon, test_start, test_end
                )
            )
            validation_years = 5
            fold = f"{test_start}-{test_end}"
            for method, (validation_probabilities, test_probabilities) in (
                prediction_sets.items()
            ):
                threshold_result = select_threshold(
                    validation,
                    validation_probabilities,
                    validation_years,
                )
                test = test.copy()
                test.attrs["training_prevalence"] = float(
                    samples.loc[
                        samples["forecast_origin"] <= training_cutoff,
                        "onset_within_horizon",
                    ].mean()
                )
                results.append(
                    evaluate_test(
                        test,
                        test_probabilities,
                        threshold_result["threshold"],
                        horizon,
                        fold,
                        method,
                        threshold_result,
                        evaluation_years=test_end - test_start + 1,
                    )
                )
                for sample, probability in zip(
                    test.itertuples(index=False), test_probabilities
                ):
                    prediction_rows.append(
                        {
                            "horizon_months": horizon,
                            "test_period": fold,
                            "method": method,
                            "forecast_origin": str(sample.forecast_origin),
                            "onset_within_horizon": int(
                                sample.onset_within_horizon
                            ),
                            "onset_months_within_horizon": json.dumps(
                                sample.onset_months_within_horizon
                            ),
                            "predicted_probability": float(probability),
                            "selected_threshold": threshold_result["threshold"],
                            "predicted_warning": int(
                                probability >= threshold_result["threshold"]
                            ),
                        }
                    )

    result_frame = pd.DataFrame(results)
    prediction_frame = pd.DataFrame(prediction_rows)
    result_frame.to_csv(output_dir / "fold_warning_budget_metrics.csv", index=False)
    prediction_frame.to_csv(
        output_dir / "outer_warning_predictions.csv", index=False
    )
    summary = (
        result_frame.groupby(["horizon_months", "method"], as_index=False)
        .agg(
            folds=("test_period", "nunique"),
            folds_meeting_test_limits=("test_false_warning_limits_met", "sum"),
            test_unique_onsets=("test_unique_onsets", "sum"),
            test_detected_unique_onsets=("test_detected_unique_onsets", "sum"),
            test_false_warning_months=("test_false_warning_months", "sum"),
            test_false_warning_episodes=("test_false_warning_episodes", "sum"),
        )
    )
    summary["pooled_unique_event_recall"] = (
        summary["test_detected_unique_onsets"] / summary["test_unique_onsets"]
    )
    summary.to_csv(output_dir / "pooled_warning_budget_summary.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_hash,
        "audit_passed": bool(audit["all_checks_passed"]),
        "source_period": {
            "start_month": audit["dataset"]["start_month"],
            "end_month": audit["dataset"]["end_month"],
        },
        "threshold_selection": (
            "For each fold, select on the immediately preceding five-year "
            "validation period, maximizing unique-onset event recall subject "
            "to the false-warning limits; ties prefer fewer false-warning "
            "months, fewer warning episodes, then a higher threshold. The "
            "same model fit is used for validation and subsequent test."
        ),
        "warning_episode_definition": (
            "A warning episode is a run of warnings on consecutive monthly "
            "forecast origins. It is false only if none of its forecast "
            "windows contains an onset."
        ),
        "false_warning_month_definition": (
            "A warned forecast origin whose forward outcome window contains "
            "no onset."
        ),
        "limits": {
            "false_warning_episodes_per_year": MAX_FALSE_WARNING_EPISODES_PER_YEAR,
            "false_warning_months_per_year": MAX_FALSE_WARNING_MONTHS_PER_YEAR,
        },
        "limitations": [
            "Previously examined historical folds: exploratory, not independent validation.",
            "Thresholds are selected on small validation event counts.",
            "Overlapping forecast windows are dependent.",
            "The same onset may be covered by multiple warning origins.",
            "Historical predictor values do not prove issue-time availability.",
            "The final test fold excludes forecast origins with incomplete horizon outcomes.",
        ],
        "outputs": [
            "fold_warning_budget_metrics.csv",
            "outer_warning_predictions.csv",
            "pooled_warning_budget_summary.csv",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Warning-budget evaluation written to: {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
