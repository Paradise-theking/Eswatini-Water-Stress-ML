from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_nested_drought_onset_backtest as backtest


ROOT = Path(__file__).resolve().parents[1]
REFRESH_DIR = (
    ROOT
    / "outputs"
    / "expanded"
    / "shadow_evaluation_data"
    / "refresh_through_2026-09"
)
INPUT_FILE = REFRESH_DIR / "mnjoli_water_stress_monthly_updated_swba.csv"
AUDIT_REPORT = REFRESH_DIR / "data_audit" / "audit_report.json"
OUTPUT_DIR = REFRESH_DIR / "unseen_2026_evaluation"
TRAINING_END = pd.Period("2020-12", freq="M")
VALIDATION_START_YEAR = 2021
VALIDATION_END_YEAR = 2025
TEST_YEAR = 2026


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_refreshed_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not INPUT_FILE.is_file() or not AUDIT_REPORT.is_file():
        raise FileNotFoundError(
            "Refresh and audit the 2026 data before running the unseen holdout."
        )
    audit = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
    if not audit.get("all_checks_passed"):
        raise ValueError("The refreshed-data audit did not pass.")
    input_hash = sha256_file(INPUT_FILE)
    if input_hash != audit.get("input_sha256"):
        raise ValueError("The audited 2026 holdout file has changed.")

    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=[backtest.DATE_COLUMN],
    )
    samples = backtest.build_samples(data)
    if int(samples["onset_label"].sum()) != int(
        audit["onsets"]["onset_events"]
    ):
        raise ValueError("Rebuilt labels disagree with the refreshed audit.")
    if samples["target_period"].dt.year.max() != TEST_YEAR:
        raise ValueError("No 2026 target months are present in the audited data.")
    return data, samples, audit


def analyze_historical_warning_budget() -> pd.DataFrame:
    predictions_path = (
        ROOT
        / "outputs"
        / "expanded"
        / "drought_onset_model_preparation"
        / "nested_chronological_backtest_1month"
        / "outer_test_predictions.csv"
    )
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Missing historical backtest predictions: {predictions_path}"
        )
    predictions = pd.read_csv(predictions_path, parse_dates=["target_month"])
    results: list[dict[str, Any]] = []

    for period, fold in predictions.groupby("test_period", sort=True):
        labels = fold["actual_onset"].to_numpy(dtype=int)
        probabilities = fold["predicted_probability"].to_numpy(dtype=float)
        years = int(fold["target_month"].dt.year.nunique())
        candidates = np.unique(
            np.concatenate(([0.0], probabilities, [1.000001]))
        )
        feasible = []
        for threshold in candidates:
            predicted = probabilities >= threshold
            false_months = int(np.sum(predicted & (labels == 0)))
            warning_frame = pd.DataFrame(
                {
                    "target_period": pd.PeriodIndex(
                        fold["target_period"], freq="M"
                    ),
                    "target_month": fold["target_month"],
                    "actual_onset": labels,
                    "predicted_onset": predicted.astype(int),
                }
            )
            episodes = backtest.episode_counts(warning_frame)
            false_episodes = int(episodes["false_warning_episodes"])
            if (
                false_episodes / years
                <= backtest.MAX_FALSE_WARNING_EPISODES_PER_YEAR
                and false_months / years
                <= backtest.MAX_FALSE_WARNING_MONTHS_PER_YEAR
            ):
                feasible.append(
                    {
                        "threshold": float(threshold),
                        "hits": int(np.sum(predicted & (labels == 1))),
                        "false_warning_months": false_months,
                        "false_warning_episodes": false_episodes,
                        "false_warning_months_per_year": false_months / years,
                        "false_warning_episodes_per_year": (
                            false_episodes / years
                        ),
                        "warnings": int(predicted.sum()),
                        "recall": (
                            float(np.sum(predicted & (labels == 1)) / labels.sum())
                            if labels.sum()
                            else float("nan")
                        ),
                    }
                )
        if not feasible:
            raise RuntimeError(f"No feasible threshold found in fold {period}.")
        # This is intentionally a post-hoc diagnostic upper bound, not a valid
        # estimate of a threshold that could have been selected without test
        # labels.
        oracle = max(
            feasible,
            key=lambda row: (
                row["recall"],
                -row["false_warning_months"],
                row["threshold"],
            ),
        )
        locked = backtest.episode_counts(
            fold.assign(
                predicted_onset=(
                    fold["predicted_probability"] >= fold["threshold"]
                ).astype(int)
            )
        )
        selected_threshold = float(fold["threshold"].iloc[0])
        selected_predictions = probabilities >= selected_threshold
        selected_false_months = int(
            np.sum(selected_predictions & (labels == 0))
        )
        selected_hits = int(np.sum(selected_predictions & (labels == 1)))
        results.append(
            {
                "test_period": period,
                "test_samples": int(len(fold)),
                "observed_onsets": int(labels.sum()),
                "validation_selected_threshold": selected_threshold,
                "locked_test_hits": selected_hits,
                "locked_test_misses": int(labels.sum() - selected_hits),
                "locked_test_false_warning_months": selected_false_months,
                "locked_test_false_warning_months_per_year": (
                    selected_false_months / years
                ),
                "locked_test_false_warning_episodes": (
                    locked["false_warning_episodes"]
                ),
                "locked_test_false_warning_episodes_per_year": (
                    locked["false_warning_episodes"] / years
                ),
                "posthoc_max_hits_within_both_caps": oracle["hits"],
                "posthoc_max_recall_within_both_caps": oracle["recall"],
                "posthoc_threshold": oracle["threshold"],
                "posthoc_false_warning_months": (
                    oracle["false_warning_months"]
                ),
                "posthoc_false_warning_episodes": (
                    oracle["false_warning_episodes"]
                ),
                "posthoc_warnings": oracle["warnings"],
                "posthoc_note": (
                    "Optimistic diagnostic using test labels to find the "
                    "best feasible threshold; not valid as model performance."
                ),
            }
        )
    return pd.DataFrame(results)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    historical_feasibility = analyze_historical_warning_budget()
    historical_feasibility.to_csv(
        OUTPUT_DIR / "historical_warning_budget_feasibility.csv",
        index=False,
    )

    data, samples, audit = load_refreshed_data()
    training = samples.loc[
        samples["target_period"] <= TRAINING_END
    ].copy()
    validation = samples.loc[
        samples["target_period"].dt.year.between(
            VALIDATION_START_YEAR, VALIDATION_END_YEAR
        )
    ].copy()
    test = samples.loc[
        samples["target_period"].dt.year == TEST_YEAR
    ].copy()
    if test.empty:
        raise ValueError("No eligible 2026 holdout samples are available.")
    if training["onset_label"].nunique() != 2:
        raise ValueError("Pre-2021 training data lacks both outcome classes.")
    if validation["onset_label"].sum() == 0:
        raise ValueError("2021-2025 validation data has no onset to tune policy.")

    selected, candidate_rows = backtest.select_candidate(
        data,
        training,
        VALIDATION_START_YEAR,
    )
    features = backtest.FEATURE_SETS[selected["feature_set"]]
    columns = backtest.model_columns(features)
    training_x = backtest.transform_features(
        training, data, TRAINING_END
    )
    validation_x = backtest.transform_features(
        validation, data, TRAINING_END
    )
    test_x = backtest.transform_features(
        test, data, TRAINING_END
    )
    model = backtest.make_model(
        float(selected["C"]),
        None
        if selected["class_weight"] == "none"
        else selected["class_weight"],
    )
    model.fit(training_x[columns], training_x["onset_label"])
    validation_probabilities = model.predict_proba(
        validation_x[columns]
    )[:, 1]
    threshold = backtest.select_recall_threshold(
        validation_x["onset_label"].to_numpy(dtype=int),
        validation_probabilities,
        validation_x["target_period"],
        evaluation_years=VALIDATION_END_YEAR - VALIDATION_START_YEAR + 1,
    )
    test_probabilities = model.predict_proba(test_x[columns])[:, 1]
    test_labels = test_x["onset_label"].to_numpy(dtype=int)
    test_predictions = (test_probabilities >= threshold["threshold"]).astype(int)
    predictions = pd.DataFrame(
        {
            "target_period": test_x["target_period"].astype(str),
            "forecast_origin": test_x["forecast_origin"].dt.strftime("%Y-%m"),
            "actual_onset": test_labels,
            "predicted_probability": test_probabilities,
            "threshold": threshold["threshold"],
            "predicted_warning": test_predictions,
        }
    )
    predictions.to_csv(
        OUTPUT_DIR / "unseen_2026_predictions.csv",
        index=False,
    )

    labels_count = int(test_labels.sum())
    warnings_count = int(test_predictions.sum())
    false_months = int(
        np.sum((test_predictions == 1) & (test_labels == 0))
    )
    periods = pd.PeriodIndex(test_x["target_period"], freq="M")
    warning_frame = pd.DataFrame(
        {
            "target_period": periods,
            "target_month": periods.to_timestamp(),
            "actual_onset": test_labels,
            "predicted_onset": test_predictions,
        }
    )
    episodes = backtest.episode_counts(warning_frame)
    probability_metrics = backtest.probability_metrics(
        test_labels,
        test_probabilities,
        np.full(len(test), float(training["onset_label"].mean())),
    )
    probability_metrics = {
        key: (
            float(value)
            if np.isfinite(value)
            else None
        )
        for key, value in probability_metrics.items()
    }
    summary = {
        "evaluation": "Unseen 2026 chronological holdout",
        "input_sha256": sha256_file(INPUT_FILE),
        "audit_sha256": audit["input_sha256"],
        "audit_passed": bool(audit["all_checks_passed"]),
        "data_available_through": audit["dataset"]["end_month"],
        "test_period": f"2026-01 through {audit['dataset']['end_month']}",
        "test_months": int(len(test)),
        "observed_onsets": labels_count,
        "model_training_target_period": f"start through {TRAINING_END}",
        "validation_period": (
            f"{VALIDATION_START_YEAR}-{VALIDATION_END_YEAR}"
        ),
        "selected_feature_set": selected["feature_set"],
        "selected_features": list(features),
        "C": selected["C"],
        "class_weight": selected["class_weight"],
        "validation_selected_threshold": threshold["threshold"],
        "validation_recall": threshold["recall"],
        "validation_predicted_warnings": (
            threshold["predicted_warnings"]
        ),
        "validation_false_warning_episodes": (
            threshold["false_warning_episodes"]
        ),
        "validation_false_warning_episodes_per_year": (
            threshold["false_warning_episodes_per_year"]
        ),
        "validation_false_warning_months": (
            threshold["false_warning_months"]
        ),
        "validation_false_warning_months_per_year": (
            threshold["false_warning_months_per_year"]
        ),
        "warnings_in_unseen_period": warnings_count,
        "hits_in_unseen_period": int(
            np.sum((test_predictions == 1) & (test_labels == 1))
        ),
        "misses_in_unseen_period": int(
            np.sum((test_predictions == 0) & (test_labels == 1))
        ),
        "false_warning_months_in_unseen_period": false_months,
        "false_warning_episodes_in_unseen_period": (
            episodes["false_warning_episodes"]
        ),
        "false_warning_episode_rate_per_year": None,
        "probability_metrics": probability_metrics,
        "interpretation": (
            "The 2026 labels were excluded from feature/model selection and "
            "threshold selection. No onset has occurred in the observed "
            "holdout, so recall and detection skill cannot be estimated. "
            "False-warning counts are descriptive for these partial-year "
            "months and do not establish the annual warning-budget rate."
        ),
        "threshold_policy": {
            "priority": "maximize recall",
            "max_false_warning_episodes_per_year": (
                backtest.MAX_FALSE_WARNING_EPISODES_PER_YEAR
            ),
            "max_false_warning_months_per_year": (
                backtest.MAX_FALSE_WARNING_MONTHS_PER_YEAR
            ),
        },
        "candidate_grid_size": len(candidate_rows),
        "outputs": {
            "historical_feasibility": str(
                OUTPUT_DIR / "historical_warning_budget_feasibility.csv"
            ),
            "2026_predictions": str(
                OUTPUT_DIR / "unseen_2026_predictions.csv"
            ),
        },
    }
    (OUTPUT_DIR / "unseen_2026_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print("\nHistorical warning-budget feasibility:")
    print(historical_feasibility.to_string(index=False))


if __name__ == "__main__":
    main()
