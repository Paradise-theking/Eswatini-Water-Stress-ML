"""Matched-origin non-ML baselines for Paper 2 diagnostics."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score,
)

from paper2_matched_diagnostics import (
    DIRECTIONS, HORIZONS, OUT, STATES, make_folds, prepare_data,
)

BASELINE_OUT = OUT / "matched_baselines"
BINARY_TASKS = {"drought_state", "onset"}
LABELS = {"multistate": STATES, "direction": DIRECTIONS}


def build_targets(swba, state, horizon):
    future_swba = swba.shift(-horizon)
    future_state = state.shift(-horizon)
    previous_swba = swba.shift(-(horizon - 1))

    onset = ((future_swba <= -1) & (previous_swba > -1)).astype(float)
    onset.loc[future_swba.isna() | previous_swba.isna()] = np.nan

    drought = (future_swba <= -1).astype(float)
    drought.loc[future_swba.isna()] = np.nan

    direction = np.sign(future_state - state)
    return {
        "drought_state": drought,
        "onset": onset,
        "multistate": future_state,
        "direction": direction,
    }


def smooth_distribution(counts, prior, alpha=1.0):
    counts = np.asarray(counts, dtype=float)
    prior = np.asarray(prior, dtype=float)
    return (counts + alpha * prior) / (counts.sum() + alpha)


def get_probabilities(task, baseline, train_idx, origin_idx, target_idx,
                      target, state, dates, labels):
    y_train = target.iloc[train_idx]
    valid = y_train.notna().to_numpy()
    train_idx = np.asarray(train_idx)[valid]
    y_train = target.iloc[train_idx].astype(int)
    classes = list(labels) if labels is not None else [0, 1]

    if labels is None:
        prior_positive = float(y_train.mean())
        prior = np.array([1.0 - prior_positive, prior_positive])
    else:
        counts = np.array([(y_train == label).sum() for label in classes], dtype=float)
        prior = counts / counts.sum() if counts.sum() else np.repeat(1 / len(classes), len(classes))

    if baseline == "persistence":
        if task == "drought_state":
            p = float(state.iloc[origin_idx] == 2) if pd.notna(state.iloc[origin_idx]) else prior[1]
            return np.array([1 - p, p])
        if task == "onset":
            return np.array([1.0, 0.0])
        if task == "multistate":
            origin_state = state.iloc[origin_idx]
            return np.eye(3)[int(origin_state)] if pd.notna(origin_state) else prior
        return np.array([0.0, 1.0, 0.0])

    if baseline == "training_prevalence":
        return prior

    if baseline == "target_month_climatology":
        target_month = dates[target_idx].month
        month_mask = np.array([dates[idx].month == target_month for idx in train_idx])
        month_y = y_train.to_numpy()[month_mask]
        if labels is None:
            month_counts = np.array([(month_y == 0).sum(), (month_y == 1).sum()])
        else:
            month_counts = np.array([(month_y == label).sum() for label in classes])
        return smooth_distribution(month_counts, prior, alpha=12.0)

    if baseline == "origin_state_transition":
        origin_state = state.iloc[origin_idx]
        if pd.isna(origin_state):
            return prior
        state_mask = np.array([state.iloc[idx] == origin_state for idx in train_idx])
        conditional_y = y_train.to_numpy()[state_mask]
        if labels is None:
            conditional_counts = np.array([
                (conditional_y == 0).sum(), (conditional_y == 1).sum()
            ])
        else:
            conditional_counts = np.array([
                (conditional_y == label).sum() for label in classes
            ])
        return smooth_distribution(conditional_counts, prior, alpha=float(len(classes)))

    raise ValueError(f"Unknown baseline: {baseline}")


def main():
    BASELINE_OUT.mkdir(parents=True, exist_ok=True)
    _, swba, state, _ = prepare_data()
    dates = swba.index
    folds = make_folds(len(swba))
    prediction_file = OUT / "matched_validation_predictions.csv"
    if not prediction_file.exists():
        raise FileNotFoundError("Run scripts/paper2_matched_diagnostics.py first.")

    model_predictions = pd.read_csv(prediction_file, parse_dates=["origin", "target_date"])
    rows = []
    baseline_names = [
        "persistence",
        "training_prevalence",
        "target_month_climatology",
        "origin_state_transition",
    ]

    for horizon in HORIZONS:
        targets = build_targets(swba, state, horizon)
        for fold_number, (train_idx, validation_idx) in enumerate(folds, start=1):
            safe_train = train_idx[train_idx + horizon < validation_idx[0]]
            for task, target in targets.items():
                model_rows = model_predictions[
                    (model_predictions.task == task)
                    & (model_predictions.horizon == horizon)
                    & (model_predictions.fold == fold_number)
                ]
                if model_rows.empty:
                    continue

                train_valid = target.iloc[safe_train].notna().to_numpy()
                usable_train = safe_train[train_valid]
                if not len(usable_train):
                    continue

                labels = LABELS.get(task)
                for row in model_rows.itertuples(index=False):
                    origin_idx = dates.get_loc(row.origin)
                    target_idx = dates.get_loc(row.target_date)
                    actual = int(row.actual)
                    for baseline in baseline_names:
                        probs = get_probabilities(
                            task, baseline, usable_train, origin_idx, target_idx,
                            target, state, dates, labels,
                        )
                        if task in BINARY_TASKS:
                            positive_probability = float(probs[1])
                            predicted = int(positive_probability >= 0.5)
                            rows.append({
                                "task": task, "horizon": horizon, "fold": fold_number,
                                "origin": row.origin.date().isoformat(),
                                "target_date": row.target_date.date().isoformat(),
                                "actual": actual, "baseline": baseline,
                                "predicted": predicted,
                                "probability": positive_probability,
                                "probabilities": "",
                            })
                        else:
                            predicted = int(labels[int(np.argmax(probs))])
                            rows.append({
                                "task": task, "horizon": horizon, "fold": fold_number,
                                "origin": row.origin.date().isoformat(),
                                "target_date": row.target_date.date().isoformat(),
                                "actual": actual, "baseline": baseline,
                                "predicted": predicted, "probability": np.nan,
                                "probabilities": json.dumps(probs.tolist()),
                            })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No matched baseline predictions were generated.")

    frame.to_csv(BASELINE_OUT / "matched_baseline_predictions.csv", index=False)
    summaries = []
    for (task, horizon, baseline), group in frame.groupby(["task", "horizon", "baseline"]):
        actual = group.actual.astype(int).to_numpy()
        predicted = group.predicted.astype(int).to_numpy()
        result = {
            "task": task, "horizon": int(horizon), "baseline": baseline,
            "n": int(len(group)),
        }
        if task in BINARY_TASKS:
            probability = group.probability.astype(float).to_numpy()
            tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
            result.update({
                "events": int(actual.sum()), "tp": int(tp), "fp": int(fp),
                "fn": int(fn), "tn": int(tn),
                "precision": precision_score(actual, predicted, zero_division=0),
                "recall": recall_score(actual, predicted, zero_division=0),
                "f1": f1_score(actual, predicted, zero_division=0),
                "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
                "pr_auc": average_precision_score(actual, probability) if actual.sum() else np.nan,
                "brier": brier_score_loss(actual, probability),
            })
        else:
            labels = LABELS[task]
            probability = np.vstack(group.probabilities.map(json.loads).to_numpy())
            result.update({
                "accuracy": accuracy_score(actual, predicted),
                "macro_f1": f1_score(actual, predicted, labels=labels, average="macro", zero_division=0),
                "log_loss": log_loss(actual, probability, labels=labels),
                "brier": float(np.mean(np.sum((
                    np.eye(len(labels))[[labels.index(int(v)) for v in actual]] - probability
                ) ** 2, axis=1))),
            })
        summaries.append(result)

    pd.DataFrame(summaries).to_csv(BASELINE_OUT / "matched_baseline_summary.csv", index=False)
    metadata = {
        "comparison": "Baselines are evaluated on the exact task/horizon/fold/origin/target rows present in the matched logistic-regression prediction file.",
        "baselines": {
            "persistence": "Carry origin drought state forward; onset is not forecast; multistate carries origin class; direction predicts no change.",
            "training_prevalence": "Fold-local empirical class distribution.",
            "target_month_climatology": "Fold-local target-calendar-month class distribution with 12 prior pseudo-observations weighted by the training-fold class prior.",
            "origin_state_transition": "Fold-local target distribution conditional on the observed origin state, smoothed toward the training-fold class prior.",
        },
        "no_final_test_used": True,
        "warning": "Exploratory baseline audit. Small event counts and probability calibration limitations remain. Do not compare against results from different validation origins.",
    }
    (BASELINE_OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
