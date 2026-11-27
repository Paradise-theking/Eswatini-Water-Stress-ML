"""Cumulative drought-onset forecasting against a fold-local state-transition baseline.

For each origin t and horizon h, the target is 1 iff at least one new
transition from non-drought to drought occurs in t+1,...,t+h. A target is
missing unless every SWBA value from t through t+h is observed. Validation
uses the Paper 2 five-fold expanding chronological design and matched origins.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/expanded/mnjoli_water_stress_monthly_1981_2025_swba.csv"
OUT = ROOT / "outputs/research/paper2_cumulative_onset"
HORIZONS = [1, 3, 6]
N_FOLDS = 5
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]
DRY_THRESHOLD = -1.0
TRANSITION_PRIOR_STRENGTH = 5.0


def load_data():
    df = (pd.read_csv(SOURCE, parse_dates=["month_date"])
          .sort_values("month_date").set_index("month_date").asfreq("MS"))
    swba = pd.to_numeric(df["swba_3month"], errors="coerce")
    climate = [
        "precipitation_mm", "pet_mm", "water_balance_mm",
        "temperature_mean_c", "temperature_min_c", "temperature_max_c",
        "dewpoint_c", "surface_pressure", "u_component_of_wind_10m",
        "v_component_of_wind_10m", "surface_solar_radiation_downwards_sum",
        "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2",
        "runoff_sum", "surface_runoff_sum", "water_balance_3month",
        "water_balance_6month",
    ]
    climate = [c for c in climate if c in df.columns]
    if not climate:
        raise RuntimeError("No expected hydroclimatic predictors found.")
    X = df[climate].copy()
    for c in climate:
        X[f"{c}_lag1"] = df[c].shift(1)
        X[f"{c}_lag3"] = df[c].shift(3)
    X["swba_origin"] = swba
    for lag in (1, 3, 6):
        X[f"swba_lag{lag}"] = swba.shift(lag)
    X["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    return df, swba, X


def cumulative_onset_target(swba, horizon):
    """Return nullable target; any missing SWBA in the future window stays missing."""
    values = swba.to_numpy(dtype=float)
    y = np.full(len(values), np.nan, dtype=float)
    for i in range(len(values) - horizon):
        window = values[i:i + horizon + 1]  # includes origin for onset predecessor
        if not np.isfinite(window).all():
            continue
        event = any(
            values[j] <= DRY_THRESHOLD and values[j - 1] > DRY_THRESHOLD
            for j in range(i + 1, i + horizon + 1)
        )
        y[i] = float(event)
    return pd.Series(y, index=swba.index, name=f"cumulative_onset_{horizon}m")


def make_folds(n_rows):
    first_validation = int(n_rows * 0.60)
    edges = np.linspace(first_validation, n_rows, N_FOLDS + 1, dtype=int)
    return [
        (np.arange(0, edges[k], dtype=int),
         np.arange(edges[k], edges[k + 1], dtype=int))
        for k in range(N_FOLDS) if edges[k + 1] > edges[k]
    ]


def make_model(class_weight):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0, max_iter=3000, class_weight=class_weight
        )),
    ])


def score_binary(y, p, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "events": int(y.sum()),
        "event_fraction": float(y.mean()),
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_alarms_per_100_origins": float(100 * fp / len(y)),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, swba, X = load_data()
    folds = make_folds(len(df))
    prediction_rows, fold_rows, window_rows = [], [], []

    for horizon in HORIZONS:
        target = cumulative_onset_target(swba, horizon)
        for fold, (train_idx, val_idx) in enumerate(folds, start=1):
            # Purge training origins whose cumulative target window reaches
            # the first validation origin. This is stricter than merely
            # preventing target-month overlap.
            train_idx = train_idx[train_idx + horizon < val_idx[0]]
            train_idx = train_idx[target.iloc[train_idx].notna().to_numpy()]
            val_idx = val_idx[target.iloc[val_idx].notna().to_numpy()]
            if not len(train_idx) or not len(val_idx):
                continue

            y_train = target.iloc[train_idx].astype(int).to_numpy()
            y_val = target.iloc[val_idx].astype(int).to_numpy()
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            origin_states = np.select(
                [swba.iloc[train_idx].to_numpy() <= -1,
                 swba.iloc[train_idx].to_numpy() <= -0.5],
                [2, 1], default=0
            )
            val_states = np.select(
                [swba.iloc[val_idx].to_numpy() <= -1,
                 swba.iloc[val_idx].to_numpy() <= -0.5],
                [2, 1], default=0
            )
            prior = float(y_train.mean())

            # Conditional state-transition benchmark: fold-local empirical
            # cumulative-onset frequency given the observed origin severity
            # state, shrunk toward the fold event prevalence.
            p_transition = []
            for state in val_states:
                group = y_train[origin_states == state]
                p = (float(group.sum()) + TRANSITION_PRIOR_STRENGTH * prior) / (
                    len(group) + TRANSITION_PRIOR_STRENGTH
                )
                p_transition.append(p)
            model_probs = {"state_transition": np.asarray(p_transition)}
            model_probs["training_prevalence"] = np.full(len(val_idx), prior)

            for label, weight in [("logistic_unweighted", None),
                                  ("logistic_balanced", "balanced")]:
                if np.unique(y_train).size < 2:
                    continue
                estimator = make_model(weight)
                estimator.fit(X_train, y_train)
                classes = list(estimator.named_steps["classifier"].classes_)
                model_probs[label] = estimator.predict_proba(X_val)[:, classes.index(1)]

            fold_meta = {
                "horizon_months": horizon, "fold": fold,
                "train_origin_start": df.index[train_idx[0]].date().isoformat(),
                "train_origin_end": df.index[train_idx[-1]].date().isoformat(),
                "validation_origin_start": df.index[val_idx[0]].date().isoformat(),
                "validation_origin_end": df.index[val_idx[-1]].date().isoformat(),
                "n_train": int(len(train_idx)), "n_validation": int(len(val_idx)),
                "train_events": int(y_train.sum()), "validation_events": int(y_val.sum()),
                "train_prevalence": prior,
            }
            window_rows.append(fold_meta)

            for model_name, probs in model_probs.items():
                metrics = score_binary(y_val, probs)
                fold_rows.append({**fold_meta, "model": model_name, **metrics})
                for j, idx in enumerate(val_idx):
                    prediction_rows.append({
                        "horizon_months": horizon, "fold": fold,
                        "origin": df.index[idx].date().isoformat(),
                        "target_window_start": df.index[idx + 1].date().isoformat(),
                        "target_window_end": df.index[idx + horizon].date().isoformat(),
                        "origin_state": int(val_states[j]),
                        "actual": int(y_val[j]), "model": model_name,
                        "probability": float(probs[j]),
                    })

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise RuntimeError("No cumulative-onset predictions generated.")
    predictions.to_csv(OUT / "cumulative_onset_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "cumulative_onset_fold_metrics.csv", index=False)
    pd.DataFrame(window_rows).to_csv(OUT / "cumulative_onset_fold_windows.csv", index=False)

    pooled = []
    for (h, model), group in predictions.groupby(["horizon_months", "model"]):
        result = score_binary(group.actual, group.probability)
        pooled.append({"horizon_months": int(h), "model": model, **result})
    pd.DataFrame(pooled).to_csv(OUT / "cumulative_onset_pooled_metrics.csv", index=False)

    threshold_rows = []
    for (h, model), group in predictions.groupby(["horizon_months", "model"]):
        for threshold in THRESHOLDS:
            threshold_rows.append({
                "horizon_months": int(h), "model": model,
                **score_binary(group.actual, group.probability, threshold)
            })
    pd.DataFrame(threshold_rows).to_csv(OUT / "cumulative_onset_threshold_sensitivity.csv", index=False)

    metadata = {
        "target": "1 if at least one new transition SWBA> -1 to SWBA<= -1 occurs in t+1,...,t+h; otherwise 0",
        "missing_label_rule": "Target remains missing unless every monthly SWBA value from origin t through t+h is observed; missing future labels are never recoded as non-events.",
        "horizons_months": HORIZONS,
        "validation": "Same five-fold expanding chronological design: first 60% of full monthly timeline is initial training; remaining timeline is split into five contiguous validation blocks.",
        "purging": "Training origin retained only if origin_index+h is strictly earlier than first validation origin.",
        "models": ["unweighted logistic regression", "balanced logistic regression", "conditional origin-state transition frequency", "training-fold event prevalence"],
        "transition_baseline": f"Fold-local cumulative-onset frequency conditional on origin severity state, shrunk toward fold prevalence with prior strength {TRANSITION_PRIOR_STRENGTH}.",
        "features": "Origin-time climate variables, lagged climate, observed origin SWBA and lags, month harmonics; no post-origin values used as predictors.",
        "probability_threshold": 0.5,
        "threshold_sensitivity": THRESHOLDS,
        "final_test_used": False,
        "interpretation": "Exploratory validation only. Overlapping cumulative windows induce dependence; fold-level and pooled metrics must be interpreted with sparse event counts and uncertainty.",
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(pooled).to_string(index=False))
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
