from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.feature_engineering import MODEL_FEATURES, add_forecast_features


ROOT = Path(__file__).resolve().parents[2]
MONTHLY_PATH = ROOT / "data" / "processed" / "mnjoli_water_stress_monthly.csv"
MODELING_PATH = ROOT / "data" / "processed" / "mnjoli_water_stress_modeling_dataset.csv"
SPEC_PATH = ROOT / "data" / "processed" / "final_model_specification.json"
MODEL_PATH = ROOT / "models" / "water_stress_ridge.joblib"
CHIRPS_COMPARISON_PATH = ROOT / "chirps_monthly_comparison_2015_2026.csv"
OUTPUT_PATH = ROOT / "chirps_v3_model_sensitivity_2015_2025.csv"


def load_inputs():
    required = [MONTHLY_PATH, MODELING_PATH, SPEC_PATH, MODEL_PATH, CHIRPS_COMPARISON_PATH]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required input files are missing: " + ", ".join(missing)
            + ". Run compare_chirps_monthly first and ensure repository data/model files are present."
        )

    monthly = pd.read_csv(MONTHLY_PATH, parse_dates=["month_date"])
    targets = pd.read_csv(
        MODELING_PATH,
        parse_dates=["month_date", "forecast_date"],
    )
    with SPEC_PATH.open("r", encoding="utf-8") as stream:
        specification = json.load(stream)
    comparison = pd.read_csv(CHIRPS_COMPARISON_PATH)

    if "target_t_plus_1" not in targets:
        raise ValueError("Modeling dataset has no target_t_plus_1 column.")
    return monthly, targets, specification, comparison


def make_features(monthly: pd.DataFrame) -> pd.DataFrame:
    engineered = add_forecast_features(monthly)
    absent = [name for name in MODEL_FEATURES if name not in engineered]
    if absent:
        raise ValueError(f"Feature engineering did not create required features: {absent}")
    return engineered


def metrics(observed: pd.Series, predicted: np.ndarray) -> dict:
    return {
        "n": int(len(observed)),
        "MAE": float(mean_absolute_error(observed, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(observed, predicted))),
        "R2": float(r2_score(observed, predicted)),
    }


def main():
    print("=" * 76)
    print("WATERWATCH CHIRPS V3 MODEL-INPUT SENSITIVITY TEST")
    print("Offline experiment: same ERA5 variables, same model, same target and dates")
    print("=" * 76)

    monthly, targets, specification, comparison = load_inputs()
    model = joblib.load(MODEL_PATH)

    if len(MODEL_FEATURES) != int(specification["number_of_features"]):
        raise ValueError("MODEL_FEATURES count does not match final model specification.")

    comparison["month_date"] = pd.to_datetime(comparison["month"] + "-01")
    comparison = comparison[[
        "month_date",
        "chirps_v3_sat_rainfall_mm",
        "chirps_v3_rnl_rainfall_mm",
    ]]

    base = monthly.sort_values("month_date").copy()
    base["month_date"] = pd.to_datetime(base["month_date"])
    base_features = make_features(base)

    target_dates = targets[[
        "month_date", "forecast_date", "target_t_plus_1"
    ]].copy()
    target_dates["month_date"] = pd.to_datetime(target_dates["month_date"])
    target_dates["forecast_date"] = pd.to_datetime(target_dates["forecast_date"])

    test_start = pd.Timestamp(specification["test_period"]["start"])
    test_end = pd.Timestamp(specification["test_period"]["end"])
    test_targets = target_dates[
        target_dates["forecast_date"].between(test_start, test_end)
    ].copy()

    if test_targets.empty:
        raise ValueError("No target rows fall within the specified held-out test period.")

    variants = {
        "chirps_v2_baseline": None,
        "chirps_v3_sat": "chirps_v3_sat_rainfall_mm",
        "chirps_v3_rnl": "chirps_v3_rnl_rainfall_mm",
    }

    prediction_frames = []
    feature_differences = []

    for variant, rainfall_column in variants.items():
        variant_monthly = base.copy()
        if rainfall_column is not None:
            replacement = comparison[["month_date", rainfall_column]].rename(
                columns={rainfall_column: "replacement_precipitation_mm"}
            )
            variant_monthly = variant_monthly.merge(
                replacement, on="month_date", how="left", validate="one_to_one"
            )
            missing = int(variant_monthly["replacement_precipitation_mm"].isna().sum())
            if missing:
                raise ValueError(
                    f"{variant} has {missing} months without matched CHIRPS v3 rainfall."
                )
            variant_monthly["precipitation_mm"] = variant_monthly[
                "replacement_precipitation_mm"
            ]
            variant_monthly = variant_monthly.drop(columns=["replacement_precipitation_mm"])

        engineered = make_features(variant_monthly)
        usable = engineered.dropna(subset=MODEL_FEATURES).copy()
        usable["prediction"] = model.predict(usable[MODEL_FEATURES])
        usable["variant"] = variant

        aligned = test_targets.merge(
            usable[["month_date", "prediction"]],
            on="month_date",
            how="inner",
            validate="one_to_one",
        )
        if len(aligned) != len(test_targets):
            raise ValueError(
                f"{variant}: matched {len(aligned)} of {len(test_targets)} held-out target rows."
            )

        aligned["variant"] = variant
        aligned["prediction_error"] = aligned["prediction"] - aligned["target_t_plus_1"]
        prediction_frames.append(aligned)

        feature_table = usable[["month_date"] + MODEL_FEATURES]
        feature_differences.append((variant, feature_table))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(OUTPUT_PATH, index=False)

    print(f"\nHeld-out forecast dates: {test_start:%Y-%m} to {test_end:%Y-%m}")
    print(f"Observed test cases: {len(test_targets)}")
    print("\nPerformance against the unchanged observed target:")
    for variant in variants:
        subset = predictions[predictions["variant"] == variant]
        result = metrics(subset["target_t_plus_1"], subset["prediction"].to_numpy())
        print(
            f"{variant:22s} n={result['n']:2d}  "
            f"MAE={result['MAE']:.4f}  RMSE={result['RMSE']:.4f}  R2={result['R2']:.4f}"
        )

    baseline = predictions[predictions["variant"] == "chirps_v2_baseline"][
        ["month_date", "prediction"]
    ].rename(columns={"prediction": "prediction_v2"})
    for variant in ("chirps_v3_sat", "chirps_v3_rnl"):
        current = predictions[predictions["variant"] == variant][
            ["month_date", "prediction"]
        ].rename(columns={"prediction": f"prediction_{variant}"})
        paired = baseline.merge(current, on="month_date", validate="one_to_one")
        delta = paired[f"prediction_{variant}"] - paired["prediction_v2"]
        print(f"\n{variant} forecast sensitivity versus v2:")
        print(f"  Mean absolute prediction change: {delta.abs().mean():.6f} SWBA")
        print(f"  Maximum absolute prediction change: {delta.abs().max():.6f} SWBA")
        print(f"  Mean signed prediction change: {delta.mean():+.6f} SWBA")

    print("\nFeature-level sensitivity over the held-out forecast origins:")
    feature_baseline = feature_differences[0][1]
    test_origins = set(test_targets["month_date"])
    feature_baseline = feature_baseline[feature_baseline["month_date"].isin(test_origins)]
    for variant, table in feature_differences[1:]:
        table = table[table["month_date"].isin(test_origins)]
        paired = feature_baseline.merge(
            table, on="month_date", suffixes=("_v2", f"_{variant}"),
            validate="one_to_one",
        )
        changes = []
        for feature in MODEL_FEATURES:
            delta = paired[f"{feature}_{variant}"] - paired[f"{feature}_v2"]
            changes.append((feature, float(delta.abs().mean()), float(delta.abs().max())))
        changed = [item for item in changes if item[1] > 1e-10]
        print(f"\n{variant}: {len(changed)} of {len(MODEL_FEATURES)} features changed")
        for feature, mean_abs, max_abs in sorted(changed, key=lambda item: item[1], reverse=True):
            print(f"  {feature:32s} mean_abs={mean_abs:.6f} max_abs={max_abs:.6f}")

    print(f"\nSaved predictions: {OUTPUT_PATH}")
    print("NOTE: This is a controlled source-substitution sensitivity experiment, not model retraining.")
    print("NOTE: It preserves the deployed model artifact and does not modify production files.")


if __name__ == "__main__":
    main()
