from pathlib import Path
import pandas as pd

BASE = Path("outputs/expanded")

INPUT = (
    BASE / "drought_onset_robustness"
    / "matched_drought_onset_event_data.csv"
)

OUTPUT = (
    BASE / "drought_onset_robustness_comparison"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "anomaly_precipitation_mm",
    "anomaly_pet_mm",
    "anomaly_volumetric_soil_water_layer_1",
    "anomaly_water_balance_3month",
    "anomaly_temperature_max_c",
]

df = pd.read_csv(INPUT)

for lead in [1, 2, 3]:

    subset = df[
        (df["lead_months"] == lead)
        & (df["feature"].isin(FEATURES))
    ].copy()

    case_matrix = subset.pivot(
        index="onset_month",
        columns="feature",
        values="case_value",
    )

    control_matrix = subset.pivot(
        index="onset_month",
        columns="feature",
        values="matched_control_mean",
    )

    case_corr = case_matrix.corr(method="spearman")
    control_corr = control_matrix.corr(method="spearman")

    case_corr.to_csv(
        OUTPUT / f"onset_predictor_correlations_{lead}m.csv"
    )

    control_corr.to_csv(
        OUTPUT / f"control_predictor_correlations_{lead}m.csv"
    )

    print()
    print("=" * 75)
    print(f"{lead}-MONTH LEAD: DROUGHT-ONSET PREDICTOR CORRELATIONS")
    print("=" * 75)

    print("\nDrought-onset observations:")
    print(case_corr.round(3).to_string())

    print("\nMatched-control averages:")
    print(control_corr.round(3).to_string())

print()
print("Correlation matrices saved in:")
print(OUTPUT)
