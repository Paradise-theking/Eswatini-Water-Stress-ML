from pathlib import Path
import pandas as pd
import numpy as np

base = Path("outputs/expanded")

file_3 = (
    base / "drought_onset_robustness"
    / "matched_drought_onset_statistical_summary.csv"
)

file_6 = (
    base / "drought_onset_robustness_6month"
    / "matched_drought_onset_statistical_summary.csv"
)

out_dir = base / "drought_onset_robustness_comparison"
out_dir.mkdir(parents=True, exist_ok=True)

a = pd.read_csv(file_3)
b = pd.read_csv(file_6)

a = a.add_suffix("_3month")
b = b.add_suffix("_6month")

merged = a.merge(
    b,
    left_on=["lead_months_3month", "feature_3month"],
    right_on=["lead_months_6month", "feature_6month"],
    how="inner",
)

merged["significant_3month"] = (
    merged["bh_adjusted_p_value_3month"] < 0.05
)

merged["significant_6month"] = (
    merged["bh_adjusted_p_value_6month"] < 0.05
)

merged["ci_excludes_zero_3month"] = (
    (merged["bootstrap_95ci_lower_3month"] > 0)
    | (merged["bootstrap_95ci_upper_3month"] < 0)
)

merged["ci_excludes_zero_6month"] = (
    (merged["bootstrap_95ci_lower_6month"] > 0)
    | (merged["bootstrap_95ci_upper_6month"] < 0)
)

merged["same_effect_direction"] = (
    np.sign(merged["mean_paired_difference_3month"])
    == np.sign(merged["mean_paired_difference_6month"])
)

merged["stable_strong_signal"] = (
    merged["significant_3month"]
    & merged["significant_6month"]
    & merged["ci_excludes_zero_3month"]
    & merged["ci_excludes_zero_6month"]
    & merged["same_effect_direction"]
)

merged["stable_statistical_signal"] = (
    merged["significant_3month"]
    & merged["significant_6month"]
    & merged["same_effect_direction"]
)

merged["absolute_effect_3month"] = (
    merged["paired_effect_size_3month"].abs()
)

merged["absolute_effect_6month"] = (
    merged["paired_effect_size_6month"].abs()
)

merged = merged.sort_values(
    ["lead_months_3month", "stable_strong_signal",
     "absolute_effect_3month"],
    ascending=[True, False, False],
)

output = out_dir / "drought_onset_signal_stability.csv"
merged.to_csv(output, index=False)

print("=" * 75)
print("DROUGHT-ONSET SIGNAL STABILITY")
print("=" * 75)

for lead in [1, 2, 3]:

    subset = merged[
        merged["lead_months_3month"] == lead
    ]

    strong = subset[subset["stable_strong_signal"]]

    statistical = subset[
        subset["stable_statistical_signal"]
    ]

    print()
    print(f"{lead}-MONTH LEAD")
    print("-" * 75)
    print(f"Features compared: {len(subset)}")
    print(f"Stable statistical signals: {len(statistical)}")
    print(f"Stable strong signals: {len(strong)}")

    if not strong.empty:

        columns = [
            "feature_3month",
            "mean_paired_difference_3month",
            "mean_paired_difference_6month",
            "paired_effect_size_3month",
            "paired_effect_size_6month",
        ]

        print()
        print(
            strong[columns].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

print()
print("Saved:", output)
