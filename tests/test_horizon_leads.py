import unittest

import numpy as np
import pandas as pd

from scripts.evaluate_true_lead_horizons import (
    BASE_PREDICTORS,
    build_horizon_dataset,
)


class HorizonLeadTests(unittest.TestCase):
    def make_source(self):
        dates = pd.date_range("2000-01-01", periods=30, freq="MS")
        data = {"month_date": dates, "swba_3month": np.arange(30, dtype=float)}
        for column_index, column in enumerate(BASE_PREDICTORS):
            data[column] = np.arange(30, dtype=float) + column_index * 100
        return pd.DataFrame(data)

    def test_lead_one_two_three_align_target_and_origin(self):
        source = self.make_source()
        for lead in (1, 2, 3):
            with self.subTest(lead=lead):
                frame, features = build_horizon_dataset(source, lead)
                first = frame.iloc[0]
                source_origin_index = 12  # maximum required lag
                self.assertEqual(
                    first["forecast_origin_month"],
                    source.loc[source_origin_index, "month_date"],
                )
                self.assertEqual(
                    first["target_month"],
                    source.loc[source_origin_index + lead, "month_date"],
                )
                self.assertEqual(
                    first["target_swba_3month"],
                    source.loc[source_origin_index + lead, "swba_3month"],
                )
                self.assertEqual(
                    first["precipitation_mm_lag_0"],
                    source.loc[source_origin_index, "precipitation_mm"],
                )
                self.assertEqual(len(features), len(BASE_PREDICTORS) * 5 + 2)

    def test_rejects_zero_or_negative_lead(self):
        with self.assertRaises(ValueError):
            build_horizon_dataset(self.make_source(), 0)

    def test_rejects_duplicate_months(self):
        source = self.make_source()
        source.loc[1, "month_date"] = source.loc[0, "month_date"]
        with self.assertRaises(ValueError):
            build_horizon_dataset(source, 1)


if __name__ == "__main__":
    unittest.main()
