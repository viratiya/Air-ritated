from __future__ import annotations

import unittest

import joblib
import numpy as np

from app import valid_value
from src.config import FINAL_MODEL_FILE, METADATA_FILE, SENTINEL_VALUE, TARGET
from src.data_loader import load_raw_data
from src.features import engineer_features
from src.predict import predict_dataframe


class Air-ritatedSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test, cls.labels, cls.dictionary = load_raw_data(allow_download=False)
        cls.model = joblib.load(FINAL_MODEL_FILE)
        cls.metadata = joblib.load(METADATA_FILE)

    def test_schema(self):
        self.assertIn(TARGET, self.train.columns)
        self.assertNotIn(TARGET, self.test.columns)
        self.assertEqual(len(self.test), len(self.labels))

    def test_sentinel_becomes_nan(self):
        row = self.test.iloc[[0]].copy()
        row.loc[row.index[0], "PT08.S1(CO)"] = SENTINEL_VALUE
        features = engineer_features(row, self.metadata["feature_columns"])
        self.assertTrue(np.isnan(features.iloc[0]["PT08.S1(CO)"]))

    def test_valid_value_rejects_datetime_strings(self):
        self.assertFalse(valid_value("2005-03-23 21:00:00"))
        self.assertTrue(valid_value(12.5))

    def test_feature_order(self):
        features = engineer_features(self.test.iloc[:3], self.metadata["feature_columns"])
        self.assertEqual(features.columns.tolist(), self.metadata["feature_columns"])

    def test_model_prediction(self):
        result = predict_dataframe(self.test.iloc[:5])
        self.assertEqual(len(result), 5)
        self.assertTrue(np.isfinite(result).all())

    def test_scenario_prediction(self):
        features = engineer_features(self.test.iloc[[10]], self.metadata["feature_columns"])
        features.loc[features.index[0], "T"] = self.metadata["feature_medians"]["T"]
        prediction = self.model.predict(features)
        self.assertTrue(np.isfinite(prediction[0]))


if __name__ == "__main__":
    unittest.main()
