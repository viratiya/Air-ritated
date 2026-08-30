from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
TEST_LABELS_FILE = DATA_DIR / "test_labels.csv"
DICTIONARY_FILE = DATA_DIR / "data_dictionary.csv"

TARGET = "CO(GT)"
DATE_COLUMN = "Date"
TIME_COLUMN = "Time"
DATETIME_COLUMN = "DateTime"
SENTINEL_VALUE = -200.0
RANDOM_STATE = 42
VALIDATION_FRACTION = 0.20
HIGH_MISSINGNESS_THRESHOLD = 0.80

FINAL_MODEL_FILE = MODEL_DIR / "final_model.joblib"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessing_pipeline.joblib"
METADATA_FILE = MODEL_DIR / "model_metadata.joblib"
METRICS_FILE = OUTPUT_DIR / "metrics.json"
COMPARISON_FILE = OUTPUT_DIR / "model_comparison.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "test_predictions.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "feature_importance.csv"
AUDIT_FILE = OUTPUT_DIR / "data_audit.json"


def ensure_output_directories() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
