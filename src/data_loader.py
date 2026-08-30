from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    AUDIT_FILE,
    DATA_DIR,
    DATETIME_COLUMN,
    DATE_COLUMN,
    DICTIONARY_FILE,
    SENTINEL_VALUE,
    TARGET,
    TEST_FILE,
    TEST_LABELS_FILE,
    TIME_COLUMN,
    TRAIN_FILE,
)

DATASET_REPOSITORY = "https://github.com/viciouss28/HACK_ML_DATASET.git"


def ensure_dataset(data_dir: Path = DATA_DIR, allow_download: bool = True) -> None:
    """Ensure all four organizer files exist, optionally cloning their repository."""
    required = ["train.csv", "test.csv", "test_labels.csv", "data_dictionary.csv"]
    if all((data_dir / name).exists() for name in required):
        return
    if not allow_download:
        missing = [name for name in required if not (data_dir / name).exists()]
        raise FileNotFoundError(f"Missing dataset files: {', '.join(missing)}")
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="Air-ritated_data_") as tmp:
        repo_dir = Path(tmp) / "repo"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", DATASET_REPOSITORY, str(repo_dir)],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            raise RuntimeError(
                "Dataset is missing and automatic download failed. Copy the four CSV files "
                "from 01_AirQuality into the project's data/ directory."
            ) from exc
        source = repo_dir / "01_AirQuality"
        for name in required:
            shutil.copy2(source / name, data_dir / name)


def load_raw_data(allow_download: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dataset(allow_download=allow_download)
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    labels = pd.read_csv(TEST_LABELS_FILE)
    dictionary = pd.read_csv(DICTIONARY_FILE)
    validate_schema(train, test, labels)
    return train, test, labels, dictionary


def validate_schema(train: pd.DataFrame, test: pd.DataFrame, labels: pd.DataFrame) -> None:
    required_time = {DATE_COLUMN, TIME_COLUMN}
    if not required_time.issubset(train.columns) or not required_time.issubset(test.columns):
        raise ValueError("train.csv and test.csv must contain Date and Time columns.")
    if TARGET not in train.columns or TARGET in test.columns:
        raise ValueError("Expected CO(GT) in train.csv only.")
    if set(train.columns) - {TARGET} != set(test.columns):
        raise ValueError("Train/test feature schemas do not match.")
    if TARGET not in labels.columns or DATETIME_COLUMN not in labels.columns:
        raise ValueError("test_labels.csv must contain DateTime and CO(GT).")
    if len(test) != len(labels):
        raise ValueError("test.csv and test_labels.csv row counts do not match.")


def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result[DATETIME_COLUMN] = pd.to_datetime(
        result[DATE_COLUMN].astype(str) + " " + result[TIME_COLUMN].astype(str), errors="coerce"
    )
    return result


def _safe_number(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def build_data_audit(train: pd.DataFrame, test: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    train_dt = add_datetime(train)
    test_dt = add_datetime(test)
    numeric_features = [c for c in test.columns if c not in {DATE_COLUMN, TIME_COLUMN}]
    valid_target = train.loc[train[TARGET] != SENTINEL_VALUE, TARGET]
    sentinel_train = {c: int((pd.to_numeric(train[c], errors="coerce") == SENTINEL_VALUE).sum()) for c in train.select_dtypes("number")}
    sentinel_test = {c: int((pd.to_numeric(test[c], errors="coerce") == SENTINEL_VALUE).sum()) for c in test.select_dtypes("number")}
    correlations = (
        train.replace(SENTINEL_VALUE, np.nan)[numeric_features + [TARGET]].corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .abs()
        .sort_values(ascending=False)
    )
    shift_rows = []
    clean_train = train.replace(SENTINEL_VALUE, np.nan)
    clean_test = test.replace(SENTINEL_VALUE, np.nan)
    for col in numeric_features:
        tr_mean, te_mean = clean_train[col].mean(), clean_test[col].mean()
        tr_std = clean_train[col].std()
        smd = abs(te_mean - tr_mean) / tr_std if pd.notna(tr_std) and tr_std > 0 else np.nan
        shift_rows.append({"feature": col, "standardized_mean_difference": _safe_number(smd)})
    shift_rows.sort(key=lambda row: row["standardized_mean_difference"] or -1, reverse=True)
    aligned = pd.to_datetime(labels[DATETIME_COLUMN], errors="coerce").reset_index(drop=True).equals(
        test_dt[DATETIME_COLUMN].reset_index(drop=True)
    )
    audit: dict[str, Any] = {
        "shapes": {"train": list(train.shape), "test": list(test.shape), "test_labels": list(labels.shape)},
        "columns": {"train": train.columns.tolist(), "test": test.columns.tolist(), "test_labels": labels.columns.tolist()},
        "dtypes": {"train": train.dtypes.astype(str).to_dict(), "test": test.dtypes.astype(str).to_dict()},
        "target": TARGET,
        "schema_differences": {"train_only": [TARGET], "test_only": []},
        "date_ranges": {
            "train": [str(train_dt[DATETIME_COLUMN].min()), str(train_dt[DATETIME_COLUMN].max())],
            "test": [str(test_dt[DATETIME_COLUMN].min()), str(test_dt[DATETIME_COLUMN].max())],
        },
        "chronology": {
            "train_ordered": bool(train_dt[DATETIME_COLUMN].is_monotonic_increasing),
            "test_ordered": bool(test_dt[DATETIME_COLUMN].is_monotonic_increasing),
            "strictly_separated": bool(train_dt[DATETIME_COLUMN].max() < test_dt[DATETIME_COLUMN].min()),
            "hourly_median_gap_train": str(train_dt[DATETIME_COLUMN].sort_values().diff().median()),
            "test_label_datetime_alignment": aligned,
        },
        "ordinary_nulls": {"train": train.isna().sum().astype(int).to_dict(), "test": test.isna().sum().astype(int).to_dict()},
        "sentinel_minus_200": {"train": sentinel_train, "test": sentinel_test, "test_labels_target": int((labels[TARGET] == SENTINEL_VALUE).sum())},
        "duplicate_rows": {"train": int(train.duplicated().sum()), "test": int(test.duplicated().sum()), "test_labels": int(labels.duplicated().sum())},
        "target_distribution_valid_only": {k: _safe_number(v) for k, v in valid_target.describe().to_dict().items()},
        "top_absolute_target_correlations": {k: _safe_number(v) for k, v in correlations.head(10).to_dict().items()},
        "largest_train_test_shifts": shift_rows[:10],
        "leakage_notes": [
            "test_labels.csv is reserved for one final holdout evaluation and is never used for fitting or selection.",
            "All learned imputation statistics are fitted inside each training fold or on the final training set.",
            "No target-derived lag or rolling features are used.",
            "Sensor readings are contemporaneous inputs, so this product is best described as CO estimation/nowcasting, not a future-horizon forecast.",
        ],
    }
    return audit


def write_data_audit(train: pd.DataFrame, test: pd.DataFrame, labels: pd.DataFrame, path: Path = AUDIT_FILE) -> dict[str, Any]:
    audit = build_data_audit(train, test, labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit
