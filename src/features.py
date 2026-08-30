from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATETIME_COLUMN, DATE_COLUMN, SENTINEL_VALUE, TIME_COLUMN

CALENDAR_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]


def engineer_features(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Create leakage-safe contemporaneous and cyclical calendar features."""
    result = df.copy()
    dt = pd.to_datetime(
        result[DATE_COLUMN].astype(str) + " " + result[TIME_COLUMN].astype(str), errors="coerce"
    )
    result["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
    result["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
    result["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)
    result["month_sin"] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12.0)
    result["month_cos"] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12.0)
    result["is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
    result = result.drop(columns=[DATE_COLUMN, TIME_COLUMN, DATETIME_COLUMN], errors="ignore")
    result = result.apply(pd.to_numeric, errors="coerce")
    result = result.replace(SENTINEL_VALUE, np.nan)
    if feature_columns is not None:
        for column in feature_columns:
            if column not in result.columns:
                result[column] = np.nan
        result = result.reindex(columns=feature_columns)
    return result


def select_feature_columns(train_features: pd.DataFrame, missingness_threshold: float = 0.80) -> tuple[list[str], list[str]]:
    missing_fraction = train_features.isna().mean()
    dropped = missing_fraction[missing_fraction >= missingness_threshold].index.tolist()
    kept = [c for c in train_features.columns if c not in dropped]
    return kept, dropped
