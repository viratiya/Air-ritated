from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import FINAL_MODEL_FILE, METADATA_FILE
from .features import engineer_features


def load_model_bundle(model_path: Path = FINAL_MODEL_FILE, metadata_path: Path = METADATA_FILE):
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Trained artifacts are missing. Run: python -m src.train")
    return joblib.load(model_path), joblib.load(metadata_path)


def predict_dataframe(df: pd.DataFrame) -> pd.Series:
    model, metadata = load_model_bundle()
    features = engineer_features(df, feature_columns=metadata["feature_columns"])
    return pd.Series(model.predict(features), index=df.index, name="Predicted_CO(GT)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Air-ritated CO predictions for a CSV.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/custom_predictions.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input_csv)
    result = frame.copy()
    result["Predicted_CO(GT)"] = predict_dataframe(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
