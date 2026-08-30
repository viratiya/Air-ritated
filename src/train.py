from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .config import (
    COMPARISON_FILE,
    FINAL_MODEL_FILE,
    HIGH_MISSINGNESS_THRESHOLD,
    IMPORTANCE_FILE,
    METADATA_FILE,
    METRICS_FILE,
    PREDICTIONS_FILE,
    PREPROCESSOR_FILE,
    RANDOM_STATE,
    SENTINEL_VALUE,
    TARGET,
    VALIDATION_FRACTION,
    ensure_output_directories,
)
from .data_loader import add_datetime, load_raw_data, write_data_audit
from .evaluate import regression_metrics
from .features import CALENDAR_FEATURES, engineer_features, select_feature_columns
from .preprocessing import make_preprocessor

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class Candidate:
    name: str
    pipeline: Pipeline
    parameters: dict[str, list] | None = None
    search_iterations: int = 0


def _pipeline(model, scale: bool = False) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor(scale=scale)), ("model", model)])


def build_candidates(quick: bool = False) -> list[Candidate]:
    tree_count = 180 if quick else 400
    iterations = 180 if quick else 450
    candidates = [
        Candidate("Dummy Mean", _pipeline(DummyRegressor(strategy="mean"))),
        Candidate("Linear Regression", _pipeline(LinearRegression(), scale=True)),
        Candidate(
            "Random Forest",
            _pipeline(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            {
                "model__n_estimators": [tree_count, tree_count + 150],
                "model__max_features": [0.65, 0.85, 1.0],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_depth": [None, 18, 28],
            },
            2 if quick else 4,
        ),
        Candidate(
            "Extra Trees",
            _pipeline(ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            {
                "model__n_estimators": [tree_count, tree_count + 150],
                "model__max_features": [0.65, 0.85, 1.0],
                "model__min_samples_leaf": [1, 2, 3],
                "model__max_depth": [None, 20, 30],
            },
            2 if quick else 5,
        ),
        Candidate(
            "HistGradientBoosting",
            _pipeline(HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            {
                "model__learning_rate": [0.035, 0.06, 0.09],
                "model__max_iter": [iterations, iterations + 150],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__l2_regularization": [0.0, 0.2, 1.0],
                "model__min_samples_leaf": [12, 20, 30],
            },
            2 if quick else 5,
        ),
    ]
    try:
        from xgboost import XGBRegressor

        candidates.append(
            Candidate(
                "XGBoost",
                _pipeline(
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=0,
                    )
                ),
                {
                    "model__n_estimators": [iterations, iterations + 200],
                    "model__max_depth": [3, 5, 7],
                    "model__learning_rate": [0.025, 0.05, 0.08],
                    "model__subsample": [0.75, 0.9, 1.0],
                    "model__colsample_bytree": [0.7, 0.9, 1.0],
                    "model__reg_lambda": [1.0, 3.0, 8.0],
                },
                2 if quick else 5,
            )
        )
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor

        candidates.append(
            Candidate(
                "CatBoost",
                _pipeline(
                    CatBoostRegressor(
                        loss_function="RMSE",
                        random_seed=RANDOM_STATE,
                        verbose=False,
                        allow_writing_files=False,
                        thread_count=-1,
                    )
                ),
                {
                    "model__iterations": [iterations, iterations + 200],
                    "model__depth": [5, 7, 9],
                    "model__learning_rate": [0.025, 0.05, 0.08],
                    "model__l2_leaf_reg": [2.0, 5.0, 9.0],
                },
                2 if quick else 4,
            )
        )
    except ImportError:
        pass
    return candidates


def _fit_candidate(candidate: Candidate, X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
    if not candidate.parameters or candidate.search_iterations <= 0:
        model = clone(candidate.pipeline).fit(X, y)
        return model, {}
    splitter = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        clone(candidate.pipeline),
        candidate.parameters,
        n_iter=candidate.search_iterations,
        scoring="neg_mean_absolute_error",
        cv=splitter,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def _chronological_split(X: pd.DataFrame, y: pd.Series, datetimes: pd.Series):
    cut = int(len(X) * (1.0 - VALIDATION_FRACTION))
    return (
        X.iloc[:cut].copy(),
        X.iloc[cut:].copy(),
        y.iloc[:cut].copy(),
        y.iloc[cut:].copy(),
        datetimes.iloc[:cut].copy(),
        datetimes.iloc[cut:].copy(),
    )


def _choose_calendar_features(
    X_dev: pd.DataFrame,
    X_val: pd.DataFrame,
    y_dev: pd.Series,
    y_val: pd.Series,
    quick: bool,
) -> tuple[list[str], pd.DataFrame]:
    all_columns = X_dev.columns.tolist()
    base_columns = [c for c in all_columns if c not in CALENDAR_FEATURES]
    probe = _pipeline(
        ExtraTreesRegressor(
            n_estimators=160 if quick else 320,
            max_features=0.85,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    rows = []
    for name, columns in [("Sensors + calendar", all_columns), ("Sensors only", base_columns)]:
        start = time.perf_counter()
        fitted = clone(probe).fit(X_dev[columns], y_dev)
        metrics = regression_metrics(y_val, fitted.predict(X_val[columns]))
        rows.append({"Feature Set": name, **metrics, "Training Time (s)": time.perf_counter() - start})
    experiment = pd.DataFrame(rows).sort_values("MAE", ignore_index=True)
    selected = all_columns if experiment.iloc[0]["Feature Set"] == "Sensors + calendar" else base_columns
    return selected, experiment


def _feature_importance(model, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring="neg_mean_absolute_error",
        n_repeats=6,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_val.columns,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    return importance


def run_training(quick: bool = False) -> dict:
    ensure_output_directories()
    train, test, labels, _dictionary = load_raw_data()

    # The holdout labels remain sealed until model and feature selection are complete.
    train_dt = add_datetime(train).sort_values("DateTime", kind="stable").reset_index(drop=True)
    valid_train = train_dt.loc[train_dt[TARGET] != SENTINEL_VALUE].reset_index(drop=True)
    y = valid_train[TARGET].astype(float)
    X_all = engineer_features(valid_train.drop(columns=[TARGET]))
    datetimes = valid_train["DateTime"]
    X_dev_raw, X_val_raw, y_dev, y_val, dt_dev, dt_val = _chronological_split(X_all, y, datetimes)

    kept, dropped = select_feature_columns(X_dev_raw, HIGH_MISSINGNESS_THRESHOLD)
    X_dev_raw, X_val_raw = X_dev_raw[kept], X_val_raw[kept]
    feature_columns, feature_experiment = _choose_calendar_features(X_dev_raw, X_val_raw, y_dev, y_val, quick)
    feature_experiment.to_csv(Path(COMPARISON_FILE).with_name("feature_set_experiment.csv"), index=False)
    X_dev, X_val = X_dev_raw[feature_columns], X_val_raw[feature_columns]

    comparison_rows: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    best_parameters: dict[str, dict] = {}

    for candidate in build_candidates(quick=quick):
        start = time.perf_counter()
        try:
            fitted, params = _fit_candidate(candidate, X_dev, y_dev)
            prediction = np.asarray(fitted.predict(X_val), dtype=float)
            metrics = regression_metrics(y_val, prediction)
            elapsed = time.perf_counter() - start
            comparison_rows.append(
                {"Model": candidate.name, **metrics, "Training Time (s)": elapsed, "Status": "completed"}
            )
            fitted_models[candidate.name] = fitted
            validation_predictions[candidate.name] = prediction
            best_parameters[candidate.name] = params
            print(f"{candidate.name:22s} MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f}", flush=True)
        except Exception as exc:
            comparison_rows.append(
                {
                    "Model": candidate.name,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "Training Time (s)": time.perf_counter() - start,
                    "Status": f"skipped: {type(exc).__name__}",
                }
            )
            print(f"{candidate.name:22s} skipped ({type(exc).__name__}: {exc})", flush=True)

    completed = pd.DataFrame(comparison_rows).dropna(subset=["MAE"]).sort_values("MAE", ignore_index=True)
    if completed.empty:
        raise RuntimeError("No candidate model completed successfully.")

    # Test a simple top-three average. Keep it only if it genuinely improves validation MAE.
    top_names = completed["Model"].head(min(3, len(completed))).tolist()
    if len(top_names) >= 2:
        ensemble_pred = np.mean([validation_predictions[name] for name in top_names], axis=0)
        ensemble_metrics = regression_metrics(y_val, ensemble_pred)
        comparison_rows.append(
            {
                "Model": "Top-3 Voting Ensemble",
                **ensemble_metrics,
                "Training Time (s)": sum(
                    row["Training Time (s)"] for row in comparison_rows if row["Model"] in top_names
                ),
                "Status": "completed",
            }
        )
        if ensemble_metrics["MAE"] < completed.iloc[0]["MAE"]:
            validation_model = VotingRegressor(
                estimators=[(f"model_{i}", clone(fitted_models[name])) for i, name in enumerate(top_names)]
            ).fit(X_dev, y_dev)
            winning_name = "Top-3 Voting Ensemble"
            winning_validation = ensemble_metrics
            final_template = VotingRegressor(
                estimators=[(f"model_{i}", clone(fitted_models[name])) for i, name in enumerate(top_names)]
            )
        else:
            winning_name = str(completed.iloc[0]["Model"])
            validation_model = fitted_models[winning_name]
            winning_validation = {key: float(completed.iloc[0][key]) for key in ["MAE", "RMSE", "R2"]}
            final_template = clone(validation_model)
    else:
        winning_name = str(completed.iloc[0]["Model"])
        validation_model = fitted_models[winning_name]
        winning_validation = {key: float(completed.iloc[0][key]) for key in ["MAE", "RMSE", "R2"]}
        final_template = clone(validation_model)

    comparison = pd.DataFrame(comparison_rows).sort_values("MAE", na_position="last", ignore_index=True)
    comparison.to_csv(COMPARISON_FILE, index=False)
    importance = _feature_importance(validation_model, X_val, y_val)
    importance.to_csv(IMPORTANCE_FILE, index=False)

    # Lock the complete design, then fit once on every valid training observation.
    X_full = X_all.reindex(columns=feature_columns)
    final_model = clone(final_template).fit(X_full, y)
    joblib.dump(final_model, FINAL_MODEL_FILE)
    standalone_preprocessor = make_preprocessor(scale=False).fit(X_full)
    joblib.dump(standalone_preprocessor, PREPROCESSOR_FILE)

    clean_full = X_full.replace(SENTINEL_VALUE, np.nan)
    metadata = {
        "project": "Air-ritated",
        "winning_model": winning_name,
        "feature_columns": feature_columns,
        "dropped_high_missingness_features": dropped,
        "feature_medians": clean_full.median(numeric_only=True).to_dict(),
        "feature_p01": clean_full.quantile(0.01, numeric_only=True).to_dict(),
        "feature_p99": clean_full.quantile(0.99, numeric_only=True).to_dict(),
        "train_rows_raw": int(len(train)),
        "train_rows_valid_target": int(len(valid_train)),
        "development_rows": int(len(X_dev)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(test)),
        "validation_start": str(dt_val.min()),
        "validation_end": str(dt_val.max()),
        "training_start": str(datetimes.min()),
        "training_end": str(datetimes.max()),
        "validation_strategy": "Last 20% of valid-target training observations; expanding-window TimeSeriesSplit inside development data for tuning.",
        "best_parameters": best_parameters,
        "calendar_features_selected": any(c in feature_columns for c in CALENDAR_FEATURES),
        "sentinel_value": SENTINEL_VALUE,
        "co_unit": "mg/m³",
    }
    joblib.dump(metadata, METADATA_FILE)

    # Only now unseal test_labels.csv for the one final holdout evaluation.
    test_features = engineer_features(test, feature_columns=feature_columns)
    test_predictions = np.asarray(final_model.predict(test_features), dtype=float)
    label_values = pd.to_numeric(labels[TARGET], errors="coerce")
    valid_holdout = label_values.notna() & (label_values != SENTINEL_VALUE)
    holdout_metrics = regression_metrics(label_values.loc[valid_holdout], test_predictions[valid_holdout.to_numpy()])

    output_predictions = pd.DataFrame(
        {
            "DateTime": labels["DateTime"],
            "Predicted_CO(GT)": test_predictions,
            "Actual_CO(GT)": label_values,
            "Valid_Holdout_Label": valid_holdout,
        }
    )
    output_predictions.to_csv(PREDICTIONS_FILE, index=False)

    metrics_payload = {
        "winning_model": winning_name,
        "validation": winning_validation,
        "holdout": holdout_metrics,
        "evaluation_rows": {"validation": int(len(y_val)), "holdout_valid": int(valid_holdout.sum()), "holdout_total": int(len(test))},
        "feature_set": "Sensors + calendar" if metadata["calendar_features_selected"] else "Sensors only",
    }
    Path(METRICS_FILE).write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    write_data_audit(train, test, labels)

    print(f"\nWinner: {winning_name}", flush=True)
    print(f"Validation: {winning_validation}", flush=True)
    print(f"Final holdout: {holdout_metrics}", flush=True)
    return metrics_payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate Air-ritated.")
    parser.add_argument("--quick", action="store_true", help="Use fewer search iterations for a fast smoke run.")
    args = parser.parse_args()
    run_training(quick=args.quick)


if __name__ == "__main__":
    main()
