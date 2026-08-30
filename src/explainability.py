from __future__ import annotations

import numpy as np
import pandas as pd


def load_global_importance(path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["feature", "importance", "importance_std"])


def local_perturbation_contributions(
    model,
    row: pd.DataFrame,
    medians: dict[str, float],
    top_n: int = 8,
) -> tuple[float, pd.DataFrame]:
    """Model-behavior explanation using one-feature-at-a-time median perturbations.

    This is not a causal explanation. It measures how the model output changes when
    one input is replaced by its training median while all other inputs stay fixed.
    """
    base = float(model.predict(row)[0])
    rows = []
    for feature in row.columns:
        changed = row.copy()
        changed.loc[changed.index[0], feature] = medians.get(feature, np.nan)
        counterfactual = float(model.predict(changed)[0])
        rows.append(
            {
                "feature": feature,
                "contribution": base - counterfactual,
                "selected_value": row.iloc[0][feature],
                "reference_median": medians.get(feature, np.nan),
            }
        )
    result = pd.DataFrame(rows)
    result["absolute_contribution"] = result["contribution"].abs()
    return base, result.nlargest(top_n, "absolute_contribution").reset_index(drop=True)


def plain_english_explanation(contributions: pd.DataFrame) -> str:
    if contributions.empty:
        return "A local explanation is unavailable for this observation."
    strongest = contributions.iloc[0]
    direction = "above" if strongest["contribution"] > 0 else "below"
    return (
        f"{strongest['feature']} had the largest local association: relative to replacing it "
        f"with its training median, it moved this model estimate {direction} the reference estimate. "
        "This describes model behavior, not a causal effect on air quality."
    )
