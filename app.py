from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.charts import COLORS, polish
from dashboard.theme import apply_theme, callout, hero, kpi
from src.config import (
    AUDIT_FILE,
    COMPARISON_FILE,
    FINAL_MODEL_FILE,
    IMPORTANCE_FILE,
    METADATA_FILE,
    METRICS_FILE,
    PREDICTIONS_FILE,
    SENTINEL_VALUE,
    TARGET,
)
from src.data_loader import add_datetime, load_raw_data
from src.explainability import local_perturbation_contributions, plain_english_explanation
from src.features import CALENDAR_FEATURES, engineer_features

st.set_page_config(page_title="Air-ritated", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")
apply_theme()


@st.cache_resource
def load_bundle():
    if not FINAL_MODEL_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError("Trained model artifacts were not found. Run `python -m src.train` first.")
    return joblib.load(FINAL_MODEL_FILE), joblib.load(METADATA_FILE)


@st.cache_data
def load_dashboard_data():
    train, test, labels, dictionary = load_raw_data(allow_download=False)
    predictions = pd.read_csv(PREDICTIONS_FILE)
    comparison = pd.read_csv(COMPARISON_FILE)
    importance = pd.read_csv(IMPORTANCE_FILE)
    metrics = json.loads(Path(METRICS_FILE).read_text(encoding="utf-8"))
    audit = json.loads(Path(AUDIT_FILE).read_text(encoding="utf-8"))
    return train, test, labels, dictionary, predictions, comparison, importance, metrics, audit


def valid_value(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric != SENTINEL_VALUE


def fmt(value, decimals: int = 2, suffix: str = "") -> str:
    return "Unavailable" if not valid_value(value) else f"{float(value):,.{decimals}f}{suffix}"


def completeness_for_row(row: pd.Series, raw_features: list[str]) -> tuple[float, int, int]:
    valid = sum(valid_value(row.get(col, np.nan)) for col in raw_features)
    return 100.0 * valid / len(raw_features), valid, len(raw_features)


def relative_band(value: float, target: pd.Series) -> str:
    q33, q67 = target.quantile([0.33, 0.67])
    if value < q33:
        return "Lower observed range"
    if value < q67:
        return "Typical observed range"
    return "Elevated relative to dataset"


try:
    model, metadata = load_bundle()
    train, test, labels, dictionary, predictions, comparison, importance, metrics, audit = load_dashboard_data()
except Exception as exc:
    hero()
    st.error(f"Air-ritated could not start: {exc}")
    st.info("From the project directory, run `python -m src.train`, then `streamlit run app.py`.")
    st.stop()

hero()

valid_train = train.loc[train[TARGET] != SENTINEL_VALUE].copy()
valid_target = valid_train[TARGET]
test_dt = add_datetime(test)
raw_features = [c for c in test.columns if c not in {"Date", "Time", "NMHC(GT)"}]
selectable_times = test_dt["DateTime"].dt.strftime("%d %b %Y · %H:%M")

with st.sidebar:
    st.markdown("### Observation navigator")
    selected_index = st.slider("Hourly test observation", 0, len(test) - 1, min(420, len(test) - 1))
    st.caption(selectable_times.iloc[selected_index])
    st.markdown("---")
    st.markdown("### Model status")
    st.success(f"Active model: {metadata['winning_model']}")
    st.caption(f"{metadata['train_rows_valid_target']:,} valid training observations")
    st.caption("Holdout labels remained sealed until final model selection.")
    st.markdown("---")
    st.caption("CO is shown in mg/m³. Relative range labels come from this dataset—not medical or regulatory thresholds.")

tabs = st.tabs(
    [
        "Command Center",
        "Explainable AI",
        "Pollution Patterns",
        "Scenario Lab",
        "Data Quality",
        "Model Lab",
    ]
)

selected_raw = test.iloc[selected_index]
selected_features = engineer_features(test.iloc[[selected_index]], metadata["feature_columns"])
selected_prediction = float(model.predict(selected_features)[0])
actual_value = predictions.iloc[selected_index]["Actual_CO(GT)"]
quality_score, valid_count, feature_count = completeness_for_row(selected_raw, raw_features)

with tabs[0]:
    st.subheader("Command Center")
    st.caption("A selected hourly observation from the organizer's chronological holdout window.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("CO estimate", f"{selected_prediction:.2f} mg/m³", relative_band(selected_prediction, valid_target))
    with c2:
        kpi("Actual CO", fmt(actual_value, 2, " mg/m³"), "Shown for final evaluation only")
    with c3:
        kpi("Input completeness", f"{quality_score:.0f}%", f"{valid_count} of {feature_count} required sensor/environment inputs valid")
    with c4:
        absolute_error = abs(selected_prediction - actual_value) if valid_value(actual_value) else np.nan
        kpi("Absolute error", fmt(absolute_error, 2, " mg/m³"), "Unavailable where reference label is -200")

    st.markdown("#### Environmental snapshot")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Temperature", fmt(selected_raw["T"], 1, " °C"))
    s2.metric("Relative humidity", fmt(selected_raw["RH"], 1, "%"))
    s3.metric("NOx", fmt(selected_raw["NOx(GT)"], 1, " ppb"))
    s4.metric("NO₂", fmt(selected_raw["NO2(GT)"], 1, " μg/m³"))
    s5.metric("CO sensor response", fmt(selected_raw["PT08.S1(CO)"], 1))

    left = max(0, selected_index - 72)
    right = min(len(test), selected_index + 73)
    window = predictions.iloc[left:right].copy()
    window["DateTime"] = pd.to_datetime(window["DateTime"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["DateTime"], y=window["Predicted_CO(GT)"], name="Predicted", line=dict(color=COLORS["teal"], width=2.5)))
    valid_window = window[window["Valid_Holdout_Label"]]
    fig.add_trace(go.Scatter(x=valid_window["DateTime"], y=valid_window["Actual_CO(GT)"], name="Actual", line=dict(color=COLORS["amber"], width=1.5), opacity=.78))
    fig.add_vline(x=pd.Timestamp(test_dt.iloc[selected_index]["DateTime"]).timestamp() * 1000, line_dash="dot", line_color=COLORS["mint"])
    fig.update_layout(title="CO estimate around selected observation", xaxis_title=None, yaxis_title="CO (mg/m³)")
    st.plotly_chart(polish(fig, 420), width="stretch")
    callout("Air-ritated estimates contemporaneous CO from sensor and environmental inputs. It does not yet predict a future horizon.")

with tabs[1]:
    st.subheader("Why did Air-ritated make this estimate?")
    callout("Importance and contributions describe model associations. They do not prove that changing a sensor or pollutant would cause CO to change.", warning=True)
    left_col, right_col = st.columns([1, 1.05])
    with left_col:
        global_top = importance.head(12).sort_values("importance")
        fig = px.bar(global_top, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=["#134e4a", "#2dd4bf", "#ccfbf1"])
        fig.update_layout(title="Global permutation importance", coloraxis_showscale=False, xaxis_title="Increase in validation MAE when shuffled", yaxis_title=None)
        st.plotly_chart(polish(fig, 470), width="stretch")
        st.caption("Calculated on the chronological validation window, not on the final test labels.")
    with right_col:
        base, local = local_perturbation_contributions(model, selected_features, metadata["feature_medians"], top_n=10)
        local_plot = local.sort_values("contribution")
        local_plot["Direction"] = np.where(local_plot["contribution"] >= 0, "Raises estimate", "Lowers estimate")
        fig = px.bar(local_plot, x="contribution", y="feature", orientation="h", color="Direction", color_discrete_map={"Raises estimate": COLORS["coral"], "Lowers estimate": COLORS["blue"]})
        fig.update_layout(title=f"Local contribution · estimate {base:.2f} mg/m³", xaxis_title="Change versus median-reference estimate", yaxis_title=None)
        st.plotly_chart(polish(fig, 470), width="stretch")
        st.write(plain_english_explanation(local))
        with st.expander("How this local explanation works"):
            st.write("Each feature is replaced—one at a time—with its training median. The bar is the difference between the selected prediction and that reference prediction. Interactions mean bars are not additive, and this is not causal analysis.")

with tabs[2]:
    st.subheader("Pollution Patterns")
    pattern = add_datetime(valid_train)
    pattern["hour"] = pattern["DateTime"].dt.hour
    pattern["day_of_week"] = pattern["DateTime"].dt.day_name()
    daily = pattern.set_index("DateTime")[TARGET].resample("D").agg(["mean", "median"]).reset_index()
    p1, p2 = st.columns(2)
    with p1:
        fig = px.line(daily, x="DateTime", y=["mean", "median"], color_discrete_sequence=[COLORS["teal"], COLORS["amber"]])
        fig.update_layout(title="Daily CO pattern", xaxis_title=None, yaxis_title="CO (mg/m³)", legend_title=None)
        st.plotly_chart(polish(fig), width="stretch")
    with p2:
        hourly = pattern.groupby("hour")[TARGET].agg(["mean", "median", "count"]).reset_index()
        fig = px.line(hourly, x="hour", y=["mean", "median"], markers=True, color_discrete_sequence=[COLORS["teal"], COLORS["amber"]])
        fig.update_layout(title="CO by hour of day", xaxis_title="Hour", yaxis_title="CO (mg/m³)", legend_title=None)
        st.plotly_chart(polish(fig), width="stretch")
    clean_numeric = pattern.drop(columns=["Date", "Time", "DateTime"], errors="ignore").replace(SENTINEL_VALUE, np.nan)
    corr = clean_numeric.corr(numeric_only=True)
    display_features = importance.head(8)["feature"].tolist()
    display_features = [c for c in display_features if c in corr.columns and c not in CALENDAR_FEATURES]
    heatmap_cols = [TARGET] + display_features[:7]
    fig = px.imshow(corr.loc[heatmap_cols, heatmap_cols], text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(title="Correlation map · valid readings only")
    st.plotly_chart(polish(fig, 520), width="stretch")
    strongest_hour = int(hourly.loc[hourly["mean"].idxmax(), "hour"])
    strongest_corr = corr[TARGET].drop(TARGET).abs().idxmax()
    strongest_corr_value = float(corr.loc[strongest_corr, TARGET])
    callout(f"Calculated insight: the highest mean hourly CO in the training data occurs around {strongest_hour:02d}:00. The strongest linear association with CO is {strongest_corr} (r = {strongest_corr_value:.2f}). Association is not causation.")

with tabs[3]:
    st.subheader("Sensor Scenario Lab")
    callout("Scenario estimates show model behavior under modified inputs. They must not be interpreted as causal effects.", warning=True)
    st.caption("Controls are restricted to the 1st–99th percentile training range to discourage unrealistic extrapolation.")
    scenario_features = ["PT08.S1(CO)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH"]
    scenario = selected_features.copy()
    cols = st.columns(2)
    for idx, feature in enumerate(scenario_features):
        low = float(metadata["feature_p01"][feature])
        high = float(metadata["feature_p99"][feature])
        current = scenario.iloc[0][feature]
        if not valid_value(current):
            current = float(metadata["feature_medians"][feature])
        step = max((high - low) / 100.0, 0.01)
        scenario.loc[scenario.index[0], feature] = cols[idx % 2].slider(feature, low, high, float(np.clip(current, low, high)), step=step, key=f"scenario_{feature}")
    scenario_prediction = float(model.predict(scenario)[0])
    change = scenario_prediction - selected_prediction
    percent = 100 * change / max(abs(selected_prediction), 1e-9)
    a, b, c = st.columns(3)
    with a:
        kpi("Original estimate", f"{selected_prediction:.2f} mg/m³", "Selected observation")
    with b:
        kpi("Scenario estimate", f"{scenario_prediction:.2f} mg/m³", "Modified model inputs")
    with c:
        kpi("Model-estimated change", f"{change:+.2f} mg/m³", f"{percent:+.1f}% versus original")
    delta_df = pd.DataFrame({"State": ["Original", "Scenario"], "CO estimate": [selected_prediction, scenario_prediction]})
    fig = px.bar(delta_df, x="State", y="CO estimate", color="State", text_auto=".2f", color_discrete_sequence=[COLORS["muted"], COLORS["teal"]])
    fig.update_layout(showlegend=False, yaxis_title="Predicted CO (mg/m³)")
    st.plotly_chart(polish(fig, 350), width="stretch")

with tabs[4]:
    st.subheader("Data Quality & Sensor Health")
    st.caption("Transparent quality status for the selected input—not a calibrated prediction confidence score.")
    q1, q2, q3 = st.columns(3)
    q1.metric("Input completeness", f"{quality_score:.0f}%")
    q2.metric("Invalid / missing inputs", feature_count - valid_count)
    q3.metric("Training target rows excluded", f"{audit['sentinel_minus_200']['train'][TARGET]:,}")
    health_rows = []
    for feature in raw_features:
        value = selected_raw.get(feature, np.nan)
        if not valid_value(value):
            status, detail = "Missing / invalid", "-200 or NaN; median-imputed by pipeline"
            reading_display = "—"
        else:
            try:
                numeric_value = float(value)
                reading_display = f"{numeric_value:.2f}"
            except (ValueError, TypeError):
                reading_display = "—"
                status, detail = "Missing / invalid", "-200 or NaN; median-imputed by pipeline"
            else:
                low = metadata["feature_p01"].get(feature)
                high = metadata["feature_p99"].get(feature)
                if low is not None and high is not None and not (low <= numeric_value <= high):
                    status, detail = "Unusual", "Outside training 1st–99th percentile"
                else:
                    status, detail = "Healthy", "Valid and within reference range"
        health_rows.append({"Input": feature, "Reading": reading_display, "Status": status, "Interpretation": detail})
    health = pd.DataFrame(health_rows)
    st.dataframe(
        health.style.map(lambda x: "color:#fb7185" if x == "Missing / invalid" else ("color:#f59e0b" if x == "Unusual" else "color:#79e7cd"), subset=["Status"]),
        width="stretch",
        hide_index=True,
    )
    missing = pd.DataFrame(
        {
            "Feature": raw_features,
            "Train invalid %": [100 * audit["sentinel_minus_200"]["train"].get(c, 0) / len(train) for c in raw_features],
            "Test invalid %": [100 * audit["sentinel_minus_200"]["test"].get(c, 0) / len(test) for c in raw_features],
        }
    ).sort_values("Train invalid %", ascending=False)
    fig = px.bar(missing, x="Feature", y=["Train invalid %", "Test invalid %"], barmode="group", color_discrete_sequence=[COLORS["amber"], COLORS["teal"]])
    fig.update_layout(title="Invalid-reading rates after recognizing the -200 sentinel", xaxis_title=None, yaxis_title="Rows (%)", legend_title=None)
    st.plotly_chart(polish(fig, 430), width="stretch")
    callout(f"Quality score = valid required raw inputs ÷ {feature_count} required raw inputs × 100. It measures completeness, not certainty.")

with tabs[5]:
    st.subheader("Model Lab")
    v = metrics["validation"]
    h = metrics["holdout"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Winning model", metrics["winning_model"])
    m2.metric("Holdout MAE", f"{h['MAE']:.3f}")
    m3.metric("Holdout RMSE", f"{h['RMSE']:.3f}")
    m4.metric("Holdout R²", f"{h['R2']:.3f}")
    st.markdown("#### Candidate comparison · chronological validation")
    show_comparison = comparison.copy()
    for col in ["MAE", "RMSE", "R2", "Training Time (s)"]:
        show_comparison[col] = show_comparison[col].round(4)
    st.dataframe(show_comparison, width="stretch", hide_index=True)
    valid_predictions = predictions[predictions["Valid_Holdout_Label"]].copy()
    valid_predictions["Residual"] = valid_predictions["Actual_CO(GT)"] - valid_predictions["Predicted_CO(GT)"]
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(valid_predictions, x="Actual_CO(GT)", y="Predicted_CO(GT)", opacity=.42, color_discrete_sequence=[COLORS["teal"]])
        line_min = min(valid_predictions["Actual_CO(GT)"].min(), valid_predictions["Predicted_CO(GT)"].min())
        line_max = max(valid_predictions["Actual_CO(GT)"].max(), valid_predictions["Predicted_CO(GT)"].max())
        fig.add_trace(go.Scatter(x=[line_min, line_max], y=[line_min, line_max], mode="lines", name="Ideal", line=dict(dash="dash", color=COLORS["amber"])))
        fig.update_layout(title="Holdout predicted vs actual", xaxis_title="Actual CO", yaxis_title="Predicted CO")
        st.plotly_chart(polish(fig), width="stretch")
    with c2:
        fig = px.scatter(valid_predictions, x="Predicted_CO(GT)", y="Residual", opacity=.42, color_discrete_sequence=[COLORS["blue"]])
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["amber"])
        fig.update_layout(title="Holdout residuals", xaxis_title="Predicted CO", yaxis_title="Actual − predicted")
        st.plotly_chart(polish(fig), width="stretch")
    fig = px.histogram(valid_predictions, x="Residual", nbins=45, marginal="box", color_discrete_sequence=[COLORS["teal"]])
    fig.update_layout(title="Residual distribution", xaxis_title="Actual − predicted (mg/m³)", yaxis_title="Observations")
    st.plotly_chart(polish(fig, 390), width="stretch")
    with st.expander("Validation and preprocessing details", expanded=True):
        st.write(f"**Validation:** {metadata['validation_strategy']}")
        st.write(f"**Development / validation / test:** {metadata['development_rows']:,} / {metadata['validation_rows']:,} / {metadata['test_rows']:,} rows")
        st.write("**Preprocessing:** confirmed -200 missing sentinel → NaN; invalid target rows excluded; median imputation learned inside pipelines; high-missingness NMHC(GT) dropped; cyclical time features selected through a validation ablation.")
        st.write("**Leakage guard:** no target lags; no test-label use for tuning, feature selection, preprocessing, model selection, or ensemble selection.")

st.markdown("---")
st.caption("Air-ritated · Hack-ML Track 1 · Model associations are not causal conclusions · UCI Air Quality source conventions")
