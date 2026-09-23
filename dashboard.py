import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ------------------------------------------------------------------
# 1. Page configuration
# ------------------------------------------------------------------
st.set_page_config(page_title="EV Battery AI Simulator", layout="wide", page_icon="🔋")

EOL_THRESHOLD = 80        # SoH (%) below which the battery is considered end-of-life
WARNING_THRESHOLD = 85    # early-warning band
MAX_CYCLES = 3000
FEATURES = ["Cycles", "Temp", "FastCharge", "DoD"]


# ------------------------------------------------------------------
# 2. Battery degradation physics (synthetic ground truth)
# ------------------------------------------------------------------
def simulate_soh(cycles, temp, fast_charge, dod):
    """Simplified degradation model: every stress factor now has a real impact,
    so the fleet contains both healthy AND degraded batteries."""
    cycle_fade = 0.006 * cycles * (0.5 + dod / 100)        # deeper discharge = faster wear
    heat_fade = 0.5 * np.maximum(0, temp - 30)              # heat above 30°C accelerates ageing
    cold_fade = 0.15 * np.maximum(0, 10 - temp)             # very cold climates also hurt
    fc_fade = fast_charge * 0.08 * (1 + cycles / 2000)      # fast charging hurts more on old cells
    return 100 - cycle_fade - heat_fade - cold_fade - fc_fade


# ------------------------------------------------------------------
# 3. Data generation + AI model (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_data_and_model():
    rng = np.random.default_rng(42)
    n = 3000

    df = pd.DataFrame({
        "Cycles": rng.integers(100, MAX_CYCLES, n),
        "Temp": np.clip(rng.normal(28, 12, n), -5, 55),
        "FastCharge": rng.uniform(0, 100, n),
        "DoD": rng.uniform(10, 95, n),
    })
    soh = simulate_soh(df["Cycles"], df["Temp"], df["FastCharge"], df["DoD"])
    df["SOH"] = np.clip(soh + rng.normal(0, 1, n), 0, 100)

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["SOH"], test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    r2 = r2_score(y_test, model.predict(X_test))
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    return df, model, r2, importance


df, model, r2, importance = load_data_and_model()

# ------------------------------------------------------------------
# 4. Sidebar controls
# ------------------------------------------------------------------
st.sidebar.title("🔋 Battery Settings")
st.sidebar.markdown("Adjust the parameters to simulate battery degradation.")

input_cycles = st.sidebar.slider("Charge Cycles (Usage)", 100, MAX_CYCLES, 1500, step=50)
input_temp = st.sidebar.slider("Avg Climate Temp (°C)", -5, 55, 38)
input_fc = st.sidebar.slider("Fast Charging Frequency (%)", 0, 100, 60)
input_dod = st.sidebar.slider("Depth of Discharge (%)", 10, 90, 70)

st.sidebar.divider()
st.sidebar.caption(f"Model accuracy (R² on unseen data): **{r2:.3f}**")

# ------------------------------------------------------------------
# 5. Live prediction
# ------------------------------------------------------------------
user_input = pd.DataFrame([[input_cycles, input_temp, input_fc, input_dod]], columns=FEATURES)
prediction = float(np.clip(model.predict(user_input)[0], 0, 100))

# Degradation curve for the selected conditions + remaining useful life (cycles until 80%)
curve_cycles = np.arange(100, MAX_CYCLES + 1, 20)
curve_df = pd.DataFrame({
    "Cycles": curve_cycles,
    "Temp": input_temp,
    "FastCharge": input_fc,
    "DoD": input_dod,
})[FEATURES]
curve_soh = np.clip(model.predict(curve_df), 0, 100)

future = curve_cycles >= input_cycles
crossing = np.where(future & (curve_soh < EOL_THRESHOLD))[0]
if prediction < EOL_THRESHOLD:
    remaining_cycles, rul_text = 0, "0 cycles (replace now)"
elif crossing.size:
    remaining_cycles = int(curve_cycles[crossing[0]] - input_cycles)
    rul_text = f"~{remaining_cycles:,} cycles"
else:
    remaining_cycles, rul_text = None, f"> {MAX_CYCLES - input_cycles:,} cycles"

if prediction >= WARNING_THRESHOLD:
    status, delta, delta_color = "Healthy", "Optimal", "normal"
elif prediction >= EOL_THRESHOLD:
    status, delta, delta_color = "Watch Closely", "Near threshold", "off"
else:
    status, delta, delta_color = "Replacement Needed", "Degraded", "inverse"

# ------------------------------------------------------------------
# 6. Header
# ------------------------------------------------------------------
st.title("⚡ EV Fleet Health & AI Predictive Analytics")
st.markdown(
    "Predicts the **State of Health (SoH)** and **Remaining Useful Life (RUL)** of Lithium-Ion "
    "batteries from driver behaviour and climate data."
)

col1, col2 = st.columns([1, 2])

with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={"text": "Predicted Health (SoH)", "font": {"size": 20}},
        number={"suffix": "%", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "white"},
            "steps": [
                {"range": [0, EOL_THRESHOLD], "color": "#FF4B4B"},
                {"range": [EOL_THRESHOLD, WARNING_THRESHOLD], "color": "#FFA421"},
                {"range": [WARNING_THRESHOLD, 100], "color": "#00CC96"},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.8, "value": EOL_THRESHOLD},
        },
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    m1, m2, m3 = st.columns(3)
    m1.metric("System Status", status, delta=delta, delta_color=delta_color)
    m2.metric("Estimated Capacity Loss", f"{100 - prediction:.1f} %")
    m3.metric("Remaining Useful Life", rul_text)

    if prediction < EOL_THRESHOLD:
        st.error("⚠️ Battery is below the 80% safety threshold. Replacement recommended.")
    elif prediction < WARNING_THRESHOLD:
        st.warning("Battery is approaching the 80% threshold. Consider reducing fast charging and heat exposure.")
    else:
        st.success("Battery is in good condition under these operating conditions.")

st.divider()

# ------------------------------------------------------------------
# 7. Analysis tabs
# ------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📉 Degradation Forecast", "📊 Fleet Overview", "🧠 What Drives Ageing?"])

with tab1:
    st.markdown("How this battery's health is expected to decline as cycles accumulate under the chosen conditions.")
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=curve_cycles, y=curve_soh, mode="lines", name="Predicted SoH",
        line=dict(color="#00CC96", width=3),
    ))
    fig_curve.add_hline(y=EOL_THRESHOLD, line_dash="dash", line_color="#FF4B4B",
                        annotation_text="80% end-of-life threshold")
    fig_curve.add_trace(go.Scatter(
        x=[input_cycles], y=[prediction], mode="markers", name="Current state",
        marker=dict(size=14, color="white", symbol="star", line=dict(width=1, color="black")),
    ))
    fig_curve.update_layout(
        template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Charge Cycles", yaxis_title="State of Health (%)", yaxis_range=[40, 100],
        height=420,
    )
    st.plotly_chart(fig_curve, use_container_width=True)

with tab2:
    st.markdown(
        "Each dot is a battery in the fleet. **Red** dots have fallen below the 80% safety threshold; "
        "the white star is the battery you are simulating."
    )
    fleet = df.copy()  # never modify the cached dataframe
    fleet["Status"] = np.where(fleet["SOH"] >= EOL_THRESHOLD, "Healthy (≥80%)", "Risk Zone (<80%)")

    healthy_pct = (fleet["SOH"] >= EOL_THRESHOLD).mean() * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Fleet size", f"{len(fleet):,}")
    c2.metric("Healthy batteries", f"{healthy_pct:.0f} %")
    c3.metric("In risk zone", f"{100 - healthy_pct:.0f} %")

    x_choice = st.radio("X-axis", ["Temp", "Cycles", "FastCharge", "DoD"], horizontal=True)
    fig_scatter = px.scatter(
        fleet.sample(1500, random_state=1), x=x_choice, y="SOH", color="Status",
        color_discrete_map={"Healthy (≥80%)": "#00CC96", "Risk Zone (<80%)": "#FF4B4B"},
        hover_data=["Cycles", "Temp", "FastCharge", "DoD"], opacity=0.65,
        labels={"SOH": "State of Health (%)"},
    )
    fig_scatter.add_hline(y=EOL_THRESHOLD, line_dash="dash", line_color="white")
    fig_scatter.add_trace(go.Scatter(
        x=[user_input[x_choice].iloc[0]], y=[prediction], mode="markers", name="Your battery",
        marker=dict(size=16, color="white", symbol="star", line=dict(width=1, color="black")),
    ))
    fig_scatter.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.markdown("Which factors the AI model relies on most when predicting battery health.")
    fig_imp = px.bar(
        x=importance.values * 100, y=importance.index, orientation="h",
        labels={"x": "Importance (%)", "y": ""}, color_discrete_sequence=["#00CC96"],
    )
    fig_imp.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", height=350)
    st.plotly_chart(fig_imp, use_container_width=True)