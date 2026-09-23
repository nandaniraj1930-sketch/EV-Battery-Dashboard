import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# 1. Page Configuration (Premium Dark Theme & Wide Layout)
st.set_page_config(page_title="EV Battery AI Simulator", layout="wide", page_icon="🔋")

# 2. Data Generation & AI Model Setup
@st.cache_data
def load_data_and_model():
    np.random.seed(42)
    n = 2000
    cycles = np.random.randint(100, 2000, n)
    temps = np.random.normal(25, 10, n)
    fast_charge = np.random.uniform(0, 100, n)
    dod = np.random.uniform(20, 90, n)
    
    soh = 100 - (cycles * 0.005) - (np.maximum(0, temps - 30) * 0.2) - (fast_charge * 0.05) - (dod * 0.02)
    soh = np.clip(soh + np.random.normal(0, 1, n), 0, 100)
    
    df = pd.DataFrame({'Cycles': cycles, 'Temp': temps, 'FastCharge': fast_charge, 'DoD': dod, 'SOH': soh})
    
    X = df[['Cycles', 'Temp', 'FastCharge', 'DoD']]
    y = df['SOH']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return df, model

df, model = load_data_and_model()

# 3. INTERACTIVE SIDEBAR (Left Control Panel)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063206.png", width=100)
st.sidebar.title("🔋 Battery Settings")
st.sidebar.markdown("Adjust parameters to simulate battery degradation.")

input_cycles = st.sidebar.slider("Charge Cycles (Usage)", 100, 2000, 800)
input_temp = st.sidebar.slider("Avg Climate Temp (°C)", 0, 50, 35)
input_fc = st.sidebar.slider("Fast Charging Frequency (%)", 0, 100, 40)
input_dod = st.sidebar.slider("Depth of Discharge (%)", 10, 90, 60)

# 4. MAIN DASHBOARD HEADER
st.title("⚡ EV Fleet Health & AI Predictive Analytics")
st.markdown("This interactive AI tool predicts the **Remaining Useful Life (SoH)** of Lithium-Ion batteries based on driver behavior and climate data.")

# Run AI Prediction Live
prediction = model.predict([[input_cycles, input_temp, input_fc, input_dod]])[0]

# 5. UNIQUE FEATURE: LIVE GAUGE CHART (Speedometer)
col1, col2 = st.columns([1, 2])

with col1:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        title = {'text': "Predicted Health (SoH)", 'font': {'size': 20}},
        number = {'suffix': "%"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white"},
            'bgcolor': "#1E1E1E",
            'steps': [
                {'range': [0, 80], 'color': "#FF4B4B"}, # Red for Danger
                {'range': [80, 100], 'color': "#00CC96"}], # Green for Healthy
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric(label="System Status", value="Healthy" if prediction >= 80 else "Critical Replacement Needed", delta="Optimal" if prediction >= 80 else "Degraded", delta_color="normal" if prediction >= 80 else "inverse")
    st.metric(label="Estimated Range Loss", value=f"{100 - prediction:.1f} %")

st.divider()

# 6. ATTRACTIVE VISUALIZATION: RED/GREEN HEATMAP
st.markdown("### 📊 Fleet Degradation Overview")
st.markdown("The scatter plot below highlights dangerous charging temperatures. **Red** indicates batteries that have fallen below the 80% safety threshold.")

df['Status'] = np.where(df['SOH'] >= 80, 'Healthy (>80%)', 'Risk Zone (<80%)')
color_map = {'Healthy (>80%)': '#00CC96', 'Risk Zone (<80%)': '#FF4B4B'}

fig_scatter = px.scatter(df, x="Temp", y="SOH", color="Status", color_discrete_map=color_map,
                 hover_data=['Cycles', 'FastCharge'], opacity=0.7,
                 labels={"Temp": "Climate Temperature (°C)", "SOH": "State of Health (%)"})

fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_scatter, use_container_width=True)