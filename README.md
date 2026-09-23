# ⚡ EV Fleet Health & Degradation AI Simulator

## 📌 Project Overview
An interactive Machine Learning web application designed to predict the Remaining Useful Life and State of Health (SoH) of Lithium-Ion EV batteries. Instead of static reports, this project provides a real-time simulation engine where fleet managers can adjust climate and charging parameters to instantly see battery degradation risks.

## 📸 Dashboard Preview


## 🛠️ Tech Stack & Methodology
* **Language & UI:** Python, Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest Regressor)
* **Data Visualization:** Plotly (Interactive Gauge & Heatmaps)
* **Data Pipeline:** End-to-end processing of non-linear degradation physics equations and simulated fleet parameters.

## 🚀 Key Features
1. **Live Inference Engine:** Users can tweak temperature and charging habits to see live health predictions.
2. **Risk Speedometer:** Visual threshold indicator for immediate battery replacement decisions (Critical < 80% SoH).
3. **Interactive Fleet Heatmap:** Dynamic scatter plot isolating high-risk vehicles based on climate data.
