import streamlit as st
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from models.PINNModel import PINNModel

# Load trained model
model = PINNModel(input_dim=3, output_dim=8)
model.load_state_dict(torch.load("pinn_model.pt"))
model.eval()

# Load scalers and parsed data
co2_df = pd.read_csv("./Parsed_Data/combined_co2_concentration.csv")
co2_df.rename(columns={"Current (A)": "Current_Set (A)"}, inplace=True)
cv_df = pd.read_csv("./Parsed_Data/combined_current_voltage.csv")
ph_df = pd.read_csv("./Parsed_Data/combined_ph_conductivity.csv")

df = co2_df.merge(cv_df, on=["Timestamp", "Current_Set (A)", "Gas_Flow_Rate (m3/h)"])
df = df.merge(ph_df, on=["Timestamp", "Current_Set (A)", "Gas_Flow_Rate (m3/h)"])

input_cols = ["Current_Set (A)", "Gas_Flow_Rate (m3/h)", "Timestamp"]
output_cols = [
    "eta_CO2_des", "SEEC", "CO2_out (%)", "Voltage (V)",
    "pH1", "Conductivity1 (mS/cm)", "Temp_pH1 (°C)", "Solvent_Flow (ml/min)"
]

X = df[input_cols].astype(float).values
Y = df[output_cols].astype(float).values

input_scaler = StandardScaler().fit(X)
output_scaler = StandardScaler().fit(Y)

# UI
st.set_page_config(layout="wide", page_title="CO₂ Capture ML App")
st.title("🌿 Physics-Informed CO₂ Capture Predictor")

with st.sidebar:
    st.header("🛠️ Set Experimental Inputs")

    current = st.slider("Current (A)", 1.0, 50.0, 6.0, step=0.1)
    flow_rate = st.slider("Gas Flow Rate (m³/h)", 0.1, 5.0, 0.5, step=0.01)
    duration = st.slider("Run Duration (seconds)", 0, 21600, 3600, step=60)

    st.markdown("---")
    st.markdown("### 🧪 Fixed Constants")
    st.markdown("- Membrane Area: `0.01 m²`")
    st.markdown("- Faraday Constant: `96485 C/mol`")
    st.markdown("- KOH Concentration: `1–2 mol/kg H₂O`")
    st.markdown("- Electrolyte: `0.05 molal K₂SO₄`")
    st.markdown("- Stack: `BMED with BPM + CEM`")
    st.markdown("- Gas Composition: `15% CO₂ in air`")

# Prepare input over time
timestamps = np.arange(0, duration + 1, step=60)
input_array = np.column_stack([np.full_like(timestamps, current), np.full_like(timestamps, flow_rate), timestamps])
X_scaled = input_scaler.transform(input_array)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Predict
with torch.no_grad():
    preds = model(X_tensor).numpy()
    preds = output_scaler.inverse_transform(preds)

# Convert to DataFrame
columns = [
    "CO₂ Desorption Efficiency (η)",
    "Specific Electrical Energy Consumption (SEEC)",
    "CO₂ Outlet Concentration (%)",
    "Voltage (V)",
    "pH",
    "Conductivity (mS/cm)",
    "Temperature (°C)",
    "Solvent Flow Rate (ml/min)"
]
df_out = pd.DataFrame(preds, columns=columns)
df_out["Time (s)"] = timestamps

# Summary Stats
mae = np.mean(np.abs(preds - output_scaler.inverse_transform(output_scaler.transform(preds))), axis=0)
rmse = np.sqrt(np.mean((preds - output_scaler.inverse_transform(output_scaler.transform(preds)))**2, axis=0))
r2 = 1 - np.sum((preds - output_scaler.inverse_transform(output_scaler.transform(preds)))**2, axis=0) / np.sum((preds - np.mean(preds, axis=0))**2, axis=0)

# Display
st.subheader("📈 Prediction Results (Final Time Step)")
latest = df_out.iloc[-1]
for name, value in zip(columns, latest[:-1]):
    color = "green" if value > 0 else "red"
    st.markdown(f"- **{name}**: <span style='color:{color}'>{value:.4f}</span>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("📊 Model Performance Metrics")

stats_df = pd.DataFrame({
    "Output": columns,
    "MAE": mae,
    "RMSE": rmse,
    "R² Score": r2
})
st.dataframe(stats_df.style.highlight_max(axis=0, color="lightgreen"))

# Plot
st.markdown("---")
st.subheader("📉 Time Series Predictions")

def plot_line(y, label, ylabel):
    fig, ax = plt.subplots()
    ax.plot(timestamps, y, color="green")
    ax.set_title(label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    st.pyplot(fig)

col1, col2, col3 = st.columns(3)
with col1:
    plot_line(df_out["CO₂ Outlet Concentration (%)"], "CO₂ Outlet Over Time", "%")

with col2:
    plot_line(df_out["Voltage (V)"], "Voltage Over Time", "V")

with col3:
    plot_line(df_out["pH"], "pH Over Time", "")
