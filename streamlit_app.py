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

# Load scalers
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

for col in input_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=input_cols)
X = df[input_cols].astype(float).values

# Ensure all output columns exist
available_outputs = [col for col in output_cols if col in df.columns]
missing_outputs = [col for col in output_cols if col not in df.columns]

if missing_outputs:
    st.warning(f"⚠️ These output columns were missing: {missing_outputs}")

df = df.dropna(subset=available_outputs)
Y = df[available_outputs].astype(float).values

input_scaler = StandardScaler().fit(X)
output_scaler = StandardScaler().fit(Y)

# UI
st.set_page_config(layout="wide", page_title="CO₂ Capture ML App")
st.title("🌿 Physics-Informed CO₂ Capture Predictor")

# Input sliders
current = st.slider("Current (A)", 4.0, 6.0, 6.0, step=0.1)
flow_rate = st.slider("Gas Flow Rate (m³/h)", 0.1, 0.5, 0.5, step=0.01)
duration = st.slider("Run Duration (seconds)", 0, 7200, 3600, step=60)

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
    "eta_CO2_des",
    "SEEC",
    "CO2_out (%)",
    "Voltage (V)",
    "pH1",
    "Conductivity1 (mS/cm)",
    "Temp_pH1 (°C)",
    "Solvent_Flow (ml/min)"
]
df_out = pd.DataFrame(preds, columns=columns)
df_out["Time (s)"] = timestamps

# Summary
st.subheader("📈 Final Predictions")
latest = df_out.iloc[-1]
for name, value in zip(columns, latest[:-1]):
    color = "green" if value > 0 else "red"
    st.markdown(f"- **{name}**: <span style='color:{color}'>{value:.4f}</span>", unsafe_allow_html=True)

# Plotting
def plot_line(y, label, ylabel):
    fig, ax = plt.subplots()
    ax.plot(timestamps, y, color="green")
    ax.set_title(label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    st.pyplot(fig)

st.markdown("---")
st.subheader("📉 Time Series Plots")
col1, col2, col3 = st.columns(3)
with col1:
    plot_line(df_out["CO2_out (%)"], "CO₂ Outlet Over Time", "%")
with col2:
    plot_line(df_out["Voltage (V)"], "Voltage Over Time", "V")
with col3:
    plot_line(df_out["pH1"], "pH Over Time", "")
