import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from models.PINNModel import PINNModel

st.set_page_config(page_title="CO₂ Capture Predictor", layout="wide")
st.title("🧪 Physics-Informed Neural Network for CO₂ Capture")

# --- Sidebar Constants ---
st.sidebar.header("⚙️ Constants")
st.sidebar.markdown("""
- **Membrane area**: 0.01 m²  
- **Faraday constant**: 96485 C/mol  
- **Inlet CO₂ concentration**: 15%  
- **Electrolyte**: 0.05 molal K₂SO₄  
- **Stack**: BMED (BPM + CEM)
""")

# --- Load parsed data for scaler fitting ---
co2_df = pd.read_csv("./Parsed_Data/combined_co2_concentration.csv")
cv_df = pd.read_csv("./Parsed_Data/combined_current_voltage.csv")
ph_df = pd.read_csv("./Parsed_Data/combined_ph_conductivity.csv")
co2_df.rename(columns={"Current (A)": "Current_Set (A)"}, inplace=True)

# Merge and clean
df = pd.merge(co2_df, cv_df, on=["Timestamp", "Current_Set (A)", "Gas_Flow_Rate (m3/h)"])
df = pd.merge(df, ph_df, on=["Timestamp", "Current_Set (A)", "Gas_Flow_Rate (m3/h)"])
df["Timestamp"] = pd.to_timedelta(df["Timestamp"]).dt.total_seconds()

# Targets
def compute_targets(row):
    try:
        eta = max((row["CO2_out (%)"] - 15) / 15, 0.0)
        seec = (row["Voltage (V)"] * row["Measured_Current (A)"]) / (abs(row["CO2_out (%)"] - 15) + 1e-8)
        return pd.Series([eta, seec])
    except:
        return pd.Series([np.nan, np.nan])

if all(col in df.columns for col in ["Voltage (V)", "Measured_Current (A)", "CO2_out (%)"]):
    df[["eta_CO2_des", "SEEC"]] = df.apply(compute_targets, axis=1)
    df = df.dropna(subset=["eta_CO2_des", "SEEC", "pH1", "Conductivity1 (mS/cm)", "Solvent_Flow (ml/min)"])
else:
    st.error("❌ Required columns for computing `eta_CO2_des` and `SEEC` are missing.")
    st.stop()

# Inputs and outputs
input_cols = ["Current_Set (A)", "Gas_Flow_Rate (m3/h)", "Timestamp"]
output_cols = [
    "eta_CO2_des", "SEEC", "CO2_out (%)", "Voltage (V)",
    "pH1", "Conductivity1 (mS/cm)", "Temp_pH1 (°C)", "Solvent_Flow (ml/min)"
]

X = df[input_cols].astype(float).values
Y = df[output_cols].astype(float).values

input_scaler = StandardScaler().fit(X)
output_scaler = StandardScaler().fit(Y)

# --- Load model ---
model = PINNModel(input_dim=3, output_dim=8)
model.load_state_dict(torch.load("pinn_model.pt"))
model.eval()

# --- User Inputs ---
st.subheader("🔧 Input Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    current = st.slider("Current (A)", 1.0, 50.0, 6.0, step=0.1)
with col2:
    flow = st.slider("Gas Flow Rate (m³/h)", 0.1, 5.0, 0.5, step=0.1)
with col3:
    duration = st.slider("Run Duration (seconds)", 0, 21600, 3600, step=60)

timestamps = np.arange(0, duration + 1, step=60)
input_arr = np.column_stack([
    np.full_like(timestamps, current),
    np.full_like(timestamps, flow),
    timestamps
])

X_scaled = input_scaler.transform(input_arr)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# --- Predict ---
with torch.no_grad():
    preds = model(X_tensor).numpy()
    preds = output_scaler.inverse_transform(preds)

pred_df = pd.DataFrame(preds, columns=output_cols)
pred_df["Time (s)"] = timestamps

# --- Final Predictions ---
st.subheader("📊 Final Predicted Values")
latest = pred_df.iloc[-1]
for col in output_cols:
    st.markdown(f"**{col}**: `{latest[col]:.4f}`")

# --- Graphs ---
st.markdown("---")
st.subheader("📈 Time-Series Output Graphs")
g1, g2, g3 = st.columns(3)

def plot_graph(ax, x, y, label):
    ax.plot(x, y, color="green")
    ax.set_title(label)
    ax.set_xlabel("Time (s)")
    ax.grid(True)

with g1:
    fig, ax = plt.subplots()
    plot_graph(ax, pred_df["Time (s)"], pred_df["CO2_out (%)"], "CO₂ Outlet (%)")
    st.pyplot(fig)

with g2:
    fig, ax = plt.subplots()
    plot_graph(ax, pred_df["Time (s)"], pred_df["Voltage (V)"], "Voltage (V)")
    st.pyplot(fig)

with g3:
    fig, ax = plt.subplots()
    plot_graph(ax, pred_df["Time (s)"], pred_df["pH1"], "pH1")
    st.pyplot(fig)
