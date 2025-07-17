import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.PINNModel import PINNModel

# --- Constants ---
CO2_INLET_PCT = 15
EPS = 1e-8

# --- Load data ---
print("🔄 Loading data...")
co2_df = pd.read_csv("./Parsed_Data/combined_co2_concentration.csv")
co2_df.rename(columns={"Current (A)": "Current_Set (A)"}, inplace=True)
cv_df = pd.read_csv("./Parsed_Data/combined_current_voltage.csv", low_memory=False)
ph_df = pd.read_csv("./Parsed_Data/combined_ph_conductivity.csv", low_memory=False)

df = pd.merge(co2_df, cv_df, on=["Timestamp", "Current_Set (A)", "Gas_Flow_Rate (m3/h)"], how="inner")
df = pd.merge(df, ph_df, on=["Timestamp", "Current_Set (A)", "Gas_Flow_Rate (m3/h)"], how="inner")
df["Timestamp"] = pd.to_timedelta(df["Timestamp"]).dt.total_seconds()

def compute_targets(row):
    try:
        eta = max((row["CO2_out (%)"] - CO2_INLET_PCT) / CO2_INLET_PCT, 0.0)
        seec = (row["Voltage (V)"] * row["Measured_Current (A)"]) / (abs(row["CO2_out (%)"] - CO2_INLET_PCT) + EPS)
        return pd.Series([eta, seec])
    except:
        return pd.Series([np.nan, np.nan])

df[["eta_CO2_des", "SEEC"]] = df.apply(compute_targets, axis=1)
df = df.dropna(subset=["eta_CO2_des", "SEEC", "pH1", "Conductivity1 (mS/cm)", "Solvent_Flow (ml/min)"])

input_cols = ["Current_Set (A)", "Gas_Flow_Rate (m3/h)", "Timestamp"]
output_cols = [
    "eta_CO2_des", "SEEC", "CO2_out (%)", "Voltage (V)",
    "pH1", "Conductivity1 (mS/cm)", "Temp_pH1 (°C)", "Solvent_Flow (ml/min)"
]

# --- Normalize like training ---
X = df[input_cols].astype(float).values
Y = df[output_cols].astype(float).values

input_scaler = StandardScaler()
X_scaled = input_scaler.fit_transform(X)

output_scaler = StandardScaler()
Y_scaled = output_scaler.fit_transform(Y)

# --- Load model ---
model = PINNModel(input_dim=3, output_dim=8)
model.load_state_dict(torch.load("pinn_model.pt"))
model.eval()

# --- Predict on full dataset ---
with torch.no_grad():
    preds_scaled = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
    preds_real = output_scaler.inverse_transform(preds_scaled)

# --- Eval metrics ---
mae = mean_absolute_error(Y, preds_real, multioutput='raw_values')
rmse = np.sqrt(mean_squared_error(Y, preds_real, multioutput='raw_values'))
r2 = r2_score(Y, preds_real, multioutput='raw_values')

print("\n📊 Evaluation Metrics (Validation Set):")
for i, name in enumerate(output_cols):
    print(f"🔹 {name:<25} | MAE: {mae[i]:.4f} | RMSE: {rmse[i]:.4f} | R²: {r2[i]:.4f}")

# --- Manual prediction ---
print("\n🧪 Manual Prediction Mode")
try:
    current = float(input("Enter Current (A): "))
    flow_rate = float(input("Enter Gas Flow Rate (m3/h): "))
    time_sec = float(input("Enter Time (s): "))
except ValueError:
    print("❌ Invalid input. Please enter numbers.")
    exit()

# Scale input, predict, and inverse scale output
input_arr = np.array([[current, flow_rate, time_sec]])
input_scaled = input_scaler.transform(input_arr)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

with torch.no_grad():
    pred_scaled = model(input_tensor).numpy()
    pred_real = output_scaler.inverse_transform(pred_scaled)[0]

print("\n🔮 Predicted Outputs:")
for i, name in enumerate(output_cols):
    print(f"  {name:<25}: {pred_real[i]:.4f}")
