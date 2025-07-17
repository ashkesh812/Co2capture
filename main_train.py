import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from models.PINNModel import PINNModel

# --- Constants ---
FARADAY = 96485  # C/mol
MEMBRANE_AREA = 0.01  # m²
CO2_INLET_PCT = 15
EPS = 1e-8

# --- Load parsed datasets ---
print("🔄 Loading data...")
co2_df = pd.read_csv("./Parsed_Data/combined_co2_concentration.csv")
co2_df.rename(columns={"Current (A)": "Current_Set (A)"}, inplace=True)

cv_df = pd.read_csv("./Parsed_Data/combined_current_voltage.csv", low_memory=False)
ph_df = pd.read_csv("./Parsed_Data/combined_ph_conductivity.csv", low_memory=False)

# --- Merge datasets ---
df = pd.merge(co2_df, cv_df, on=["Timestamp", "Current_Set (A)", "Gas_Flow_Rate (m3/h)"], how="inner")
df = pd.merge(df, ph_df, on=["Timestamp", "Current_Set (A)", "Gas_Flow_Rate (m3/h)"], how="inner")

# --- Compute targets ---
def compute_targets(row):
    try:
        eta = max((row["CO2_out (%)"] - CO2_INLET_PCT) / CO2_INLET_PCT, 0.0)
        seec = (row["Voltage (V)"] * row["Measured_Current (A)"]) / (abs(row["CO2_out (%)"] - CO2_INLET_PCT) + EPS)
        return pd.Series([eta, seec])
    except:
        return pd.Series([np.nan, np.nan])

df[["eta_CO2_des", "SEEC"]] = df.apply(compute_targets, axis=1)
df = df.dropna(subset=["eta_CO2_des", "SEEC", "pH1", "Conductivity1 (mS/cm)", "Solvent_Flow (ml/min)"])
df["Timestamp"] = pd.to_timedelta(df["Timestamp"]).dt.total_seconds()

# --- Prepare inputs and outputs ---
input_cols = ["Current_Set (A)", "Gas_Flow_Rate (m3/h)", "Timestamp"]
output_cols = [
    "eta_CO2_des", "SEEC", "CO2_out (%)", "Voltage (V)",
    "pH1", "Conductivity1 (mS/cm)", "Temp_pH1 (°C)", "Solvent_Flow (ml/min)"
]

X = df[input_cols].astype(float).values
Y = df[output_cols].astype(float).values

# --- Normalize inputs and outputs ---
input_scaler = StandardScaler()
X = input_scaler.fit_transform(X)

output_scaler = StandardScaler()
Y = output_scaler.fit_transform(Y)

# --- Split data ---
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)

# --- Model setup ---
model = PINNModel(input_dim=3, output_dim=8)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# --- Training loop ---
print("🚀 Training model...")
epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    preds = model(X_train)
    loss = criterion(preds, Y_train)

    # Physics-informed components (fixed indexing)
    eta_pred = preds[:, 0]
    seec_pred = preds[:, 1]
    co2_out_pred = preds[:, 2]
    voltage_pred = preds[:, 3]
    current_input = X_train[:, 0]

    phys_loss_eta = torch.mean(torch.relu(-(eta_pred - (co2_out_pred - CO2_INLET_PCT) / (CO2_INLET_PCT + EPS)))**2)
    phys_loss_seec = torch.mean((seec_pred - (voltage_pred * current_input) / (torch.abs(co2_out_pred - CO2_INLET_PCT) + EPS))**2)

    total_loss = loss + 1.0 * phys_loss_eta + 1.0 * phys_loss_seec
    total_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Validation
    model.eval()
    val_loss = criterion(model(X_val), Y_val).item()

    # Print every 5 epochs
    if epoch % 5 == 0 or epoch == epochs - 1:
        print(f"[{epoch:03d}] Train Loss: {total_loss.item():.5f} | Val Loss: {val_loss:.5f}")

# --- Save model ---
torch.save(model.state_dict(), "pinn_model.pt")
print("✅ Model trained and saved to pinn_model.pt")
