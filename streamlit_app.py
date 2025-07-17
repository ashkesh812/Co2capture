import streamlit as st
import numpy as np
import pandas as pd
import torch
from models.PINNModel import PINNModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Load and prepare training data for scalers and evaluation ---
st.set_page_config(layout="wide")
st.title("🔬 Electrochemical CO₂ Capture Prediction Platform")

@st.cache_resource
def load_and_prepare_data():
    co2_df = pd.read_csv("./Parsed_Data/combined_co2_concentration.csv")
    co2_df.rename(columns={"Current (A)": "Current_Set (A)"}, inplace=True)
    cv_df = pd.read_csv("./Parsed_Data/combined_current_voltage.csv")
    ph_df = pd.read_csv("./Parsed_Data/combined_ph_conductivity.csv")

    df = pd.merge(co2_df, cv_df, on=["Timestamp", "Current_Set (A)", "Gas_Flow_Rate (m3/h)"], how="inner")
    df = pd.merge(df, ph_df, on=["Timestamp", "Current_Set (A)", "Gas_Flow_Rate (m3/h)"], how="inner")
    df["Timestamp"] = pd.to_timedelta(df["Timestamp"]).dt.total_seconds()

    CO2_INLET_PCT = 15
    EPS = 1e-8

    def compute_targets(row):
        eta = max((row["CO2_out (%)"] - CO2_INLET_PCT) / (CO2_INLET_PCT + EPS), 0.0)
        seec = (row["Voltage (V)"] * row["Measured_Current (A)"]) / (abs(row["CO2_out (%)"] - CO2_INLET_PCT) + EPS)
        return pd.Series([eta, seec])

    df[["eta_CO2_des", "SEEC"]] = df.apply(compute_targets, axis=1)
    df = df.dropna(subset=["eta_CO2_des", "SEEC", "pH1", "Conductivity1 (mS/cm)", "Solvent_Flow (ml/min)"])

    input_cols = ["Current_Set (A)", "Gas_Flow_Rate (m3/h)", "Timestamp"]
    output_cols = [
        "eta_CO2_des", "SEEC", "CO2_out (%)", "Voltage (V)",
        "pH1", "Conductivity1 (mS/cm)", "Temp_pH1 (°C)", "Solvent_Flow (ml/min)"
    ]

    X = df[input_cols].astype(float).values
    Y = df[output_cols].astype(float).values

    X_scaler = StandardScaler().fit(X)
    Y_scaler = StandardScaler().fit(Y)

    return df, X, Y, X_scaler, Y_scaler, input_cols, output_cols

def load_model():
    model = PINNModel(input_dim=3, output_dim=8)
    model.load_state_dict(torch.load("pinn_model.pt"))
    model.eval()
    return model

# --- Load resources ---
df, X_raw, Y_true, X_scaler, Y_scaler, input_cols, output_cols = load_and_prepare_data()
model = load_model()

# --- Sidebar Inputs ---
st.sidebar.header("🔧 Set Experimental Inputs")
current = st.sidebar.slider("Current (A)", min_value=4.0, max_value=6.5, step=0.1, value=6.0)
flow_rate = st.sidebar.slider("Gas Flow Rate (m3/h)", min_value=0.3, max_value=0.5, step=0.05, value=0.5)
duration = st.sidebar.slider("Run Duration (seconds)", min_value=600, max_value=7200, step=300, value=3600)
time_step = 60  # Predict every 60 seconds

# --- Time vector and input generation ---
time_vector = np.arange(0, duration + 1, time_step)
input_matrix = np.array([[current, flow_rate, t] for t in time_vector])
X_scaled = X_scaler.transform(input_matrix)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# --- Make predictions ---
with torch.no_grad():
    preds_scaled = model(X_tensor).numpy()
    preds_real = Y_scaler.inverse_transform(preds_scaled)

pred_df = pd.DataFrame(preds_real, columns=output_cols)
pred_df["Time (s)"] = time_vector

# --- Show Predictions at Final Time ---
st.subheader("📌 Final Predicted Outputs")
final = pred_df.iloc[-1]
cols = st.columns(4)
for i, col in enumerate(output_cols):
    color = "green" if i % 2 == 0 else "red"
    with cols[i % 4]:
        st.markdown(f"<span style='color:{color}; font-size:20px'>{col}</span>", unsafe_allow_html=True)
        st.metric(label="", value=f"{final[col]:.4f}")

# --- Evaluation Stats (global stats) ---
with torch.no_grad():
    full_preds_scaled = model(torch.tensor(X_scaler.transform(X_raw), dtype=torch.float32)).numpy()
    full_preds_real = Y_scaler.inverse_transform(full_preds_scaled)

mae = mean_absolute_error(Y_true, full_preds_real, multioutput='raw_values')
rmse = np.sqrt(mean_squared_error(Y_true, full_preds_real, multioutput='raw_values'))
r2 = r2_score(Y_true, full_preds_real, multioutput='raw_values')

st.subheader("📊 Model Performance (Global Evaluation)")
st.dataframe(pd.DataFrame({
    "Output": output_cols,
    "MAE": np.round(mae, 4),
    "RMSE": np.round(rmse, 4),
    "R²": np.round(r2, 4)
}))

# --- Graphs ---
st.subheader("📈 Key Trends Over Time")
plot_cols = ["eta_CO2_des", "Voltage (V)", "pH1"]
for var in plot_cols:
    fig, ax = plt.subplots()
    ax.plot(pred_df["Time (s)"], pred_df[var], marker='o', linewidth=2, color='green')
    ax.set_title(f"{var} vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(var)
    st.pyplot(fig)
