import os
import re
import pandas as pd
from datetime import timedelta

def parse_current_voltage_file(file_path):
    filename = os.path.basename(file_path)

    # Extract current and flow from filename
    current_match = re.search(r"(\d+\.?\d*)A", filename)
    flow_match = re.search(r"(\d+\.?\d*)m3h", filename)
    current = float(current_match.group(1)) if current_match else None
    gas_flow_rate = float(flow_match.group(1)) if flow_match else None

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            raw = pd.read_csv(file_path, skiprows=1, header=0, engine="python", on_bad_lines="skip")
        elif ext == ".xlsx":
            raw = pd.read_excel(file_path, skiprows=1, header=0)
        else:
            print(f"❌ Unsupported file type: {filename}")
            return None
    except Exception as e:
        print(f"❌ Failed to load {filename}: {e}")
        return None

    # Clean up column names
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw.dropna(axis=1, how="all")

    # Required columns
    required = ["Time", "Current", "VoltSup", "Press", "Mfm2"]
    for col in required:
        if col not in df.columns:
            print(f"❌ Missing required column '{col}' in {filename}")
            return None

    df = df.dropna(subset=["Time"]).copy()
    df["Timestamp"] = [str(timedelta(seconds=i)) for i in range(len(df))]

    # Add experimental inputs
    df["Current_Set (A)"] = current
    df["Gas_Flow_Rate (m3/h)"] = gas_flow_rate

    # Rename and organize
    df = df.rename(columns={
        "Current": "Measured_Current (A)",
        "VoltSup": "Voltage (V)",
        "Press": "Pressure (bar)",
        "Mfm2": "Gas_Flow_Mfm2 (Ln/min)",
        "Refelec": "Ref_Electrode (V)"  # optional
    })

    cols = [
        "Timestamp",
        "Current_Set (A)",
        "Gas_Flow_Rate (m3/h)",
        "Measured_Current (A)",
        "Voltage (V)",
        "Pressure (bar)",
        "Gas_Flow_Mfm2 (Ln/min)"
    ]

    if "Ref_Electrode (V)" in df.columns:
        cols.append("Ref_Electrode (V)")

    return df[cols]

def batch_parse_current_voltage_files(folder_path):
    all_dfs = []
    for fname in os.listdir(folder_path):
        if fname.endswith((".csv", ".xlsx")):
            fpath = os.path.join(folder_path, fname)
            df = parse_current_voltage_file(fpath)
            if df is not None:
                all_dfs.append(df)
                print(f"✅ Parsed: {fname}")
            else:
                print(f"❌ Skipped: {fname}")
    if not all_dfs:
        print("❌ No valid current–voltage files found.")
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)
