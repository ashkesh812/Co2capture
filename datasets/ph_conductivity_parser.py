import os
import re
import pandas as pd
from datetime import timedelta

def parse_ph_conductivity_file(file_path):
    filename = os.path.basename(file_path)

    # Extract current and gas flow rate from filename if available
    current_match = re.search(r"(\d+\.?\d*)A", filename)
    flow_match = re.search(r"(\d+\.?\d*)m3h", filename)
    current = float(current_match.group(1)) if current_match else None
    gas_flow_rate = float(flow_match.group(1)) if flow_match else None

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".xlsx":
            df_raw = pd.read_excel(file_path, skiprows=1, header=0)
        elif ext == ".csv":
            df_raw = pd.read_csv(file_path, skiprows=1, header=0, engine="python", on_bad_lines="skip")
        else:
            print(f"❌ Unsupported file type: {filename}")
            return None
    except Exception as e:
        print(f"❌ Failed to load {filename}: {e}")
        return None

    # Clean column names
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    df = df_raw.dropna(axis=1, how="all")

    # Drop extra label rows
    if "Time" not in df.columns:
        print(f"❌ Missing 'Time' column in {filename}")
        return None

    df = df.dropna(subset=["Time"]).copy()
    df["Timestamp"] = [str(timedelta(seconds=i)) for i in range(len(df))]

    # Add extracted inputs
    df["Current_Set (A)"] = current
    df["Gas_Flow_Rate (m3/h)"] = gas_flow_rate

    rename_map = {
        "Ph1": "pH1",
        "Ph2": "pH2",
        "Cond1": "Conductivity1 (mS/cm)",
        "Cond2": "Conductivity2 (mS/cm)",
        "MFM1": "Solvent_Flow (ml/min)",
        "TempPH1": "Temp_pH1 (°C)",
        "TempPH2": "Temp_pH2 (°C)",
        "MfmOmega": "Gas_Flow_Actual (Ln/min)"
    }

    df = df.rename(columns=rename_map)

    output_columns = ["Timestamp", "Current_Set (A)", "Gas_Flow_Rate (m3/h)"]
    for col in rename_map.values():
        if col in df.columns:
            output_columns.append(col)

    return df[output_columns]

def batch_parse_ph_conductivity_files(folder_path):
    all_dfs = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".xlsx") or fname.endswith(".csv"):
            fpath = os.path.join(folder_path, fname)
            df = parse_ph_conductivity_file(fpath)
            if df is not None:
                all_dfs.append(df)
                print(f"✅ Parsed: {fname}")
            else:
                print(f"❌ Skipped: {fname}")
    if not all_dfs:
        print("❌ No valid files found.")
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)
