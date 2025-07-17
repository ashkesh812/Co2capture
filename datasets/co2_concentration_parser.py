import os
import re
import pandas as pd
from datetime import timedelta

def parse_co2_concentration_file(file_path):
    filename = os.path.basename(file_path)
    current_match = re.search(r"(\d+\.?\d*)A", filename)
    flow_match = re.search(r"(\d+\.?\d*)m3h", filename)
    current = float(current_match.group(1)) if current_match else None
    gas_flow_rate = float(flow_match.group(1)) if flow_match else None

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            raw_data = pd.read_csv(file_path, skiprows=9, engine="python", on_bad_lines="skip")
        elif ext == ".xlsx":
            raw_data = pd.read_excel(file_path, skiprows=9, header=0)
        else:
            print(f"❌ Unsupported file type: {filename}")
            return None
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

    # Drop empty columns
    raw_data = raw_data.dropna(axis=1, how="all")

    # Normalize and identify columns
    col_map = {}
    for col in raw_data.columns:
        if isinstance(col, str):
            col_clean = col.strip()
            if "U3730359" in col_clean and "Carbon dioxide concentration" in col_clean:
                col_map[col] = "CO2_in (%)"
            elif "U3930116" in col_clean and "Carbon dioxide concentration" in col_clean:
                col_map[col] = "CO2_out (%)"
            elif "Timestamp" in col_clean:
                col_map[col] = "Timestamp"

    df = raw_data.rename(columns=col_map)

    # Verify required columns exist
    if not all(k in df.columns for k in ["Timestamp", "CO2_in (%)", "CO2_out (%)"]):
        print(f"❌ Missing required columns in {filename}")
        return None

    df = df.dropna(subset=["Timestamp"]).copy()
    df["Elapsed_Time"] = [str(timedelta(seconds=i)) for i in range(len(df))]
    df = df.drop(columns=["Timestamp"]).rename(columns={"Elapsed_Time": "Timestamp"})
    df["Current (A)"] = current
    df["Gas_Flow_Rate (m3/h)"] = gas_flow_rate

    return df[["Timestamp", "Current (A)", "Gas_Flow_Rate (m3/h)", "CO2_in (%)", "CO2_out (%)"]]

def batch_parse_co2_files(folder_path):
    all_dfs = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.csv') or fname.endswith('.xlsx'):
            fpath = os.path.join(folder_path, fname)
            df = parse_co2_concentration_file(fpath)
            if df is not None:
                all_dfs.append(df)
                print(f"✅ Parsed: {fname}")
            else:
                print(f"❌ Skipped: {fname}")
    return pd.concat(all_dfs, ignore_index=True)
