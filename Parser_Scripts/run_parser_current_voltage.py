from datasets.current_voltage_parser import batch_parse_current_voltage_files

if __name__ == "__main__":
    folder_path = "./Data/Current-voltage"
    df_all = batch_parse_current_voltage_files(folder_path)
    df_all.to_csv("combined_current_voltage.csv", index=False)
    print("✅ Done! Saved to combined_current_voltage.csv")
