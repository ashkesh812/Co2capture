from datasets.ph_conductivity_parser import batch_parse_ph_conductivity_files

if __name__ == "__main__":
    folder_path = "./Data/pH:conductivity:flowrate"
    df_all = batch_parse_ph_conductivity_files(folder_path)
    df_all.to_csv("combined_ph_conductivity.csv", index=False)
    print("✅ Done! Saved to combined_ph_conductivity.csv")
