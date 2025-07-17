import sys
import os

# Make sure Python can find local modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import from the renamed parser file
from datasets.co2_concentration_parser import batch_parse_co2_files

if __name__ == "__main__":
    # Folder containing your 9 CO2 concentration files
    folder_path = "./Data/Co2 Concentration"

    # Parse all files and combine the data
    df_all = batch_parse_co2_files(folder_path)

    # Save the result
    df_all.to_csv("combined_co2_concentration.csv", index=False)

    print("✅ Done! Saved to combined_co2_concentration.csv")
