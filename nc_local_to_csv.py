import os
import xarray as xr
import pandas as pd
import json
from tqdm import tqdm  # progress bar

# ---------------- Configuration ----------------
input_folder = r"c:\Users\aim\Desktop\copy sih\sih divy\nc_Argo"
output_csv = r"c:\Users\aim\Desktop\copy sih\sih divy\raw_csv\raw_merged_data.csv"
processed_file_json = r"c:\Users\aim\Desktop\copy sih\sih divy\processed_files.json"
# ------------------------------------------------

# Load list of already processed files
if os.path.exists(processed_file_json):
    with open(processed_file_json, "r") as f:
        processed_files = set(json.load(f))
else:
    processed_files = set()

# Scan all .nc files in input folder
nc_files = [f for f in os.listdir(input_folder) if f.endswith(".nc")]
new_files = [f for f in nc_files if f not in processed_files]

if not new_files:
    print("✅ No new files to process.")
else:
    print(f"📂 Found {len(new_files)} new NetCDF files")

    for file in tqdm(new_files, desc="Processing new files"):
        try:
            file_path = os.path.join(input_folder, file)
            ds = xr.open_dataset(file_path)

            # Convert NetCDF to DataFrame
            df = ds.to_dataframe().reset_index()
            df["source_file"] = file  # Track origin file

            # Append directly to CSV (memory-efficient)
            if os.path.exists(output_csv):
                df.to_csv(output_csv, mode="a", index=False, header=False)
            else:
                df.to_csv(output_csv, index=False)

            # Update processed files JSON after each file (safe resumable)
            processed_files.add(file)
            with open(processed_file_json, "w") as f:
                json.dump(list(processed_files), f)

        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")

    print(f"✅ Merged CSV updated: {output_csv}")
