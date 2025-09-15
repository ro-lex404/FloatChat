import os
import pandas as pd
from tqdm import tqdm

# ---------------- Configuration ----------------
merged_csv = r"c:\Users\aim\Desktop\copy sih\sih divy\raw_csv\raw_merged_data.csv"
output_folder = r"c:\Users\aim\Desktop\copy sih\sih divy\raw_csv\preprocessed"
os.makedirs(output_folder, exist_ok=True)

metadata_csv = os.path.join(output_folder, "metadata.csv")
measurements_csv = os.path.join(output_folder, "measurements.csv")

metadata_cols = [
    'platform_number', 'float_serial_no', 'project_name', 'pi_name', 'platform_type',
    'direction', 'data_centre', 'cycle_number'
]

measurements_cols = [
    'platform_number', 'cycle_number', 'time', 'latitude', 'longitude',
    'pres', 'temp', 'psal', 'pres_qc', 'temp_qc', 'psal_qc',
    'pres_adjusted', 'temp_adjusted', 'psal_adjusted',
    'pres_adjusted_qc', 'temp_adjusted_qc', 'psal_adjusted_qc',
    'source_file'
]

byte_cols = ['data_type', 'project_name', 'platform_type', 'pi_name']

chunksize = 100_000  # Adjust based on RAM

# ---------------- Track duplicates ----------------
seen_metadata = set()
seen_measurements = set()

# ---------------- Start preprocessing ----------------
print("📂 Starting preprocessing...")

with pd.read_csv(merged_csv, chunksize=chunksize) as reader:
    for chunk in tqdm(reader, desc="Processing chunks"):
        # 1️⃣ Convert 'juld' to datetime safely
        if 'juld' in chunk.columns:
            try:
                if pd.api.types.is_numeric_dtype(chunk['juld']):
                    chunk['time'] = pd.to_datetime(chunk['juld'], unit='D', origin=pd.Timestamp('1950-01-01'))
                else:
                    chunk['time'] = pd.to_datetime(chunk['juld'], errors='coerce')
                chunk.drop(columns=['juld'], inplace=True)
            except Exception as e:
                print(f"⚠️ Error converting 'juld': {e}")
                chunk['time'] = pd.NaT

        # 2️⃣ Replace ARGO missing values (-2147483647) with NaN
        chunk.replace(-2147483647, pd.NA, inplace=True)

        # 3️⃣ Decode byte columns
        for col in byte_cols:
            if col in chunk.columns:
                chunk[col] = chunk[col].apply(lambda x: x.decode('utf-8').strip() if isinstance(x, bytes) else x)

        # 4️⃣ Process metadata
        meta_chunk = chunk[metadata_cols].drop_duplicates(subset=['platform_number', 'cycle_number'])
        # Fast deduplication using set comprehension
        meta_chunk = meta_chunk[[not ((row.platform_number, row.cycle_number) in seen_metadata) 
                                 for row in meta_chunk.itertuples(index=False)]]
        for row in meta_chunk.itertuples(index=False):
            seen_metadata.add((row.platform_number, row.cycle_number))
        meta_chunk.to_csv(metadata_csv, mode='a', header=not os.path.exists(metadata_csv), index=False)

        # 5️⃣ Process measurements
        meas_chunk = chunk[measurements_cols].drop_duplicates(subset=['platform_number', 'cycle_number', 'pres'])
        meas_chunk = meas_chunk[[not ((row.platform_number, row.cycle_number, row.pres) in seen_measurements) 
                                 for row in meas_chunk.itertuples(index=False)]]
        for row in meas_chunk.itertuples(index=False):
            seen_measurements.add((row.platform_number, row.cycle_number, row.pres))
        meas_chunk.to_csv(measurements_csv, mode='a', header=not os.path.exists(measurements_csv), index=False)

print("✅ Preprocessing complete!")
print(f"Metadata saved: {metadata_csv}")
print(f"Measurements saved: {measurements_csv}")
