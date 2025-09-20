import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

# --- Configuration ---
INPUT_CSV_FILE = 'argo_metadata1.csv'
FAISS_INDEX_FILE = 'argo_faiss.index'
METADATA_FILE = 'argo_profile_summaries.csv'
MODEL_NAME = 'intfloat/e5-base-v2'

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Load and Aggregate Data ---
logging.info(f"Loading data from {INPUT_CSV_FILE}...")
try:
    df = pd.read_csv(INPUT_CSV_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
except FileNotFoundError:
    logging.error(f"Error: The file {INPUT_CSV_FILE} was not found.")
    exit()

logging.info("Aggregating data for each float profile (cycle)...")
profile_groups = df.groupby(['float_id', 'cycle_number'])

agg_df = profile_groups.agg(
    latitude=('latitude', 'first'),
    longitude=('longitude', 'first'),
    datetime=('datetime', 'first'),
    temp_min=('temperature', 'min'),
    temp_max=('temperature', 'max'),
    salinity_mean=('salinity', 'mean')
).reset_index()

logging.info(f"Aggregated data into {len(agg_df)} unique profiles.")

# --- 2. Generate Textual Summaries ---
logging.info("Generating textual summaries for each profile...")

def create_summary_sentence(row):
    lat_dir = 'N' if row['latitude'] >= 0 else 'S'
    lon_dir = 'E' if row['longitude'] >= 0 else 'W'
    lat_val = abs(row['latitude'])
    lon_val = abs(row['longitude'])
    
    temp_info = f"temperatures from {row['temp_min']:.1f}°C to {row['temp_max']:.1f}°C" if pd.notna(row['temp_min']) else "no temperature data"
    sal_info = f"salinities around {row['salinity_mean']:.1f} PSU" if pd.notna(row['salinity_mean']) else "no salinity data"

    return (
        f"ARGO float {row['float_id']} on cycle {row['cycle_number']} reported data on "
        f"{row['datetime'].strftime('%Y-%m-%d')} at location {lat_val:.1f}°{lat_dir}, {lon_val:.1f}°{lon_dir}. "
        f"The profile measured {temp_info} and {sal_info}."
    )

agg_df['summary_text'] = agg_df.apply(create_summary_sentence, axis=1)

# --- 3. Create Vector Embeddings ---
logging.info(f"Loading sentence transformer model: '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)

logging.info("Converting text summaries to vector embeddings...")
# CRITICAL: Use the EXACT same parameters as in app_4.py
sentences = agg_df['summary_text'].tolist()
embeddings = model.encode(
    sentences, 
    show_progress_bar=True, 
    normalize_embeddings=True,  # This must match app_4.py
    convert_to_numpy=True       # This must match app_4.py
)

logging.info(f"Created {len(embeddings)} embeddings with dimension {embeddings.shape[1]}.")

# --- 4. Build and Save FAISS Index ---
logging.info("Building FAISS index...")
d = embeddings.shape[1]

# Use IndexFlatL2
index = faiss.IndexFlatL2(d)

# Add the vectors to the index - ensure they are float32
index.add(embeddings.astype('float32'))

logging.info(f"FAISS index built with {index.ntotal} total vectors.")

# Save the index to disk
faiss.write_index(index, FAISS_INDEX_FILE)
logging.info(f"✅ FAISS index saved to {FAISS_INDEX_FILE}")

# --- 5. Save Metadata ---
agg_df.to_csv(METADATA_FILE, index=False)
logging.info(f"✅ Metadata and summaries saved to {METADATA_FILE}")