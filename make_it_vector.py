import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sql_to_py import load_data   # <-- make sure this merges measurements + metadata
import os

def prepare_and_save(batch_size=250):
    # 1. Load merged unique data from Postgres
    df = load_data()

    # 2. Save directory
    save_dir = r"C:\Users\aim\Desktop\copy sih\sih divy\vector_db"
    os.makedirs(save_dir, exist_ok=True)

    embeddings_file = os.path.join(save_dir, "embeddings.npy")
    metadata_file = os.path.join(save_dir, "metadata.parquet")

    # 3. Handle already processed data
    if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
        existing_embeddings = np.load(embeddings_file)
        existing_metadata = pd.read_parquet(metadata_file)

        # Drop rows already embedded before
        df = df[~df.apply(tuple, 1).isin(existing_metadata.apply(tuple, 1))]
        print(f"⏩ Skipping {len(existing_metadata)} already processed rows")

        embeddings_list = existing_embeddings.tolist()
    else:
        existing_metadata = pd.DataFrame()
        embeddings_list = []

    if df.empty:
        print("✅ No new rows to process!")
        return

    # 4. Build text content for embeddings
    df["content"] = df.apply(
        lambda row: (
            f"Platform: {row['platform_number']}, "
            f"Cycle: {row['cycle_number']}, "
            f"Project: {row.get('project_name', 'N/A')}, "
            f"PI: {row.get('pi_name', 'N/A')}, "
            f"Platform Type: {row.get('platform_type', 'N/A')}, "
            f"Direction: {row.get('direction', 'N/A')}, "
            f"Pressure: {row.get('pres', 'N/A')}, "
            f"Temperature: {row.get('temp', 'N/A')}, "
            f"Salinity: {row.get('psal', 'N/A')}, "
            f"Latitude: {row.get('latitude', 'N/A')}, "
            f"Longitude: {row.get('longitude', 'N/A')}, "
            f"Time: {row.get('time', 'N/A')}"
        ),
        axis=1,
    )

    # 5. Load embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 6. Encode in batches
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch_df = df.iloc[start:end]

        # Generate embeddings
        batch_embeddings = model.encode(
            batch_df["content"].tolist(),
            show_progress_bar=True
        )

        embeddings_list.extend(batch_embeddings.tolist())

        # Save progress
        all_embeddings = np.array(embeddings_list)
        all_metadata = pd.concat(
            [existing_metadata, df.iloc[:end]], ignore_index=True
        )

        np.save(embeddings_file, all_embeddings)
        all_metadata.to_parquet(metadata_file, index=False)

        print(f"💾 Saved batch {start}-{end} ({len(all_embeddings)} total rows)")

    print(f"✅ All embeddings processed and saved in {save_dir}")

if __name__ == "__main__":
    prepare_and_save()
