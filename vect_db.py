import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load metadata
metadata_path = "argo_metadata.csv"
df = pd.read_csv(metadata_path)

# Recreate the 'text' column
df['text'] = df.apply(lambda row: f"Float {row['float_id']} at {row['latitude']}, {row['longitude']} on {row['datetime']}", axis=1)

# Load embedding model
embedding_model = SentenceTransformer("intfloat/e5-base-v2")

# Encode all rows
print("Encoding metadata with embedding model...")
embeddings = embedding_model.encode(df["text"].tolist(), convert_to_numpy=True, show_progress_bar=True)

# Ensure float32 for FAISS
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
d = embeddings.shape[1]  # should be 768
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Save index
faiss.write_index(index, "argo_index.faiss")
print(f"FAISS index rebuilt with dimension {d} and {index.ntotal} vectors")