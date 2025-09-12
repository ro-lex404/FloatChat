import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load your CSV
df = pd.read_csv("argo_metadata.csv")

# Create a text column for embedding
df["text"] = df.apply(lambda row: f"Float {row['float_id']} at {row['latitude']}, {row['longitude']} on {row['datetime']}", axis=1)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True)

# Save embeddings for later use
np.save("argo_embeddings.npy", embeddings)


#FAISS DB
# Create FAISS index
dimension = embeddings.shape[1]  # embedding size
index = faiss.IndexFlatL2(dimension)  # L2 distance

# Add embeddings to index
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "argo_index.faiss")
print("✅ FAISS index created and saved.")

# Load index
index = faiss.read_index("argo_index.faiss")

#Querying
# Example query
query = "Find floats at 10N, 60E on 2013"
query_embedding = model.encode([query], convert_to_numpy=True)

# Search
k = 5  # top results
distances, indices = index.search(query_embedding, k)

# Show results
print(df.iloc[indices[0]])