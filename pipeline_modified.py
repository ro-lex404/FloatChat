import requests
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import time
from requests.exceptions import ConnectionError

# -- Load Assets --
print("Loading CSV, FAISS index, and embedding model...")
df = pd.read_csv("argo_metadata.csv")

# === CRITICAL FIX: RECREATE THE 'text' COLUMN ===
print("Recreating 'text' column for consistency...")
df['text'] = df.apply(lambda row: f"Float {row['float_id']} at {row['latitude']}, {row['longitude']} on {row['datetime']}", axis=1)
# ================================================

index = faiss.read_index("argo_index.faiss")
embedding_model = SentenceTransformer("intfloat/e5-base-v2")
print("✅ Assets loaded.\n")

# -- User Query --
query = "What ARGO floats were near -41N, 96E in 2013 and what patterns do they show?"
target_lat, target_lon = -41, 96

# -- HYBRID SEARCH: FILTER FIRST, THEN RANK --
print("Filtering data to relevant area and year...")
df['datetime'] = pd.to_datetime(df['datetime'])
geo_filtered_df = df[
    (df['datetime'].dt.year == 2013) &
    (df['latitude'].between(target_lat - 15, target_lat + 15)) &
    (df['longitude'].between(target_lon - 15, target_lon + 15))
].copy()

if len(geo_filtered_df) == 0:
    print("❌ No data found in the general area for 2013. Cannot perform search.")
    exit()

# Now this line will work because geo_filtered_df has the 'text' column
print("Performing vector search on filtered data...")
filtered_embeddings = embedding_model.encode(geo_filtered_df['text'].tolist(), convert_to_numpy=True)
filtered_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
filtered_index.add(filtered_embeddings)

# Search the filtered index
query_embedding = embedding_model.encode([query], convert_to_numpy=True)
distances, filtered_indices = filtered_index.search(query_embedding, k=5) # Get top 5 from filtered data

# Retrieve the actual rows from the filtered DataFrame
retrieved_rows = geo_filtered_df.iloc[filtered_indices[0]]

# -- Prepare Context --
retrieved_rows = retrieved_rows.drop_duplicates(subset=['float_id', 'latitude', 'longitude', 'datetime'])
context_lines = []
for _, row in retrieved_rows.iterrows():
    formatted_date = row['datetime'].strftime('%Y-%m-%d %H:%M')
    context_lines.append(f"Float {row['float_id']} was at {row['latitude']}°N, {row['longitude']}°E on {formatted_date}.")
context = "\n".join(context_lines)

print(f"Retrieved Context (after geographical filter):\n{context}\n")

# -- Create a STRICTER Prompt --
prompt = f"""You are an expert oceanographer. Analyze the provided ARGO float data and answer the question based solely on it. If the data does not contain enough information to answer the question, simply state that.

**Context Data:**
{context}

**Question:**
{query}

**Answer:** (be concise, one paragraph)
"""

# -- Send to Ollama with Retry Logic --
url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3.2:1b",
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.1,
        "num_predict": 250
    }
}

max_retries = 3
retry_delay = 5  # seconds

for attempt in range(1, max_retries + 1):
    try:
        print(f"Attempt {attempt}: Sending request to Ollama...")
        response = requests.post(url, json=payload, timeout=60)  # 60 second timeout
        response.raise_for_status()  # Raises an exception for 4xx/5xx status codes

        # If successful, parse and print the response
        full_response = response.json()
        answer = full_response['response']
        print("\n🤖 ANSWER:")
        print(answer)
        break  # Exit the retry loop on success

    except ConnectionError:
        print(f"   Server not ready. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
    except requests.exceptions.HTTPError as e:
        print(f"   HTTP Error: {e}")
        break
    except Exception as e:
        print(f"   An unexpected error occurred: {e}")
        break
else:
    # This block runs if the loop never 'break's (i.e., all retries failed)
    print("❌ All connection attempts failed. Please ensure 'ollama serve' is running.")