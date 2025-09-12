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
index = faiss.read_index("argo_index.faiss")
embedding_model = SentenceTransformer("thenlper/gte-small")
print("✅ Assets loaded.\n")

# -- User Query --
query = "What ARGO floats were near -41N, 96E in 2013 and what patterns do they show?"

# -- FAISS Search --
query_embedding = embedding_model.encode([query], convert_to_numpy=True)
distances, indices = index.search(query_embedding, 5)
retrieved_rows = df.iloc[indices[0]]

# ... (previous code for loading assets and search) ...

# -- Prepare Context BETTER --
# First, drop duplicate rows based on key columns to avoid redundant info
retrieved_rows = retrieved_rows.drop_duplicates(subset=['float_id', 'latitude', 'longitude', 'datetime'])

context_lines = []
for _, row in retrieved_rows.iterrows():
    # Format the datetime to be more readable
    formatted_date = pd.to_datetime(row['datetime']).strftime('%Y-%m-%d %H:%M')
    context_lines.append(f"Float {row['float_id']} was at {row['latitude']}°N, {row['longitude']}°E on {formatted_date}.")
context = "\n".join(context_lines)

print(f"Retrieved Context:\n{context}\n")

# -- Create a STRICTER Prompt --
prompt = f"""You are an expert oceanographer. Analyze the provided ARGO float data and answer the question based solely on it. If the data does not contain enough information to answer the question, simply state that.

**Context Data:**
{context}

**Question:**
{query}

**Answer:** (be concise, one paragraph)
"""

# ... (rest of your code) ...

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