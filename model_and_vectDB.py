from openai import OpenAI
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load CSV and FAISS index
df = pd.read_csv("argo_metadata.csv")
index = faiss.read_index("argo_index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: User query
query = "What ARGO floats were near 10N, 70E in 2013 and what patterns do they show?"

# Step 2: Convert query to embedding
query_embedding = model.encode([query], convert_to_numpy=True)

# Step 3: Retrieve top matches from FAISS
distances, indices = index.search(query_embedding, 5)
retrieved_rows = df.iloc[indices[0]]

# Step 4: Prepare context for LLM
context = "\n".join(retrieved_rows["text"].tolist())

# Step 5: Send to LLM (Example: OpenAI GPT)
client = OpenAI(api_key="YOUR_API_KEY")
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert oceanographer."},
        {"role": "user", "content": f"Here is ARGO float data:\n{context}\n\nAnswer the question: {query}"}
    ]
)

# Step 6: Print LLM's answer
print(response.choices[0].message["content"])
