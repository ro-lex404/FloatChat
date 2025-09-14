import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai  # ✅ Correct import
import traceback

# --- API Key Configuration ---
try:
    API_KEY = os.environ["GOOGLE_API_KEY"]
except KeyError:
    raise RuntimeError("The 'GOOGLE_API_KEY' environment variable is not set. Please set it before running the application.")

# --- Load Assets & Configuration ---
try:
    print("Loading FAISS index and metadata...")
    index_path = "argo_index.faiss"
    metadata_path = "argo_metadata.csv"
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    index = faiss.read_index(index_path)
    print("FAISS index loaded successfully")
    
    df = pd.read_csv(metadata_path)
    print("Metadata loaded successfully")
    
    # Recreate 'text' column (ensure consistent structure)
    df['text'] = df.apply(
        lambda row: f"Float {row['float_id']} at {row['latitude']}, {row['longitude']} on {row['datetime']}", axis=1
    )
    
    # Load embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("intfloat/e5-base-v2")
    print("Embedding model loaded successfully")

    # Initialize Gemini
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    print("Gemini client and model initialized")
    
except Exception as e:
    print(f"Error during initialization: {e}")
    raise RuntimeError(f"Failed to initialize application: {e}")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="FloatChat API",
    description="Backend for querying ARGO float data using RAG with a FAISS vector database."
)

# Pydantic model for request payload
class QueryPayload(BaseModel):
    query: str

# --- Query Normalization Layer ---
def normalize_query(user_query: str) -> str:
    """
    Rewrite user query into the structured form that matches FAISS index data.
    Example:
        Input:  "Which float id was at -46N, 70E on 2013?"
        Output: "Float at -46, 70 on 2013"
    """
    try:
        prompt = f"""
        You are a query reformulation assistant. 
        Rewrite the user query into a structured form that matches the metadata format:
        "Float <float_id> at <latitude>, <longitude> on <datetime>".
        The float_id is unknown, so only rewrite location and datetime into that style.

        User query: {user_query}

        Normalized query:
        """
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        return user_query
    except Exception as e:
        print("Query normalization failed:", e)
        return user_query

# --- RAG Pipeline ---
def run_rag_pipeline(query: str):
    """
    Executes full RAG pipeline for a given query.
    """
    print(f"Processing query: {query}")
    start_time = time.time()

    # Step 0: Normalize query
    normalized_query = normalize_query(query)
    print("Normalized query:", normalized_query)
    
    # Step 1: Embed normalized query
    query_embedding = embedding_model.encode([normalized_query], convert_to_numpy=True)
    
    # Step 2: Search FAISS index
    k = 5
    print("Index dimension:", index.d)
    print("Query embedding dimension:", query_embedding.shape[1])
    distances, indices = index.search(query_embedding, k)
    retrieved_rows = df.iloc[indices[0]]

    # Step 3: Prepare context
    context_lines = []
    for _, row in retrieved_rows.iterrows():
        formatted_date = pd.to_datetime(row['datetime']).strftime('%Y-%m-%d %H:%M')
        context_lines.append(
            f"Float {row['float_id']} at {row['latitude']}N, {row['longitude']}E on {formatted_date}. "
            f"Additional context: {row['text']}."
        )
    context = "\n".join(context_lines)

    print("Context prepared for LLM:", context)

    # Step 4: Construct final prompt
    prompt_template = """
    You are an expert marine scientist. Answer user questions about ARGO floats
    using ONLY the provided context data.

    *Context Data:*
    {context}

    *Question:*
    {query}

    *Answer:* (concise, one paragraph)
    """
    prompt = prompt_template.format(context=context, query=query)

    # Step 5: Query Gemini
    print("Sending request to Gemini 2.5 Flash...")
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        print("Gemini API call failed:", e)
        traceback.print_exc()
        raise RuntimeError("Failed to get response from Gemini API")

    # Extract answer robustly
    answer = None
    if hasattr(response, "text") and response.text:
        answer = response.text
    elif hasattr(response, "candidates") and response.candidates:
        try:
            answer = response.candidates[0].content.parts[0].text
        except Exception:
            pass
    if not answer:
        print("DEBUG: Raw Gemini response:", response)
        answer = "No answer could be extracted from the Gemini response."

    end_time = time.time()
    print(f"Response received successfully in {end_time - start_time:.2f} seconds")
    return {"answer": answer, "context": context}

# --- API Endpoints ---
@app.post("/query")
async def handle_query(payload: QueryPayload):
    if not payload.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = run_rag_pipeline(payload.query)
        return result
    except Exception as e:
        error_msg = f"An error occurred in the RAG pipeline: {e}"
        print(error_msg)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)
