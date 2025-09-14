import os
import time
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- API Key Configuration ---
try:
    API_KEY = os.environ["GOOGLE_API_KEY"]
except KeyError:
    # A custom exception to provide a clearer error message
    raise RuntimeError("The 'GOOGLE_API_KEY' environment variable is not set. Please set it before running the application.")

# --- Global Client & Model Initialization ---
# Initialize the Gemini API client and model once at startup
try:
    genai.configure(api_key=API_KEY)
    client = genai.Client()
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("Gemini client and model initialized successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to initialize Gemini API client: {e}")

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
    
    df['text'] = df.apply(
        lambda row: f"Float {row['float_id']} at {row['latitude']}, {row['longitude']} on {row['datetime']}", axis=1)
    
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("intfloat/e5-base-v2")
    print("Embedding model loaded successfully")

except Exception as e:
    print(f"Error during initialization: {e}")
    raise RuntimeError(f"Failed to initialize application assets: {e}")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="FloatChat API",
    description="Backend for querying ARGO float data using RAG with a FAISS vector database."
)

class QueryPayload(BaseModel):
    query: str

def run_rag_pipeline(query: str):
    """
    Executes the full RAG pipeline for a given query.
    """
    print(f"Processing query: {query}")
    start_time = time.time()
    
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    
    k = 5
    distances, indices = index.search(query_embedding, k)
    retrieved_rows = df.iloc[indices[0]]

    context_lines = []
    for _, row in retrieved_rows.iterrows():
        formatted_date = pd.to_datetime(row['datetime']).strftime('%Y-%m-%d %H:%M')
        context_lines.append(f"Float {row['float_id']} at {row['latitude']}N, {row['longitude']}E on {formatted_date}. Additional context: {row['text']}.")
    
    context = "\n".join(context_lines)

    prompt_template = """
    You are an expert marine scientist. Your task is to answer user questions about ARGO oceanographic floats using only the provided context data.
    If the data does not contain enough information to answer the question, simply state that you cannot answer based on the provided data.

    *Context Data:*
    {context}

    *Question:*
    {query}

    *Answer:* (be concise, one paragraph)
    """
    prompt = prompt_template.format(context=context, query=query)

    print("Sending request to Gemini 2.5 Flash...")
    try:
        response = model.generate_content(prompt)
    except Exception as e:
        print("DEBUG: Gemini API call failed:", e)
        traceback.print_exc()
        raise RuntimeError("Failed to get response from Gemini API")

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
