import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import google.generativeai as genai
from typing import List, Optional
from datetime import datetime

# --- API Key Configuration ---
API_KEY = os.getenv("GEMINI_API_KEY")

# --- Load Assets & Configuration ---
try:
    print("Loading FAISS index and processed metadata...")
    # Use local paths for the FAISS index and metadata
    index_path = "argo_index.faiss"
    metadata_path = "argo_metadata.csv"

    # Check if files exist
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    index = faiss.read_index(index_path)
    print("FAISS index loaded successfully")
    
    df = pd.read_csv(metadata_path)
    print("Metadata loaded successfully")
    
    # Recreate the 'text' column for consistency with the vector DB logic
    df['text'] = df.apply(lambda row: f"Float {row['float_id']} at {row['latitude']}, {row['longitude']} on {row['datetime']}", axis=1)
    
    # Use local embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("intfloat/e5-base-v2")
    print("Embedding model loaded successfully")

    # Configure the Gemini API client using the API key
    print("Initializing Gemini client...")
    genai.configure(api_key=API_KEY)
    print("Gemini client and model initialized")
    
except Exception as e:
    print(f"Error during initialization: {e}")
    raise RuntimeError(f"Failed to initialize application: {e}")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="FloatChat API",
    description="Backend for querying ARGO float data using RAG with a FAISS vector database."
)

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryPayload(BaseModel):
    query: str

class FloatProfile(BaseModel):
    id: int
    float_id: str
    latitude: float
    longitude: float
    datetime: str
    temperature: Optional[float] = None
    salinity: Optional[float] = None
    pressure: Optional[float] = None

class FloatTimeSeries(BaseModel):
    datetime: str
    temperature: Optional[float] = None
    salinity: Optional[float] = None
    pressure: Optional[float] = None

# --- API Endpoints ---

@app.post("/query")
async def handle_query(payload: QueryPayload):
    """
    Endpoint to process a user's natural language query.
    """
    if not payload.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    result = run_rag_pipeline(payload.query)
    return result

@app.get("/api/floats", response_model=List[FloatProfile])
async def get_all_floats(limit: int = 200):
    """Get a list of float profiles for the map"""
    try:
        # Convert the dataframe to the required format
        result = []
        for idx, row in df.head(limit).iterrows():
            result.append({
                "id": idx,
                "float_id": str(row.get('float_id', '')),
                "latitude": float(row.get('latitude', 0)),
                "longitude": float(row.get('longitude', 0)),
                "datetime": str(row.get('datetime', '')),
                "temperature": float(row.get('temperature', 0)) if pd.notna(row.get('temperature')) else None,
                "salinity": float(row.get('salinity', 0)) if pd.notna(row.get('salinity')) else None,
                "pressure": float(row.get('pressure', 0)) if pd.notna(row.get('pressure')) else None
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving float data: {str(e)}")

@app.get("/api/float/{float_id}", response_model=List[FloatTimeSeries])
async def get_float_timeseries(float_id: str):
    """Get all data for a specific float to plot its time series"""
    try:
        # Filter dataframe for the specific float_id
        float_data = df[df['float_id'] == float_id]
        if float_data.empty:
            raise HTTPException(status_code=404, detail=f"Float {float_id} not found")
        
        result = []
        for _, row in float_data.iterrows():
            result.append({
                "datetime": str(row.get('datetime', '')),
                "temperature": float(row.get('temperature', 0)) if pd.notna(row.get('temperature')) else None,
                "salinity": float(row.get('salinity', 0)) if pd.notna(row.get('salinity')) else None,
                "pressure": float(row.get('pressure', 0)) if pd.notna(row.get('pressure')) else None
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving time series: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the backend is running"""
    return {"status": "healthy", "message": "Backend is running"}

# --- RAG Pipeline Function ---
def run_rag_pipeline(query: str):
    """
    Executes the full RAG pipeline for a given query.
    """
    try:
        print(f"Processing query: {query}")
        start_time = time.time()
        
        # Step 1: Embed the query
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        
        # Step 2: Search the FAISS index
        k = 5  # Number of top results to retrieve
        distances, indices = index.search(query_embedding, k)
        retrieved_rows = df.iloc[indices[0]]

        # Step 3: Prepare context for the LLM
        context_lines = []
        for _, row in retrieved_rows.iterrows():
            formatted_date = pd.to_datetime(row['datetime']).strftime('%Y-%m-%d %H:%M')
            context_lines.append(f"Float {row['float_id']} at {row['latitude']}N, {row['longitude']}E on {formatted_date}. Additional context: {row['text']}.")
        
        context = "\n".join(context_lines)

        # Step 4: Construct the prompt
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

        # Step 5: Send to the LLM (Gemini)
        print("Sending request to Gemini...")
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        
        answer = response.text
        
        end_time = time.time()
        print(f"Response received successfully in {end_time - start_time:.2f} seconds")
        return {"answer": answer, "context": context}

    except Exception as e:
        error_msg = f"Gemini API error occurred: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)