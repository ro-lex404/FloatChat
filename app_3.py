# app.py (Hybrid: Fast map from CSV, detailed charts from API)
import os
import time
import re
import logging
from typing import List, Optional, Dict, Any
import asyncio

import pandas as pd
import faiss
import numpy as np
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from async_lru import alru_cache

# --- RE-INTRODUCE ARGOPY for detailed float data ---
from argopy import DataFetcher as ArgoDataFetcher

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize FastAPI App ---
app = FastAPI(
    title="FloatChat & Hybrid Data API",
    description="Backend serving map data from CSV and detailed float data from live API."
)

# --- Add CORS Middleware (no changes) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global variables (no changes) ---
index = None
df_metadata = None
embedding_model = None

# --- Application Initialization (no changes) ---
def initialize_application():
    global index, df_metadata, embedding_model
    try:
        logger.info("Loading FAISS index and metadata from argo_metadata.csv...")
        index = faiss.read_index("argo_index.faiss")
        df_metadata = pd.read_csv("argo_metadata.csv")
        df_metadata['float_id'] = df_metadata['float_id'].astype(str)
        df_metadata['datetime'] = pd.to_datetime(df_metadata['datetime'])
        df_metadata['text'] = df_metadata.apply(
            lambda row: f"Float {row['float_id']} at {row['latitude']:.3f}, {row['longitude']:.3f} on {row['datetime']}",
            axis=1
        )
        logger.info("FAISS index and metadata loaded.")
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer("intfloat/e5-base-v2")
        logger.info("Embedding model loaded.")
        API_KEY = os.getenv("GEMINI_API_KEY")
        if not API_KEY: raise ValueError("GEMINI_API_KEY not set.")
        genai.configure(api_key=API_KEY)
        logger.info("Gemini client initialized.")
    except Exception as e:
        logger.error(f"FATAL: Error during initialization: {e}")
        raise RuntimeError(f"Failed to initialize application: {e}") from e

initialize_application()

class QueryPayload(BaseModel):
    query: str

# ===============================================================
# --- SECTION 1: DATA ENDPOINTS ---
# ===============================================================

@app.get("/api/live/map_data", summary="Get float positions from local CSV")
@alru_cache(maxsize=10)
async def get_live_map_data(region: str = "-180,180,-90,90"):
    logger.info("Serving latest float positions from static metadata file.")
    try:
        latest_static = (df_metadata.sort_values('datetime', ascending=False)
                       .drop_duplicates('float_id')
                       .reset_index(drop=True))
        result = latest_static[['float_id', 'latitude', 'longitude', 'datetime']].to_dict('records')
        logger.info(f"Returning {len(result)} unique floats from CSV.")
        return result
    except Exception as e:
        logger.error(f"Failed to serve map data from CSV: {e}")
        return []

@app.get("/api/live/float/{float_id}", summary="Get time-series for a float from LIVE API")
@alru_cache(maxsize=100)
async def get_live_float_data(float_id: str):
    logger.info(f"Fetching LIVE historical data for float {float_id}...")
    
    def fetch_data_sync(fid):
        try:
            fetcher = ArgoDataFetcher(src='erddap', cache=True, timeout=25)
            ds = fetcher.float(int(fid)).to_dataframe()
            if ds.empty:
                return None
            
            ds.rename(columns={
                'TIME': 'datetime', 'TEMP': 'temperature',
                'PSAL': 'salinity', 'PRES': 'pressure'
            }, inplace=True)
            
            chart_cols = ['datetime', 'temperature', 'salinity', 'pressure']
            existing_cols = [col for col in chart_cols if col in ds.columns]
            
            return ds[existing_cols].to_dict('records')
        except Exception as e:
            logger.error(f"Argopy fetch failed for float {fid}: {e}")
            return None

    data = await asyncio.to_thread(fetch_data_sync, float_id.strip())
    
    if data is None:
        logger.warning(f"No live data found for float {float_id}.")
        raise HTTPException(status_code=404, detail="No live data found for this float.")
    
    logger.info(f"Successfully fetched {len(data)} measurements for float {float_id} from live API.")
    return data

# ===============================================================
# --- FIX: Added full implementation for the RAG pipeline ---
# ===============================================================

def standardize_query(user_query: str) -> str:
    """Standardize user query for better search results."""
    try:
        query_lower = user_query.lower()
        float_ids = re.findall(r'float\s+(\d+)', query_lower)
        if float_ids:
            return f"Float {float_ids[0]} data"
        patterns = {'temp': 'temperature', 'sal': 'salinity', 'pres': 'pressure', 'hawaii': 'Hawaii region', 'pacific': 'Pacific Ocean'}
        for pattern, replacement in patterns.items():
            if re.search(pattern, query_lower):
                return re.sub(pattern, replacement, query_lower)
        return user_query
    except Exception as e:
        logger.warning(f"Query standardization failed: {e}")
        return user_query

async def run_rag_pipeline(query: str) -> Dict[str, Any]:
    """Run the RAG pipeline using local data and AI."""
    try:
        start_time = time.time()
        standardized_query = standardize_query(query)
        query_embedding = embedding_model.encode([f"query: {standardized_query}"], convert_to_numpy=True)
        
        k = 10
        distances, indices = index.search(query_embedding, k)
        
        valid_indices = [idx for idx in indices[0] if idx != -1 and idx < len(df_metadata)]
        retrieved_rows = df_metadata.iloc[valid_indices]
        
        context_lines = [row['text'] for _, row in retrieved_rows.iterrows()]
        context = "\n".join(context_lines[:5])

        prompt_template = f"""
        You are an oceanography expert. Answer concisely using the provided context from the database.
        Context (float locations and times):
        {context}
        Question: {query}
        Answer briefly:"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = await asyncio.to_thread(model.generate_content, prompt_template)
        answer = response.text
        
        end_time = time.time()
        logger.info(f"RAG pipeline completed in {end_time - start_time:.2f}s")
        
        return {"answer": answer, "context": context}
    except Exception as e:
        error_msg = f"Error in RAG pipeline: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg) from e

@app.post("/query", summary="Process a natural language query via RAG")
async def handle_query(payload: QueryPayload):
    """Handle natural language queries."""
    if not payload.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return await run_rag_pipeline(payload.query)

# ===============================================================
# --- FIX: Added full implementation for the health check ---
# ===============================================================

@app.get("/health", summary="Health check")
async def health_check():
    """Provides a detailed health check of the API components."""
    return {
        "status": "healthy",
        "mode": "Hybrid (CSV for map, API for details)",
        "metadata_loaded": df_metadata is not None and not df_metadata.empty,
        "index_loaded": index is not None,
        "model_loaded": embedding_model is not None
    }

# --- Main execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)