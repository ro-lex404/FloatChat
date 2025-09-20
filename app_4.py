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

# API calling for ingesting raw data
from argopy import DataFetcher as ArgoDataFetcher

# useful for tracing in case of error handling
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Fast API 
app = FastAPI(
    title="FloatChat & Hybrid Data API",
    description="Backend serving map data from CSV and detailed float data from live API."
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Global Variable 
faiss_index = None
df_profile_summaries = None
df_map_data = None
embedding_model = None

# Application initialization
def initialize_application():
    global faiss_index, df_profile_summaries, df_map_data, embedding_model
    try:
        logger.info("Loading FAISS index from argo_faiss.index...")
        faiss_index = faiss.read_index("argo_faiss.index")
        
        # Loading profile summaries for CSV files with ample input data
        logger.info("Loading profile summaries from argo_profile_summaries.csv...")
        df_profile_summaries = pd.read_csv("argo_profile_summaries.csv")
        df_profile_summaries['float_id'] = df_profile_summaries['float_id'].astype(str)
        
        # Creating map data from profile summaries (for backward compatibility)
        logger.info("Creating map data from profile summaries...")
        df_map_data = df_profile_summaries.copy()
        if 'datetime' in df_map_data.columns:
            df_map_data['datetime'] = pd.to_datetime(df_map_data['datetime'])
        
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer("intfloat/e5-base-v2")
        
        API_KEY = os.getenv("GEMINI_API_KEY")
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY not set.")
        genai.configure(api_key=API_KEY)
        logger.info("Application initialized successfully with new data files.")
        
    except Exception as e:
        logger.error(f"FATAL: Error during initialization: {e}")
        raise RuntimeError(f"Failed to initialize application: {e}") from e

initialize_application()

class QueryPayload(BaseModel):
    query: str

# Data endpoints  
@app.get("/api/live/map_data", summary="Get float positions from profile summaries")
@alru_cache(maxsize=10)
async def get_live_map_data(region: str = "-180,180,-90,90"):
    logger.info("Serving latest float positions from profile summaries.")
    try:
        # Use the profile summaries data for map positions
        if df_map_data is not None and not df_map_data.empty:
            # Getting latest entry for each float
            latest_static = (df_map_data.sort_values('datetime', ascending=False)
                           .drop_duplicates('float_id')
                           .reset_index(drop=True))
            
            # Select required columns (handle different column names)
            result_cols = ['float_id', 'latitude', 'longitude', 'datetime']
            available_cols = [col for col in result_cols if col in latest_static.columns]
            
            result = latest_static[available_cols].to_dict('records')
            logger.info(f"Returning {len(result)} unique floats from profile summaries.")
            return result
        else:
            logger.warning("No map data available from profile summaries.")
            return []
    except Exception as e:
        logger.error(f"Failed to serve map data from profile summaries: {e}")
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

# RAG pipelining with ample summaries
async def run_rag_pipeline(query: str) -> Dict[str, Any]:
    try:
        start_time = time.time()
        
        # Encode the queryembeddings
        query_embedding = embedding_model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        
        # Retreive top 5 most relevant profiles
        k = 5
        distances, indices = faiss_index.search(np.array([query_embedding]).astype('float32'), k)
        
        # Get valid indices and retreive rows
        valid_indices = [idx for idx in indices[0] if idx != -1 and idx < len(df_profile_summaries)]
        retrieved_rows = df_profile_summaries.iloc[valid_indices]
        
        # Use the rich summary text for context
        context_lines = []
        for _, row in retrieved_rows.iterrows():
            if 'summary_text' in row and pd.notna(row['summary_text']):
                context_lines.append(row['summary_text'])
        
        context = "\n---\n".join(context_lines)
        
        # Check if we have any context
        if not context.strip():
            logger.warning("No relevant context found for query")
            return {
                "answer": "I couldn't find any relevant data to answer your question in the available ARGO float database.",
                "context": "",
                "retrieved_count": 0
            }

        prompt_template = f"""
        You are an expert oceanographer. Answer the user's question based *only* on the following context provided from the ARGO float database.
        If the context does not contain the answer, say that you cannot find the information in the available data.

        Context:
        {context}

        Question: {query}

        Answer:"""

        # Defining the sync function with prompt_template as parameter
        def generate_content_sync(prompt):
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                return response
            except Exception as e:
                logger.error(f"Gemini API error: {str(e)}")
                raise

        # Pass prompt_template as argument
        response = await asyncio.to_thread(generate_content_sync, prompt_template)
        answer = response.text
        
        end_time = time.time()
        logger.info(f"RAG pipeline completed in {end_time - start_time:.2f}s")
        
        return {
            "answer": answer,
            "context": context,
            "retrieved_count": len(retrieved_rows)
        }
    except Exception as e:
        error_msg = f"Error in RAG pipeline: {str(e)}"
        logger.error(error_msg, exc_info=True)  # Add exc_info for full traceback
        raise HTTPException(status_code=500, detail=error_msg) from e

# RAG pipelining with Ample summaries
@app.post("/query", summary="Process a natural language query via RAG")
async def handle_query(payload: QueryPayload):
    """Handle natural language queries."""
    if not payload.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return await run_rag_pipeline(payload.query)

# Checking if the Server is runnnign or not 
# Helpfu for error detection
@app.get("/health", summary="Health check")
async def health_check():
    return {
        "status": "healthy",
        "mode": "Hybrid (Profile summaries for map, API for details)",
        "profile_summaries_loaded": df_profile_summaries is not None and not df_profile_summaries.empty,
        "faiss_index_loaded": faiss_index is not None,
        "model_loaded": embedding_model is not None,
        "total_profiles": len(df_profile_summaries) if df_profile_summaries is not None else 0,
        "unique_floats": df_profile_summaries['float_id'].nunique() if df_profile_summaries is not None else 0
    }

# Executing the main fn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
