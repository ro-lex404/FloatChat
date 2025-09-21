#IMPORTANT: ENSURE THAT YOU HAVE A VALID GEMINI_API_KEY SET IN YOUR ENVIRONMENT VARIABLES
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
#especially imp in our case as there are many places where things can go wrong
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FloatChat & Hybrid Data API",
    description="Backend serving map data from CSV and detailed float data from multiple API sources."
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],#streamlit on 8501
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Global Variable 
faiss_index = None
df_profile_summaries = None
df_map_data = None
embedding_model = None

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
        #use intfloat/e5-base-v2 as it is small but good enough for our use case
        #flexibility is most essential here
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
        # map is loaded from static points from the csv file
        if df_map_data is not None and not df_map_data.empty:
            # Getting latest entry for each float
            latest_static = (df_map_data.sort_values('datetime', ascending=False)
                           .drop_duplicates('float_id')
                           .reset_index(drop=True))
            
            # Select and handling different column names
            result_cols = ['float_id', 'latitude', 'longitude', 'datetime']
            available_cols = [col for col in result_cols if col in latest_static.columns]
            
            result = latest_static[available_cols].to_dict('records')
            #basically returns total number of unique floats
            logger.info(f"Returning {len(result)} unique floats from profile summaries.")
            return result
        else:
            logger.warning("No map data available from profile summaries.")
            return []
    except Exception as e:
        logger.error(f"Failed to serve map data from profile summaries: {e}")
        return []
    
#start of sequentially accessing each data source
def fetch_from_source(fetcher_source, numeric_fid):
    """Try to fetch data from a specific source"""
    try:
        logger.info(f"Trying to fetch data from {fetcher_source} for float {numeric_fid}")
        fetcher = ArgoDataFetcher(src=fetcher_source, cache=True, timeout=30)
        ds = fetcher.float(numeric_fid).to_dataframe()
        
        if ds is None or ds.empty:
            logger.warning(f"No data returned from {fetcher_source} for float {numeric_fid}")
            return None
        
        logger.info(f"Successfully fetched data from {fetcher_source} for float {numeric_fid}: {ds.shape}")
        return ds
    except Exception as e:
        logger.warning(f"Failed to fetch from {fetcher_source}: {str(e)}")
        return None

# FALLBACK IN CASE ANY SOURCE IS DOWN 
@app.get("/api/live/float/{float_id}", summary="Get time-series for a float from multiple API sources")
@alru_cache(maxsize=100)
async def get_live_float_data(float_id: str):
    logger.info(f"Fetching historical data for float {float_id} from multiple sources...")
    
    def fetch_data_sync(fid):
        try:
            clean_fid = str(fid).strip()
            logger.info(f"Processing request for clean float ID: {clean_fid}")
            
            try:
                numeric_fid = int(clean_fid)
            except ValueError:
                logger.error(f"Invalid float ID format: {clean_fid} - must be numeric")
                return None
            
            # try multiple data sources to try to minimize the downtime.
            sources = ['gdac', 'erddap', 'argovis']
            ds = None
            
            for source in sources:
                ds = fetch_from_source(source, numeric_fid)
                if ds is not None and not ds.empty:
                    logger.info(f"Using data from {source} for float {numeric_fid}")
                    break
            
            if ds is None or ds.empty:
                logger.warning(f"No data returned from any source for float {numeric_fid}")
                return None
            
            logger.info(f"Raw data shape for float {numeric_fid}: {ds.shape}")
            logger.info(f"Available columns: {list(ds.columns)}")
            
            # Map the actual columns of the data to some standard names
            column_mapping = {
                'datetime': 'datetime', 
                'temp': 'temperature',
                'psal': 'salinity', 
                'pres': 'pressure',
                'TEMP': 'temperature',
                'PSAL': 'salinity',
                'PRES': 'pressure',
                'TIME': 'datetime',
                'JULD': 'datetime',
                'LATITUDE': 'latitude',
                'LONGITUDE': 'longitude'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in ds.columns:
                    ds.rename(columns={old_name: new_name}, inplace=True)
            
            #julian datetimes are a standard in scientific applications like oceanography
            #convert to standard datetime
            if 'datetime' not in ds.columns and 'JULD' in ds.columns:
                try:
                    ds['datetime'] = pd.to_datetime(ds['JULD'], unit='D', origin='julian')
                except Exception as e:
                    logger.warning(f"Could not convert JULD to datetime: {e}")
            
            ds = ds.replace({np.nan: None})
            
            # Columns we need for plotting
            chart_cols = ['datetime', 'temperature', 'salinity', 'pressure', 'latitude', 'longitude']
            existing_cols = [col for col in chart_cols if col in ds.columns]
            
            if not existing_cols:
                logger.warning(f"No usable columns found for float {numeric_fid}. Available: {list(ds.columns)}")
                return None
            
            logger.info(f"Using columns for float {numeric_fid}: {existing_cols}")
            
            records = ds[existing_cols].to_dict('records')
            
            for record in records:
                # Handle datetime conversion
                #Different sources have different datetime formats,
                #
                if 'datetime' in record and record['datetime'] is not None:
                    try:
                        if hasattr(record['datetime'], 'isoformat'):
                            record['datetime'] = record['datetime'].isoformat()
                        elif isinstance(record['datetime'], str):
                            # Ensure proper datetime format
                            dt = pd.to_datetime(record['datetime'])
                            record['datetime'] = dt.isoformat()
                        elif isinstance(record['datetime'], (int, float)):
                            # Handle Julian day format
                            dt = pd.to_datetime(record['datetime'], unit='D', origin='julian')
                            record['datetime'] = dt.isoformat()
                    except Exception as dt_error:
                        logger.warning(f"Datetime conversion error: {dt_error}")
                        record['datetime'] = None
                
                # Ensure numeric fields are properly formatted
                for field in ['temperature', 'salinity', 'pressure', 'latitude', 'longitude']:
                    if field in record and record[field] is not None:
                        try:
                            if pd.isna(record[field]):
                                record[field] = None
                            else:
                                record[field] = float(record[field])
                        except (ValueError, TypeError):
                            record[field] = None
            
            # Remove records with no useful data
            valid_records = []
            for record in records:
                has_data = any(
                    record.get(field) is not None 
                    for field in ['temperature', 'salinity', 'pressure']
                )
                if has_data:
                    valid_records.append(record)
            
            logger.info(f"Returning {len(valid_records)} valid records for float {numeric_fid}")
            return valid_records
            
        except Exception as e:
            # All data sources(gdac, erddap, argovis) are down
            logger.error(f"All data source fetch failed for float {fid}: {str(e)}", exc_info=True)
            return None

    # Execute the fetch in a thread
    data = await asyncio.to_thread(fetch_data_sync, float_id.strip())
    
    if data is None:
        logger.warning(f"No live data found for float {float_id}. Trying fallback to profile summaries.")
        
        # Fallback to profile summaries data
        try:
            if df_profile_summaries is not None and not df_profile_summaries.empty:
                float_data = df_profile_summaries[df_profile_summaries['float_id'] == float_id]
                if not float_data.empty:
                    # Create mock time-series data from profile summaries
                    records = []
                    for _, row in float_data.iterrows():
                        record = {
                            'datetime': row.get('datetime', None),
                            'temperature': row.get('temperature', None),
                            'salinity': row.get('salinity', None),
                            'pressure': row.get('pressure', None),
                            'latitude': row.get('latitude', None),
                            'longitude': row.get('longitude', None),
                            'source': 'fallback'
                        }
                        # Handle datetime serialization
                        if record['datetime'] is not None and hasattr(record['datetime'], 'isoformat'):
                            record['datetime'] = record['datetime'].isoformat()
                        records.append(record)
                    
                    logger.info(f"Returning {len(records)} fallback records for float {float_id}")
                    return records
        except Exception as fallback_error:
            logger.error(f"Fallback data also failed: {fallback_error}")
    
    if not data:
        logger.warning(f"No data available for float {float_id} from any source")
        return []
    
    logger.info(f"Successfully fetched {len(data)} measurements for float {float_id}")
    return data

# ENDPOINTS FOR DATA SOURCES

#ADDED PURELY TO SEE WHICH DATA SOURCES ARE WORKING, FREQUENT OUTAGE IN ERDDAP
@app.get("/api/status", summary="Check status of all data sources")
async def get_api_status():
    """Check status of all available data sources"""
    source_status = {}
    sources = ['gdac', 'erddap', 'argovis']
    
    for source in sources:
        try:
            # Test with a known float ID
            test_float_id = "1901512"  # Example float ID
            fetcher = ArgoDataFetcher(src=source, cache=False, timeout=10)
            test_data = fetcher.float(int(test_float_id)).to_dataframe()
            
            source_status[source] = {
                "status": "online",
                "available": test_data is not None and not test_data.empty,
                "test_float_data": test_data is not None and not test_data.empty
            }
        except Exception as e:
            source_status[source] = {
                "status": "offline",
                "available": False,
                "error": str(e)[:100]  
            }
    
    return {
        "status": "online" if any(s["status"] == "online" for s in source_status.values()) else "offline",
        "sources": source_status,
        "profile_summaries_available": df_profile_summaries is not None and not df_profile_summaries.empty,
        "total_profile_records": len(df_profile_summaries) if df_profile_summaries is not None else 0
    }

# Add a debug endpoint to check which floats are available from profile summaries
#THIS DOES NOT TAKE INTO ACCOUNT SERVER OUTAGES, JUST SHOWS WHAT FLOAT IDS ARE IN THE CSV
@app.get("/api/debug/available_floats", summary="Debug endpoint to check available float IDs")
async def get_available_floats():
    """Debug endpoint to see what float IDs are available in our data"""
    try:
        if df_profile_summaries is not None and not df_profile_summaries.empty:
            unique_floats = df_profile_summaries['float_id'].unique()
            sample_floats = sorted(unique_floats)[:20]  # First 20 for debugging
            
            return {
                "total_floats": len(unique_floats),
                "sample_float_ids": sample_floats,
                "data_columns": list(df_profile_summaries.columns) if df_profile_summaries is not None else []
            }
        else:
            return {"error": "No profile summaries loaded"}
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return {"error": str(e)}

async def standardize_query_with_gemini(query: str) -> str:
    """
    Use Gemini AI to standardize user queries into a structured format that matches the profile summary structure.
    This provides more intelligent and context-aware standardization.
    """
    try:
        # gemini standardizes query
        standardization_prompt = f"""
        You are an expert query standardization system for ARGO float oceanographic data.
        
        Convert the user's natural language query into a standardized format that would match
        the structure of ARGO float profile summaries. The summaries follow this pattern:
        
        "ARGO float [FLOAT_ID] on cycle [CYCLE] reported data on [DATE] at location [LAT]°[DIR], [LON]°[DIR]. 
        The profile measured temperatures from [MIN_TEMP]°C to [MAX_TEMP]°C and salinities around [SALINITY] PSU."
        
        Examples of standardization:
        - "show me temperature for float 1901302" → "ARGO float 1901302 temperature data"
        - "what's the salinity in bay of bengal?" → "ARGO floats in Bay of Bengal salinity"
        - "compare float 123 and 456" → "comparison between ARGO float 123 and ARGO float 456"
        - "data from august 2020" → "ARGO float data from August 2020"
        - "data of float 123 and 456" → "ARGO float 123 and ARGO float 456 data"
        
        Key rules:
        1. Always include "ARGO float" or "ARGO floats" in the standardized query
        2. Preserve all specific parameters (temperature, salinity, pressure, location, time)
        3. Remove conversational filler words ("show me", "what is", etc.)
        4. Maintain the original intent but make it search-friendly
        5. If location is mentioned, include it in standardized form
        6. If time period is mentioned, include it properly formatted
        
        User query: "{query}"
        
        Standardized query:"""
        
        def generate_standardization_sync(prompt):
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini standardization error: {str(e)}")
                return query  # Fallback to original query
        
        # Execute the standardization in a thread
        standardized_query = await asyncio.to_thread(generate_standardization_sync, standardization_prompt)
        
        # Basic cleanup to ensure consistency
        standardized_query = re.sub(r'\s+', ' ', standardized_query).strip()
        
        logger.info(f"Query standardized by Gemini: '{query}' -> '{standardized_query}'")
        return standardized_query
        
    except Exception as e:
        logger.warning(f"Gemini query standardization failed: {e}, using original query")
        return query

# RAG pipelining with ample summaries, uses standardization from above
async def run_rag_pipeline(query: str) -> Dict[str, Any]:
    try:
        start_time = time.time()
        
        standardized_query = await standardize_query_with_gemini(query)
        
        # encode to embeddings using intfloat/e5-base-v2
        query_embedding = embedding_model.encode(standardized_query, normalize_embeddings=True, convert_to_numpy=True)
        
        # Retrieve all top 10 most relevant profiles(more context)
        k = 10
        distances, indices = faiss_index.search(np.array([query_embedding]).astype('float32'), k)
        
        # Get valid indices and retrieve rows
        valid_indices = [idx for idx in indices[0] if idx != -1 and idx < len(df_profile_summaries)]
        retrieved_rows = df_profile_summaries.iloc[valid_indices]
        
        # Use summary_text from profile summaries
        context_lines = []
        for _, row in retrieved_rows.iterrows():
            if 'summary_text' in row and pd.notna(row['summary_text']):
                context_lines.append(row['summary_text'])
        
        context = "\n---\n".join(context_lines)
        
        # any context is available or not
        if not context.strip():
            logger.warning("No relevant context found for query")
            return {
                "answer": "I couldn't find any relevant data to answer your question in the available ARGO float database.",
                "context": "",
                "retrieved_count": 0,
                "original_query": query,
                "standardized_query": standardized_query
            }

        prompt_template = f"""
        You are an expert oceanographer. Answer the user's question based *only* on the following context provided from the ARGO float database.
        If the context does not contain the answer, say that you cannot find the information in the available data.

        Original user question: {query}
        Standardized search query: {standardized_query}
        
        Context from ARGO float database:
        {context}

        Question: {query}

        Provide a comprehensive answer using only the information from the context. If the context contains
        multiple relevant profiles, synthesize the information into a coherent response.

        Answer:"""

        # define the sync function with prompt_template parameter
        def generate_content_sync(prompt):
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(prompt)
                return response
            except Exception as e:
                logger.error(f"Gemini API error: {str(e)}")
                raise

        # pass prompt_template 
        response = await asyncio.to_thread(generate_content_sync, prompt_template)
        answer = response.text
        
        end_time = time.time()
        logger.info(f"RAG pipeline completed in {end_time - start_time:.2f}s")
        
        return {
            "answer": answer,
            "context": context,
            "retrieved_count": len(retrieved_rows),
            "original_query": query,
            "standardized_query": standardized_query
        }
    except Exception as e:
        error_msg = f"Error in RAG pipeline: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg) from e

#queryresponse includes things like original query, standardized query, context used etc
class QueryResponse(BaseModel):
    answer: str
    context: str
    retrieved_count: int
    original_query: str
    standardized_query: str

# RAG endpoint - update to use the new response model
@app.post("/query", summary="Process a natural language query via RAG", response_model=QueryResponse)
async def handle_query(payload: QueryPayload):
    """Handle natural language queries."""
    if not payload.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    result = await run_rag_pipeline(payload.query)
    return result

# Health endpoint to check whichever data sources are working
#ADDED TO SHOW IF PROFILE SUMMARIES ARE LOADED, FAISS INDEX AND EMBEDDING MODEL
@app.get("/health", summary="Health check")
#runs on localhost:8000/health
async def health_check():
    try:
        # Get status of all sources
        status_response = await get_api_status()
        
        return {
            "status": "healthy",
            "mode": "Hybrid (Profile summaries for map, API for details)",
            "profile_summaries_loaded": df_profile_summaries is not None and not df_profile_summaries.empty,
            "faiss_index_loaded": faiss_index is not None,
            "model_loaded": embedding_model is not None,
            "total_profiles": len(df_profile_summaries) if df_profile_summaries is not None else 0,
            "unique_floats": df_profile_summaries['float_id'].nunique() if df_profile_summaries is not None else 0,
            "data_sources": status_response["sources"],
            "sample_float_ids": df_profile_summaries['float_id'].head(10).tolist() if df_profile_summaries is not None else []
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
#runs on localhost:8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)