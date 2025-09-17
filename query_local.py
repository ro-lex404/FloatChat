import re
import pandas as pd
import spacy
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import dateparser
import os
import numpy as np

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
    print("Warning: spaCy model not found. Using fallback parser.")

# Configuration - update these paths to match your actual file locations
DATA_DIR = "data"  # Directory where your CSV files are stored
METADATA_FILE = os.path.join(DATA_DIR, "argo_metadata.csv")
PROFILES_FILE = os.path.join(DATA_DIR, "argo.csv")

def load_argo_data():
    """Load ARGO data from CSV files and merge them"""
    try:
        # Load metadata
        metadata_df = pd.read_csv(METADATA_FILE)
        print(f"Loaded metadata: {len(metadata_df)} records")
        
        # Load profiles
        profiles_df = pd.read_csv(PROFILES_FILE)
        print(f"Loaded profiles: {len(profiles_df)} records")
        
        # Merge the dataframes - adjust the merge key based on your actual column names
        merge_columns = []
        for col in ['platform_number', 'float_id', 'wmo_id', 'cycle']:
            if col in metadata_df.columns and col in profiles_df.columns:
                merge_columns.append(col)
                break
        
        if not merge_columns:
            print("Warning: No common columns found for merging. Using metadata only.")
            return metadata_df
        
        # Merge the data
        merged_df = pd.merge(metadata_df, profiles_df, on=merge_columns[0], how='inner')
        print(f"Merged data: {len(merged_df)} records")
        
        return merged_df
        
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        print("Falling back to mock data")
        return get_mock_argo_data()
    except Exception as e:
        print(f"Error processing CSV files: {e}")
        return get_mock_argo_data()

def get_mock_argo_data():
    """Fallback mock data if CSV files are not available"""
    print("Using mock data for demonstration")
    data = {
        'platform_number': [290, 290, 290, 291, 291, 292, 292, 293, 294, 295],
        'date': [
            '2020-01-15', '2020-01-16', '2020-01-17', 
            '2020-01-10', '2020-01-11', '2020-02-01', 
            '2020-02-02', '2020-02-03', '2020-03-10', '2020-03-15'
        ],
        'latitude': [13.08, 13.10, 13.12, 12.98, 12.96, 13.20, 13.22, 13.24, 19.08, 18.92],
        'longitude': [80.28, 80.30, 80.32, 80.18, 80.16, 80.40, 80.42, 80.44, 72.88, 72.82],
        'temp': [28.5, 28.7, 28.6, 29.1, 29.0, 28.2, 28.3, 28.1, 26.5, 26.8],
        'salinity': [34.5, 34.6, 34.5, 34.7, 34.6, 34.3, 34.4, 34.2, 35.1, 35.2],
        'pressure': [10.2, 15.5, 20.1, 9.8, 14.7, 11.2, 16.8, 21.5, 8.5, 12.3],
        'region': ['Bay of Bengal', 'Bay of Bengal', 'Bay of Bengal', 
                  'Bay of Bengal', 'Bay of Bengal', 'Bay of Bengal',
                  'Bay of Bengal', 'Bay of Bengal', 'Arabian Sea', 'Arabian Sea']
    }
    return pd.DataFrame(data)

def extract_entities_with_spacy(query: str) -> Dict[str, Any]:
    """Use spaCy to extract entities from natural language query"""
    if not nlp:
        return {}
    
    doc = nlp(query)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"]:  # Location entities
            entities["location"] = ent.text
        elif ent.label_ == "DATE":
            entities["date"] = ent.text
        elif ent.label_ == "CARDINAL":  # Numbers
            entities["limit"] = int(ent.text)
        elif ent.label_ == "QUANTITY":
            if any(word in ent.text for word in ['degree', 'celsius', 'temp']):
                entities["parameter"] = "temperature"
    
    return entities

def parse_user_query(raw_query: str) -> Dict[str, Any]:
    """More flexible natural language query parsing"""
    query_lower = raw_query.lower()
    
    # Extract using spaCy
    nlp_entities = extract_entities_with_spacy(raw_query)
    
    # Extract parameter with very flexible matching
    parameter = None
    if any(word in query_lower for word in ['salinity', 'salt', 'saline']):
        parameter = 'salinity'
    elif any(word in query_lower for word in ['temperature', 'temp', 'heat', 'warm', 'cold', 'degree', 'celsius']):
        parameter = 'temp'
    elif any(word in query_lower for word in ['pressure', 'depth', 'deep', 'shallow', 'barometric']):
        parameter = 'pressure'
    elif any(word in query_lower for word in ['data', 'measurement', 'reading', 'value']):
        parameter = 'general'  # General data request
    
    # Extract location with very flexible matching
    location = nlp_entities.get("location")
    if not location:
        # Match any location mention
        location_patterns = [
            r'(?:near|around|close to|in|at|from)\s+([\w\s]+)',
            r'(\b(?:atlantic|pacific|indian|arctic|southern)\s+ocean\b)',
            r'(\b(?:bay of bengal|arabian sea|red sea|mediterranean|caribbean)\b)',
            r'(\b(?:chennai|mumbai|goa|kolkata|maldives|sri lanka|india|australia)\b)',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                break
    
    # Extract date range with very flexible parsing
    start_date, end_date = None, None
    
    # Try to parse any date mention
    if 'date' in nlp_entities:
        parsed_date = dateparser.parse(nlp_entities['date'])
        if parsed_date:
            start_date = parsed_date.strftime('%Y-%m-01')
            end_date = (parsed_date + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Look for relative time expressions
    time_keywords = {
        'last year': (datetime.now() - timedelta(days=365), datetime.now()),
        'last month': (datetime.now() - timedelta(days=30), datetime.now()),
        'last week': (datetime.now() - timedelta(days=7), datetime.now()),
        'recent': (datetime.now() - timedelta(days=90), datetime.now()),
        'recently': (datetime.now() - timedelta(days=90), datetime.now()),
    }
    
    for keyword, (start, end) in time_keywords.items():
        if keyword in query_lower and not start_date:
            start_date = start.strftime('%Y-%m-%d')
            end_date = end.strftime('%Y-%m-%d')
            break
    
    # Default to broader time range if no specific date
    if not start_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')  # 3 years
    
    # Extract result limit
    limit = nlp_entities.get("limit", 20)  # Default to 20 results
    limit_pattern = r'(?:show|display|get|find)\s+(\d+)\s+(?:results|records|points|measurements)'
    limit_match = re.search(limit_pattern, query_lower, re.IGNORECASE)
    if limit_match:
        limit = int(limit_match.group(1))
    
    return {
        "parameter": parameter,
        "location": location,
        "start_date": start_date,
        "end_date": end_date,
        "limit": min(limit, 100),  # Cap at 100 results
        "raw_query": raw_query
    }

def safe_sample(df, n, replace=True):
    """Safely sample from a DataFrame without causing size errors"""
    if len(df) == 0:
        return df
    n_samples = min(n, len(df))
    if replace:
        return df.sample(n=n_samples, replace=True)
    else:
        return df.sample(n=n_samples, replace=False)

def query_pipeline(user_query: str, location: Optional[str], 
                  start_date: str, end_date: str, top_k: int = 20) -> pd.DataFrame:
    """Query pipeline that uses real ARGO data from CSV files"""
    # Load the actual ARGO data
    df = load_argo_data()
    
    # Convert date columns if needed - adjust based on your actual column names
    date_columns = []
    for col in ['date', 'time', 'timestamp', 'datetime', 'measurement_date']:
        if col in df.columns:
            date_columns.append(col)
            break
    
    if date_columns:
        try:
            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]])
            # Filter by date range
            df = df[(df[date_columns[0]] >= pd.to_datetime(start_date)) & 
                    (df[date_columns[0]] <= pd.to_datetime(end_date))]
        except Exception as e:
            print(f"Error processing dates: {e}")
            # Continue without date filtering if there's an error
    
    # Flexible location matching based on actual column names
    if location:
        location_lower = location.lower()
        
        # Check for location columns in your data
        location_columns = []
        for col in ['region', 'location', 'ocean', 'sea', 'latitude', 'longitude', 'geo_region']:
            if col in df.columns:
                location_columns.append(col)
        
        if location_columns:
            # For numeric columns (lat/lon), do approximate matching
            if 'latitude' in location_columns and 'longitude' in location_columns:
                # Simple geographic approximation - you might want to enhance this
                if 'chennai' in location_lower:
                    df = df[
                        (df['latitude'] > 12.5) & (df['latitude'] < 13.5) &
                        (df['longitude'] > 80.0) & (df['longitude'] < 81.0)
                    ]
                elif 'mumbai' in location_lower:
                    df = df[
                        (df['latitude'] > 18.5) & (df['latitude'] < 19.5) &
                        (df['longitude'] > 72.5) & (df['longitude'] < 73.5)
                    ]
            else:
                # For text columns, do text matching
                for col in location_columns:
                    if col in ['region', 'location', 'ocean', 'sea']:
                        try:
                            df = df[df[col].str.contains(location_lower, case=False, na=False)]
                        except Exception as e:
                            print(f"Error filtering by location: {e}")
                        break
    
    # Filter by parameter if specified
    if user_query and user_query != 'general':
        user_query_lower = user_query.lower()
        
        # Map query terms to actual column names in your data
        column_mapping = {
            'salinity': ['salinity', 'salt', 'saline', 'psu'],
            'temp': ['temperature', 'temp', 'water_temp', 'sea_temp'],
            'pressure': ['pressure', 'depth', 'pres', 'dbars']
        }
        
        selected_columns = ['platform_number']
        # Add date and location columns if available
        for col in ['date', 'time', 'timestamp', 'datetime', 'latitude', 'longitude']:
            if col in df.columns:
                selected_columns.append(col)
        
        if user_query_lower in column_mapping:
            for possible_col in column_mapping[user_query_lower]:
                if possible_col in df.columns:
                    selected_columns.append(possible_col)
                    break
        
        # Also include region if available
        if 'region' in df.columns:
            selected_columns.append('region')
        
        # Only keep columns that actually exist in the dataframe
        selected_columns = [col for col in selected_columns if col in df.columns]
        if selected_columns:
            df = df[selected_columns]
    
    # Always return some data even if filters are too restrictive
    if len(df) == 0:
        print("No results found with filters, returning sample data")
        df = load_argo_data()
        # Use safe sampling to avoid the error
        df = safe_sample(df, min(20, top_k), replace=False)
    
    # Return top_k results using safe sampling
    return safe_sample(df, top_k, replace=False)

def results_to_json(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert DataFrame to JSON format for API response"""
    if df.empty:
        return {
            "status": "success",
            "message": "No results found",
            "results": []
        }
    
    return {
        "status": "success",
        "results": df.to_dict('records'),
        "count": len(df)
    }

# Test the enhanced parser with real data
if __name__ == "__main__":
    # Test various natural language queries
    test_queries = [
        "Show me ocean temperature data",
        "What's the salinity in the Pacific Ocean?",
        "Recent pressure measurements near India",
        "Ocean data from last year",
        "Temperature readings from the Atlantic",
    ]
    
    # First, let's examine the structure of your CSV files
    try:
        metadata_df = pd.read_csv(METADATA_FILE)
        print("Metadata columns:", list(metadata_df.columns))
        print("Metadata sample:")
        print(metadata_df.head(2))
        
        profiles_df = pd.read_csv(PROFILES_FILE)
        print("\nProfiles columns:", list(profiles_df.columns))
        print("Profiles sample:")
        print(profiles_df.head(2))
        
    except Exception as e:
        print(f"Could not examine CSV files: {e}")
        print("Using mock data structure")
    
    # Test queries
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        parsed = parse_user_query(query)
        print(f"   Parsed: {parsed}")
        
        df = query_pipeline(
            user_query=parsed["parameter"] or "general",
            location=parsed["location"],
            start_date=parsed["start_date"],
            end_date=parsed["end_date"],
            top_k=parsed.get("limit", 10)
        )
        
        print(f"   Results: {len(df)} records")
        if not df.empty:
            print(f"   Columns: {list(df.columns)}")
            if len(df) > 0:
                print(f"   Sample: {dict(df.iloc[0])}")