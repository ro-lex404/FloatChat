import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import urllib.parse

# --- UI Configuration ---
st.set_page_config(
    page_title="FloatChat AI - Dashboard",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 FloatChat AI Dashboard")
st.markdown("Interactive visualization of ARGO oceanographic float data")

# --- Initialize Session State ---
if "selected_float" not in st.session_state:
    st.session_state.selected_float = None
if "float_data" not in st.session_state:
    st.session_state.float_data = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Functions ---
def call_backend(query: str):
    """Sends a query to the FastAPI backend and returns the response."""
    backend_url = "http://localhost:8000/query"
    try:
        response = requests.post(
            backend_url,
            json={"query": query},
            timeout=200
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("The request took too long to complete.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the backend: {e}")
        return None

def get_map_data():
    """Fetch float data for the map visualization"""
    try:
        response = requests.get("http://localhost:8000/api/floats?limit=200", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to load map data: {e}")
        return []

def get_float_timeseries(float_id):
    """Fetch time series data for a specific float"""
    try:
        response = requests.get(f"http://localhost:8000/api/float/{float_id}", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to load time series data: {e}")
        return []

# Get map data
map_data = get_map_data()

# --- Check for URL parameter selection ---
try:
    # Try new Streamlit API first
    query_params = st.query_params
    if "selected_float" in query_params:
        selected_float = query_params["selected_float"]
        if selected_float and st.session_state.selected_float != selected_float:
            st.session_state.selected_float = selected_float
            st.session_state.float_data = None
except:
    # Fallback to experimental API
    try:
        query_params = st.experimental_get_query_params()
        if "selected_float" in query_params:
            selected_float = query_params["selected_float"][0]
            if selected_float and st.session_state.selected_float != selected_float:
                st.session_state.selected_float = selected_float
                st.session_state.float_data = None
    except:
        pass

# --- Main App Layout ---
tab1, tab2 = st.tabs(["🗺️ Map Dashboard", "💬 Chat Interface"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Float Locations")
        
        if map_data:
            # Create HTML for Leaflet map with URL-based selection
            leaflet_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
                <style>
                    #map {{ height: 500px; width: 100%; }}
                    .leaflet-popup-content {{ 
                        font-family: Arial, sans-serif; 
                        max-width: 300px;
                    }}
                    .select-btn {{
                        margin-top: 8px; 
                        padding: 8px 12px; 
                        background: #4CAF50; 
                        color: white; 
                        border: none; 
                        border-radius: 4px; 
                        cursor: pointer;
                        width: 100%;
                        font-size: 14px;
                    }}
                    .select-btn:hover {{
                        background: #45a049;
                    }}
                </style>
            </head>
            <body>
                <div id="map"></div>
                <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
                <script>
                    const map = L.map('map').setView([20, 0], 2);
                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        attribution: '© OpenStreetMap contributors'
                    }}).addTo(map);
                    
                    const floats = {json.dumps(map_data)};
                    
                    function selectFloat(floatId) {{
                        // Update the URL with the selected float
                        const currentUrl = new URL(window.parent.location.href);
                        currentUrl.searchParams.set('selected_float', floatId);
                        window.parent.history.replaceState({{}}, '', currentUrl);
                        
                        // Reload the page to trigger Streamlit's parameter detection
                        window.parent.location.reload();
                    }}
                    
                    floats.forEach(float => {{
                        const popupContent = `
                            <div style="max-width: 280px;">
                                <b>Float ID:</b> ${{float.float_id}}<br>
                                <b>Date:</b> ${{new Date(float.datetime).toLocaleString()}}<br>
                                <b>Temp:</b> ${{float.temperature !== null ? float.temperature.toFixed(2) + '°C' : 'N/A'}}<br>
                                <b>Salinity:</b> ${{float.salinity !== null ? float.salinity.toFixed(3) + ' PSU' : 'N/A'}}<br>
                                <button class="select-btn" onclick="selectFloat('${{float.float_id}}')">
                                    📍 Select This Float
                                </button>
                            </div>
                        `;
                        
                        const marker = L.marker([float.latitude, float.longitude])
                            .addTo(map)
                            .bindPopup(popupContent);
                    }});
                </script>
            </body>
            </html>
            """
            
            # Display the map
            st.components.v1.html(leaflet_html, height=520)
            
        else:
            st.warning("No map data available. Please ensure the backend is running.")
    
    with col2:
        st.header("Float Details")
        
        if st.session_state.selected_float:
            st.success(f"Selected Float: **{st.session_state.selected_float}**")
            
            if st.session_state.float_data:
                timeseries_data = st.session_state.float_data
                df_ts = pd.DataFrame(timeseries_data)
                df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])
                
                # Display metrics
                col21, col22, col23 = st.columns(3)
                with col21:
                    latest = df_ts.iloc[-1] if len(df_ts) > 0 else {}
                    st.metric("Latest Temp", f"{latest.get('temperature', 'N/A'):.2f}°C" if latest.get('temperature') else "N/A")
                with col22:
                    st.metric("Latest Salinity", f"{latest.get('salinity', 'N/A'):.3f} PSU" if latest.get('salinity') else "N/A")
                with col23:
                    st.metric("Data Points", len(df_ts))
                
                # Display time series chart using Streamlit native charts
                if not df_ts.empty:
                    if 'temperature' in df_ts.columns:
                        st.subheader("Temperature Time Series")
                        chart_data = df_ts[['datetime', 'temperature']].dropna()
                        if not chart_data.empty:
                            st.line_chart(chart_data.set_index('datetime'))
                    
                    if 'salinity' in df_ts.columns:
                        st.subheader("Salinity Time Series")
                        chart_data = df_ts[['datetime', 'salinity']].dropna()
                        if not chart_data.empty:
                            st.line_chart(chart_data.set_index('datetime'))
                
                # Raw data table
                with st.expander("View Raw Data"):
                    st.dataframe(df_ts[['datetime', 'temperature', 'salinity', 'pressure']].dropna(how='all'))
            
            else:
                st.info("Loading time series data...")
                timeseries_data = get_float_timeseries(st.session_state.selected_float)
                if timeseries_data:
                    st.session_state.float_data = timeseries_data
                    st.rerun()
                else:
                    st.error("Failed to load time series data")
                    
            # Add a clear selection button
            if st.button("Clear Selection"):
                st.session_state.selected_float = None
                st.session_state.float_data = None
                # Also clear the URL parameter
                try:
                    st.query_params.clear()
                except:
                    try:
                        st.experimental_set_query_params({})
                    except:
                        pass
                st.rerun()
                
        else:
            st.info("👆 Select a float from the map to view details")
            st.info("Click on any float marker and then click the 'Select This Float' button in the popup")

with tab2:
    st.header("Chat with Float Data")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("What is the average temperature of the Indian Ocean?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = call_backend(prompt)
                if result:
                    st.markdown(result["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                    
                    # Show context in expander
                    with st.expander("View retrieved context"):
                        st.text_area(
                            "Context", 
                            value=result.get("context", "No context available"),
                            height=200,
                            label_visibility="collapsed"
                        )

# Add a simple manual selection in sidebar
with st.sidebar:
    st.header("Quick Float Selection")
    if map_data:
        float_options = [f["float_id"] for f in map_data]
        selected_float = st.selectbox("Select a float:", options=float_options, key="sidebar_float_select")
        if st.button("Load Selected Float", key="sidebar_load_btn"):
            st.session_state.selected_float = selected_float
            st.session_state.float_data = None
            st.rerun()