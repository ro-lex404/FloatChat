# streamlit_app.py (With Selected Float Marker)
import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import time
import plotly.express as px

# --- UI Configuration ---
st.set_page_config(
    page_title="ARGO Float Data Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State (no changes) ---
if "selected_float" not in st.session_state:
    st.session_state.selected_float = None
if "float_data" not in st.session_state:
    st.session_state.float_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "map_data" not in st.session_state:
    st.session_state.map_data = None
if "last_map_fetch" not in st.session_state:
    st.session_state.last_map_fetch = 0

# --- API Communication & CSS (no changes) ---
API_BASE_URL = "http://127.0.0.1:8000"

@st.cache_data(ttl=300)
def get_map_data_cached():
    try:
        response = requests.get(f"{API_BASE_URL}/api/live/map_data", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to load map data: {e}")
        return []

def get_float_timeseries(float_id):
    try:
        response = requests.get(f"{API_BASE_URL}/api/live/float/{float_id}", timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def send_chat_query(query):
    try:
        response = requests.post(f"{API_BASE_URL}/query", json={"query": query}, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": f"Sorry, I couldn't process your request: {str(e)}"}

# --- Custom CSS for a professional look ---
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #0b5394;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0b5394;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        color: #0b5394;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #cfe2f3;
        font-weight: 600;
    }
    
    /* Cards */
    .card {
        color:black;
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #cfe2f3;
    }
    
    /* Float detail card */
    .float-card {
        background-color: #e6f2ff;
        border-left: 4px solid #0b5394;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Buttons */
    .stButton button {
        width: 100%;
        border-radius: 6px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
        background: linear-gradient(135deg, #0b5394 0%, #3d85c6 100%);
        color: white;
        border: none;
        padding: 10px;
        font-weight: 500;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, #0b5394 0%, #3d85c6 100%);
        color: white;
    }
    
    /* Chat messages */
    .chat-message-user {
        color:black;
        background-color: #e6f2ff;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        border-left: 4px solid #0b5394;
        font-size: 14px;
    }
    
    .chat-message-assistant {
        color:black;
        background-color: #f0f7ff;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        border-left: 4px solid #3d85c6;
        font-size: 14px;
    }
    
    /* Info boxes */
    .info-box {
        color:black;
        background-color: #e6f2ff;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid #0b5394;
        font-size: 14px;
    }
    
    /* Warning boxes */
    .warning-box {
        color:black;
        background-color: #fff3cd;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid #ffc107;
        font-size: 14px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0b5555 0%, #3d85c6 100%);
        color: white;
        border-bottom: none;
    }
    
    /* Select box styling */
    .stSelectbox div div {
        border-radius: 6px;
        border: 1px solid #ced4da;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Dataframe styling */
    .dataframe {
        font-size: 14px;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online {
        background-color: #28a745;
    }
    .status-offline {
        background-color: #dc3545;
    }

    /* --- MODIFICATION START --- */
    /* Metric component styling */
    .stMetric {
        background: linear-gradient(135deg, #0b5394 0%, #3d85c6 100%);
        color: white;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        min-height: 85px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .stMetric label {
        font-size: 14px !important;
        color: white !important; /* Ensure label is white */
    }
    
    .stMetric div {
        font-size: 20px !important;
        font-weight: bold;
        color: white !important; /* Ensure value is white */
    }
    /* --- MODIFICATION END --- */
</style>
""", unsafe_allow_html=True)
# --- Main App Layout ---
st.markdown('<h1 class="main-header">🌊 ARGO Float Data Explorer</h1>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🌍 Map Dashboard", "💬 AI Chat Interface"])

with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Live Float Locations</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Map Data", key="refresh_map", use_container_width=True):
            st.cache_data.clear()
            st.session_state.map_data = None
            st.rerun()
        
        current_time = time.time()
        if st.session_state.map_data is None or (current_time - st.session_state.last_map_fetch) > 300:
            with st.spinner("Loading float data..."):
                map_data = get_map_data_cached()
                st.session_state.map_data = map_data
                st.session_state.last_map_fetch = current_time
        else:
            map_data = st.session_state.map_data

        if map_data:
            df_map = pd.DataFrame(map_data)
            # New, corrected line
            df_map['datetime'] = pd.to_datetime(df_map['datetime'], errors='coerce')
            unique_floats = df_map.sort_values('datetime', ascending=False).drop_duplicates('float_id')
            
            # Display metrics in a row - INSIDE the card
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                # --- MODIFICATION START ---
                # Removed the markdown wrapper
                st.metric("🌊 Unique Floats", len(unique_floats))
                # --- MODIFICATION END ---
            with col_metric2:
                # --- MODIFICATION START ---
                latest_date = unique_floats['datetime'].max().strftime('%Y-%m-%d')
                st.metric("📅 Latest Data", latest_date)
                # --- MODIFICATION END ---
            with col_metric3:
                # --- MODIFICATION START ---
                region_count = len(unique_floats[unique_floats['longitude'].between(30, 120) & 
                                                unique_floats['latitude'].between(-30, 30)])
                st.metric("🌏 Indian Ocean", region_count)
                # --- MODIFICATION END ---
            
           # 1. Base layer for ALL floats (a neutral blue color)
            all_floats_layer = pdk.Layer(
                "ScatterplotLayer",
                data=unique_floats,
                get_position=["longitude", "latitude"],
                get_color=[11, 83, 148, 160],  # Blue, slightly transparent
                get_radius=50000, # Slightly smaller base radius
                pickable=True,
                auto_highlight=True,
            )
            
            # 2. Define the initial view of the map
            view_state = pdk.ViewState(
                latitude=20, # Centered more globally
                longitude=80,
                zoom=1.5,
                pitch=0,
            )
            
            # 3. Create a list to hold our map layers
            layers = [all_floats_layer]
            
            # 4. If a float is selected, create a special marker layer for it
            if st.session_state.selected_float:
                selected_float_df = unique_floats[unique_floats['float_id'] == st.session_state.selected_float]
                
                if not selected_float_df.empty:
                    selected_float_data = selected_float_df.iloc[0]
                    
                    # Create a prominent red layer for the selected float
                    selected_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=selected_float_df,
                        get_position=["longitude", "latitude"],
                        get_color=[255, 0, 0, 255],  # Bright red, fully opaque
                        get_radius=80000, # Make it larger to stand out
                        pickable=False, # No need to pick this one
                    )
                    layers.append(selected_layer)
                    
                    # 5. Update the view to zoom into the selected float
                    view_state.latitude = selected_float_data['latitude']
                    view_state.longitude = selected_float_data['longitude']
                    view_state.zoom = 5 # Zoom in closer
            
            # 6. Create the Deck with the list of layers
            deck = pdk.Deck(
                layers=layers, # Use the dynamic list of layers
                initial_view_state=view_state,
                tooltip={
                    "html": """
                    <div style="padding: 10px; background-color: #0b5394; color: white; border-radius: 5px;">
                    <b>Float ID:</b> {float_id}<br/>
                    <b>Click to select this float.</b>
                    </div>
                    """,
                },
                map_style="light",
                height=400
            )
            
            st.pydeck_chart(deck)
            
            # Float selection
            st.markdown("---")
            st.markdown('<div class="section-header">Select a Float for Detailed Analysis</div>', unsafe_allow_html=True)
            
            float_options = [""] + sorted(unique_floats['float_id'].tolist())
            selected_id = st.selectbox(
                "Choose a Float ID:",
                options=float_options,
                index=0,
                key="float_select"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📊 Load Float Data", key="load_btn", use_container_width=True) and selected_id:
                    st.session_state.selected_float = selected_id
                    st.session_state.float_data = None
            with col_btn2:
                if st.button("🗑️ Clear Selection", key="clear_btn", use_container_width=True):
                    st.session_state.selected_float = None
                    st.session_state.float_data = None

            st.markdown('</div>', unsafe_allow_html=True)  # Close the card

        else:
            st.markdown('<div class="warning-box">No map data available. Check if the backend API is running.</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">The app will use cached data from the CSV file if API is unavailable.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Float Details</div>', unsafe_allow_html=True)
        
        if st.session_state.selected_float:
            float_id = st.session_state.selected_float
            st.markdown(f'<div class="float-card">', unsafe_allow_html=True)
            st.success(f"Selected Float: **{float_id}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.session_state.float_data is None:
                with st.spinner(f"Loading detailed data for float {float_id}..."):
                    data = get_float_timeseries(float_id)
                    if data:
                        st.session_state.float_data = data
                        st.rerun()
                    else:
                        st.markdown('<div class="warning-box">No detailed data available for this float</div>', unsafe_allow_html=True)
            else:
                df_ts = pd.DataFrame(st.session_state.float_data)
                if not df_ts.empty:
                    df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])
                    df_ts.sort_values('datetime', inplace=True)
                    
                    # Display metrics - INSIDE the card
                    st.markdown('<div class="section-header" style="font-size: 1.2rem;">Measurements Summary</div>', unsafe_allow_html=True)
                    cols = st.columns(3)
                    with cols[0]:
                        # --- MODIFICATION START ---
                        st.metric("📈 Measurements", len(df_ts))
                        # --- MODIFICATION END ---
                    if not df_ts['temperature'].isna().all():
                        with cols[1]:
                            # --- MODIFICATION START ---
                            avg_temp = df_ts['temperature'].mean()
                            st.metric("🌡️ Avg Temp", f"{avg_temp:.2f}°C")
                            # --- MODIFICATION END ---
                    if not df_ts['salinity'].isna().all():
                        with cols[2]:
                            # --- MODIFICATION START ---
                            avg_salinity = df_ts['salinity'].mean()
                            st.metric("🧂 Avg Salinity", f"{avg_salinity:.2f} PSU")
                            # --- MODIFICATION END ---
                    
                    # SEPARATE charts for temperature and salinity
                    if not df_ts['temperature'].isna().all():
                        st.markdown('<div class="section-header" style="font-size: 1.2rem;">Temperature Data</div>', unsafe_allow_html=True)
                        fig_temp = px.line(df_ts, x='datetime', y='temperature', 
                                         title="Temperature Over Time")
                        fig_temp.update_layout(
                            height=250, 
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#2c3e50'),
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        fig_temp.update_traces(line=dict(color='#e74c3c'))
                        st.plotly_chart(fig_temp, use_container_width=True)
                    
                    if not df_ts['salinity'].isna().all():
                        st.markdown('<div class="section-header" style="font-size: 1.2rem;">Salinity Data</div>', unsafe_allow_html=True)
                        fig_salinity = px.line(df_ts, x='datetime', y='salinity', 
                                             title="Salinity Over Time")
                        fig_salinity.update_layout(
                            height=250, 
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#2c3e50'),
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        fig_salinity.update_traces(line=dict(color='#3498db'))
                        st.plotly_chart(fig_salinity, use_container_width=True)
                    
                    # Show pressure data if available
                    if not df_ts['pressure'].isna().all():
                        st.markdown('<div class="section-header" style="font-size: 1.2rem;">Pressure Data</div>', unsafe_allow_html=True)
                        fig_pressure = px.line(df_ts, x='datetime', y='pressure', 
                                             title="Pressure Over Time")
                        fig_pressure.update_layout(
                            height=200, 
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#2c3e50'),
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_pressure, use_container_width=True)
                    
                    with st.expander("📋 Sample Data (first 10 rows)"):
                        st.dataframe(df_ts.head(10).style.format({
                            'temperature': '{:.2f}',
                            'salinity': '{:.2f}',
                            'pressure': '{:.1f}'
                        }))
                else:
                    st.markdown('<div class="warning-box">No measurement data available for this float</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">👈 Select a float from the map to view detailed data</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close the card

with tab2:
    st.markdown('<div class="section-header">💬 AI-Powered Chat Interface</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about ARGO floats, ocean data, or specific measurements")
    # Chat examples
    with st.expander("💡 Example Queries"):
        st.markdown("""
        - Show me salinity profiles near the equator in September 2013
        - Compare temperature parameters in the Arabian Sea for the last 6 months
        - What are the nearest ARGO floats to the Indian Ocean?
        - Show me float data from the Bay of Bengal
        - Display temperature trends for float 1900410
        """)
    
    # Simple chat interface with improved styling
    for msg in st.session_state.chat_history[-6:]:  # Show only last 6 messages
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
    
    if prompt := st.chat_input("Ask about ARGO floats..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("Analyzing your question..."):
            response = send_chat_query(prompt)
            reply = response.get("answer", "No response available")
            
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">ℹ️ About SIH 2025</div>', unsafe_allow_html=True)
    st.markdown("""
    **AI-Powered Conversational System for ARGO Float Data**
    
    This tool enables users to query, explore, and visualize oceanographic information using natural language.
    """)
    
    st.markdown("### 📊 Data Sources")
    st.info("""
    - **Live API**: Real-time float positions and measurements
    - **Static Metadata**: Fallback data from processed NetCDF files
    - **AI Integration**: RAG pipeline with Gemini API for intelligent queries
    - **Vector Database**: FAISS for efficient similarity search
    """)
    
    st.markdown("### 🛠️ Technical Stack")
    st.info("""
    - **Backend**: FastAPI with async processing
    - **Frontend**: Streamlit for interactive visualization
    - **Database**: Structured data from NetCDF files
    - **AI**: Retrieval-Augmented Generation (RAG) pipeline
    """)
    
    st.markdown("### 🎯 Key Features")
    st.info("""
    - Natural language query processing
    - Geospatial visualization of float data
    - Time-series analysis of ocean parameters
    - Interactive chat interface
    """)
    
    st.markdown("### 🔧 Controls")
    if st.button("🔄 Clear All Cache", use_container_width=True):
        st.cache_data.clear()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📈 System Status")
    
    # Improved status display with better float count logic
    if st.session_state.map_data:
        st.success("✅ Connected to data source")
        try:
            if isinstance(st.session_state.map_data, list):
                float_count = len(st.session_state.map_data)
                unique_floats = len(set([float_item.get('float_id', '') for float_item in st.session_state.map_data]))
                st.info(f"📊 {float_count} data points from {unique_floats} unique floats")
            else:
                st.info("📊 Map data loaded successfully")
        except Exception as e:
            st.info("📊 Map data loaded successfully")
    else:
        st.warning("⚠️ Using fallback data")
    
    st.markdown("---")
    st.markdown("**Built for Smart India Hackathon 2025**")
    st.markdown('</div>', unsafe_allow_html=True)