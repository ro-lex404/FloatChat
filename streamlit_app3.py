# MODIFIED streamlit_app2.py to include voice feature
import streamlit as st
import requests
import pandas as pd
import pydeck as pdk
import time
import plotly.express as px
from spr_speech import speech_module, continuous_listening
import threading
import queue as thread_queue
from typing import Optional

# --- UI Configuration ---
st.set_page_config(
    page_title="ARGO Float Data Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
# Ensures all necessary keys are in the session state from the start.
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
if 'listening' not in st.session_state:
    st.session_state.listening = False
if 'recognized_text' not in st.session_state:
    st.session_state.recognized_text = ""
if 'speech_queue' not in st.session_state:
    st.session_state.speech_queue = thread_queue.Queue()
if 'continuous_listener' not in st.session_state:
    st.session_state.continuous_listener = None
if 'last_speech_result' not in st.session_state:
    st.session_state.last_speech_result = None
if 'stop_listening_event' not in st.session_state:
    st.session_state.stop_listening_event = None

# Thread-safe queue for communication between speech thread and main app thread.
speech_results_queue = thread_queue.Queue()

# --- API Communication ---
API_BASE_URL = "http://127.0.0.1:8000"

@st.cache_data(ttl=300)
def get_map_data_cached():
    """Fetches and caches live float locations for the map."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/live/map_data", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to load map data: {e}")
        return []

# In your streamlit_app3.py - Replace the speech processing functions:

def continuous_listening_thread(stop_event, stop_phrase="stop listening"):
    """
    Runs the listening generator in a thread.
    Its ONLY job is to take text from the generator and put it on the queue.
    """
    try:
        for text_fragment in continuous_listening(stop_event, stop_phrase):
            if text_fragment and text_fragment.strip():
                # Put each valid fragment from the generator into the queue
                speech_results_queue.put({"type": "continuous", "text": text_fragment})
    except Exception as e:
        print(f"Error in continuous_listening_thread: {e}")
        speech_results_queue.put({"type": "error", "message": str(e)})
    finally:
        # Always put a "stopped" message so the main thread knows to clean up.
        print("Thread is sending 'stopped' message to queue.")
        speech_results_queue.put({"type": "stopped"})

def process_speech_results():
    """
    Processes results from the speech queue. Includes final text cleanup.
    """
    processed_something = False
    current_text = st.session_state.get("recognized_text", "")
    
    while not speech_results_queue.empty():
        try:
            result = speech_results_queue.get_nowait()
            print(f"Processing speech result from queue: {result}")

            if result["type"] == "continuous":
                new_fragment = result.get("text", "").strip()
                if new_fragment:
                    updated_text = (current_text + " " + new_fragment).strip()
                    st.session_state.recognized_text = updated_text
                    st.session_state.user_input_box_voice = updated_text
                    current_text = updated_text
                st.session_state.last_speech_result = "success"
                processed_something = True

            elif result["type"] == "stopped":
                print("Processing 'stopped' message. Finalizing text and cleaning up state.")

                # --- NEW & IMPORTANT: Final text cleanup logic ---
                final_text = st.session_state.get("recognized_text", "")
                stop_phrases_to_clean = ["stop listening", "stop listen", "stop"]
                for phrase in stop_phrases_to_clean:
                    if final_text.endswith(phrase):
                        final_text = final_text[:-len(phrase)].strip()
                
                # Update the state with the *cleaned* text
                st.session_state.recognized_text = final_text
                st.session_state.user_input_box_voice = final_text
                # --- END NEW LOGIC ---

                st.session_state.listening = False
                if st.session_state.continuous_listener and st.session_state.continuous_listener.is_alive():
                    st.session_state.continuous_listener.join(timeout=0.5)
                
                st.session_state.continuous_listener = None
                st.session_state.stop_listening_event = None
                processed_something = True

            elif result["type"] == "error":
                st.session_state.last_speech_result = f"Speech Error: {result['message']}"
                st.session_state.listening = False
                # Cleanup on error as well
                if st.session_state.continuous_listener and st.session_state.continuous_listener.is_alive():
                    st.session_state.continuous_listener.join(timeout=0.5)
                st.session_state.continuous_listener = None
                st.session_state.stop_listening_event = None
                processed_something = True

        except thread_queue.Empty:
            break
            
    return processed_something
    
def start_continuous_listening():
    """Start continuous listening in a separate thread with a stop event."""
    if st.session_state.continuous_listener is None or not st.session_state.continuous_listener.is_alive():
        # Don't clear previous text, allow accumulation across recordings
        st.session_state.stop_listening_event = threading.Event()
        
        st.session_state.continuous_listener = threading.Thread(
            target=continuous_listening_thread,
            args=(st.session_state.stop_listening_event, "stop listening"),
            daemon=True
        )
        st.session_state.continuous_listener.start()
        st.session_state.listening = True
        st.session_state.last_speech_result = None
        print("Started continuous listening")

def stop_continuous_listening():
    """Signal the listening thread to stop and clean up."""
    try:
        # Signal thread to stop
        if st.session_state.stop_listening_event:
            st.session_state.stop_listening_event.set()
        
        # Ensure the final text is in the input box
        if st.session_state.get("recognized_text"):
            st.session_state.user_input_box_voice = st.session_state.recognized_text.strip()
        
        # Clean up thread
        if st.session_state.continuous_listener and st.session_state.continuous_listener.is_alive():
            st.session_state.continuous_listener.join(timeout=1.0)
            
        # Update UI state
        st.session_state.listening = False
        st.session_state.continuous_listener = None
        st.session_state.stop_listening_event = None
        print("Stopped continuous listening")
    except Exception as e:
        print(f"Error in stop_continuous_listening: {e}")
        # Ensure state is cleaned up even if error occurs
        st.session_state.listening = False
        st.session_state.continuous_listener = None
        st.session_state.stop_listening_event = None

# --- Data Fetching Functions ---
def get_float_timeseries(float_id):
    """Fetches detailed time-series data for a specific float ID."""
    try:
        float_id_str = str(float_id).strip()
        st.info(f"🔍 Requesting data for float ID: {float_id_str}")

        response = requests.get(f"{API_BASE_URL}/api/live/float/{float_id_str}", timeout=30)
        if response.status_code == 404:
            st.warning(f"⚠️ Float {float_id_str} not found.")
            return []
        response.raise_for_status() # Raises an exception for other bad statuses (500, 403, etc.)

        data = response.json()
        if not data:
            st.warning(f"⚠️ No measurements found for float {float_id_str}")
            return []
        
        st.success(f"✅ Retrieved {len(data)} measurements for float {float_id_str}")
        return data

    except requests.exceptions.Timeout:
        st.error(f"⏰ Timeout while fetching data for float {float_id}. The API might be slow.")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Connection error. Is the backend API running?")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API request error: {str(e)}")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
    return []


def send_chat_query(query):
    """Sends a query to the AI chat backend."""
    try:
        response = requests.post(f"{API_BASE_URL}/query", json={"query": query}, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": f"Sorry, I couldn't process your request: {str(e)}"}

# --- Main App Execution ---

# Process any pending speech results at the start of each script run.
speech_processed = process_speech_results()

# CSS for styling the application.
st.markdown("""
<style>
/* [Your existing CSS remains unchanged] */
.main {
background-color: #f8f9fa;
}
.main-header {
font-size: 2.5rem;
color: #0b5394;
text-align: center;
margin-bottom: 1.5rem;
font-weight: 700;
padding-bottom: 0.5rem;
border-bottom: 2px solid #0b5394;
}

.section-header {
font-size: 1.4rem;
color: #0b5394;
margin-bottom: 1rem;
padding-bottom: 0.5rem;
border-bottom: 1px solid #cfe2f3;
font-weight: 600;
}

.card {
color: black;
background-color: white;
border-radius: 8px;
padding: 20px;
margin-bottom: 20px;
box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
border: 1px solid #cfe2f3;
}

.float-card {
background-color: #e6f2ff;
border-left: 4px solid #0b5394;
padding: 15px;
border-radius: 8px;
margin-bottom: 15px;
}

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

.chat-message-user {
color: black;
background-color: #e6f2ff;
padding: 12px 16px;
border-radius: 12px;
margin-bottom: 12px;
border-left: 4px solid #0b5394;
font-size: 14px;
}

.chat-message-assistant {
color: black;
background-color: #f0f7ff;
padding: 12px 16px;
border-radius: 12px;
margin-bottom: 12px;
border-left: 4px solid #3d85c6;
font-size: 14px;
}

.info-box {
color: black;
background-color: #e6f2ff;
padding: 12px 16px;
border-radius: 8px;
margin-bottom: 15px;
border-left: 4px solid #0b5394;
font-size: 14px;
}

.warning-box {
color: black;
background-color: #fff3cd;
padding: 12px 16px;
border-radius: 8px;
margin-bottom: 15px;
border-left: 4px solid #ffc107;
font-size: 14px;
}

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

.stSelectbox div div {
border-radius: 6px;
border: 1px solid #ced4da;
}

.css-1d391kg {
background-color: #f8f9fa;
}

.dataframe {
font-size: 14px;
}

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
color: white !important;
}

.stMetric div {
font-size: 20px !important;
font-weight: bold;
color: white !important;
}

.compact-chart {
height: 250px;
}
</style>
""", unsafe_allow_html=True)

# --- Main App Layout ---
st.markdown('<h1 class="main-header">🌊 ARGO Float Data Explorer</h1>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🌍 Map Dashboard", "💬 AI Chat"])

# --- Map Dashboard Tab ---
with tab1:
    # Your existing map dashboard code remains the same
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
            if 'datetime' in df_map.columns:
                df_map['datetime'] = pd.to_datetime(df_map['datetime'], errors='coerce')
            unique_floats = df_map.sort_values('datetime', ascending=False).drop_duplicates('float_id')
            unique_floats['float_id'] = unique_floats['float_id'].astype(str)

            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("🌊 Unique Floats", len(unique_floats))
            with col_metric2:
                if 'datetime' in unique_floats.columns and not unique_floats['datetime'].isna().all():
                    latest_date = unique_floats['datetime'].max().strftime('%Y-%m-%d')
                    st.metric("📅 Latest Data", latest_date)
                else:
                    st.metric("📅 Latest Data", "N/A")
            with col_metric3:
                if 'longitude' in unique_floats.columns and 'latitude' in unique_floats.columns:
                    region_count = len(unique_floats[unique_floats['longitude'].between(30, 120) &
                                    unique_floats['latitude'].between(-30, 30)])
                    st.metric("🌍 Indian Ocean", region_count)
                else:
                    st.metric("🌍 Indian Ocean", "N/A")

            if 'longitude' in unique_floats.columns and 'latitude' in unique_floats.columns:
                all_floats_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=unique_floats,
                    get_position=["longitude", "latitude"],
                    get_color=[11, 83, 148, 160],
                    get_radius=50000,
                    pickable=True,
                    auto_highlight=True,
                )

                view_state = pdk.ViewState(latitude=20, longitude=80, zoom=1.5, pitch=0)
                layers = [all_floats_layer]

                if st.session_state.selected_float:
                    selected_float_df = unique_floats[unique_floats['float_id'] == str(st.session_state.selected_float)]
                    if not selected_float_df.empty:
                        selected_float_data = selected_float_df.iloc[0]
                        selected_layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=selected_float_df,
                            get_position=["longitude", "latitude"],
                            get_color=[255, 0, 0, 255], # Red
                            get_radius=80000,
                            pickable=False,
                        )
                        layers.append(selected_layer)
                        view_state.latitude = selected_float_data['latitude']
                        view_state.longitude = selected_float_data['longitude']
                        view_state.zoom = 5

                deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip={
                        "html": """
                        <div style="padding: 10px; background-color: #0b5394; color: white; border-radius: 5px;">
                        <b>Float ID:</b> {float_id}<br/>
                        <b>Lat:</b> {latitude}<br/>
                        <b>Lon:</b> {longitude}<br/>
                        <b>Click to select this float.</b>
                        </div>
                        """,
                    },
                    map_style="light",
                    height=400
                )
                
                st.pydeck_chart(deck, use_container_width=True)
            else:
                st.warning("Map data missing required latitude/longitude columns.")

            st.markdown("---")
            st.markdown('<div class="section-header">Select a Float for Detailed Analysis</div>', unsafe_allow_html=True)
            
            float_options = [""] + sorted(unique_floats['float_id'].tolist())
            current_index = 0
            if st.session_state.selected_float:
                try:
                    current_index = float_options.index(str(st.session_state.selected_float))
                except ValueError:
                    current_index = 0
            
            selected_id = st.selectbox(
                "Choose a Float ID:",
                options=float_options,
                index=current_index,
                key="float_select"
            )

            if selected_id and selected_id != st.session_state.selected_float:
                st.session_state.selected_float = selected_id
                st.session_state.float_data = None
                st.rerun()

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📊 Load Float Data", key="load_btn", use_container_width=True) and selected_id:
                    st.session_state.selected_float = selected_id
                    st.session_state.float_data = None
                    st.rerun()
            with col_btn2:
                if st.button("🗑️ Clear Selection", key="clear_btn", use_container_width=True):
                    st.session_state.selected_float = None
                    st.session_state.float_data = None
                    st.rerun()

        else:
            st.markdown('<div class="warning-box">No map data available. Check if the backend API is running.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Float Details</div>', unsafe_allow_html=True)

        if st.session_state.selected_float:
            float_id = st.session_state.selected_float
            st.markdown(f'<div class="float-card">Selected Float: <strong>{float_id}</strong></div>', unsafe_allow_html=True)
            
            if st.session_state.float_data is None:
                with st.spinner(f"Loading detailed data for float {float_id}..."):
                    st.session_state.float_data = get_float_timeseries(float_id)
                    st.rerun()

            if st.session_state.float_data:
                df_ts = pd.DataFrame(st.session_state.float_data)
                
                # Dynamic column finding for datetime, temp, salinity, pressure
                datetime_col = next((c for c in df_ts.columns if 'date' in c.lower() or 'time' in c.lower()), None)
                temp_col = next((c for c in df_ts.columns if 'temp' in c.lower()), None)
                salinity_col = next((c for c in df_ts.columns if 'sal' in c.lower()), None)
                pressure_col = next((c for c in df_ts.columns if 'pres' in c.lower()), None)
                
                if datetime_col:
                    df_ts[datetime_col] = pd.to_datetime(df_ts[datetime_col], errors='coerce')
                    df_ts.sort_values(datetime_col, inplace=True)
                
                st.markdown('<div class="section-header" style="font-size: 1.2rem;">Measurements Summary</div>', unsafe_allow_html=True)
                cols_m = st.columns(3)
                cols_m[0].metric("📈 Measurements", len(df_ts))
                if temp_col and not df_ts[temp_col].isna().all():
                    cols_m[1].metric("🌡️ Avg Temp", f"{df_ts[temp_col].mean():.2f}°C")
                if salinity_col and not df_ts[salinity_col].isna().all():
                    cols_m[2].metric("🧂 Avg Salinity", f"{df_ts[salinity_col].mean():.2f} PSU")

                # Function to create a compact plotly chart
                def create_compact_chart(df, x_col, y_col, title, color):
                    if y_col and not df[y_col].isna().all() and x_col:
                        st.markdown(f"**{title}**")
                        fig = px.line(df, x=x_col, y=y_col)
                        fig.update_layout(
                            height=200, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                            font={"color": "#2c3e50"}, margin={"l": 20, "r": 20, "t": 20, "b": 20}, showlegend=False
                        )
                        fig.update_traces(line={"color": color})
                        st.plotly_chart(fig, use_container_width=True)

                create_compact_chart(df_ts, datetime_col, temp_col, "Temperature Data", '#e74c3c')
                create_compact_chart(df_ts, datetime_col, salinity_col, "Salinity Data", '#3498db')
                create_compact_chart(df_ts, datetime_col, pressure_col, "Pressure Data", '#2ecc71')

                with st.expander("📋 Sample Data"):
                    st.dataframe(df_ts.head(10), use_container_width=True)
            else:
                st.markdown('<div class="warning-box">⚠️ No detailed data available for this float.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">👈 Select a float from the map or dropdown to view its detailed data.</div>', unsafe_allow_html=True)


# --- AI Chat Tab ---
with tab2:
    st.markdown('<div class="section-header">💬 AI-Powered Chat Interface</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about ARGO floats, ocean data, or specific measurements.")

    # --- Simplified Speech Recognition Controls (inside Chat tab only) ---
    col_input, col_button = st.columns([4, 1])

    with col_input:
        # Initialize the text input value from session state
        if "user_input_box_voice" not in st.session_state:
            st.session_state.user_input_box_voice = st.session_state.get("recognized_text", "")
        
        user_input = st.text_input(
            "Type your message or use voice input:",
            key="user_input_box_voice",
            placeholder="Click the mic to start recording..."
        )

    with col_button:
        st.write("")
        st.write("")
        # Toggle recording: start/stop continuous listening. The continuous listener will set
        # st.session_state.recognized_text when the stop phrase is detected.
        if st.session_state.get("listening", False):
            if st.button("Stop Recording", key="stop_continuous", use_container_width=True):
                stop_continuous_listening()
                st.rerun()
        else:
            if st.button("🎤 Record", key="start_continuous", use_container_width=True):
                start_continuous_listening()
                st.rerun()

        # Visual feedback while listening
        if st.session_state.get("listening", False):
            # This block now correctly handles the UI updates while listening.
            
            # First, a safety check: if the thread has died for some reason, fix the state.
            if not (st.session_state.continuous_listener and st.session_state.continuous_listener.is_alive()):
                st.session_state.listening = False
                st.rerun()

            st.info("🎤 Listening... Say 'stop listening' to finish.")
            
            # This uses st.spinner as a context manager for a cleaner look
            with st.spinner("Capturing speech..."):
                # The loop now runs for a longer time, making it feel more responsive.
                # It will be interrupted by a rerun as soon as new speech is detected.
                for _ in range(25): # Increased range for a longer listening window (5 seconds)
                    if not speech_results_queue.empty():
                        st.rerun()
                    time.sleep(0.2)

            # If the loop finishes and we are still listening, it means the user was quiet.
            # We can trigger one last rerun to keep the loop going if needed.
            if st.session_state.get("listening", False):
                st.rerun()
                
        if st.session_state.get("last_speech_result") and st.session_state.last_speech_result != "success":
            st.warning(f"Speech recognition: {st.session_state.last_speech_result}")

    # --- Send / Clear Controls ---
    col_send1, col_send2 = st.columns([3, 1])
    with col_send1:
        if st.button("📤 Send Message", use_container_width=True) and user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("Analyzing your question..."):
                response = send_chat_query(user_input)
                reply = response.get("answer", "No response available")
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            # clear recognized_text after sending
            st.session_state.recognized_text = ""
            st.session_state.last_speech_result = None
            st.rerun()

    with col_send2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.recognized_text = ""
            st.session_state.last_speech_result = None
            st.rerun()

    # --- Chat History Display ---
    st.markdown("---")
    st.markdown("### 💭 Conversation")

    chat_container = st.container()
    with chat_container:
        # Display up to the last 6 messages
        for msg in st.session_state.get("chat_history", [])[-6:]:
            if msg.get("role") == "user":
                st.markdown(f'<div class="chat-message-user">👤💬 {msg.get("content")}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-assistant">🐟🫧 {msg.get("content")}</div>', unsafe_allow_html=True)

    if not st.session_state.get("chat_history"):
        with st.expander("💡 Example Queries"):
            st.markdown("""
            - *"Show me salinity profiles near the equator in August 2020"*
            - *"What are the nearest ARGO floats to the Indian Ocean?"*
            - *"Show me float data from the Bay of Bengal"*
            - *"Display temperature trends for float 2902212"*
            - *"Compare salinity data between different ocean regions"*
            """)

    # Debug info only inside chat tab (can be removed later)
    with st.expander("🔧 Debug Info (Remove Later)"):
        st.write("Session State:")
        st.json({
            "listening": st.session_state.get("listening", False),
            "recognized_text": st.session_state.get("recognized_text", ""),
            "last_speech_result": st.session_state.get("last_speech_result"),
            "continuous_listener_alive": st.session_state.continuous_listener.is_alive() if st.session_state.get("continuous_listener") else False
        })
        st.write(f"Speech queue size: {speech_results_queue.qsize()}")

# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="section-header">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    **AI-Powered Conversational System for ARGO Float Data**

    This tool enables users to query, explore, and visualize oceanographic information using natural language.
    """)

    st.markdown("### 📊 Data Sources")
    st.info("""
    - **Live API**: Real-time float positions and measurements
    - **Static Metadata**: Fallback data from processed NetCDF files
    - **AI Integration**: RAG pipeline with Gemini API for intelligent queries
    """)

    st.markdown("### 🛠️ Technical Stack")
    st.info("""
    - **Backend**: FastAPI with async processing
    - **Frontend**: Streamlit for interactive visualization
    - **AI**: Retrieval-Augmented Generation (RAG) pipeline
    """)
    
    st.markdown("### 🔧 Controls")
    if st.button("🔄 Clear All Cache & State", use_container_width=True):
        st.cache_data.clear()
        # A more robust way to clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("### 📈 System Status")

    if st.session_state.map_data:
        st.success("✅ Connected to data source")
        try:
            if isinstance(st.session_state.map_data, list) and st.session_state.map_data:
                float_count = len(st.session_state.map_data)
                unique_floats = len({f.get('float_id') for f in st.session_state.map_data})
                st.info(f"📊 {float_count} data points from {unique_floats} unique floats loaded.")
            else:
                st.info("📊 Map data loaded.")
        except Exception:
            st.info("📊 Map data loaded.")
    else:
        st.warning("⚠️ No live data. Check API connection.")

    st.markdown("---")
    st.markdown("**Built for Smart India Hackathon 2025**")

# --- Final Rerun ---
# If speech processing updated the state, rerun the script to ensure the UI reflects the changes instantly.
if speech_processed:
    st.rerun()