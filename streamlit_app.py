# This script creates a Streamlit UI that serves as the frontend for the application.
# It allows a user to type a query, sends it to the FastAPI backend, and displays the response.

import streamlit as st
import requests
import json
import time

# --- UI Configuration ---
st.set_page_config(
    page_title="FloatChat AI",
    page_icon="🤖",
    layout="wide",
)

st.title("🌊 FloatChat AI")
st.markdown("Ask a question about ARGO oceanographic floats.")

# --- Functions ---
def call_backend(query: str):
    """
    Sends a query to the FastAPI backend and returns the response.
    """
    backend_url = "http://localhost:8000/query"
    try:
        # Increased timeout to 200 seconds to match backend
        response = requests.post(
            backend_url,
            json={"query": query},
            timeout=200  # Increased from 60 to 200 seconds
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("The request took too long to complete. This is expected on systems with limited RAM. Please try again with all other applications closed.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the backend. Please ensure the backend is running. Details: {e}")
        return None

# --- Main App Logic ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is the average temperature of the Indian Ocean?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display a loading spinner while the API call is in progress
    with st.chat_message("assistant"):
        with st.spinner("Thinking... (This may take 1-2 minutes on limited RAM)"):
            # Call the backend API
            result = call_backend(prompt)
            if result:
                st.markdown(result["answer"])
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})