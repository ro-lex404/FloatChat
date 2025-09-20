# 🌊 ARGO FloatChat: AI-Powered Ocean Data Explorer

![ARGO-FloatChat](https://img.shields.io/badge/ARGO-FloatChat-blue?style=for-the-badge&logo=ocean)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)

A sophisticated conversational AI system for exploring and analyzing ARGO float oceanographic data. This application, built for the Smart India Hackathon 2025, combines a real-time data visualization dashboard with a powerful natural language query interface.

## 📋 Table of Contents
- [🚀 Key Features](#-key-features)
- [🏗️ Architecture Overview](#️-architecture-overview)
- [🛠️ Technical Stack](#️-technical-stack)
- [📦 Installation & Setup](#-installation--setup)
- [🎯 Usage](#-usage)
- [🔍 API Endpoints](#-api-endpoints)
- [🌟 Acknowledgments](#-acknowledgments)

## 🚀 Key Features

### 🌍 Interactive Map Dashboard
- **Hybrid Data Loading**: Initial map view is populated instantly using pre-processed profile summaries for a fast user experience.
- **Smart Selection & Highlighting**: Click any float on the map or select from a dropdown to load its detailed data. The selected float is highlighted with a distinct, larger red marker, and the map automatically zooms in.
- **Multi-Layer Visualization**: A base layer in `PyDeck` shows all active floats in blue, with a dynamic highlight layer for the selected float, providing clear visual context.
- **At-a-Glance Metrics**: The dashboard features styled metric cards displaying the total number of unique floats, the latest data timestamp, and a real-time count of floats in the Indian Ocean region.

### 📊 Advanced Data Visualization
- **Live Time-Series Analysis**: When a float is selected, its detailed time-series data is fetched live via the `Argopy` library.
- **Multi-Parameter Charts**: Separate, cleanly styled `Plotly` charts are generated for Temperature, Salinity, and Pressure, allowing for focused analysis.
- **Statistical Summaries**: Key statistics like total measurements, average temperature, and average salinity are calculated and displayed for the selected float.

### 💬 AI-Powered Chat Interface
- **Natural Language Queries**: A dedicated chat tab allows users to ask complex questions about ocean data in plain English.
- **RAG Pipeline**: Utilizes a Retrieval-Augmented Generation (RAG) pipeline with a `FAISS` vector database to find the most relevant data to answer user questions.
- **Gemini 1.5 Flash Integration**: Powered by Google's `gemini-1.5-flash` model for intelligent, context-aware, and accurate responses based on retrieved data.
- **Conversational Memory**: The interface displays the last 6 messages, providing context for follow-up questions.

## 🏗️ Architecture Overview

The system uses a modern, decoupled frontend-backend architecture for performance and scalability. The backend handles data processing and AI, while the frontend focuses on user interaction and visualization.

```
Frontend (Streamlit) → Backend (FastAPI) → Data Sources
     │                         │
     ├── Map Visualization     ├── Profile Summaries (CSV + FAISS)
     ├── Chat Interface        ├── Live ARGO API (Argopy/Erddap)
     ├── Charts & Metrics      ├── Gemini AI Integration
     └── User Controls         └── Async Data Processing
```

## 🛠️ Technical Stack

### Backend
- **Framework**: FastAPI for high-performance, asynchronous API endpoints
- **Data Retrieval**: Argopy to fetch live, detailed time-series data from the ERDDAP source
- **Vector Search**: FAISS for efficient similarity search on ocean data summaries
- **AI & Embeddings**: Google Gemini 1.5 Flash for generation and SentenceTransformer (intfloat/e5-base-v2) for text embeddings
- **Caching**: async-lru for in-memory caching of expensive API calls
- **Server**: Uvicorn

### Frontend
- **Framework**: Streamlit for rapid, interactive web app development
- **Mapping**: PyDeck for creating multi-layered, interactive geospatial maps
- **Charting**: Plotly Express for generating responsive and aesthetic time-series charts

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- A Google Gemini API Key

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Set your Google Gemini API key as an environment variable.

```bash
# On Linux/macOS
export GEMINI_API_KEY="your-gemini-api-key-here"

# On Windows (Command Prompt)
set GEMINI_API_KEY="your-gemini-api-key-here"
```

### 4. Prepare Data Files
The RAG system requires a vector index and pre-processed summary files. Run the provided data preparation script to generate them.

```bash
# This script will create the necessary index and summary files
python create_vector_db.py
```

This will generate `argo_faiss.index` and `argo_profile_summaries.csv`.

### 5. Run the Application
You need to run the backend and frontend in two separate terminals.

**Terminal 1: Start the FastAPI Backend**
```bash
python app_4.py
```
The API will be available at http://localhost:8000.

**Terminal 2: Start the Streamlit Frontend**
```bash
streamlit run streamlit_app2.py
```
The application will open in your browser at http://localhost:8501.

## 🎯 Usage

### Exploring the Map Dashboard
1. Navigate to the 🌍 Map Dashboard tab
2. Interact with the global map to see float locations. Use the Refresh button if needed
3. Click a float on the map or use the "Choose a Float ID" dropdown and click "Load Float Data"
4. The right-hand panel will automatically load and display detailed charts and statistics for the selected float
5. Click "Clear Selection" to reset the view

### Using the AI Chat
1. Navigate to the 💬 AI Chat Interface tab
2. Type a question into the input box. Use the "Example Queries" expander for ideas
3. The AI will use the ARGO database context to formulate a response

## 🔍 API Endpoints

The FastAPI backend provides the following key endpoints:

- **GET /api/live/map_data**: Provides summarized location data for all floats from `argo_profile_summaries.csv` to populate the initial map view
- **GET /api/live/float/{float_id}**: Fetches detailed, live time-series data (temp, salinity, pressure) for a specific float ID using Argopy
- **POST /query**: Accepts a natural language query and processes it through the RAG pipeline to generate an answer
- **GET /health**: A health check endpoint to verify that all components (FAISS index, data files, models) are loaded correctly

You can view interactive API documentation at http://localhost:8000/docs.

## 🌟 Acknowledgments

- The ARGO Program for providing the comprehensive, open-source oceanographic data that powers this application
- Google for the powerful Gemini models that enable our natural language interface
- The developers and communities behind FastAPI, Streamlit, FAISS, and the entire Python data science ecosystem
- Smart India Hackathon 2025 for providing the platform and opportunity to build innovative solutions
