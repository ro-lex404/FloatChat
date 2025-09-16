from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import uvicorn
import pandas as pd
import plotly.express as px
import base64
from io import BytesIO
from mcp_tools import query_handler

# ----------------------------
# Input schema for POST
# ----------------------------
class QueryRequest(BaseModel):
    raw_query: str

app = FastAPI()

# ----------------------------
# 1) POST endpoint (for LLMs / API clients)
# ----------------------------
@app.post("/query_argo_data")
def query_argo_data(req: QueryRequest):
    return query_handler(req.raw_query)

# ----------------------------
# 2) GET endpoint (browser JSON)
# ----------------------------

@app.get("/query_argo_data")
def query_argo_data_get(raw_query: str):
    return query_handler(raw_query)

# ----------------------------
# 3) HTML endpoint (browser table)
# ----------------------------
@app.get("/query_html", response_class=HTMLResponse)
def query_argo_data_html(raw_query: str):
    results = query_handler(raw_query)

    if "results" not in results or not results["results"]:
        return "<h3>⚠️ No results found.</h3>"

    rows = results["results"]

    # Build HTML table
    table = "<table border='1' style='border-collapse: collapse; padding: 5px;'>"
    table += "<tr>" + "".join(f"<th>{col}</th>" for col in rows[0].keys()) + "</tr>"
    for row in rows:
        table += "<tr>" + "".join(f"<td>{row[col]}</td>" for col in row.keys()) + "</tr>"
    table += "</table>"

    return f"<h2>Query: {raw_query}</h2>{table}"

# ----------------------------
# 4) Dashboard endpoint (UI + Graphs)
# ----------------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(query: str = "Show me salinity in January 2020 near Chennai"):
    results = query_handler(query)

    if "results" not in results or not results["results"]:
        return "<h3>⚠️ No results found.</h3>"

    df = pd.DataFrame(results["results"])

    # ✅ If dataframe is empty, show message instead of blank screen
    if df.empty:
        return f"""
        <h2>🌊 Argo Data Explorer Dashboard</h2>
        <form method="get" action="/dashboard">
            <input type="text" name="query" value="{query}" size="80"/>
            <input type="submit" value="Run"/>
        </form>
        <p>⚠️ No results found for: {query}</p>
        """

    # Map graph if lat/lon present
    if "latitude" in df and "longitude" in df:
        fig = px.scatter_geo(
            df,
            lat="latitude",
            lon="longitude",
            hover_name="platform_number",
            color="temp" if "temp" in df else None,
            title="🌍 Argo Float Locations"
        )
        buf = BytesIO()
        fig.write_image(buf, format="png")
        graph_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        img_html = f"<img src='data:image/png;base64,{graph_b64}' width='800'/>"
    else:
        img_html = ""

    # Data table
    table = df.head(20).to_html(index=False)

    return f"""
    <h2>🌊 Argo Data Explorer Dashboard</h2>
    <form method="get" action="/dashboard">
        <input type="text" name="query" value="{query}" size="80"/>
        <input type="submit" value="Run"/>
    </form>
    <h3>Results for: {query}</h3>
    {table}
    <br>
    {img_html}
    """

# ----------------------------
# 5) Health check
# ----------------------------
@app.get("/ping")
def ping():
    return {"status": "✅ Server is running"}

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
