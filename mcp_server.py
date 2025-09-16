from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import uvicorn
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
    """
    Exposed tool for LLMs via MCP (expects POST JSON).
    """
    return query_handler(req.raw_query)


# ----------------------------
# 2) GET endpoint (browser JSON)
# ----------------------------
@app.get("/query_argo_data")
def query_argo_data_get(raw_query: str):
    """
    Browser/URL friendly JSON.
    Example: http://127.0.0.1:8000/query_argo_data?raw_query=Show+me+salinity
    """
    return query_handler(raw_query)


# ----------------------------
# 3) HTML endpoint (browser human-friendly)
# ----------------------------
@app.get("/query_html", response_class=HTMLResponse)
def query_argo_data_html(raw_query: str):
    """
    Human-friendly HTML view of query results.
    Example: http://127.0.0.1:8000/query_html?raw_query=Show+me+salinity
    """
    results = query_handler(raw_query)

    if "results" not in results or not results["results"]:
        return "<h3>⚠️ No results found.</h3>"

    rows = results["results"]

    # Build table
    table = "<table border='1' style='border-collapse: collapse; padding: 5px;'>"
    table += "<tr>" + "".join(f"<th>{col}</th>" for col in rows[0].keys()) + "</tr>"
    for row in rows:
        table += "<tr>" + "".join(f"<td>{row[col]}</td>" for col in row.keys()) + "</tr>"
    table += "</table>"

    # Add optional graph if exists
    graph_html = ""
    if "plot" in results:
        graph_html = f"<h4>Graph:</h4><img src='data:image/png;base64,{results['plot']}' />"

    return f"<h2>Query: {raw_query}</h2>{table}{graph_html}"


# ----------------------------
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(query: str = "Show me salinity in January 2020 near Chennai"):
    ...

# 4) Health check
# ----------------------------
@app.get("/ping")
def ping():
    return {"status": "✅ Server is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
