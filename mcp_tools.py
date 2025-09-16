from query_local import parse_user_query, query_pipeline, results_to_json

def query_handler(raw_query: str):
    """
    MCP tool handler for querying Argo data.
    Takes a natural language query string, parses it, runs pipeline, returns JSON.
    """
    parsed = parse_user_query(raw_query)
    
    df = query_pipeline(
        user_query=parsed["parameter"] or raw_query,
        location=parsed["location"],
        start_date=parsed["start_date"],
        end_date=parsed["end_date"],
        top_k=5
    )
    
    return results_to_json(df)
