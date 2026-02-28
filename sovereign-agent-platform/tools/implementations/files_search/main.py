def run(args: dict) -> dict:
    query = args.get("query", "")
    return {
        "results": [
            {
                "chunk_id": "stub-chunk-1",
                "content": f"Stub search result for query: {query}",
                "score": 0.0,
                "source_uri": "stub://local",
            }
        ]
    }
