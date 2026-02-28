def run(args: dict) -> dict:
    chunk_id = args["chunk_id"]
    return {
        "chunk_id": chunk_id,
        "content": "Stub chunk content",
        "document_title": "Stub Document",
        "source_uri": "stub://local",
        "offset_start": 0,
        "offset_end": 0,
    }
