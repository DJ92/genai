from __future__ import annotations

import asyncio
import os

import asyncpg
import requests


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


async def _run_search(query: str, scope: str, k: int) -> dict:
    postgres_dsn = os.getenv("POSTGRES_DSN", "postgresql://agent:agent@localhost:5432/agentdb")
    model_gateway_url = os.getenv("MODEL_GATEWAY_URL", "http://localhost:8001").rstrip("/")

    embed_response = requests.post(
        f"{model_gateway_url}/v1/embeddings",
        json={"input": query},
        timeout=30,
    )
    embed_response.raise_for_status()
    payload = embed_response.json()
    embedding = payload.get("data", [{}])[0].get("embedding", [])
    if not embedding:
        return {"results": []}

    connection = await asyncpg.connect(postgres_dsn)
    try:
        rows = await connection.fetch(
            """
            SELECT
                c.id::text AS chunk_id,
                c.content AS content,
                d.source_uri AS source_uri,
                (e.embedding <=> $1::vector) AS distance
            FROM embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE d.scope = $2
            ORDER BY e.embedding <=> $1::vector
            LIMIT $3
            """,
            _vector_literal(embedding),
            scope,
            k,
        )
    finally:
        await connection.close()

    return {
        "results": [
            {
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "score": max(0.0, 1.0 - float(row["distance"])),
                "source_uri": row["source_uri"],
            }
            for row in rows
        ]
    }


def run(args: dict) -> dict:
    query = args.get("query", "")
    scope = args.get("scope", "personal")
    k = int(args.get("k", 5))
    return asyncio.run(_run_search(query=query, scope=scope, k=k))

