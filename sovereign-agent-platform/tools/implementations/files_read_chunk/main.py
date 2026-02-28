from __future__ import annotations

import asyncio
import os

import asyncpg


async def _read_chunk(chunk_id: str) -> dict:
    postgres_dsn = os.getenv("POSTGRES_DSN", "postgresql://agent:agent@localhost:5432/agentdb")
    connection = await asyncpg.connect(postgres_dsn)
    try:
        row = await connection.fetchrow(
            """
            SELECT
                c.id::text AS chunk_id,
                c.content AS content,
                c.offset_start AS offset_start,
                c.offset_end AS offset_end,
                d.title AS document_title,
                d.source_uri AS source_uri
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = $1::uuid
            """,
            chunk_id,
        )
    finally:
        await connection.close()

    if not row:
        raise ValueError(f"chunk not found: {chunk_id}")

    return {
        "chunk_id": row["chunk_id"],
        "content": row["content"],
        "document_title": row["document_title"] or "",
        "source_uri": row["source_uri"],
        "offset_start": row["offset_start"] or 0,
        "offset_end": row["offset_end"] or 0,
    }


def run(args: dict) -> dict:
    chunk_id = args["chunk_id"]
    return asyncio.run(_read_chunk(chunk_id))

