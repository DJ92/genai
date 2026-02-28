from __future__ import annotations

import logging

import asyncpg

from app.adapters.model_gateway_client import ModelGatewayClient
from app.schemas.chat import Citation

logger = logging.getLogger(__name__)


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


async def retrieve(
    *,
    query: str,
    allowed_scopes: list[str],
    db_pool: asyncpg.Pool,
    model_gateway_client: ModelGatewayClient,
    k: int = 8,
) -> list[Citation]:
    if not allowed_scopes:
        return []

    query_embedding = await model_gateway_client.embed(query)
    if not query_embedding:
        return []

    vector_param = _vector_literal(query_embedding)
    sql = """
        SELECT
            c.id::text AS chunk_id,
            c.document_id::text AS document_id,
            c.content AS content,
            c.offset_start AS offset_start,
            c.offset_end AS offset_end,
            d.title AS document_title,
            d.source_uri AS source_uri,
            (e.embedding <=> $1::vector) AS distance
        FROM embeddings e
        JOIN chunks c ON e.chunk_id = c.id
        JOIN documents d ON c.document_id = d.id
        WHERE d.scope = ANY($2::text[])
        ORDER BY e.embedding <=> $1::vector
        LIMIT $3
    """

    rows = await db_pool.fetch(sql, vector_param, allowed_scopes, k)
    citations: list[Citation] = []
    for row in rows:
        distance = row["distance"] if row["distance"] is not None else 1.0
        citations.append(
            Citation(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                document_title=row["document_title"],
                source_uri=row["source_uri"],
                content=row["content"],
                score=max(0.0, 1.0 - float(distance)),
                offsets=(row["offset_start"], row["offset_end"]),
            )
        )
    return citations
