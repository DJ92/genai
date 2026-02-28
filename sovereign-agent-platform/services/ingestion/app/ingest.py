from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path

import asyncpg

from app.chunkers.heading_aware import chunk_document
from app.embedders.via_model_gateway import ModelGatewayEmbedder
from app.parsers.docx import parse_docx
from app.parsers.html import parse_html
from app.parsers.md import parse_markdown
from app.parsers.pdf import parse_pdf

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".html", ".htm"}
MIME_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into sovereign-agent-platform")
    parser.add_argument("--path", required=True, help="Path to folder or file")
    parser.add_argument("--scope", default="personal", help="Scope label for documents")
    parser.add_argument(
        "--postgres-dsn",
        default="postgresql://agent:agent@localhost:5432/agentdb",
        help="Postgres DSN",
    )
    parser.add_argument(
        "--model-gateway-url",
        default="http://localhost:8001",
        help="Model gateway URL",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="Chunk token limit")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parser_for(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf
    if suffix == ".docx":
        return parse_docx
    if suffix == ".md":
        return parse_markdown
    if suffix in {".html", ".htm"}:
        return parse_html
    raise ValueError(f"unsupported extension: {suffix}")


def gather_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_SUFFIXES else []
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files)


def vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(value) for value in embedding) + "]"


async def upsert_document(
    connection: asyncpg.Connection,
    *,
    file_path: Path,
    file_sha256: str,
    scope: str,
    mime_type: str,
) -> tuple[str, bool]:
    source_uri = str(file_path.resolve())
    existing = await connection.fetchrow(
        """
        SELECT id::text AS id, sha256
        FROM documents
        WHERE source_uri = $1
        ORDER BY created_at DESC
        LIMIT 1
        """,
        source_uri,
    )
    if existing and existing["sha256"] == file_sha256:
        return existing["id"], True

    document_id = await connection.fetchval(
        """
        INSERT INTO documents (source_uri, sha256, mime_type, scope, title)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id::text
        """,
        source_uri,
        file_sha256,
        mime_type,
        scope,
        file_path.name,
    )
    return document_id, False


async def ingest_file(
    connection: asyncpg.Connection,
    embedder: ModelGatewayEmbedder,
    *,
    file_path: Path,
    scope: str,
    max_tokens: int,
) -> dict:
    file_sha = sha256_file(file_path)
    mime_type = MIME_BY_SUFFIX[file_path.suffix.lower()]
    document_id, unchanged = await upsert_document(
        connection, file_path=file_path, file_sha256=file_sha, scope=scope, mime_type=mime_type
    )
    if unchanged:
        return {"file": str(file_path), "status": "skipped", "reason": "hash unchanged"}

    text = parser_for(file_path)(str(file_path))
    chunks = chunk_document(text, max_tokens=max_tokens)
    inserted = 0

    for index, chunk in enumerate(chunks):
        chunk_id = await connection.fetchval(
            """
            INSERT INTO chunks (document_id, chunk_index, content, content_sha256, offset_start, offset_end)
            VALUES ($1::uuid, $2, $3, $4, $5, $6)
            RETURNING id::text
            """,
            document_id,
            index,
            chunk.content,
            content_sha256(chunk.content),
            chunk.offset_start,
            chunk.offset_end,
        )
        embedding = await embedder.embed(chunk.content)
        if embedding:
            await connection.execute(
                """
                INSERT INTO embeddings (chunk_id, embedding_model_id, embedding)
                VALUES ($1::uuid, $2, $3::vector)
                ON CONFLICT (chunk_id, embedding_model_id)
                DO UPDATE SET embedding = EXCLUDED.embedding, created_at = now()
                """,
                chunk_id,
                "default",
                vector_literal(embedding),
            )
        inserted += 1

    return {"file": str(file_path), "status": "ingested", "document_id": document_id, "chunks": inserted}


async def run_ingestion(args: argparse.Namespace) -> list[dict]:
    root = Path(args.path)
    if not root.exists():
        raise FileNotFoundError(f"path does not exist: {root}")

    files = gather_files(root)
    embedder = ModelGatewayEmbedder(args.model_gateway_url)
    results: list[dict] = []

    connection = await asyncpg.connect(args.postgres_dsn)
    try:
        for path in files:
            result = await ingest_file(
                connection,
                embedder,
                file_path=path,
                scope=args.scope,
                max_tokens=args.max_tokens,
            )
            results.append(result)
    finally:
        await connection.close()

    return results


def main() -> None:
    args = parse_args()
    results = asyncio.run(run_ingestion(args))
    summary = {
        "path": args.path,
        "scope": args.scope,
        "processed": len(results),
        "ingested": sum(1 for item in results if item["status"] == "ingested"),
        "skipped": sum(1 for item in results if item["status"] == "skipped"),
        "results": results,
    }
    print(summary)


if __name__ == "__main__":
    main()

