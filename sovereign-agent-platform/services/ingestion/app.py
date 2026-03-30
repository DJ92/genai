from __future__ import annotations

from typing import Any

from platform_core.store import FileBackedStore


def health(store: FileBackedStore | None = None) -> dict[str, Any]:
    working_store = store or FileBackedStore.default()
    return {"status": "ok", "service": "ingestion", **working_store.health()}


def post_ingest(request: dict[str, Any], store: FileBackedStore | None = None) -> dict[str, Any]:
    working_store = store or FileBackedStore.default()
    return working_store.ingest_documents(request.get("documents", []))
