from __future__ import annotations

from typing import Any

from platform_core.store import FileBackedStore
from platform_core.workflows import run_merchandising_workflow


def health(store: FileBackedStore | None = None) -> dict[str, Any]:
    working_store = store or FileBackedStore.default()
    return {"status": "ok", "service": "orchestrator", **working_store.health()}


def run_job(job_id: str, store: FileBackedStore | None = None, inline: bool = True) -> dict[str, Any]:
    working_store = store or FileBackedStore.default()
    job = working_store.get_job(job_id)
    working_store.update_job(job_id, status="running")
    result = run_merchandising_workflow(job["prompt"], scopes=job["scopes"], store=working_store)
    return working_store.update_job(job_id, status="completed", result=result, inline=inline)
