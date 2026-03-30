from __future__ import annotations

from typing import Any

from platform_core.store import FileBackedStore
from services.orchestrator.app import run_job


def health(store: FileBackedStore | None = None) -> dict[str, Any]:
    working_store = store or FileBackedStore.default()
    return {"status": "ok", "service": "gateway", **working_store.health()}


def post_chat(request: dict[str, Any], store: FileBackedStore | None = None) -> dict[str, Any]:
    working_store = store or FileBackedStore.default()
    job = working_store.create_job(
        workflow=request.get("workflow", "merchandising_brief"),
        prompt=request["prompt"],
        scopes=request.get("scopes", ["personal"]),
        metadata={"request_type": "chat"},
    )
    result = run_job(job["id"], store=working_store, inline=True)
    return {
        "job_id": job["id"],
        "response": result["result"]["response"],
        "citations": result["result"]["citations"],
        "events": result["result"]["events"],
        "policy_decision": result["result"]["policy_decision"],
        "workflow_steps": result["result"]["workflow_steps"],
        "latency_ms": result["result"]["latency_ms"],
    }


def post_jobs(request: dict[str, Any], store: FileBackedStore | None = None) -> dict[str, Any]:
    working_store = store or FileBackedStore.default()
    job = working_store.create_job(
        workflow=request.get("workflow", "merchandising_brief"),
        prompt=request["prompt"],
        scopes=request.get("scopes", ["personal"]),
        metadata={"request_type": "job"},
    )
    completed = run_job(job["id"], store=working_store, inline=True)
    return completed


def get_job(job_id: str, store: FileBackedStore | None = None) -> dict[str, Any]:
    working_store = store or FileBackedStore.default()
    return working_store.get_job(job_id)
