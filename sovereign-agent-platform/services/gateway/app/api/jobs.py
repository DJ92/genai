from __future__ import annotations

import json

import asyncpg
from fastapi import APIRouter, HTTPException, Request

from app.core.config import get_settings
from app.schemas.jobs import CreateJobRequest, JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


async def _get_connection(request: Request) -> tuple[asyncpg.Connection, bool]:
    if hasattr(request.app.state, "db_pool"):
        return await request.app.state.db_pool.acquire(), True
    settings = get_settings()
    connection = await asyncpg.connect(settings.postgres_dsn)
    return connection, False


async def _release_connection(request: Request, connection: asyncpg.Connection, pooled: bool) -> None:
    if pooled:
        await request.app.state.db_pool.release(connection)
    else:
        await connection.close()


def _row_to_job_response(row: asyncpg.Record) -> JobResponse:
    raw_state = row["state"]
    if isinstance(raw_state, str):
        try:
            state = json.loads(raw_state)
            if not isinstance(state, dict):
                state = {}
        except json.JSONDecodeError:
            state = {}
    elif isinstance(raw_state, dict):
        state = raw_state
    else:
        state = {}

    return JobResponse(
        id=str(row["id"]),
        owner=row["owner"],
        workflow_name=row["workflow_name"],
        status=row["status"],
        priority=row["priority"],
        state=state,
    )


@router.post("", response_model=JobResponse)
async def create_job(payload: CreateJobRequest, request: Request) -> JobResponse:
    connection, pooled = await _get_connection(request)
    try:
        row = await connection.fetchrow(
            """
            INSERT INTO jobs (owner, status, priority, workflow_name, state)
            VALUES ($1, 'queued', $2, $3, $4::jsonb)
            RETURNING id, owner, workflow_name, status, priority, state
            """,
            payload.owner,
            payload.priority,
            payload.workflow_name,
            json.dumps(payload.state),
        )
    finally:
        await _release_connection(request, connection, pooled)
    return _row_to_job_response(row)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, request: Request) -> JobResponse:
    connection, pooled = await _get_connection(request)
    try:
        row = await connection.fetchrow(
            """
            SELECT id, owner, workflow_name, status, priority, state
            FROM jobs
            WHERE id = $1::uuid
            """,
            job_id,
        )
    finally:
        await _release_connection(request, connection, pooled)

    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _row_to_job_response(row)


@router.get("/{job_id}/events")
async def get_job_events(job_id: str, request: Request) -> dict:
    connection, pooled = await _get_connection(request)
    try:
        rows = await connection.fetch(
            """
            SELECT id::text AS id, event_type, payload, payload_sha256, created_at
            FROM events
            WHERE job_id = $1::uuid
            ORDER BY created_at ASC
            """,
            job_id,
        )
    finally:
        await _release_connection(request, connection, pooled)

    return {
        "job_id": job_id,
        "events": [
            {
                "id": row["id"],
                "event_type": row["event_type"],
                "payload": row["payload"],
                "payload_sha256": row["payload_sha256"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in rows
        ],
    }


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request) -> dict:
    connection, pooled = await _get_connection(request)
    try:
        row = await connection.fetchrow(
            """
            UPDATE jobs
            SET status = 'cancelled', updated_at = now()
            WHERE id = $1::uuid AND status IN ('queued', 'running')
            RETURNING id, status
            """,
            job_id,
        )
    finally:
        await _release_connection(request, connection, pooled)

    if row is None:
        raise HTTPException(status_code=409, detail="job cannot be cancelled in current status")
    return {"id": str(row["id"]), "status": row["status"]}
