from __future__ import annotations

import asyncio
import hashlib
import json
import logging

import asyncpg

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.workflows.plan_execute_verify import run_workflow

logger = logging.getLogger(__name__)


def _payload_sha256(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


async def _log_event(
    connection: asyncpg.Connection,
    *,
    trace_id: str,
    event_type: str,
    payload: dict,
    job_id: str | None = None,
) -> None:
    settings = get_settings()
    event_payload = {
        **payload,
        "versions": {
            "model_backend_id": "unknown",
            "prompt_version": "v0.1",
            "policy_bundle_version": settings.policy_bundle_version,
            "tool_version": "v0.1",
        },
    }
    await connection.execute(
        """
        INSERT INTO events (job_id, trace_id, event_type, payload, payload_sha256)
        VALUES ($1::uuid, $2, $3, $4::jsonb, $5)
        """,
        job_id,
        trace_id,
        event_type,
        json.dumps(event_payload),
        _payload_sha256(event_payload),
    )


async def _claim_next_job(connection: asyncpg.Connection) -> asyncpg.Record | None:
    return await connection.fetchrow(
        """
        WITH next_job AS (
            SELECT id
            FROM jobs
            WHERE status = 'queued'
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        UPDATE jobs
        SET status = 'running', updated_at = now()
        WHERE id = (SELECT id FROM next_job)
        RETURNING id::text AS id, owner, status, priority, workflow_name, state
        """
    )


async def _run_job(connection: asyncpg.Connection, job: asyncpg.Record) -> None:
    settings = get_settings()
    trace_id = f"job-{job['id']}"
    await _log_event(
        connection,
        trace_id=trace_id,
        event_type="job_transition",
        payload={"old_status": "queued", "new_status": "running", "job_id": job["id"]},
        job_id=job["id"],
    )

    try:
        result = await run_workflow(
            dict(job),
            gateway_url=settings.gateway_url,
            opa_url=settings.opa_url,
            require_approval_for_dangerous_tools=settings.require_approval_for_dangerous_tools,
            log_event=lambda event_type, payload: _log_event(
                connection,
                trace_id=trace_id,
                event_type=event_type,
                payload=payload,
                job_id=job["id"],
            ),
        )

        await connection.execute(
            """
            UPDATE jobs
            SET status = 'completed', state = $2::jsonb, updated_at = now()
            WHERE id = $1::uuid
            """,
            job["id"],
            json.dumps(result.state),
        )
        await _log_event(
            connection,
            trace_id=trace_id,
            event_type="job_transition",
            payload={"old_status": "running", "new_status": "completed", "job_id": job["id"]},
            job_id=job["id"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("workflow failed for job %s", job["id"])
        await connection.execute(
            """
            UPDATE jobs
            SET status = 'failed',
                state = COALESCE(state, '{}'::jsonb) || $2::jsonb,
                updated_at = now()
            WHERE id = $1::uuid
            """,
            job["id"],
            json.dumps({"error": str(exc)}),
        )
        await _log_event(
            connection,
            trace_id=trace_id,
            event_type="job_transition",
            payload={"old_status": "running", "new_status": "failed", "job_id": job["id"]},
            job_id=job["id"],
        )


async def worker_loop() -> None:
    settings = get_settings()
    pool = await asyncpg.create_pool(settings.postgres_dsn, min_size=1, max_size=5)
    logger.info("orchestrator worker started")
    try:
        while True:
            async with pool.acquire() as connection:
                async with connection.transaction():
                    job = await _claim_next_job(connection)
                if job is None:
                    await asyncio.sleep(settings.poll_interval_seconds)
                    continue
                await _run_job(connection, job)
    finally:
        await pool.close()


if __name__ == "__main__":
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(worker_loop())

