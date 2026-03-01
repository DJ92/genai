from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import httpx
import yaml

from eval.harness.score import score_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run golden task harness")
    parser.add_argument("--tasks", default="eval/golden/tasks.yaml")
    parser.add_argument("--output", default="eval/golden/expected")
    parser.add_argument("--gateway-url", default="http://localhost:8000")
    parser.add_argument(
        "--postgres-dsn",
        default="postgresql://agent:agent@localhost:5432/agentdb",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def _fetch_events(connection: asyncpg.Connection, trace_id: str) -> list[dict]:
    rows = await connection.fetch(
        """
        SELECT event_type, payload, created_at
        FROM events
        WHERE trace_id = $1
        ORDER BY created_at ASC
        """,
        trace_id,
    )
    events: list[dict] = []
    for row in rows:
        events.append(
            {
                "event_type": row["event_type"],
                "payload": row["payload"],
                "created_at": row["created_at"].isoformat(),
            }
        )
    return events


async def _run_live_task(
    task: dict,
    *,
    gateway_url: str,
    client: httpx.AsyncClient,
    connection: asyncpg.Connection,
) -> tuple[dict, list[dict]]:
    response = await client.post(
        f"{gateway_url.rstrip('/')}/chat",
        json={
            "messages": [{"role": "user", "content": task["prompt"]}],
            "scopes": task.get("scopes", ["personal"]),
        },
    )
    response.raise_for_status()
    response_payload = response.json()
    trace_id = response_payload.get("trace_id")
    if not trace_id:
        return response_payload, []
    events = await _fetch_events(connection, trace_id)
    return response_payload, events


async def run() -> int:
    args = parse_args()
    tasks = yaml.safe_load(Path(args.tasks).read_text(encoding="utf-8"))

    results = []

    connection = None
    if not args.dry_run:
        connection = await asyncpg.connect(args.postgres_dsn)

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            for task in tasks:
                if args.dry_run:
                    response = {
                        "trace_id": f"dry-{task['id']}",
                        "response": task["prompt"],
                        "citations": [],
                    }
                    events: list[dict] = []
                else:
                    try:
                        response, events = await _run_live_task(
                            task,
                            gateway_url=args.gateway_url,
                            client=client,
                            connection=connection,
                        )
                    except Exception as exc:  # noqa: BLE001
                        response = {"trace_id": None, "response": "", "citations": []}
                        events = []
                        results.append(
                            {
                                "task_id": task["id"],
                                "description": task.get("description", ""),
                                "trace_id": None,
                                "pass": False,
                                "failures": [f"task execution error: {exc}"],
                            }
                        )
                        continue

                verdict = score_task(task, response, events)
                results.append(
                    {
                        "task_id": task["id"],
                        "description": task.get("description", ""),
                        "trace_id": response.get("trace_id"),
                        **verdict,
                    }
                )
    finally:
        if connection is not None:
            await connection.close()

    summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "total": len(results),
        "passed": sum(1 for result in results if result["pass"]),
        "failed": sum(1 for result in results if not result["pass"]),
        "results": results,
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"report_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not args.dry_run and summary["failed"] > 0:
        return 1
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
