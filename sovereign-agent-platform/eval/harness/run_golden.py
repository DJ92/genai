from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from eval.harness.score import score_task
from platform_core.store import FileBackedStore
from services.gateway.app import post_chat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run golden task harness")
    parser.add_argument("--tasks", default="eval/golden/platform_tasks.yaml")
    parser.add_argument("--output", default="eval/golden/expected")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def execute_tasks(
    tasks_path: str | Path,
    output_dir: str | Path,
    dry_run: bool = False,
    store: FileBackedStore | None = None,
) -> dict:
    tasks = yaml.safe_load(Path(tasks_path).read_text(encoding="utf-8"))
    working_store = store or FileBackedStore.default()

    results = []
    for task in tasks:
        if dry_run:
            response = {"response": "dry-run", "citations": [], "policy_decision": "dry-run"}
            events = []
        else:
            response = post_chat(
                {
                    "prompt": task["prompt"],
                    "scopes": task.get("scopes", ["personal"]),
                },
                store=working_store,
            )
            events = response.get("events", [])

        verdict = score_task(task, response, events)
        results.append({"task_id": task["id"], **verdict})

    summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "total": len(results),
        "passed": sum(1 for result in results if result["pass"]),
        "failed": sum(1 for result in results if not result["pass"]),
        "results": results,
    }

    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    report_path = resolved_output / f"report_{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = execute_tasks(args.tasks, args.output, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
