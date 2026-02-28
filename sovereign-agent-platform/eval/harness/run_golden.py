import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from eval.harness.score import score_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run golden task harness")
    parser.add_argument("--tasks", default="eval/golden/tasks.yaml")
    parser.add_argument("--output", default="eval/golden/expected")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = yaml.safe_load(Path(args.tasks).read_text(encoding="utf-8"))

    results = []
    for task in tasks:
        if args.dry_run:
            response = {"response": "dry-run", "citations": []}
            events = []
        else:
            response = {"response": "not_implemented", "citations": []}
            events = []

        verdict = score_task(task, response, events)
        results.append({"task_id": task["id"], **verdict})

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


if __name__ == "__main__":
    main()
