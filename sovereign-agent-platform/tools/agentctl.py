from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.harness.run_golden import execute_tasks
from platform_core.policy import validate_tool_specs
from platform_core.store import FileBackedStore
from services.gateway.app import get_job, post_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Developer CLI for the Sovereign Agent Platform.")
    parser.add_argument("--state-path", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Create and run a workflow job.")
    run_parser.add_argument("--prompt", required=True)
    run_parser.add_argument("--workflow", default="merchandising_brief")
    run_parser.add_argument("--scopes", nargs="+", default=["personal"])

    eval_parser = subparsers.add_parser("eval", help="Run the golden evaluation tasks.")
    eval_parser.add_argument("--tasks", default="eval/golden/tasks.yaml")
    eval_parser.add_argument("--output", default="eval/golden/expected")

    jobs_parser = subparsers.add_parser("jobs", help="Inspect workflow jobs.")
    jobs_subparsers = jobs_parser.add_subparsers(dest="jobs_command", required=True)
    jobs_subparsers.add_parser("list", help="List jobs")

    tool_parser = subparsers.add_parser("tools", help="Inspect or validate tool specs.")
    tool_subparsers = tool_parser.add_subparsers(dest="tools_command", required=True)
    validate_parser = tool_subparsers.add_parser("validate", help="Validate JSON tool specs")
    validate_parser.add_argument("--spec-dir", default="tools/specs")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = FileBackedStore.default(args.state_path)

    if args.command == "run":
        payload = post_jobs(
            {
                "prompt": args.prompt,
                "workflow": args.workflow,
                "scopes": args.scopes,
            },
            store=store,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "eval":
        summary = execute_tasks(args.tasks, args.output, dry_run=False, store=store)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "jobs" and args.jobs_command == "list":
        jobs = [get_job(job["id"], store=store) for job in store.list_jobs()]
        print(json.dumps({"jobs": jobs}, indent=2))
        return

    if args.command == "tools" and args.tools_command == "validate":
        report = validate_tool_specs(Path(args.spec_dir))
        print(json.dumps(report, indent=2))
        return

    raise ValueError("Unsupported command")


if __name__ == "__main__":
    main()
