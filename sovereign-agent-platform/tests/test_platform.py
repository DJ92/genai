from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from platform_core.store import FileBackedStore
from services.gateway.app import get_job, post_chat, post_jobs
from services.ingestion.app import post_ingest


def test_chat_returns_budget_fact_and_citations(tmp_path: Path) -> None:
    store = FileBackedStore.default(tmp_path / "state.json")
    response = post_chat(
        {"prompt": "What did the meeting notes from January say about the budget?", "scopes": ["personal"]},
        store=store,
    )
    assert "$50,000" in response["response"]
    assert response["citations"]


def test_policy_gates_web_fetch(tmp_path: Path) -> None:
    store = FileBackedStore.default(tmp_path / "state.json")
    response = post_chat(
        {"prompt": "Fetch the contents of https://example.com/data.json", "scopes": ["personal"]},
        store=store,
    )
    assert response["policy_decision"] == "needs_approval"
    tool_calls = [event for event in response["events"] if event["event_type"] == "tool_call"]
    assert tool_calls[0]["payload"]["tool_name"] == "web_fetch"


def test_jobs_and_ingestion_round_trip(tmp_path: Path) -> None:
    store = FileBackedStore.default(tmp_path / "state.json")
    post_ingest(
        {
            "documents": [
                {
                    "title": "Ops Escalation Memo",
                    "scope": "personal",
                    "content": "Escalate publish actions when inventory confidence falls below threshold.",
                }
            ]
        },
        store=store,
    )
    job = post_jobs(
        {
            "prompt": "Create a merchandising brief that references the new ops memo.",
            "workflow": "merchandising_brief",
            "scopes": ["personal"],
        },
        store=store,
    )
    fetched = get_job(job["id"], store=store)
    assert fetched["status"] == "completed"
    assert fetched["result"]["workflow_steps"]


def test_agentctl_and_golden_eval(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["SOVEREIGN_STATE_PATH"] = str(tmp_path / "cli_state.json")

    run = subprocess.run(
        ["python", "-m", "tools.agentctl", "run", "--prompt", "Create a merchandising brief for January planning notes"],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    run_payload = json.loads(run.stdout)
    assert run_payload["status"] == "completed"

    listed = subprocess.run(
        ["python", "-m", "tools.agentctl", "jobs", "list"],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    listed_payload = json.loads(listed.stdout)
    assert listed_payload["jobs"]

    tools = subprocess.run(
        ["python", "-m", "tools.agentctl", "tools", "validate"],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    tool_payload = json.loads(tools.stdout)
    assert tool_payload["checked"] >= 1

    evaluation = subprocess.run(
        ["python", "-m", "tools.agentctl", "eval", "--output", str(tmp_path / "reports")],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    eval_payload = json.loads(evaluation.stdout)
    assert eval_payload["passed"] == eval_payload["total"]
