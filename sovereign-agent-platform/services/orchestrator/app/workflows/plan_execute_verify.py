from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx

EventLogger = Callable[[str, dict], Awaitable[None]]


@dataclass
class WorkflowResult:
    summary: str
    state: dict


def _hash_payload(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _proposed_tools_from_request(job_state: dict, plan_text: str) -> list[dict]:
    if isinstance(job_state.get("proposed_tools"), list):
        return job_state["proposed_tools"]

    candidates = []
    lowered = plan_text.lower()
    if "search" in lowered:
        candidates.append({"name": "files_search", "args": {"query": job_state.get("request", "")}})
    if "chunk" in lowered or "read" in lowered:
        candidates.append({"name": "files_read_chunk", "args": {"chunk_id": "00000000-0000-0000-0000-000000000000"}})
    if "http" in lowered or "web" in lowered:
        candidates.append({"name": "web_fetch", "args": {"url": "https://example.com"}})
    return candidates


async def _chat(gateway_url: str, *, prompt: str, scopes: list[str] | None = None) -> dict:
    payload = {"messages": [{"role": "user", "content": prompt}], "scopes": scopes or ["personal"]}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{gateway_url.rstrip('/')}/chat", json=payload)
        response.raise_for_status()
        return response.json()


async def _opa_decision(
    opa_url: str,
    *,
    subject: str,
    action: str,
    resource: str,
    context: dict,
) -> dict:
    payload = {
        "input": {
            "subject": subject,
            "action": action,
            "resource": resource,
            "context": context,
        }
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(opa_url, json=payload)
        response.raise_for_status()
        return response.json().get("result", {})


async def _execute_tool_stub(tool_name: str, args: dict) -> dict:
    if tool_name == "files_search":
        return {"results": [{"chunk_id": "stub", "content": "stub result", "score": 0.0, "source_uri": "stub://"}]}
    if tool_name == "files_read_chunk":
        return {
            "chunk_id": args.get("chunk_id", "stub"),
            "content": "stub chunk",
            "document_title": "stub",
            "source_uri": "stub://",
            "offset_start": 0,
            "offset_end": 0,
        }
    if tool_name == "web_fetch":
        return {"url": args.get("url", "https://example.com"), "status_code": 200, "body": "stub web body"}
    return {"result": "unknown tool"}


async def run_workflow(
    job: dict,
    *,
    gateway_url: str,
    opa_url: str,
    require_approval_for_dangerous_tools: bool,
    log_event: EventLogger,
) -> WorkflowResult:
    raw_state = job.get("state")
    if isinstance(raw_state, str):
        try:
            parsed = json.loads(raw_state)
            job_state = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            job_state = {}
    elif isinstance(raw_state, dict):
        job_state = raw_state
    else:
        job_state = {}
    request_text = job_state.get("request", f"Run workflow {job.get('workflow_name')}")
    owner_subject = f"user:{job.get('owner', 'dj')}"

    plan_prompt = (
        "Create a step-by-step execution plan for this job. Include candidate tools and rationale.\n\n"
        f"Job request: {request_text}"
    )
    plan_response = await _chat(gateway_url, prompt=plan_prompt)
    plan_text = plan_response.get("response", "")
    proposed_tools = _proposed_tools_from_request(job_state, plan_text)
    await log_event(
        "job_plan",
        {
            "plan_text": plan_text,
            "proposed_tools": proposed_tools,
            "plan_trace_id": plan_response.get("trace_id"),
        },
    )

    preflight: list[dict] = []
    for tool in proposed_tools:
        tool_name = tool.get("name")
        args = tool.get("args", {})
        decision = await _opa_decision(
            opa_url,
            subject=owner_subject,
            action="tool.invoke",
            resource=tool_name,
            context={
                "tool_args": args,
                "require_approval_for_dangerous_tools": require_approval_for_dangerous_tools,
            },
        )
        preflight.append(
            {
                "tool": tool_name,
                "args": args,
                "decision": decision.get("decision", "deny"),
                "reason": decision.get("reason", ""),
            }
        )

    await log_event("policy_preflight", {"preflight": preflight})

    execution_results: list[dict] = []
    for item in preflight:
        if item["decision"] != "allow":
            execution_results.append(
                {
                    "tool": item["tool"],
                    "status": "skipped",
                    "reason": item["reason"] or item["decision"],
                }
            )
            continue

        tool_name = item["tool"]
        args = item.get("args", {})
        idempotency_key = _hash_payload({"tool_name": tool_name, "args": args})
        await log_event(
            "tool_call",
            {
                "tool_name": tool_name,
                "args_hash": _hash_payload({"args": args}),
                "idempotency_key": idempotency_key,
            },
        )

        attempts = 0
        last_error = ""
        tool_result = None
        while attempts < 3:
            attempts += 1
            try:
                tool_result = await _execute_tool_stub(tool_name, args)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        if tool_result is None:
            execution_results.append(
                {
                    "tool": tool_name,
                    "status": "failed",
                    "error": last_error or "execution failed",
                    "attempts": attempts,
                }
            )
            await log_event(
                "tool_result",
                {
                    "tool_name": tool_name,
                    "success": False,
                    "duration_ms": 0,
                    "output_hash": _hash_payload({"error": last_error}),
                },
            )
            continue

        execution_results.append(
            {
                "tool": tool_name,
                "status": "completed",
                "attempts": attempts,
                "result": tool_result,
            }
        )
        await log_event(
            "tool_result",
            {
                "tool_name": tool_name,
                "success": True,
                "duration_ms": 0,
                "output_hash": _hash_payload(tool_result),
            },
        )

    verify_prompt = (
        "Verify this execution result for consistency, schema correctness, and citation completeness.\n\n"
        f"Plan: {plan_text}\n\nResults: {json.dumps(execution_results)}"
    )
    verify_response = await _chat(gateway_url, prompt=verify_prompt)
    await log_event(
        "job_verify",
        {
            "verify_response": verify_response.get("response"),
            "verify_trace_id": verify_response.get("trace_id"),
        },
    )

    summary_prompt = (
        "Create a final concise summary artifact for this workflow execution.\n\n"
        f"Request: {request_text}\n\nExecution: {json.dumps(execution_results)}\n\n"
        f"Verification: {verify_response.get('response', '')}"
    )
    summary_response = await _chat(gateway_url, prompt=summary_prompt)
    summary_text = summary_response.get("response", "")
    await log_event(
        "job_summary",
        {
            "summary": summary_text,
            "summary_trace_id": summary_response.get("trace_id"),
        },
    )

    return WorkflowResult(
        summary=summary_text,
        state={
            "request": request_text,
            "plan": {"text": plan_text, "tools": proposed_tools},
            "preflight": preflight,
            "execute": execution_results,
            "verify": verify_response.get("response", ""),
            "summary": summary_text,
        },
    )
