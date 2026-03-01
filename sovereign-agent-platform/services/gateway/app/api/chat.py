from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.adapters.retrieval import retrieve
from app.core.auth import get_subject
from app.core.config import get_settings
from app.core.tracing import new_trace_id, sha256_json
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.policy import PolicyDecision

router = APIRouter(prefix="/chat", tags=["chat"])


async def _log_event(request: Request, *, trace_id: str, event_type: str, payload: dict, job_id: str | None = None) -> None:
    db_pool = request.app.state.db_pool
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
    payload_sha256 = sha256_json(event_payload)
    await db_pool.execute(
        """
        INSERT INTO events (job_id, trace_id, event_type, payload, payload_sha256)
        VALUES ($1, $2, $3, $4::jsonb, $5)
        """,
        job_id,
        trace_id,
        event_type,
        json.dumps(event_payload),
        payload_sha256,
    )


async def _log_policy_decision(
    request: Request,
    *,
    trace_id: str,
    subject: str,
    action: str,
    resource: str,
    decision: PolicyDecision,
) -> None:
    settings = get_settings()
    await request.app.state.db_pool.execute(
        """
        INSERT INTO policy_decisions
            (trace_id, subject, action, resource, decision, reason, policy_bundle_version)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        trace_id,
        subject,
        action,
        resource,
        decision.decision,
        decision.reason,
        settings.policy_bundle_version,
    )
    await _log_event(
        request,
        trace_id=trace_id,
        event_type="policy_decision",
        payload={
            "action": action,
            "resource": resource,
            "decision": decision.decision,
            "reason": decision.reason,
        },
    )


def _latest_user_message(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _extract_text(model_response: dict) -> str:
    choices = model_response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def _extract_tool_calls(model_response: dict) -> list[dict]:
    choices = model_response.get("choices", [])
    if not choices:
        return []

    tool_calls = choices[0].get("message", {}).get("tool_calls", [])
    extracted: list[dict] = []
    for call in tool_calls:
        function_data = call.get("function", {})
        name = function_data.get("name") or call.get("name")
        raw_arguments = function_data.get("arguments", call.get("arguments", "{}"))
        if isinstance(raw_arguments, str):
            try:
                args = json.loads(raw_arguments)
            except json.JSONDecodeError:
                args = {"_raw": raw_arguments}
        else:
            args = raw_arguments
        extracted.append({"id": call.get("id"), "name": name, "args": args})
    return [call for call in extracted if call.get("name")]


def _tool_prompt_spec(tool_name: str, spec: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": spec.get("description", ""),
            "parameters": spec.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


@router.post("", response_model=ChatResponse)
async def chat(request_payload: ChatRequest, request: Request, subject: str = Depends(get_subject)) -> ChatResponse:
    settings = get_settings()
    trace_id = new_trace_id()

    if not hasattr(request.app.state, "db_pool"):
        raise HTTPException(status_code=500, detail="database pool not initialized")

    policy_client = request.app.state.policy_client
    model_client = request.app.state.model_gateway_client
    tool_router = request.app.state.tool_router

    messages = [message.model_dump() for message in request_payload.messages]
    latest_user_text = _latest_user_message(messages)

    await _log_event(
        request,
        trace_id=trace_id,
        event_type="chat_in",
        payload={"message": latest_user_text, "user_id": subject},
    )

    data_decision = await policy_client.decide(
        subject=subject,
        action="data.read",
        resource="scopes",
        context={"request_scopes": request_payload.scopes},
    )
    await _log_policy_decision(
        request,
        trace_id=trace_id,
        subject=subject,
        action="data.read",
        resource="scopes",
        decision=data_decision,
    )

    allowed_scopes = (
        data_decision.constraints.get("allowed_scopes", [])
        if data_decision.decision in {"allow", "needs_approval"}
        else []
    )

    requested_tools = request_payload.tools or tool_router.list_tools()
    tool_prompt_specs: list[dict] = []
    for tool_name in requested_tools:
        decision = await policy_client.decide(
            subject=subject,
            action="tool.invoke",
            resource=tool_name,
            context={
                "request_scopes": request_payload.scopes,
                "require_approval_for_dangerous_tools": settings.require_approval_for_dangerous_tools,
            },
        )
        await _log_policy_decision(
            request,
            trace_id=trace_id,
            subject=subject,
            action="tool.invoke",
            resource=tool_name,
            decision=decision,
        )
        if decision.decision in {"allow", "needs_approval"}:
            tool_prompt_specs.append(_tool_prompt_spec(tool_name, tool_router.load_spec(tool_name)))

    await _log_event(
        request,
        trace_id=trace_id,
        event_type="retrieval_query",
        payload={
            "query_hash": sha256_json({"query": latest_user_text}),
            "scopes": allowed_scopes,
            "k": settings.max_retrieval_k,
        },
    )
    citations = await retrieve(
        query=latest_user_text,
        allowed_scopes=allowed_scopes,
        db_pool=request.app.state.db_pool,
        model_gateway_client=model_client,
        k=settings.max_retrieval_k,
    )
    await _log_event(
        request,
        trace_id=trace_id,
        event_type="retrieval_results",
        payload={
            "num_results": len(citations),
            "top_scores": [citation.score for citation in citations[:3]],
        },
    )

    context_lines = []
    for citation in citations:
        context_lines.append(
            (
                f"[chunk:{citation.chunk_id}] "
                f"title={citation.document_title or 'untitled'} "
                f"source={citation.source_uri or 'unknown'} "
                f"offsets={citation.offsets}: "
                f"{citation.content or ''}"
            )
        )
    tools_summary = ", ".join(spec["function"]["name"] for spec in tool_prompt_specs) or "none"
    system_message = (
        "You are a policy-governed assistant. Use retrieved evidence when available and cite chunk IDs. "
        f"Available tools: {tools_summary}. "
        "If no local evidence exists, state that the answer is from model knowledge."
    )
    if context_lines:
        system_message += "\n\nRetrieved context:\n" + "\n".join(context_lines)

    prompt_messages = [{"role": "system", "content": system_message}, *messages]

    await _log_event(
        request,
        trace_id=trace_id,
        event_type="model_call",
        payload={
            "prompt_hash": sha256_json({"messages": prompt_messages}),
            "model_name": "default",
            "backend_id": "unknown",
        },
    )
    model_response = await model_client.chat(messages=prompt_messages, tools=tool_prompt_specs)
    await _log_event(
        request,
        trace_id=trace_id,
        event_type="model_result",
        payload={
            "response_hash": sha256_json({"response": model_response}),
            "token_usage": model_response.get("usage", {}),
            "latency_ms": model_response.get("metadata", {}).get("latency_ms"),
        },
    )

    pending_approvals: list[dict[str, Any]] = []
    tool_calls = _extract_tool_calls(model_response)
    tool_result_messages: list[dict] = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        idempotency_key = sha256_json({"tool_name": tool_name, "args": args})

        await _log_event(
            request,
            trace_id=trace_id,
            event_type="tool_call",
            payload={
                "tool_name": tool_name,
                "args_hash": sha256_json({"args": args}),
                "idempotency_key": idempotency_key,
            },
        )

        try:
            tool_router.validate_input(tool_name, args)
        except Exception as exc:  # noqa: BLE001
            tool_result_messages.append(
                {"role": "tool", "content": json.dumps({"tool": tool_name, "validation_error": str(exc)})}
            )
            continue

        decision = await policy_client.decide(
            subject=subject,
            action="tool.invoke",
            resource=tool_name,
            context={
                "request_scopes": request_payload.scopes,
                "tool_args": args,
                "require_approval_for_dangerous_tools": settings.require_approval_for_dangerous_tools,
            },
        )
        await _log_policy_decision(
            request,
            trace_id=trace_id,
            subject=subject,
            action="tool.invoke",
            resource=tool_name,
            decision=decision,
        )

        if decision.decision == "needs_approval":
            pending_approvals.append({"tool": tool_name, "args": args, "reason": decision.reason})
            continue
        if decision.decision != "allow":
            tool_result_messages.append(
                {
                    "role": "tool",
                    "content": json.dumps({"tool": tool_name, "denied": True, "reason": decision.reason}),
                }
            )
            continue

        result = tool_router.execute(tool_name, args, timeout_seconds=settings.tools_timeout_seconds)
        await _log_event(
            request,
            trace_id=trace_id,
            event_type="tool_result",
            payload={
                "tool_name": tool_name,
                "success": result.get("success", False),
                "duration_ms": result.get("metadata", {}).get("duration_ms"),
                "output_hash": sha256_json({"data": result.get("data"), "error": result.get("error")}),
            },
        )
        tool_result_messages.append(
            {"role": "tool", "content": json.dumps({"tool": tool_name, "result": result}, default=str)}
        )

    if pending_approvals:
        response_text = "Tool execution pending approval."
    else:
        response_text = _extract_text(model_response)
        if model_response.get("fallback_reason") and citations and "json" not in latest_user_text.lower():
            response_text = " ".join(citation.content or "" for citation in citations[:2]).strip() or response_text
        if tool_result_messages:
            second_response = await model_client.chat(
                messages=[
                    *prompt_messages,
                    {"role": "assistant", "content": response_text},
                    *tool_result_messages,
                    {"role": "user", "content": "Incorporate the tool results into the final answer."},
                ],
            )
            response_text = _extract_text(second_response)

    has_local_evidence = bool(citations)
    evidence_source = "local_retrieval" if has_local_evidence else "model_knowledge"
    await _log_event(
        request,
        trace_id=trace_id,
        event_type="chat_out",
        payload={
            "response": response_text,
            "citations": [citation.model_dump() for citation in citations],
            "evidence_source": evidence_source,
            "has_local_evidence": has_local_evidence,
        },
    )

    return ChatResponse(
        trace_id=trace_id,
        response=response_text,
        citations=citations,
        evidence_source=evidence_source,
        has_local_evidence=has_local_evidence,
        pending_approvals=pending_approvals,
    )
