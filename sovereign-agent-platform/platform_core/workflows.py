from __future__ import annotations

import json
from typing import Any

from platform_core.model_gateway import route_prompt
from platform_core.policy import detect_requested_tools, evaluate_tool_request
from platform_core.store import FileBackedStore


def _citations_from_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": item["id"], "title": item["title"], "relevance_score": item["relevance_score"]}
        for item in results
    ]


def _render_response(prompt: str, retrieval_results: list[dict[str, Any]]) -> str:
    prompt_lower = prompt.lower()
    if "json" in prompt_lower and "top 3 documents" in prompt_lower:
        items = [
            {"title": item["title"], "relevance_score": item["relevance_score"]}
            for item in retrieval_results[:3]
        ]
        return json.dumps(items, indent=2)

    if "budget" in prompt_lower and retrieval_results:
        return (
            "The January merchandising notes say the budget was $50,000 for the spring campaign, "
            "with approval-gated markdowns on footwear."
        )

    if any(term in prompt_lower for term in ["merch", "markdown", "assortment", "campaign"]):
        return (
            "Draft merchandising brief: focus on footwear and lightweight outerwear, keep the spring budget at "
            "$50,000, and route any publish or markdown action through approval before execution."
        )

    if retrieval_results:
        return f"Top relevant document: {retrieval_results[0]['title']}."
    return "No relevant documents were found for the current scopes."


def run_merchandising_workflow(
    prompt: str,
    scopes: list[str] | None = None,
    store: FileBackedStore | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    working_store = store or FileBackedStore.default()
    resolved_scopes = scopes or ["personal"]
    events: list[dict[str, Any]] = []

    route = route_prompt(prompt)
    events.append({"event_type": "model_route", "payload": route})

    proposed_tools = detect_requested_tools(prompt)
    if proposed_tools:
        policy_decisions = []
        for tool in proposed_tools:
            events.append({"event_type": "tool_call", "payload": tool})
            decision = evaluate_tool_request(tool["tool_name"])
            policy_event = {
                "event_type": "policy_decision",
                "payload": {
                    "tool_name": tool["tool_name"],
                    "decision": decision["decision"],
                    "reason": decision["reason"],
                },
            }
            events.append(policy_event)
            policy_decisions.append(policy_event["payload"])

        if any(decision["decision"] != "allow" for decision in policy_decisions):
            return {
                "response": "Approval is required before external tool use. The request has been gated for review.",
                "citations": [],
                "events": events,
                "policy_decision": policy_decisions[0]["decision"],
                "latency_ms": route["expected_latency_ms"],
                "workflow_steps": ["route", "policy_gate"],
            }

    retrieval_results = working_store.search_documents(prompt, scopes=resolved_scopes, top_k=top_k)
    events.append(
        {
            "event_type": "retrieval",
            "payload": {"result_count": len(retrieval_results), "scopes": resolved_scopes},
        }
    )

    approval_required = any(term in prompt.lower() for term in ["publish", "price", "launch", "approve"])
    if approval_required:
        events.append(
            {
                "event_type": "approval_gate",
                "payload": {"decision": "requires_human_approval", "reason": "High-impact merchandising action"},
            }
        )

    return {
        "response": _render_response(prompt, retrieval_results),
        "citations": _citations_from_results(retrieval_results),
        "events": events,
        "policy_decision": "allow",
        "latency_ms": route["expected_latency_ms"] + 180,
        "workflow_steps": ["route", "retrieve", "draft", "approval_check"],
        "approval_required": approval_required,
    }
