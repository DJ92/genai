from __future__ import annotations

from typing import Any


def route_prompt(prompt: str) -> dict[str, Any]:
    token_count = len(prompt.split())
    wants_json = "json" in prompt.lower()
    high_stakes = any(term in prompt.lower() for term in ["publish", "price", "launch", "approve"])
    escalate = token_count > 24 or wants_json or high_stakes
    return {
        "path": "small-model-first" if not escalate else "escalated",
        "selected_model": "small-instruct" if not escalate else "large-instruct",
        "expected_latency_ms": 3200 if not escalate else 7800,
        "reason": "Escalated for structured or higher-stakes output." if escalate else "Handled on the small model path.",
    }
