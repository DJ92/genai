from __future__ import annotations

from typing import Any

from platform_core.model_gateway import route_prompt


def health() -> dict[str, Any]:
    return {"status": "ok", "service": "model_gateway", "latency_budget_ms": 10000}


def route(request: dict[str, Any]) -> dict[str, Any]:
    return route_prompt(request["prompt"])
