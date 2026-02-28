from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings
from app.schemas.policy import PolicyDecision

logger = logging.getLogger(__name__)


class PolicyClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._url = settings.opa_url
        self._timeout = httpx.Timeout(10.0)

    async def decide(self, *, subject: str, action: str, resource: str, context: dict) -> PolicyDecision:
        payload = {
            "input": {
                "subject": subject,
                "action": action,
                "resource": resource,
                "context": context,
            }
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._url, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.exception("policy request failed")
            return PolicyDecision(decision="deny", reason=f"policy-unreachable: {exc}", constraints={})

        result = response.json().get("result") or {}
        return PolicyDecision(
            decision=result.get("decision", "deny"),
            reason=result.get("reason", "missing policy reason"),
            constraints=result.get("constraints", {}),
        )
