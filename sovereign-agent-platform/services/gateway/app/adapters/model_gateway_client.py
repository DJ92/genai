from __future__ import annotations

import httpx

from app.core.config import get_settings


class ModelGatewayClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.model_gateway_url.rstrip("/")
        self._timeout = httpx.Timeout(60.0)

    async def chat(
        self,
        *,
        messages: list[dict],
        model: str | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        metadata: dict | None = None,
    ) -> dict:
        payload: dict = {"messages": messages}
        if model:
            payload["model"] = model
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format
        if metadata:
            payload["metadata"] = metadata

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()

    async def embed(self, text: str, model: str | None = None) -> list[float]:
        payload: dict = {"input": text}
        if model:
            payload["model"] = model
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self._base_url}/v1/embeddings", json=payload)
            response.raise_for_status()
            result = response.json()

        data = result.get("data", [])
        if not data:
            return []
        return data[0].get("embedding", [])
