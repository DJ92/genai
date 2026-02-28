from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from app.core.config import get_settings


class OpenAICompatibleAdapter:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._timeout = httpx.Timeout(120.0)

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._settings.model_backend_api_key:
            headers["Authorization"] = f"Bearer {self._settings.model_backend_api_key}"
        return headers

    async def _post(self, path: str, payload: dict, retries: int = 3) -> tuple[dict, int]:
        url = f"{self._settings.model_backend_base_url.rstrip('/')}{path}"
        last_error: Exception | None = None
        for attempt in range(retries):
            started = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, headers=self._headers(), json=payload)
                    response.raise_for_status()
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    return response.json(), latency_ms
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
        if last_error is None:
            raise RuntimeError("unknown backend error")
        raise last_error

    async def chat(self, payload: dict) -> tuple[dict, int]:
        return await self._post("/v1/chat/completions", payload)

    async def embeddings(self, payload: dict) -> tuple[dict, int]:
        return await self._post("/v1/embeddings", payload)

    @staticmethod
    def normalize_tool_calls(raw_tool_calls: list[dict] | None) -> list[dict]:
        normalized: list[dict] = []
        for index, call in enumerate(raw_tool_calls or []):
            function_data = call.get("function", {})
            normalized.append(
                {
                    "id": call.get("id", f"tool_call_{index}"),
                    "name": function_data.get("name") or call.get("name"),
                    "arguments": function_data.get("arguments", call.get("arguments", "{}")),
                    "type": "function",
                }
            )
        return normalized

    @staticmethod
    def normalize_chat_response(raw: dict) -> dict:
        choices = raw.get("choices", [])
        normalized_choices: list[dict[str, Any]] = []
        for index, choice in enumerate(choices):
            message = choice.get("message", {})
            normalized_choices.append(
                {
                    "index": choice.get("index", index),
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", ""),
                        "tool_calls": OpenAICompatibleAdapter.normalize_tool_calls(
                            message.get("tool_calls")
                        ),
                    },
                    "finish_reason": choice.get("finish_reason"),
                }
            )

        return {
            "id": raw.get("id", "chatcmpl-local"),
            "object": "chat.completion",
            "created": raw.get("created"),
            "model": raw.get("model"),
            "choices": normalized_choices,
            "usage": raw.get("usage", {}),
        }

    @staticmethod
    def normalize_embeddings_response(raw: dict) -> dict:
        return {
            "object": "list",
            "data": raw.get("data", []),
            "model": raw.get("model"),
            "usage": raw.get("usage", {}),
        }

