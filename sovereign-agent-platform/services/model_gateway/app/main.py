from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import jsonschema
from fastapi import FastAPI, HTTPException

from app.adapters.openai_compatible import OpenAICompatibleAdapter
from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
adapter = OpenAICompatibleAdapter()

app = FastAPI(title="model-gateway", version="0.1.0")


def _extract_text_from_response(response: dict) -> str:
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if isinstance(item, dict))
    return str(content)


def _response_schema(payload: dict) -> dict | None:
    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return None
    if response_format.get("type") != "json_schema":
        return None
    schema = response_format.get("json_schema")
    if isinstance(schema, dict) and "schema" in schema:
        return schema.get("schema")
    if isinstance(schema, dict):
        return schema
    return None


def _validate_json_schema(content: str, schema: dict) -> None:
    parsed = json.loads(content)
    jsonschema.validate(parsed, schema)


def _repair_messages(messages: list[dict], error: str) -> list[dict]:
    repair = {
        "role": "system",
        "content": (
            "Your previous response did not satisfy required JSON schema output. "
            f"Error: {error}. Return ONLY valid JSON matching schema."
        ),
    }
    return [*messages, repair]


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "model-gateway",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/capabilities")
async def capabilities() -> dict:
    return {
        "backend_id": settings.model_backend_id,
        "default_model_name": settings.default_model_name,
        "default_embed_model_name": settings.default_embed_model_name,
        "supports_tool_calls": True,
        "supports_structured_output": True,
    }


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict) -> dict:
    request_payload = dict(payload)
    request_payload.setdefault("model", settings.default_model_name)
    schema = _response_schema(request_payload)

    last_error: Exception | None = None
    normalized: dict[str, Any] | None = None
    latency_ms = 0
    schema_valid = schema is None

    for attempt in range(3):
        try:
            raw_response, latency_ms = await adapter.chat(request_payload)
            normalized = adapter.normalize_chat_response(raw_response)

            if schema is None:
                break

            content = _extract_text_from_response(normalized)
            _validate_json_schema(content, schema)
            schema_valid = True
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            schema_valid = False
            if attempt == 2:
                break
            if schema is not None:
                request_payload["messages"] = _repair_messages(
                    request_payload.get("messages", []), str(exc)
                )

    if normalized is None:
        raise HTTPException(status_code=502, detail=f"backend chat call failed: {last_error}")
    if schema is not None and not schema_valid:
        content = _extract_text_from_response(normalized)
        try:
            _validate_json_schema(content, schema)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail=f"structured output validation failed after retries: {exc}",
            ) from exc

    normalized.setdefault("metadata", {})
    normalized["metadata"].update(
        {
            "backend_id": settings.model_backend_id,
            "model_name": request_payload.get("model", settings.default_model_name),
            "token_usage": normalized.get("usage", {}),
            "latency_ms": latency_ms,
        }
    )
    return normalized


@app.post("/v1/embeddings")
async def embeddings(payload: dict) -> dict:
    request_payload = dict(payload)
    request_payload.setdefault("model", settings.default_embed_model_name)
    try:
        raw_response, latency_ms = await adapter.embeddings(request_payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"backend embeddings call failed: {exc}") from exc

    normalized = adapter.normalize_embeddings_response(raw_response)
    normalized.setdefault("metadata", {})
    normalized["metadata"].update(
        {
            "backend_id": settings.model_backend_id,
            "model_name": request_payload.get("model", settings.default_embed_model_name),
            "token_usage": normalized.get("usage", {}),
            "latency_ms": latency_ms,
        }
    )
    return normalized
