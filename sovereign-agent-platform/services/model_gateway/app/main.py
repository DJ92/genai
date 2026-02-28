from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from app.adapters.openai_compatible import OpenAICompatibleAdapter
from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
adapter = OpenAICompatibleAdapter()

app = FastAPI(title="model-gateway", version="0.1.0")


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
    }


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict) -> dict:
    payload.setdefault("model", settings.default_model_name)
    try:
        response = await adapter.chat(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"backend chat call failed: {exc}") from exc

    response.setdefault("metadata", {})
    response["metadata"].update(
        {
            "backend_id": settings.model_backend_id,
            "model_name": payload.get("model", settings.default_model_name),
        }
    )
    return response


@app.post("/v1/embeddings")
async def embeddings(payload: dict) -> dict:
    payload.setdefault("model", settings.default_embed_model_name)
    try:
        response = await adapter.embeddings(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"backend embeddings call failed: {exc}") from exc

    response.setdefault("metadata", {})
    response["metadata"].update(
        {
            "backend_id": settings.model_backend_id,
            "model_name": payload.get("model", settings.default_embed_model_name),
        }
    )
    return response
