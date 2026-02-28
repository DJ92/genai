from __future__ import annotations

from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI

from app.adapters.model_gateway_client import ModelGatewayClient
from app.adapters.tool_router import ToolRouter
from app.api import chat, health, jobs
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.policy_client import PolicyClient

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_pool = await asyncpg.create_pool(settings.postgres_dsn, min_size=1, max_size=10)
    app.state.policy_client = PolicyClient()
    app.state.model_gateway_client = ModelGatewayClient()
    app.state.tool_router = ToolRouter()
    try:
        yield
    finally:
        await app.state.db_pool.close()


app = FastAPI(title="gateway", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(jobs.router)

