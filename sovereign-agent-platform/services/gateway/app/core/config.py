from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: str = Field(default="dev", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    postgres_dsn: str = Field(
        default="postgresql://agent:agent@localhost:5432/agentdb", alias="POSTGRES_DSN"
    )
    opa_url: str = Field(default="http://localhost:8181/v1/data/agent/decision", alias="OPA_URL")
    policy_bundle_version: str = Field(default="dev", alias="POLICY_BUNDLE_VERSION")

    model_gateway_url: str = Field(default="http://localhost:8001", alias="MODEL_GATEWAY_URL")

    default_scopes: str = Field(default="personal", alias="DEFAULT_SCOPES")
    max_retrieval_k: int = Field(default=8, alias="MAX_RETRIEVAL_K")

    tool_runner_image: str = Field(default="tool-runner:latest", alias="TOOL_RUNNER_IMAGE")
    tools_timeout_seconds: int = Field(default=120, alias="TOOLS_TIMEOUT_SECONDS")

    default_network_egress: bool = Field(default=False, alias="DEFAULT_NETWORK_EGRESS")
    require_approval_for_dangerous_tools: bool = Field(
        default=True, alias="REQUIRE_APPROVAL_FOR_DANGEROUS_TOOLS"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
