from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    postgres_dsn: str = Field(
        default="postgresql://agent:agent@localhost:5432/agentdb", alias="POSTGRES_DSN"
    )
    gateway_url: str = Field(default="http://gateway:8000", alias="GATEWAY_URL")
    opa_url: str = Field(default="http://opa:8181/v1/data/agent/decision", alias="OPA_URL")
    policy_bundle_version: str = Field(default="dev", alias="POLICY_BUNDLE_VERSION")
    require_approval_for_dangerous_tools: bool = Field(
        default=True, alias="REQUIRE_APPROVAL_FOR_DANGEROUS_TOOLS"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    poll_interval_seconds: int = Field(default=2, alias="ORCHESTRATOR_POLL_INTERVAL_SECONDS")


@lru_cache
def get_settings() -> Settings:
    return Settings()
