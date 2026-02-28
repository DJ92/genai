from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    postgres_dsn: str = Field(
        default="postgresql://agent:agent@localhost:5432/agentdb", alias="POSTGRES_DSN"
    )
    gateway_url: str = Field(default="http://gateway:8000", alias="GATEWAY_URL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()
