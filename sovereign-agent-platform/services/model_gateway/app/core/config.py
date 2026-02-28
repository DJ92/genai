from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    model_backend_base_url: str = Field(
        default="http://host.docker.internal:8002", alias="MODEL_BACKEND_BASE_URL"
    )
    model_backend_api_key: str = Field(default="", alias="MODEL_BACKEND_API_KEY")
    model_backend_id: str = Field(default="llamacpp_local_m5", alias="MODEL_BACKEND_ID")
    default_model_name: str = Field(default="local-instruct", alias="DEFAULT_MODEL_NAME")
    default_embed_model_name: str = Field(default="local-embed", alias="DEFAULT_EMBED_MODEL_NAME")


@lru_cache
def get_settings() -> Settings:
    return Settings()
