from pydantic import BaseModel


class EmbeddingsRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
