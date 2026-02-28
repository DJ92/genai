from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    response_format: dict | None = None
    tools: list[dict] | None = None
    metadata: dict = Field(default_factory=dict)
