from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    args: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    success: bool
    data: dict | list | str | None = None
    error: str | None = None
    metadata: dict = Field(default_factory=dict)
