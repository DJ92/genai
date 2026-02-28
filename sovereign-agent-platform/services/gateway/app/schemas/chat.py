from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    scopes: list[str] = Field(default_factory=lambda: ["personal"])
    metadata: dict = Field(default_factory=dict)
    tools: list[str] | None = None


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str | None = None
    source_uri: str | None = None
    content: str | None = None
    offsets: tuple[int | None, int | None] | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    trace_id: str
    response: str
    citations: list[Citation] = Field(default_factory=list)
    evidence_source: str = "model_knowledge"
    has_local_evidence: bool = False
    pending_approvals: list[dict] = Field(default_factory=list)

