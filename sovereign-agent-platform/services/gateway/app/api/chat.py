from fastapi import APIRouter, Depends

from app.core.auth import get_subject
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, subject: str = Depends(get_subject)) -> ChatResponse:
    _ = request
    _ = subject
    return ChatResponse(trace_id="pending", response="chat pipeline not yet wired")
