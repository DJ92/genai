from pydantic import BaseModel


class CapabilitiesResponse(BaseModel):
    backend_id: str
    default_model_name: str
    default_embed_model_name: str
    supports_tool_calls: bool = True
