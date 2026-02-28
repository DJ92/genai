from pydantic import BaseModel, Field


class CreateJobRequest(BaseModel):
    owner: str
    workflow_name: str
    priority: int = 0
    state: dict = Field(default_factory=dict)


class JobResponse(BaseModel):
    id: str
    owner: str
    workflow_name: str
    status: str
    priority: int
    state: dict = Field(default_factory=dict)
