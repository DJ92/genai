from pydantic import BaseModel, Field


class PolicyRequestInput(BaseModel):
    subject: str
    action: str
    resource: str
    context: dict = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    decision: str
    reason: str
    constraints: dict = Field(default_factory=dict)
