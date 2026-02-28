from dataclasses import dataclass


@dataclass
class WorkflowResult:
    summary: str
    state: dict


async def run_workflow(job: dict) -> WorkflowResult:
    return WorkflowResult(summary="workflow stub", state={"job": job})
