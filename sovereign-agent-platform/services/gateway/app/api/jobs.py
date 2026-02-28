from fastapi import APIRouter

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("")
async def create_job() -> dict:
    return {"status": "not_implemented"}


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict:
    return {"id": job_id, "status": "not_implemented"}


@router.get("/{job_id}/events")
async def get_job_events(job_id: str) -> dict:
    return {"id": job_id, "events": []}


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    return {"id": job_id, "status": "cancel_requested"}
