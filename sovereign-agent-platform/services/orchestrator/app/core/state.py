from dataclasses import dataclass


@dataclass
class JobState:
    job_id: str
    status: str
    payload: dict
