import hashlib
import json
from uuid import uuid4


def new_trace_id() -> str:
    return str(uuid4())


def sha256_json(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
