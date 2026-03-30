from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}


DEFAULT_DOCUMENTS = [
    {
        "id": "doc_jan_budget",
        "title": "January Merch Planning Notes",
        "scope": "personal",
        "content": (
            "The January meeting notes say the budget was $50,000 for the spring campaign. "
            "The team prioritized footwear, lightweight jackets, and approval-gated markdowns."
        ),
    },
    {
        "id": "doc_vendor_ops",
        "title": "Vendor Ops Weekly Update",
        "scope": "personal",
        "content": (
            "Vendor ops expects delayed inventory receipts on two footwear SKUs. "
            "Any publish or price change action requires manager approval."
        ),
    },
    {
        "id": "doc_support_playbook",
        "title": "Merchandising Workflow Playbook",
        "scope": "personal",
        "content": (
            "Use the small-model draft path for short merchandising requests. "
            "Escalate to the large model only when the request spans multiple documents or needs structured JSON output."
        ),
    },
]


@dataclass
class FileBackedStore:
    path: Path

    @classmethod
    def default(cls, path: str | Path | None = None) -> "FileBackedStore":
        explicit = Path(path) if path else None
        env_path = os.environ.get("SOVEREIGN_STATE_PATH")
        if explicit:
            target = explicit
        elif env_path:
            target = Path(env_path)
        else:
            target = Path(__file__).resolve().parents[1] / "artifacts" / "platform_state.json"
        return cls(target)

    def _seed_state(self) -> dict[str, Any]:
        return {
            "documents": deepcopy(DEFAULT_DOCUMENTS),
            "jobs": [],
            "events": [],
            "next_job_id": 1,
        }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            state = self._seed_state()
            self._save(state)
            return state
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def reset(self) -> None:
        self._save(self._seed_state())

    def health(self) -> dict[str, Any]:
        state = self._load()
        return {
            "documents": len(state["documents"]),
            "jobs": len(state["jobs"]),
            "event_log_entries": len(state["events"]),
        }

    def log_event(self, event: dict[str, Any]) -> None:
        state = self._load()
        state["events"].append({"timestamp": utc_now(), **event})
        self._save(state)

    def ingest_documents(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        state = self._load()
        created = []
        for index, document in enumerate(documents, start=1):
            doc_id = document.get("id") or f"doc_ingested_{len(state['documents']) + index:03d}"
            record = {
                "id": doc_id,
                "title": document["title"],
                "scope": document.get("scope", "personal"),
                "content": document["content"],
            }
            state["documents"].append(record)
            created.append(record)
        self._save(state)
        return {"ingested": len(created), "documents": created}

    def search_documents(self, query: str, scopes: list[str] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        state = self._load()
        scope_filter = set(scopes or ["personal"])
        query_tokens = tokenize(query)
        results = []
        for document in state["documents"]:
            if document["scope"] not in scope_filter:
                continue
            haystack = f"{document['title']} {document['content']}"
            overlap = query_tokens.intersection(tokenize(haystack))
            score = float(len(overlap))
            if "budget" in query.lower() and "$50,000" in document["content"]:
                score += 2.0
            if "json" in query.lower():
                score += 0.5
            if score > 0:
                results.append(
                    {
                        "id": document["id"],
                        "title": document["title"],
                        "scope": document["scope"],
                        "content": document["content"],
                        "relevance_score": round(score / max(len(query_tokens), 1), 3),
                    }
                )
        results.sort(key=lambda item: item["relevance_score"], reverse=True)
        return results[:top_k]

    def create_job(self, workflow: str, prompt: str, scopes: list[str], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        state = self._load()
        job_id = f"job-{state['next_job_id']:04d}"
        state["next_job_id"] += 1
        job = {
            "id": job_id,
            "workflow": workflow,
            "prompt": prompt,
            "scopes": scopes,
            "status": "queued",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "metadata": metadata or {},
            "result": None,
        }
        state["jobs"].append(job)
        self._save(state)
        return job

    def update_job(self, job_id: str, **updates: Any) -> dict[str, Any]:
        state = self._load()
        for job in state["jobs"]:
            if job["id"] == job_id:
                job.update(updates)
                job["updated_at"] = utc_now()
                self._save(state)
                return job
        raise KeyError(f"Unknown job id: {job_id}")

    def get_job(self, job_id: str) -> dict[str, Any]:
        state = self._load()
        for job in state["jobs"]:
            if job["id"] == job_id:
                return job
        raise KeyError(f"Unknown job id: {job_id}")

    def list_jobs(self) -> list[dict[str, Any]]:
        state = self._load()
        return list(state["jobs"])
