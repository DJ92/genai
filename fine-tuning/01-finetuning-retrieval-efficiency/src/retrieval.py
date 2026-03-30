from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.data import FineTuningConfig, build_document_catalog, hashed_embedding


def cosine_search(query_matrix: np.ndarray, doc_matrix: np.ndarray, top_k: int) -> np.ndarray:
    scores = query_matrix @ doc_matrix.T
    return np.argsort(-scores, axis=1)[:, :top_k]


def recall_at_k(targets: np.ndarray, ranked: np.ndarray) -> float:
    hits = sum(target in row for target, row in zip(targets, ranked))
    return hits / max(len(targets), 1)


def mrr_at_k(targets: np.ndarray, ranked: np.ndarray) -> float:
    total = 0.0
    for target, row in zip(targets, ranked):
        reciprocal = 0.0
        for idx, candidate in enumerate(row, start=1):
            if candidate == target:
                reciprocal = 1.0 / idx
                break
        total += reciprocal
    return total / max(len(targets), 1)


def build_index_payload(records: list[dict[str, Any]], config: FineTuningConfig, dim: int) -> dict[str, Any]:
    catalog = build_document_catalog(records, config)
    doc_ids = [item["doc_id"] for item in catalog]
    doc_titles = [item["doc_title"] for item in catalog]
    doc_answers = [item["answer"] for item in catalog]
    doc_matrix = np.stack([hashed_embedding(item["document_text"], dim) for item in catalog], axis=0)
    return {
        "dim": dim,
        "doc_ids": np.asarray(doc_ids),
        "doc_titles": np.asarray(doc_titles),
        "doc_answers": np.asarray(doc_answers),
        "doc_matrix": doc_matrix.astype(np.float32),
    }


def index_path(config: FineTuningConfig, dim: int) -> Path:
    artifact_dir = Path(config.training["checkpoint_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir / f"support_index_dim{dim}.npz"


def save_index(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **payload)
    return destination


def load_index(path: str | Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def benchmark(records: list[dict[str, Any]], config: FineTuningConfig, dims: list[int], top_k: int) -> dict[str, Any]:
    doc_id_to_index = {
        item["doc_id"]: idx for idx, item in enumerate(build_document_catalog(records, config))
    }
    question_column = config.data["question_column"]
    doc_id_column = config.data["doc_id_column"]
    report = {}
    for dim in dims:
        payload = build_index_payload(records, config, dim)
        query_matrix = np.stack([hashed_embedding(record[question_column], dim) for record in records], axis=0)
        targets = np.asarray([doc_id_to_index[record[doc_id_column]] for record in records], dtype=np.int64)
        start = time.perf_counter()
        ranked = cosine_search(query_matrix, payload["doc_matrix"], top_k)
        elapsed_ms = ((time.perf_counter() - start) * 1000.0) / max(len(records), 1)
        report[str(dim)] = {
            "recall@k": recall_at_k(targets, ranked),
            "mrr@k": mrr_at_k(targets, ranked),
            "avg_latency_ms": elapsed_ms,
            "index_memory_kb": float(payload["doc_matrix"].nbytes / 1024.0),
        }
    return report
