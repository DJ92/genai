from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class FineTuningConfig:
    data: dict[str, Any]
    model: dict[str, Any]
    training: dict[str, Any]
    retrieval: dict[str, Any]


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "lora_support.yaml"


def load_config(path: str | Path | None = None) -> FineTuningConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return FineTuningConfig(
        data=payload["data"],
        model=payload["model"],
        training=payload["training"],
        retrieval=payload["retrieval"],
    )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Could not find dataset at {source}.")
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def dataset_splits(records: list[dict[str, Any]], split_column: str) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in records:
        output.setdefault(row[split_column], []).append(row)
    return output


def build_document_catalog(records: list[dict[str, Any]], config: FineTuningConfig) -> list[dict[str, Any]]:
    doc_id_col = config.data["doc_id_column"]
    title_col = config.data["doc_title_column"]
    text_col = config.data["doc_text_column"]
    answer_col = config.data["answer_column"]
    catalog = {}
    for row in records:
        catalog[row[doc_id_col]] = {
            "doc_id": row[doc_id_col],
            "doc_title": row[title_col],
            "document_text": row[text_col],
            "answer": row[answer_col],
        }
    return [catalog[key] for key in sorted(catalog)]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def hashed_embedding(text: str, dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:16], 16) % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector
