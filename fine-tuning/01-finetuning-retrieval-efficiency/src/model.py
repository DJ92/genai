from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.data import FineTuningConfig, build_document_catalog, hashed_embedding


class LoRAQueryEncoder(torch.nn.Module):
    def __init__(self, base_weight: np.ndarray, rank: int) -> None:
        super().__init__()
        input_dim, hidden_dim = base_weight.shape
        self.register_buffer("base_weight", torch.as_tensor(base_weight, dtype=torch.float32))
        self.lora_a = torch.nn.Parameter(torch.zeros(input_dim, rank, dtype=torch.float32))
        self.lora_b = torch.nn.Parameter(torch.zeros(rank, hidden_dim, dtype=torch.float32))
        torch.nn.init.normal_(self.lora_a, std=0.02)
        torch.nn.init.zeros_(self.lora_b)

    def encode_base(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(inputs @ self.base_weight, dim=-1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = inputs @ self.base_weight
        adapter = (inputs @ self.lora_a) @ self.lora_b
        return F.normalize(base + adapter, dim=-1)


def build_base_weight(dim: int, hidden_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=0.0, scale=0.05, size=(dim, hidden_dim)).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=0, keepdims=True)
    matrix /= np.where(norms == 0, 1.0, norms)
    return matrix


def encode_documents(catalog: list[dict[str, Any]], dim: int, base_weight: np.ndarray) -> tuple[list[str], list[str], list[str], np.ndarray]:
    doc_ids = [item["doc_id"] for item in catalog]
    titles = [item["doc_title"] for item in catalog]
    answers = [item["answer"] for item in catalog]
    raw = np.stack([hashed_embedding(item["document_text"], dim) for item in catalog], axis=0)
    doc_repr = raw @ base_weight
    norms = np.linalg.norm(doc_repr, axis=1, keepdims=True)
    doc_repr /= np.where(norms == 0, 1.0, norms)
    return doc_ids, titles, answers, doc_repr.astype(np.float32)


def vectorize_questions(records: list[dict[str, Any]], question_column: str, dim: int) -> np.ndarray:
    return np.stack([hashed_embedding(record[question_column], dim) for record in records], axis=0)


def target_indices(records: list[dict[str, Any]], doc_ids: list[str], doc_id_column: str) -> np.ndarray:
    index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    return np.asarray([index[record[doc_id_column]] for record in records], dtype=np.int64)


def train_adapter(records: list[dict[str, Any]], config: FineTuningConfig) -> dict[str, Any]:
    train_dim = int(config.model["train_dim"])
    hidden_dim = int(config.model["hidden_dim"])
    rank = int(config.model["rank"])
    seed = int(config.training["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    catalog = build_document_catalog(records, config)
    base_weight = build_base_weight(train_dim, hidden_dim, seed)
    doc_ids, titles, answers, doc_repr = encode_documents(catalog, train_dim, base_weight)

    questions = vectorize_questions(records, config.data["question_column"], train_dim)
    targets = target_indices(records, doc_ids, config.data["doc_id_column"])

    model = LoRAQueryEncoder(base_weight, rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training["learning_rate"]))
    doc_tensor = torch.as_tensor(doc_repr, dtype=torch.float32)
    question_tensor = torch.as_tensor(questions, dtype=torch.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.long)

    history = []
    for epoch in range(int(config.training["epochs"])):
        optimizer.zero_grad()
        query_repr = model(question_tensor)
        logits = query_repr @ doc_tensor.T
        loss = F.cross_entropy(logits, target_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            predictions = logits.argmax(dim=1)
            accuracy = float((predictions == target_tensor).float().mean().item())
            history.append({"epoch": epoch + 1, "loss": float(loss.item()), "accuracy": accuracy})

    return {
        "model_state": model.state_dict(),
        "base_weight": base_weight,
        "rank": rank,
        "doc_ids": doc_ids,
        "doc_titles": titles,
        "doc_answers": answers,
        "doc_repr": doc_repr,
        "question_column": config.data["question_column"],
        "doc_id_column": config.data["doc_id_column"],
        "train_dim": train_dim,
        "history": history,
    }


def bundle_path(config: FineTuningConfig) -> Path:
    checkpoint_dir = Path(config.training["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / "lora_support.pt"


def save_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, destination)
    return destination


def load_bundle(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def query_representations(bundle: dict[str, Any], records: list[dict[str, Any]], use_adapter: bool) -> np.ndarray:
    model = LoRAQueryEncoder(bundle["base_weight"], int(bundle["rank"]))
    model.load_state_dict(bundle["model_state"])
    model.eval()
    questions = vectorize_questions(records, bundle["question_column"], int(bundle["train_dim"]))
    question_tensor = torch.as_tensor(questions, dtype=torch.float32)
    with torch.no_grad():
        if use_adapter:
            encoded = model(question_tensor).cpu().numpy()
        else:
            encoded = model.encode_base(question_tensor).cpu().numpy()
    return encoded.astype(np.float32)
