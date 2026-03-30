from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import dataset_splits, hashed_embedding, load_config, load_jsonl
from src.model import bundle_path, load_bundle, query_representations
from src.retrieval import cosine_search, index_path, load_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end evaluation with compact retrieval plus reranking.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dim", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bundle = load_bundle(args.model or bundle_path(config))
    index_payload = load_index(index_path(config, args.dim))
    records = load_jsonl(config.data["path"])
    test_records = dataset_splits(records, config.data["split_column"])["test"]

    query_matrix = np.stack(
        [hashed_embedding(record[config.data["question_column"]], args.dim) for record in test_records],
        axis=0,
    )
    start = time.perf_counter()
    ranked = cosine_search(query_matrix, np.asarray(index_payload["doc_matrix"], dtype=np.float32), top_k=int(config.retrieval["top_k"]))
    retrieval_latency_ms = ((time.perf_counter() - start) * 1000.0) / max(len(test_records), 1)

    adapted = query_representations(bundle, test_records, use_adapter=True)
    doc_repr = np.asarray(bundle["doc_repr"], dtype=np.float32)
    doc_ids = list(bundle["doc_ids"])
    target_lookup = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    reranked_predictions = []
    for query_idx, candidates in enumerate(ranked):
        candidate_repr = doc_repr[np.asarray(candidates, dtype=np.int64)]
        scores = adapted[query_idx] @ candidate_repr.T
        reranked_predictions.append(int(candidates[int(np.argmax(scores))]))
    reranked_predictions = np.asarray(reranked_predictions, dtype=np.int64)
    targets = np.asarray(
        [target_lookup[record[config.data["doc_id_column"]]] for record in test_records],
        dtype=np.int64,
    )
    payload = {
        "dim": args.dim,
        "retrieval_recall@k": float(sum(target in row for target, row in zip(targets, ranked)) / max(len(targets), 1)),
        "reranked_top1_accuracy": float((reranked_predictions == targets).mean()),
        "average_retrieval_latency_ms": retrieval_latency_ms,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
