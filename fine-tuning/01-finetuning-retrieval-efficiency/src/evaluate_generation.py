from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import dataset_splits, load_config, load_jsonl
from src.model import bundle_path, load_bundle, query_representations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and adapted retrieval-answer quality.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def accuracy(bundle: dict, records: list[dict], use_adapter: bool) -> float:
    doc_repr = np.asarray(bundle["doc_repr"], dtype=np.float32)
    doc_ids = list(bundle["doc_ids"])
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    encoded = query_representations(bundle, records, use_adapter=use_adapter)
    predictions = (encoded @ doc_repr.T).argmax(axis=1)
    targets = np.asarray([doc_id_to_index[record[bundle["doc_id_column"]]] for record in records], dtype=np.int64)
    return float((predictions == targets).mean())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bundle = load_bundle(args.model or bundle_path(config))
    records = load_jsonl(config.data["path"])
    test_records = dataset_splits(records, config.data["split_column"])["test"]
    payload = {
        "base_top1_accuracy": accuracy(bundle, test_records, use_adapter=False),
        "adapted_top1_accuracy": accuracy(bundle, test_records, use_adapter=True),
        "test_examples": len(test_records),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
