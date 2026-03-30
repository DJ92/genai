from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_config, load_jsonl
from src.retrieval import build_index_payload, index_path, save_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a support retrieval index at a chosen embedding dimension.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dim", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    records = load_jsonl(config.data["path"])
    payload = build_index_payload(records, config, args.dim)
    path = save_index(payload, index_path(config, args.dim))
    print(
        json.dumps(
            {
                "index_path": str(path),
                "dim": args.dim,
                "documents": int(len(payload["doc_ids"])),
                "memory_kb": float(payload["doc_matrix"].nbytes / 1024.0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
