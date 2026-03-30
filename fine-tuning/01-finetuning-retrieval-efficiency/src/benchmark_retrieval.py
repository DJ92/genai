from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import dataset_splits, load_config, load_jsonl
from src.retrieval import benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark retrieval quality across embedding dimensions.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dims", default=None, help="Comma-separated dimensions, defaults to config values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    records = load_jsonl(config.data["path"])
    test_records = dataset_splits(records, config.data["split_column"])["test"]
    dims = (
        [int(item) for item in args.dims.split(",")]
        if args.dims
        else [int(value) for value in config.retrieval["dims"]]
    )
    payload = benchmark(test_records, config, dims, top_k=int(config.retrieval["top_k"]))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
