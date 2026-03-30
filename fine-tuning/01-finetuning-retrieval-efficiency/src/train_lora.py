from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import dataset_splits, load_config, load_jsonl
from src.model import bundle_path, save_bundle, train_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA-style support QA adapter.")
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    records = load_jsonl(config.data["path"])
    splits = dataset_splits(records, config.data["split_column"])
    bundle = train_adapter(splits["train"], config)
    path = save_bundle(bundle, bundle_path(config))
    print(
        json.dumps(
            {
                "checkpoint": str(path),
                "train_examples": len(splits["train"]),
                "history": bundle["history"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
