from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


def write_dataset(path: Path) -> None:
    rows = [
        {
            "doc_id": "doc_boots",
            "doc_title": "Boots Support Guide",
            "document_text": "Boot issues are fixed by resetting the ankle fit strap and tightening the heel lock.",
            "question": "How do I fix ankle fit issues on the boots?",
            "answer": "Reset the ankle fit strap and tighten the heel lock.",
            "split": "train",
        },
        {
            "doc_id": "doc_boots",
            "doc_title": "Boots Support Guide",
            "document_text": "Boot issues are fixed by resetting the ankle fit strap and tightening the heel lock.",
            "question": "What fixes heel lock problems on the boots?",
            "answer": "Tighten the heel lock and reset the ankle fit strap.",
            "split": "val",
        },
        {
            "doc_id": "doc_jackets",
            "doc_title": "Jacket Care FAQ",
            "document_text": "Wash jackets cold, skip bleach, and dry on low heat to preserve the shell fabric.",
            "question": "How should I wash the jacket shell?",
            "answer": "Wash cold and dry on low heat.",
            "split": "train",
        },
        {
            "doc_id": "doc_jackets",
            "doc_title": "Jacket Care FAQ",
            "document_text": "Wash jackets cold, skip bleach, and dry on low heat to preserve the shell fabric.",
            "question": "What drying method preserves the jacket fabric?",
            "answer": "Dry on low heat.",
            "split": "test",
        },
        {
            "doc_id": "doc_payments",
            "doc_title": "Billing Troubleshooting",
            "document_text": "Card declines are often resolved by verifying the postal code and re-saving the payment method.",
            "question": "How do I resolve a card decline on the support portal?",
            "answer": "Verify the postal code and re-save the payment method.",
            "split": "train",
        },
        {
            "doc_id": "doc_payments",
            "doc_title": "Billing Troubleshooting",
            "document_text": "Card declines are often resolved by verifying the postal code and re-saving the payment method.",
            "question": "What helps when the support portal rejects my card?",
            "answer": "Verify the postal code and re-save the payment method.",
            "split": "test",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_config(tmp_path: Path, data_path: Path) -> Path:
    base = Path(__file__).resolve().parents[1] / "configs" / "lora_support.yaml"
    config = yaml.safe_load(base.read_text())
    config["data"]["path"] = str(data_path)
    config["training"]["checkpoint_dir"] = str(tmp_path / "artifacts")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_train_and_generation_eval(tmp_path: Path) -> None:
    data_path = tmp_path / "support.jsonl"
    write_dataset(data_path)
    config_path = write_config(tmp_path, data_path)
    project_root = Path(__file__).resolve().parents[1]

    train = subprocess.run(
        ["python", "src/train_lora.py", "--config", str(config_path)],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    train_payload = json.loads(train.stdout)
    assert Path(train_payload["checkpoint"]).exists()

    evaluate = subprocess.run(
        ["python", "src/evaluate_generation.py", "--config", str(config_path), "--model", train_payload["checkpoint"]],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    evaluate_payload = json.loads(evaluate.stdout)
    assert "base_top1_accuracy" in evaluate_payload
    assert "adapted_top1_accuracy" in evaluate_payload


def test_index_benchmark_and_serve_eval(tmp_path: Path) -> None:
    data_path = tmp_path / "support.jsonl"
    write_dataset(data_path)
    config_path = write_config(tmp_path, data_path)
    project_root = Path(__file__).resolve().parents[1]

    train = subprocess.run(
        ["python", "src/train_lora.py", "--config", str(config_path)],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    checkpoint = json.loads(train.stdout)["checkpoint"]

    build_index = subprocess.run(
        ["python", "src/build_index.py", "--config", str(config_path), "--dim", "256"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    build_payload = json.loads(build_index.stdout)
    assert Path(build_payload["index_path"]).exists()

    benchmark = subprocess.run(
        ["python", "src/benchmark_retrieval.py", "--config", str(config_path), "--dims", "768,256"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    benchmark_payload = json.loads(benchmark.stdout)
    assert {"768", "256"} <= set(benchmark_payload)

    serve = subprocess.run(
        ["python", "src/serve_eval.py", "--config", str(config_path), "--model", checkpoint, "--dim", "256"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    serve_payload = json.loads(serve.stdout)
    assert "reranked_top1_accuracy" in serve_payload
