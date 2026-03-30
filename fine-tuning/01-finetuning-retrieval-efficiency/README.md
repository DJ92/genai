# Fine-Tuning + Retrieval Efficiency Lab

Production-shaped support-QA project for LoRA-style adapter training, compact retrieval indexes, and answer-quality trade-offs under latency and memory constraints.

## What This Shows

- A public-safe stand-in for internal Q&A fine-tuning work
- Low-rank adaptation on top of a frozen encoder rather than end-to-end retraining
- Retrieval-quality, memory, and latency trade-offs across `768`, `384`, and `256` dimensions
- End-to-end evaluation that combines compact retrieval with reranking

## Problem

Narrow-domain Q&A systems need two things at once: better task fit through fine-tuning and efficient retrieval paths that stay within latency budgets. This project approximates that production problem with a technical-support corpus, a LoRA-style query adapter, and retrieval benchmarks at multiple embedding dimensions.

## Dataset

- Primary dataset: Tech-support style Q&A corpus in JSONL format
- Each row contains:
  - `doc_id`
  - `doc_title`
  - `document_text`
  - `question`
  - `answer`
  - `split`

## Public Interfaces

```bash
python src/train_lora.py
python src/evaluate_generation.py
python src/build_index.py --dim 768
python src/benchmark_retrieval.py
python src/serve_eval.py --dim 256
```

## Key Results

This project produces comparison tables for:

- base vs adapted top-1 answer accuracy
- retrieval `recall@k`
- retrieval `mrr@k`
- index memory footprint
- average retrieval latency
- end-to-end reranked answer accuracy

## Architecture

```text
support corpus
  -> hashed text representations
  -> frozen encoder + LoRA-style query adapter
  -> compact retrieval indexes at multiple dimensions
  -> benchmark retrieval quality and memory
  -> end-to-end retrieval + reranking evaluation
```

## Trade-offs

- Better domain fit vs the cost of adapter training
- Higher-dimensional retrieval vs lower memory and faster lookup
- Compact retrieval candidates vs end-to-end answer quality
- Public-safe approximation vs proprietary production complexity

## Failure Modes

- Hashing collisions that blur semantically similar support topics
- Overfitting the adapter to a narrow corpus
- Retrieval quality collapsing too quickly at reduced dimensions
- End-to-end improvements hiding retrieval weaknesses behind reranking

## What I Would Improve In Production

- Swap the hashed encoder for a real embedding model and compare true LoRA checkpoints
- Add hard-negative mining and multi-document reasoning tasks
- Measure tail latency under concurrent query load
- Add evaluation slices for unseen products and unseen support intents

## Testing

```bash
pytest tests -q
```
