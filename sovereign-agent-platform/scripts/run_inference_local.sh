#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8002}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found at ${MODEL_PATH}. Download a GGUF model first." >&2
  exit 1
fi

python -m llama_cpp.server \
  --model "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}"
