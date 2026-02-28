#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
: "${MODEL_URL:=https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8002}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  mkdir -p "$(dirname "${MODEL_PATH}")"
  echo "Model not found at ${MODEL_PATH}; downloading from ${MODEL_URL}"
  curl -L "${MODEL_URL}" -o "${MODEL_PATH}"
fi

python -m llama_cpp.server \
  --model "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}"
