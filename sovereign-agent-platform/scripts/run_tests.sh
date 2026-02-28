#!/usr/bin/env bash
set -euo pipefail

python -m compileall services eval tools
python -m eval.harness.run_golden --dry-run
