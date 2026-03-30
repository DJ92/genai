#!/usr/bin/env bash
set -euo pipefail

python -m compileall platform_core services eval tools tests
pytest tests -q
python -m eval.harness.run_golden --output eval/golden/expected
