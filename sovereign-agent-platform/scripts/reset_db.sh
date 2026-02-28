#!/usr/bin/env bash
set -euo pipefail

docker compose down

docker volume rm sovereign-agent-platform_pgdata 2>/dev/null || true

docker compose up -d postgres
