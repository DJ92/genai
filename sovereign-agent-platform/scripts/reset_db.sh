#!/usr/bin/env bash
set -euo pipefail

DOCKER_BIN="${DOCKER_BIN:-docker}"
if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1 && [[ -x "/Applications/Docker.app/Contents/Resources/bin/docker" ]]; then
  DOCKER_BIN="/Applications/Docker.app/Contents/Resources/bin/docker"
fi

"${DOCKER_BIN}" compose down

"${DOCKER_BIN}" volume rm sovereign-agent-platform_pgdata 2>/dev/null || true

"${DOCKER_BIN}" compose up -d postgres
