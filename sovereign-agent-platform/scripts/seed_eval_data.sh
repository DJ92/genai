#!/usr/bin/env bash
set -euo pipefail

DOCKER_BIN="${DOCKER_BIN:-docker}"
if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1 && [[ -x "/Applications/Docker.app/Contents/Resources/bin/docker" ]]; then
  export PATH="/Applications/Docker.app/Contents/Resources/bin:${PATH}"
  DOCKER_BIN="/Applications/Docker.app/Contents/Resources/bin/docker"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

POSTGRES_CONTAINER_ID="$(${DOCKER_BIN} compose ps -q postgres)"
if [[ -z "${POSTGRES_CONTAINER_ID}" ]]; then
  echo "postgres container is not running; start stack first (make dev-up)" >&2
  exit 1
fi

${DOCKER_BIN} exec -i "${POSTGRES_CONTAINER_ID}" psql -U agent -d agentdb <<'SQL'
BEGIN;

DELETE FROM documents WHERE source_uri LIKE 'fixture://%';

INSERT INTO documents (source_uri, sha256, mime_type, scope, title)
VALUES
  ('fixture://personal/january_meeting.md', encode(digest('fixture://personal/january_meeting.md', 'sha256'), 'hex'), 'text/markdown', 'personal', 'january_meeting.md'),
  ('fixture://personal/docs_update.md', encode(digest('fixture://personal/docs_update.md', 'sha256'), 'hex'), 'text/markdown', 'personal', 'docs_update.md'),
  ('fixture://work/deployment_checklist.md', encode(digest('fixture://work/deployment_checklist.md', 'sha256'), 'hex'), 'text/markdown', 'work', 'deployment_checklist.md'),
  ('fixture://work/planning_notes_q1_q2.md', encode(digest('fixture://work/planning_notes_q1_q2.md', 'sha256'), 'hex'), 'text/markdown', 'work', 'planning_notes_q1_q2.md'),
  ('fixture://financial/expense_policy.md', encode(digest('fixture://financial/expense_policy.md', 'sha256'), 'hex'), 'text/markdown', 'financial', 'expense_policy.md');

INSERT INTO chunks (document_id, chunk_index, content, content_sha256, offset_start, offset_end)
SELECT id, 0,
$$January meeting notes confirm the budget was $50,000 for Q1 initiatives.$$,
encode(digest($$January meeting notes confirm the budget was $50,000 for Q1 initiatives.$$, 'sha256'), 'hex'),
0, 72
FROM documents WHERE source_uri='fixture://personal/january_meeting.md';

INSERT INTO chunks (document_id, chunk_index, content, content_sha256, offset_start, offset_end)
SELECT id, 1,
$$Local project priorities are reliability and release readiness.$$,
encode(digest($$Local project priorities are reliability and release readiness.$$, 'sha256'), 'hex'),
73, 132
FROM documents WHERE source_uri='fixture://personal/january_meeting.md';

INSERT INTO chunks (document_id, chunk_index, content, content_sha256, offset_start, offset_end)
SELECT id, 0,
$$This concise update from docs includes required evidence references for local projects.$$,
encode(digest($$This concise update from docs includes required evidence references for local projects.$$, 'sha256'), 'hex'),
0, 86
FROM documents WHERE source_uri='fixture://personal/docs_update.md';

INSERT INTO chunks (document_id, chunk_index, content, content_sha256, offset_start, offset_end)
SELECT id, 0,
$$Launch meeting date is March 15, 2026 and architecture review was approved by Jordan Lee.$$,
encode(digest($$Launch meeting date is March 15, 2026 and architecture review was approved by Jordan Lee.$$, 'sha256'), 'hex'),
0, 90
FROM documents WHERE source_uri='fixture://work/deployment_checklist.md';

INSERT INTO chunks (document_id, chunk_index, content, content_sha256, offset_start, offset_end)
SELECT id, 1,
$$Deployment checklist includes smoke tests, rollback plan, and observability checks.$$,
encode(digest($$Deployment checklist includes smoke tests, rollback plan, and observability checks.$$, 'sha256'), 'hex'),
91, 172
FROM documents WHERE source_uri='fixture://work/deployment_checklist.md';

INSERT INTO chunks (document_id, chunk_index, content, content_sha256, offset_start, offset_end)
SELECT id, 0,
$$Q1 planning notes focus on reliability milestones and test hardening.$$,
encode(digest($$Q1 planning notes focus on reliability milestones and test hardening.$$, 'sha256'), 'hex'),
0, 69
FROM documents WHERE source_uri='fixture://work/planning_notes_q1_q2.md';

INSERT INTO chunks (document_id, chunk_index, content, content_sha256, offset_start, offset_end)
SELECT id, 1,
$$Q2 planning notes focus on performance tuning and rollout readiness.$$,
encode(digest($$Q2 planning notes focus on performance tuning and rollout readiness.$$, 'sha256'), 'hex'),
70, 136
FROM documents WHERE source_uri='fixture://work/planning_notes_q1_q2.md';

INSERT INTO chunks (document_id, chunk_index, content, content_sha256, offset_start, offset_end)
SELECT id, 0,
$$The most recent expense policy update sets travel reimbursement cap to $1,200.$$,
encode(digest($$The most recent expense policy update sets travel reimbursement cap to $1,200.$$, 'sha256'), 'hex'),
0, 77
FROM documents WHERE source_uri='fixture://financial/expense_policy.md';

WITH zero_vec AS (
  SELECT ('[' || string_agg('0', ',') || ']')::vector AS embedding
  FROM generate_series(1, 768)
)
INSERT INTO embeddings (chunk_id, embedding_model_id, embedding)
SELECT c.id, 'default', z.embedding
FROM chunks c
JOIN documents d ON d.id = c.document_id
CROSS JOIN zero_vec z
WHERE d.source_uri LIKE 'fixture://%'
ON CONFLICT (chunk_id, embedding_model_id)
DO UPDATE SET embedding = EXCLUDED.embedding, created_at = now();

COMMIT;
SQL

echo "Seeded eval fixture data into postgres."
