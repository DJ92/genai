# Sovereign Agent Platform

Local-first, policy-governed agent platform with a stable model-gateway boundary.

## Services

- `gateway`: chat/jobs ingress, policy enforcement, retrieval, and tool routing
- `model_gateway`: OpenAI-like abstraction over inference backends
- `orchestrator`: long-running workflow worker
- `ingestion`: document ingestion and embedding pipeline
- `postgres`: state, vectors, jobs, and event log
- `opa`: policy decision point

## Quick Start

1. Copy `.env.example` to `.env` and adjust as needed.
2. Start core services:
   - `make dev-up`
3. Verify health:
   - `curl http://localhost:8000/health`
   - `curl http://localhost:8001/health`
4. Run tests and static checks:
   - `make test`

## Notes

- The inference backend is expected to run on host port `8002`.
- Event logging is append-only by DB rule.
