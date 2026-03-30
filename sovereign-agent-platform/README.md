# Sovereign Agent Platform

Local-first, policy-governed workflow platform for AI-assisted operations, with a stable model-gateway boundary and a first-class developer CLI.

## What This Shows

- Reusable platform boundaries instead of a single agent demo
- Small-model-first workflow orchestration with escalation only when needed
- Policy gating for tool use and approval-gated actions
- Golden-task evaluation and developer tooling for repeatable workflows

## Services

- `gateway`
  - synchronous chat ingress and workflow job creation
  - public interfaces: `POST /chat`, `POST /jobs`, `GET /jobs/{id}`
- `model_gateway`
  - stable interface for prompt routing and model-path selection
- `orchestrator`
  - workflow execution, retries, and approval checks
- `ingestion`
  - document ingestion and retrieval corpus updates
  - public interface: `POST /ingest`
- `policy`
  - tool-use gating and approval decisions

## Workflow Focus

The concrete workflow in this repo is a merchandising-style assistant:

- retrieve planning notes and operational docs
- draft a merchandising brief on the small-model path
- escalate only for structured or higher-stakes output
- require approval before publish, price, or external-fetch actions

Target runtime:

- small-model path: about `3.2s`
- escalated path: about `7.8s`
- end-to-end budget: under `10s`

## Public Interfaces

### Service Endpoints

```text
POST /chat
POST /jobs
GET /jobs/{id}
POST /ingest
```

### Developer CLI

```bash
python -m tools.agentctl run --prompt "Create a merchandising brief for January planning notes"
python -m tools.agentctl eval
python -m tools.agentctl jobs list
python -m tools.agentctl tools validate
```

## Architecture

```text
prompt
  -> gateway
  -> model_gateway route selection
  -> retrieval over ingested docs
  -> policy and approval checks
  -> orchestrator workflow result
  -> stored job + event log
```

## Golden Evaluation

The golden harness validates three behaviors:

- grounded retrieval with citations
- policy gating for `web_fetch`
- structured JSON output for ranked documents

Run it with:

```bash
python -m eval.harness.run_golden --output eval/golden/expected
```

## Quick Start

1. Copy `.env.example` to `.env`.
2. Run the local checks:
   - `./scripts/run_tests.sh`
3. Inspect the workflow through the CLI:
   - `python -m tools.agentctl run --prompt "What did the meeting notes from January say about the budget?"`

## What I Would Improve In Production

- Replace the deterministic stub model path with live model backends behind the same contract
- Add durable retries and async execution rather than inline completion
- Move policy rules into a richer bundle and add scope-aware approvals
- Expand the ingestion pipeline beyond seeded documents into real document connectors
