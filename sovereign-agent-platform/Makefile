SHELL := /bin/bash

.PHONY: dev-up dev-down reset-db test eval milestone-a-test compose-config

dev-up:
	docker compose up -d --build

dev-down:
	docker compose down -v

reset-db:
	./scripts/reset_db.sh

test:
	./scripts/run_tests.sh

eval:
	python -m eval.harness.run_golden --output eval/golden/expected

milestone-a-test:
	docker compose up -d postgres opa
	curl -sS http://localhost:8181/v1/data/agent/decision | jq .

compose-config:
	docker compose config
