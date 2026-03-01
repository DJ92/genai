SHELL := /bin/bash

.PHONY: dev-up dev-down reset-db test eval milestone-a-test compose-config

DOCKER_APP_DIR := /Applications/Docker.app/Contents/Resources/bin
ifneq ("$(wildcard $(DOCKER_APP_DIR)/docker)","")
	export PATH := $(DOCKER_APP_DIR):$(PATH)
endif

DOCKER ?= docker
DOCKER_APP_BIN := /Applications/Docker.app/Contents/Resources/bin/docker
ifeq ($(shell command -v $(DOCKER) >/dev/null 2>&1; echo $$?),1)
	ifneq ("$(wildcard $(DOCKER_APP_BIN))","")
		DOCKER := $(DOCKER_APP_BIN)
	endif
endif

dev-up:
	$(DOCKER) compose up -d --build

dev-down:
	$(DOCKER) compose down -v

reset-db:
	./scripts/reset_db.sh

test:
	./scripts/run_tests.sh

eval:
	./scripts/seed_eval_data.sh
	python -m eval.harness.run_golden --output eval/golden/expected

milestone-a-test:
	$(DOCKER) compose up -d postgres opa
	curl -sS http://localhost:8181/v1/data/agent/decision | jq .

compose-config:
	$(DOCKER) compose config
