.PHONY: test test-e2e build db-up db-wait db-init db-reset

# Set DRY_RUN=1 to log commands without executing them
ifdef DRY_RUN
RUN = @echo "[dry-run]"
else
RUN =
endif

build:
	$(RUN) uv build

db-up:
	docker compose up -d

db-wait: db-up
	@echo "Waiting for PostgreSQL to be ready..."
	@for i in $$(seq 1 30); do \
		docker compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1 && break; \
		sleep 1; \
	done
	@docker compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1 || \
		(echo "PostgreSQL failed to start"; exit 1)

db-reset:
	docker compose down -v
	$(MAKE) db-init

db-init: db-wait
	docker compose exec -T postgres psql -U postgres -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"

test: db-wait
	$(RUN) uv run pytest

test-e2e: db-wait
	$(RUN) uv run pytest tests/test_pipeline_e2e.py -v -s --log-cli-level=DEBUG
