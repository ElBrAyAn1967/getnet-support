.PHONY: env up creds test check acceptance down

env:  ## create .env with a generated database password (once)
	@test -f .env || { cp env.example .env && \
	  sed -i.bak "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')/" .env && \
	  rm -f .env.bak && echo ".env created"; }

up: env  ## build and start db → ingest → api, wait until healthy
	docker compose up -d --build --wait

creds:  ## demo users and their generated passwords
	@docker compose exec -T api python -m app.demo

test:  ## unit + graph tests: no network, no database, no key
	uv run pytest

check:  ## lint, types, dead code, duplication, tests
	scripts/quality.sh

acceptance: ## end-to-end acceptance tests against the running stack
	$(MAKE) -s creds > .creds && uv run python scripts/acceptance.py http://127.0.0.1:8000 --credentials .creds

down:
	docker compose down
