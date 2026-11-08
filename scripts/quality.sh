#!/usr/bin/env bash
# A-26: the code-quality gate. Every tool must report zero findings.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "ruff";       uv run ruff check app tests scripts evals
echo "mypy";       uv run mypy app
# stop, run_manager, schema: parameters LangChain passes by name to the fake model; the signature must match.
echo "vulture";    uv run vulture app --min-confidence 80 --ignore-names stop,run_manager,schema
echo "duplicates"; uv run pylint app --disable=all --enable=duplicate-code --min-similarity-lines=6 --score=n
echo "pytest";     uv run pytest -q
echo "deploy";     uv run python scripts/check_deploy.py
echo "size";       find app -name '*.py' -exec cat {} + | wc -l
