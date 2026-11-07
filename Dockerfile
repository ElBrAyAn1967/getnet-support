# Two stages: build the locked environment (and bake the embedding model, so the container needs no download
# at runtime), then copy it into a slim image that runs as an unprivileged user.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy FASTEMBED_CACHE_PATH=/app/models
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY prompts ./prompts
COPY data ./data
RUN uv sync --frozen --no-dev \
 && .venv/bin/python -c "from app.rag.index import embed; embed(['warm up'])"

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build --chown=app:app /app /app
ENV PATH=/app/.venv/bin:$PATH FASTEMBED_CACHE_PATH=/app/models PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
