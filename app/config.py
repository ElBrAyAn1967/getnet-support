# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""The one Settings object. Validated at startup (validate_for_startup), never at import, so tests run keyless."""

import os
from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project identity (stable build id, surfaced by /health).
PROJECT_ID = "a084f047-317e-4529-9b51-56a900b6a152"

MIN_SECRET_BYTES = 32

Provider = Literal["fake", "anthropic", "openai", "azure_openai", "google_genai", "bedrock", "ollama"]
# Each SDK reads its own standard variables; startup only checks they are there (AWS uses its credential chain).
PROVIDER_ENV: dict[str, tuple[str, ...]] = {
    "fake": (), "ollama": (), "anthropic": ("ANTHROPIC_API_KEY",), "openai": ("OPENAI_API_KEY",),
    "azure_openai": ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION"),
    "google_genai": ("GOOGLE_API_KEY",), "bedrock": ("AWS_REGION",),
}


class Mode(StrEnum):
    DEMO = "demo"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_mode: Mode = Mode.DEMO
    auth_secret: SecretStr | None = None
    token_ttl_s: int = 3600
    login_max_failures: int = 5
    login_window_s: int = 900

    database_url: str = "postgresql://getnet:getnet@localhost:5432/getnet"

    llm_provider: Provider = "fake"
    llm_model: str = "claude-opus-5-5"
    judge_model: str = "claude-sonnet-5-5"  # evals: a different model grades the writer, never itself
    llm_base_url: str | None = None  # an OpenAI-compatible gateway (internal GenAI platform, vLLM, LiteLLM) or Ollama
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    request_token_budget: int = 24000  # p90 of a turn with Claude Opus 5.5 was 21.7k tokens (evals/results)
    # USD per million tokens for the cost metric — an estimate: set both from your Anthropic price list.
    price_per_mtok_in: float = 5.0
    price_per_mtok_out: float = 25.0

    router_min_confidence: float = 0.55
    history_turns: int = 6
    hitl_sync_wait_s: float = 0.0
    teams_webhook_url: SecretStr | None = None  # optional: new human cases are also posted to a Teams channel

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    retrieval_k: int = 5
    retrieval_min_similarity: float = 0.35
    ingest_delay_s: float = 0.8
    http_timeout_s: float = 15.0

    def validate_for_startup(self) -> None:
        """Fail closed: a production deployment without its secrets must not start."""
        if self.app_mode is Mode.PRODUCTION:
            secret = self.auth_secret.get_secret_value() if self.auth_secret else ""
            if len(secret.encode()) < MIN_SECRET_BYTES:
                raise RuntimeError(f"APP_MODE=production requires AUTH_SECRET of at least {MIN_SECRET_BYTES} bytes")
            if self.llm_provider == "fake":
                raise RuntimeError("APP_MODE=production cannot run the fake model; set LLM_PROVIDER")
        missing = [v for v in PROVIDER_ENV[self.llm_provider] if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"LLM_PROVIDER={self.llm_provider} requires {', '.join(missing)} in the environment")
        if self.llm_provider not in ("anthropic", "fake") and self.llm_model.startswith("claude-"):
            raise RuntimeError(f"LLM_PROVIDER={self.llm_provider}: set LLM_MODEL to one of that provider's models")


@lru_cache
def get_settings() -> Settings:
    return Settings()
