# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""The one model factory. The provider is configuration; the graph never knows which one it talks to.
Nodes call `llm.model()` through this module, so the evaluation can swap the backend (evals/claude_code.py).

Every provider is used through the same LangChain interface (`bind_tools`, `with_structured_output`, `invoke`),
and each SDK reads its own standard credentials from the environment."""

from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.config import get_settings
from app.observability import record_usage

MAX_OUTPUT_TOKENS = 1024


@lru_cache
def model(name: str | None = None) -> BaseChatModel:  # noqa: PLR0911 — one return per provider
    """The configured provider; `name` overrides the model (the evaluation judge uses a different one).
    Imports are local so only the provider in use is loaded."""
    s = get_settings()
    m = name or s.llm_model
    common: dict[str, Any] = {"temperature": 0, "timeout": s.llm_timeout_s, "max_retries": s.llm_max_retries}
    match s.llm_provider:
        case "anthropic":
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

            return ChatAnthropic(model_name=m, max_tokens=MAX_OUTPUT_TOKENS, **common)
        case "openai":
            from langchain_openai import ChatOpenAI  # noqa: PLC0415

            return ChatOpenAI(model=m, base_url=s.llm_base_url, max_tokens=MAX_OUTPUT_TOKENS, **common)
        case "azure_openai":
            from langchain_openai import AzureChatOpenAI  # noqa: PLC0415

            return AzureChatOpenAI(azure_deployment=m, max_tokens=MAX_OUTPUT_TOKENS, **common)
        case "google_genai":
            from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

            return ChatGoogleGenerativeAI(model=m, max_output_tokens=MAX_OUTPUT_TOKENS, **common)
        case "bedrock":
            from langchain_aws import ChatBedrockConverse  # noqa: PLC0415

            return ChatBedrockConverse(model=m, max_tokens=MAX_OUTPUT_TOKENS, temperature=0)  # type: ignore[call-arg]
        case "ollama":
            from langchain_ollama import ChatOllama  # noqa: PLC0415

            timeout = {"timeout": s.llm_timeout_s}
            return ChatOllama(model=m, base_url=s.llm_base_url, temperature=0, client_kwargs=timeout)
        case _:
            from app.fake_llm import FakeChatModel  # noqa: PLC0415

            return FakeChatModel()


def track(message: AIMessage, tokens: dict[str, int]) -> dict[str, int]:
    """Add a call's usage to the turn's running total and to the cost metric."""
    usage = message.usage_metadata or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    record_usage(usage["input_tokens"], usage["output_tokens"])
    return {
        "in": tokens.get("in", 0) + usage["input_tokens"],
        "out": tokens.get("out", 0) + usage["output_tokens"],
    }
