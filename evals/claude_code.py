# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Run the evaluation through the Claude Code CLI (`claude -p`) instead of the Anthropic API.

Evaluation-only: the app never imports this module. It lets anyone with Claude Code reproduce
`evals/results/*-claude-code.json` without an API key:

    python -m evals.run --judge --via-claude-code      # LLM_MODEL writes, JUDGE_MODEL grades

Each call is isolated (no tools, MCP, settings or saved session). Tool calls are requested as structured output with
the tools' JSON Schemas, an approximation of the API's native tool use; with an API key, `LLM_PROVIDER=anthropic`
runs the same graph through `app/llm.py`.
"""

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

TIMEOUT_S = 240


def _render(m: BaseMessage) -> str:
    if isinstance(m, AIMessage) and m.tool_calls:
        return "You already called: " + "; ".join(f"{c['name']}({json.dumps(c['args'])})" for c in m.tool_calls)
    if isinstance(m, ToolMessage):
        return f"Tool result:\n{m.content}"
    return str(m.content)


def _split(messages: Sequence[BaseMessage]) -> tuple[str, str]:
    system = "\n\n".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
    return system, "\n\n".join(_render(m) for m in messages if not isinstance(m, SystemMessage))


def _usage(envelope: dict[str, Any]) -> UsageMetadata:
    u = envelope.get("usage", {})
    tokens_in = sum(u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    tokens_out = u.get("output_tokens", 0)
    return UsageMetadata(input_tokens=tokens_in, output_tokens=tokens_out, total_tokens=tokens_in + tokens_out)


class ClaudeCodeModel(BaseChatModel):
    model_name: str
    tool_specs: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "claude-code-cli"

    def _call(self, system: str, prompt: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        binary = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
        if not binary:
            raise RuntimeError("Claude Code CLI not found: install it or set CLAUDE_BIN")
        cmd = [binary, "-p", prompt, "--model", self.model_name, "--system-prompt", system or " ", "--tools", "",
               "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
               "--no-session-persistence", "--output-format", "json"]
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]
        with tempfile.TemporaryDirectory() as cwd:  # an empty working directory: nothing from this repo leaks in
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, cwd=cwd, check=False)  # noqa: S603
        envelope: dict[str, Any] = json.loads(out.stdout)
        if envelope.get("is_error"):
            raise RuntimeError(f"claude -p failed: {str(envelope.get('result'))[:200]}")
        return envelope

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> Runnable[Any, AIMessage]:
        return self.model_copy(update={"tool_specs": [convert_to_openai_tool(t)["function"] for t in tools]})

    def with_structured_output(self, schema: Any, *, include_raw: bool = False, **_: Any) -> Runnable[Any, Any]:
        def run(messages: Sequence[BaseMessage]) -> Any:
            envelope = self._call(*_split(messages), schema.model_json_schema())
            try:
                parsed = schema.model_validate(envelope.get("structured_output"))
            except ValueError:
                parsed = None
            raw = AIMessage(content=str(envelope.get("result", "")), usage_metadata=_usage(envelope))
            return {"raw": raw, "parsed": parsed, "parsing_error": None} if include_raw else parsed

        return RunnableLambda(run)

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
                  **_: Any) -> ChatResult:
        system, prompt = _split(messages)
        if self.tool_specs:
            schema = {"type": "object", "required": ["tool_calls"], "properties": {"tool_calls": {
                "type": "array", "items": {"type": "object", "required": ["name", "args"], "properties": {
                    "name": {"type": "string", "enum": [s["name"] for s in self.tool_specs]},
                    "args": {"type": "object"}}}}}}
            system += ("\n\nTools you can call, with the JSON Schema of their arguments:\n"
                       + json.dumps(self.tool_specs, ensure_ascii=False)
                       + "\nReturn the tool calls to make now (an empty list if none fits).")
            envelope = self._call(system, prompt, schema)
            calls = (envelope.get("structured_output") or {}).get("tool_calls", [])
            message = AIMessage(content="", tool_calls=[
                {"name": c["name"], "args": c.get("args", {}), "id": f"call_{i}"} for i, c in enumerate(calls)])
        else:
            envelope = self._call(system, prompt)
            message = AIMessage(content=str(envelope.get("result", "")))
        message.usage_metadata = _usage(envelope)
        return ChatResult(generations=[ChatGeneration(message=message)])


def install() -> str:
    """Point app.llm.model at Claude Code for this process; returns the provider label for the report."""
    from app import llm  # noqa: PLC0415
    from app.config import get_settings  # noqa: PLC0415

    instances: dict[str, ClaudeCodeModel] = {}

    def model(name: str | None = None) -> BaseChatModel:
        key = name or get_settings().llm_model
        return instances.setdefault(key, ClaudeCodeModel(model_name=key))

    llm.model = model  # type: ignore[assignment]
    return "claude-code"
