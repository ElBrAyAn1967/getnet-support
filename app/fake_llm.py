# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Deterministic stand-in for the LLM (P4). It honours the same three contracts the graph uses from a real
model — structured output, tool calls, text — with keyword rules, so tests and the keyless demo run the SAME graph.
It is a test double, not a second orchestrator: no node knows it exists."""

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

from app.graph.state import RouteDecision

_USAGE = UsageMetadata(input_tokens=0, output_tokens=0, total_tokens=0)
_PT = re.compile(r"\b(você|não|minha|meu|qual|quais|como|quando|ontem|amanhã|máquina|maquininha|preciso|posso)\b", re.I)
_ES = re.compile(r"[¿¡]|\b(mi|qué|cómo|cuándo|ayer|mañana|puedo|necesito|máquina)\b", re.I)

# (pattern, agent, tool) — support intents are listed first: account problems start with support (router rule 1).
_INTENTS: list[tuple[str, str, str]] = [
    (r"reinici|restart|reboot", "support", "restart_terminal"),
    (r"recusad|negad|declin|rejected|rechazad|refused", "support", "get_recent_declines"),
    (r"n[ãa]o conecta|not connect|no conecta|internet|wi-?fi|offline|sinal|signal", "support", "get_terminal_status"),
    (r"dep[oó]sit|cai na conta|liquida|vendas de ontem|yesterday'?s sales|ventas de ayer", "support",
     "get_settlements"),
    (r"humano|human|atendente|pessoa|person|operador|operator|agente humano", "escalation", "create_case"),
    (r"clima|tempo em|weather|previs[ãa]o|temperatura|temperature", "knowledge", "get_weather"),
    (r"\beuro\b|d[oó]lar|dollar|c[aâ]mbio|exchange rate|cota[çc][ãa]o|tipo de cambio", "knowledge",
     "get_exchange_rate"),
]
_PRODUCTS = {r"cl[aá]ssic": "Get Clássica", r"\bsmart\b": "Get Smart"}  # a comparison → one query per product
_OFF_TOPIC = re.compile(r"\b(poem|poema|joke|piada|chiste|recipe|receita|python code)\b", re.I)
_UNSAFE = re.compile(r"\b(bomb|bomba|hack into|clon(e|ar)\b.{0,40}\b(card|cart[ãa]o|tarjeta))", re.I)
_INJECTION = re.compile(r"ignore (all |previous |your )|disregard|i am the admin|sou o admin|system prompt|"
                        r"other (customer|user)|outro cliente", re.I)
_YES = re.compile(r"^\s*(yes|sim|sí|si|ok|pode|please|por favor)\b", re.I)


def _language(text: str) -> str:
    return "pt" if _PT.search(text) else "es" if _ES.search(text) else "en"


def _latest(messages: Sequence[BaseMessage], marker: str) -> str:
    text = str(messages[-1].content)
    return text.split(marker, 1)[1].strip() if marker in text else text


def plan(messages: Sequence[BaseMessage]) -> RouteDecision:
    full = str(messages[-1].content)
    text = _latest(messages, "Latest message:")
    offered = re.search(r"human agent|atendente|agente humano", full.split("Latest message:", maxsplit=1)[0], re.I)
    steps = list(dict.fromkeys(agent for pattern, agent, _ in _INTENTS if re.search(pattern, text, re.I)))
    if _YES.match(text) and offered:
        steps = ["escalation"]
    if "get_recent_declines" in {t for p, _, t in _INTENTS if re.search(p, text, re.I)}:
        steps = ["support", "knowledge"]
    scope = "unsafe" if _UNSAFE.search(text) else "off_topic" if _OFF_TOPIC.search(text) else (
        "general" if steps == ["knowledge"] and not re.search("getnet", text, re.I) else "getnet")
    if scope in ("off_topic", "unsafe"):
        steps = []
    return RouteDecision(
        scope=scope, steps=steps[:2] or ([] if scope in ("off_topic", "unsafe") else ["knowledge"]),
        standalone_question=text, language=_language(text), confidence=0.9,
        injection_suspected=bool(_INJECTION.search(text)),
    )


def _tool_calls(text: str, bound: Sequence[str]) -> list[dict[str, Any]]:
    question = _latest([AIMessage(content=text)], "Question:").split("\n")[0]
    calls: list[dict[str, Any]] = []
    for pattern, _, tool in _INTENTS:
        if tool in bound and re.search(pattern, question, re.I):
            calls.append({"name": tool, "args": _args(tool, question), "id": f"call_{len(calls)}"})
    if "search_getnet" in bound and (not calls or "get_recent_declines" in text):
        products = [name for p, name in _PRODUCTS.items() if re.search(p, question, re.I)]
        terms = [f"O que é a maquininha {name}?" for name in products] if len(products) > 1 else [question]
        if "get_recent_declines" in text:
            terms = ["O que fazer quando a transação é recusada ou não autorizada na maquininha?"]
        for term in terms:
            calls.append({"name": "search_getnet", "args": {"query": term}, "id": f"call_{len(calls)}"})
    return calls[:3]


def _args(tool: str, question: str) -> dict[str, Any]:
    if tool == "get_weather":
        city = re.search(r"(?:in|em|en|para)\s+([A-ZÀ-Ú][\wÀ-ú]+(?:\s+[A-ZÀ-Ú][\wÀ-ú]+)*)", question)
        return {"city": city.group(1) if city else "São Paulo",
                "day_offset": 1 if re.search(r"tomorrow|amanh|mañana", question, re.I) else 0}
    if tool == "get_exchange_rate":
        return {"currency": "USD" if re.search(r"d[oó]lar|dollar", question, re.I) else "EUR"}
    if tool == "get_settlements":
        return ({"start_day_offset": -1, "end_day_offset": -1} if re.search(r"ontem|yesterday|ayer", question, re.I)
                else {"start_day_offset": -7, "end_day_offset": 0})
    if tool == "restart_terminal":
        tid = re.search(r"\b[A-Z]{2}-\d{4}-\d{2}\b", question)
        return {"terminal_id": tid.group() if tid else "GS-1988-01"}
    if tool == "create_case":
        return {"reason": "user_request", "summary": "Merchant asked to talk to a human agent."}
    return {}


_FIXED = {
    "none": {"en": "I couldn't find this in Getnet's official content. Would you like me to open a case with a "
                   "human agent?", "pt": "Não encontrei isso no conteúdo oficial da Getnet. Quer que eu abra um "
                   "chamado com um atendente?", "es": "No encontré esto en el contenido oficial de Getnet. ¿Quieres "
                   "que abra un caso con un agente humano?"},
    "clarify": {"en": "Could you tell me a bit more? For example: is it about your sales, your card machine, or a "
                      "Getnet product?", "pt": "Pode me contar um pouco mais? Por exemplo: é sobre suas vendas, sua "
                      "maquininha ou um produto Getnet?", "es": "¿Puedes contarme un poco más? Por ejemplo: ¿es sobre "
                      "tus ventas, tu máquina o un producto de Getnet?"},
    "scope": {"en": "I can help with Getnet products, your sales and your card machines, plus general questions "
                    "like weather or exchange rates.", "pt": "Posso ajudar com produtos Getnet, suas vendas e suas "
                    "maquininhas, além de dúvidas gerais como clima ou câmbio.", "es": "Puedo ayudarte con productos "
                    "de Getnet, tus ventas y tus máquinas, además de dudas generales como el clima o el cambio."},
    "pii": {"en": "I hid personal data you shared; please never send card numbers or documents in chat.",
            "pt": "Ocultei dados pessoais que você enviou; nunca envie números de cartão ou documentos no chat.",
            "es": "Oculté datos personales que compartiste; nunca envíes números de tarjeta o documentos por chat."},
}


def compose(text: str) -> str:
    language = re.search(r"^Language: (\w+)", text, re.M)
    lang = language.group(1) if language else "en"
    notices = re.search(r"^Notices: (.*)$", text, re.M)
    if notices and "clarify" in notices.group(1):
        return _FIXED["clarify"][lang]
    if re.search(r"^Scope: ", text, re.M):
        return _FIXED["scope"][lang]
    parts = []
    for block in re.split(r"\n(?=\[)", text.split("Evidence:", 1)[-1]):
        head, _, body = block.strip().partition("\n")
        if head.startswith("[") and body.strip():
            snippet = " ".join(re.split(r"(?<=[.!?])\s+", body.strip())[:2])[:400]
            marker = re.match(r"\[(\d+)\]", head)
            parts.append(f"{snippet} [{marker.group(1)}]" if marker else snippet)
    if "pii_redacted" in text:
        parts.append(_FIXED["pii"][lang])
    if case := re.search(r"^Case: (\S+)", text, re.M):
        parts.append(f"Case: {case.group(1)}")
    return "\n\n".join(parts) or _FIXED["none"][lang]


class FakeChatModel(BaseChatModel):
    bound: tuple[str, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "fake-deterministic"

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> Runnable[Any, AIMessage]:
        return self.model_copy(update={"bound": tuple(t.name for t in tools)})

    def with_structured_output(
        self, schema: Any, *, include_raw: bool = False, **_: Any
    ) -> Runnable[Any, Any]:
        def run(messages: Sequence[BaseMessage]) -> Any:
            parsed = plan(messages)
            raw = AIMessage(content=json.dumps(parsed.model_dump()), usage_metadata=_USAGE)
            return {"raw": raw, "parsed": parsed, "parsing_error": None} if include_raw else parsed

        return RunnableLambda(run)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None,
                  **_: Any) -> ChatResult:
        text = str(messages[-1].content)
        build: Callable[[], AIMessage] = (
            (lambda: AIMessage(content="", tool_calls=_tool_calls(text, self.bound)))
            if self.bound and not isinstance(messages[-1], ToolMessage)
            else (lambda: AIMessage(content=compose(text)))
        )
        message = build()
        message.usage_metadata = _USAGE
        return ChatResult(generations=[ChatGeneration(message=message)])
