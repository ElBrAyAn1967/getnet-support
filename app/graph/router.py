# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""router (G-2, G-3): one structured call plans the turn. Invalid output is retried once, then the user is asked to
clarify — the router never falls back silently to a default agent."""

from datetime import date
from typing import Any, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from app import llm
from app.config import get_settings
from app.graph.state import RouteDecision, TurnState
from app.llm import track
from app.observability import ROUTER_FALLBACKS, log, node_span
from app.prompts import load

ATTEMPTS = 2


def history(state: TurnState) -> str:
    """The last turns before the current message, already redacted."""
    previous = state["messages"][:-1][-2 * get_settings().history_turns :]
    return "\n".join(f"{m.type}: {m.content}" for m in previous) or "(none)"


def _messages(state: TurnState) -> list[BaseMessage]:
    prompt = load("router")
    system = prompt.text.format(today=date.today().isoformat(), min_confidence=get_settings().router_min_confidence)
    user = f"History:\n{history(state)}\n\nLatest message: {state['question']}"
    return [SystemMessage(system), HumanMessage(user)]


def router(state: TurnState) -> dict[str, Any]:
    with node_span("router"):
        structured = llm.model().with_structured_output(RouteDecision, include_raw=True)
        tokens, route = state["tokens"], None
        for attempt in range(ATTEMPTS):
            try:
                out = cast(dict[str, Any], structured.invoke(_messages(state)))  # include_raw=True → dict
            except (ValidationError, OutputParserException, ValueError):
                out = {"raw": None, "parsed": None}
            if out.get("raw") is not None:
                tokens = track(out["raw"], tokens)
            if (route := out.get("parsed")) is not None:
                break
            ROUTER_FALLBACKS.labels("retry" if attempt < ATTEMPTS - 1 else "clarify").inc()
        notices = list(state["notices"])
        if route is None or (route.steps and route.confidence < get_settings().router_min_confidence):
            notices.append("clarify")
        elif route.injection_suspected:
            notices.append("injection_suspected")
        if route is not None:
            log("route", steps=route.steps, scope=route.scope, confidence=route.confidence,
                injection=route.injection_suspected)
        return {"route": route, "notices": notices, "tokens": tokens}


def next_node(state: TurnState) -> str:
    """Conditional edge after the router and after every agent: the next planned agent, or the reply."""
    route = state.get("route")
    over_budget = sum(state["tokens"].values()) > get_settings().request_token_budget
    if route is None or "clarify" in state["notices"] or over_budget or state["step"] >= len(route.steps):
        return "personality"
    return route.steps[state["step"]]
