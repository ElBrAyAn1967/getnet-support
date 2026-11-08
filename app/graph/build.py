# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""The only place that knows the order of the nodes (section 3 of SPEC.md)."""

from collections.abc import Hashable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.agents import escalation, knowledge, support
from app.graph.guardrails import grounding, guardrail_in, guardrail_out
from app.graph.personality import personality
from app.graph.router import next_node, router
from app.graph.state import TurnState
from app.security.redact import redact

_NEXT: dict[Hashable, str] = {n: n for n in ("knowledge", "support", "escalation", "personality")}


def run_turn(graph: CompiledStateGraph[TurnState], message: str, user_id: str, conversation_id: str) -> Any:
    """The only way into the graph. PII is redacted HERE, before invoke: LangGraph checkpoints a turn's input before
    any node runs, so redacting inside the graph would already be too late."""
    clean = redact(message)
    return graph.invoke(
        {"question": clean.text, "pii_kinds": sorted(clean.kinds), "user_id": user_id,
         "conversation_id": conversation_id},
        {"configurable": {"thread_id": conversation_id}},
    )


def build(checkpointer: BaseCheckpointSaver[str]) -> CompiledStateGraph[TurnState]:
    g = StateGraph(TurnState)
    for name, fn in (("guardrail_in", guardrail_in), ("router", router), ("knowledge", knowledge),
                     ("support", support), ("escalation", escalation), ("personality", personality),
                     ("grounding", grounding), ("guardrail_out", guardrail_out)):
        g.add_node(name, fn)
    g.add_edge(START, "guardrail_in")
    g.add_edge("guardrail_in", "router")
    for name in ("router", "knowledge", "support", "escalation"):
        g.add_conditional_edges(name, next_node, _NEXT)
    g.add_edge("personality", "grounding")
    g.add_edge("grounding", "guardrail_out")
    g.add_edge("guardrail_out", END)
    return g.compile(checkpointer=checkpointer)
