# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""personality (G-8): the single place that writes to the merchant. It merges every agent's evidence into one
reply, in the user's language, with numbered citations."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app import llm
from app.graph.router import history
from app.graph.state import TurnState
from app.llm import track
from app.observability import node_span
from app.prompts import load
from app.tools.backend import MERCHANTS

_SCOPE_NOTE = {
    "off_topic": "The request is outside Getnet support: say kindly what you can help with.",
    "unsafe": "The request is unsafe: decline briefly, without lecturing.",
}


def _evidence(state: TurnState) -> str:
    sources = {s.n: s for s in state["sources"]}
    blocks = []
    for e in state["evidence"]:
        if e.source_n is None:
            head = f"[{e.agent} data · {e.tool}]"
        else:
            s = sources[e.source_n]
            origin = "Getnet official" if s.official else f"not Getnet — {s.title}"
            head = f"[{s.n}] {origin} · {s.title} · {s.url}"
        blocks.append(f"{head}\n{e.content}")
    return "\n\n".join(blocks) or "(no evidence)"


def compose_input(state: TurnState) -> str:
    route = state.get("route")
    merchant = MERCHANTS.get(state["user_id"])
    who = f"{merchant.owner_first_name} ({merchant.business})" if merchant else "not a registered merchant"
    lines = [
        f"Language: {route.language if route else 'en'}",
        f"Question: {route.standalone_question if route else state['question']}",
        f"Merchant: {who}",
        f"Notices: {', '.join(state['notices']) or 'none'}",
        "You can: open a case for a Getnet human agent when the merchant wants one.",
    ]
    if route and route.scope in _SCOPE_NOTE:
        lines.append(f"Scope: {_SCOPE_NOTE[route.scope]}")
    if case := state.get("case"):
        lines.append(f"Case: {case['id']} ({case['status']})")
    lines += [f"History:\n{history(state)}", f"Evidence:\n{_evidence(state)}"]
    return "\n".join(lines)


def personality(state: TurnState) -> dict[str, Any]:
    with node_span("personality"):
        route = state.get("route")
        system = load("personality").text.format(language=route.language if route else "en")
        reply = llm.model().invoke([SystemMessage(system), HumanMessage(compose_input(state))])
        return {"answer": str(reply.content).strip(), "tokens": track(reply, state["tokens"])}
