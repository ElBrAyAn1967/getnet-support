# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""knowledge (G-4), support (G-5) and escalation (G-6): one runner, three configurations.

Each agent asks the model which of ITS tools to call and with which arguments; the runner executes them with the
identity injected from the request, turns results into evidence and numbered sources, and holds write actions for
human approval (HITL-2).
"""

import json
import time
from collections.abc import Callable
from datetime import date
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app import llm, repo
from app.config import get_settings
from app.graph.state import Agent, Evidence, Source, TurnState
from app.llm import track
from app.observability import log, node_span
from app.prompts import load
from app.tools.account import SUPPORT_TOOLS, WRITE_TOOLS
from app.tools.cases import ESCALATION_TOOLS
from app.tools.knowledge import EXTERNAL_TOOLS, KNOWLEDGE_TOOLS, TOOL_ERRORS

MAX_TOOL_CALLS = 3
MAX_ROUNDS = 2
POLL_S = 0.5


class Node(Protocol):
    def __call__(self, state: TurnState) -> dict[str, Any]: ...


def _task(state: TurnState) -> str:
    """What the agent sees: the standalone question plus what earlier agents found this turn — never the raw
    history, so memory cannot leak into search queries or tool arguments."""
    route = state["route"]
    assert route is not None  # noqa: S101 — agents only run after a valid route
    earlier = "\n".join(f"- {e.tool}: {e.content}" for e in state["evidence"] if e.source_n is None)
    return f"Question: {route.standalone_question}" + (f"\n\nSupport data:\n{earlier}" if earlier else "")


def _await_decision(case_id: str) -> dict[str, Any]:
    """HITL-2: wait inside the turn, up to HITL_SYNC_WAIT_S, for the operator's decision (and, if approved, the
    result of the action, which the approval itself executes)."""
    deadline = time.monotonic() + get_settings().hitl_sync_wait_s
    while True:
        action = ((repo.get_case(case_id) or {}).get("action") or {})
        if action.get("decision") == "rejected" or "result" in action:
            return dict(action)
        if time.monotonic() >= deadline:  # stop waiting, and say so: a later approval is posted to the conversation
            return repo.merge_action(case_id, {"turn_answered": True})
        time.sleep(POLL_S)


class _Turn:
    """Accumulates one agent's outputs into the per-turn state lists."""

    def __init__(self, state: TurnState, agent: Agent) -> None:
        self.state, self.agent = state, agent
        self.evidence, self.sources = list(state["evidence"]), list(state["sources"])
        self.tools, self.notices = list(state["tools"]), list(state["notices"])
        self.case = state.get("case")

    def add(self, tool: str, content: Any, source: tuple[str, str, str, bool] | None = None) -> None:
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        n = None
        if source:
            n = len(self.sources) + 1
            self.sources.append(Source(n=n, title=source[0], url=source[1], fetched_at=source[2], official=source[3]))
        self.evidence.append(Evidence(agent=self.agent, tool=tool, content=text, source_n=n))

    def run(self, tool: BaseTool, args: dict[str, Any]) -> None:
        self.tools.append(tool.name)
        injected = {"user_id": self.state["user_id"], "conversation_id": self.state["conversation_id"]}
        fields = tool.get_input_jsonschema()["properties"]
        try:
            result = tool.invoke({**args, **{k: v for k, v in injected.items() if k in fields}})
        except TOOL_ERRORS as exc:
            log("tool_failed", tool=tool.name, error=type(exc).__name__)
            self.notices.append("tool_failure")
            self.add(tool.name, f"{tool.name} is unavailable right now")
            return
        if tool.name == "search_getnet":
            for r in result["results"]:
                self.add(tool.name, r["text"], (r["title"], r["url"], r["fetched_at"], True))
            if not result["results"]:
                self.add(tool.name, "No official Getnet content matched this question.")
        elif tool.name in EXTERNAL_TOOLS and "error" not in result:
            for r in result.get("results", [result]):
                self.add(tool.name, r, (r.get("title", r.get("source", "")), r["url"], date.today().isoformat(), False))
        elif tool.name in WRITE_TOOLS and "error" not in result:
            self._hold_for_approval(tool.name, args)
        elif tool.name == "create_case":
            self.case = result
            self.add(tool.name, f"Case {result['id']} opened for a human agent.")
        else:
            self.add(tool.name, result)

    def _hold_for_approval(self, tool: str, args: dict[str, Any]) -> None:
        summary = f"Merchant requested {tool} with {json.dumps(args)}; needs operator approval."
        self.case = repo.create_case(self.state["user_id"], self.state["conversation_id"], "action_approval",
                                     summary, {"tool": tool, "args": args})
        action = _await_decision(self.case["id"])
        if "result" in action:
            outcome = f"approved by an operator; result: {json.dumps(action['result'])}"
        elif action.get("decision") == "rejected":
            outcome = "rejected by an operator"
        else:
            outcome = "waiting for an operator; the merchant will see the outcome in this conversation"
        self.case = {**self.case, "status": "answered" if action.get("decision") else "pending_approval"}
        self.add(tool, f"{tool} {args}: {outcome} (case {self.case['id']}).")

    def update(self) -> dict[str, Any]:
        return {"evidence": self.evidence, "sources": self.sources, "tools": self.tools, "notices": self.notices,
                "case": self.case, "agents": [*self.state["agents"], self.agent], "step": self.state["step"] + 1}


def _agent(name: Agent, tools_for: Callable[[TurnState], list[BaseTool]]) -> Node:
    def node(state: TurnState) -> dict[str, Any]:
        with node_span(name):
            turn, tools = _Turn(state, name), tools_for(state)
            if not tools:
                turn.add("none", "This user is not a registered Getnet merchant, so there is no account data.")
                return turn.update()
            system = load(name).text.format(today=date.today().isoformat())
            bound, by_name, tokens = llm.model().bind_tools(tools), {t.name: t for t in tools}, state["tokens"]
            messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(_task(state))]
            for _ in range(MAX_ROUNDS):  # a second round can act on the first one's results (status → restart)
                if sum(tokens.values()) > get_settings().request_token_budget:
                    turn.notices.append("token_budget")
                    break
                reply = bound.invoke(messages)
                tokens = track(reply, tokens)
                calls = [c for c in reply.tool_calls if c["name"] in by_name][:MAX_TOOL_CALLS]
                if not calls:
                    break
                messages.append(reply)
                for call in calls:
                    before = len(turn.evidence)
                    turn.run(by_name[call["name"]], call["args"])
                    found = "\n".join(e.content for e in turn.evidence[before:])
                    messages.append(ToolMessage(found or "done", tool_call_id=call["id"]))
            return {**turn.update(), "tokens": tokens}

    return node


knowledge = _agent("knowledge", lambda _: list(KNOWLEDGE_TOOLS))
support = _agent("support", lambda s: list(SUPPORT_TOOLS) if s["known_user"] else [])
escalation = _agent("escalation", lambda _: list(ESCALATION_TOOLS))
