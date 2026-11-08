# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""guardrail_in (G-1), grounding (G-7) and guardrail_out (G-9).

PII is redacted on the way IN, so no model, log, checkpoint or case ever sees it; the output side therefore only
has to police what the model could invent: figures without evidence and competitor recommendations.
"""

import re
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.config import get_settings
from app.graph.state import TurnState
from app.observability import GROUNDING_REMOVALS, node_span
from app.security.redact import redact
from app.tools.backend import MERCHANTS
from app.tools.knowledge import COMPETITORS

_NUMBER = re.compile(r"(?<![\w\[.,])\d+(?:[.,]\d+)*(?![\d\]])")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_SMALL_COUNT = 3  # "two options", "step 1": counting words, not facts
_UNVERIFIED = {
    "en": "I left out some figures I could not confirm in the information available to me.",
    "pt": "Deixei de fora alguns números que não consegui confirmar nas informações disponíveis.",
    "es": "Dejé fuera algunas cifras que no pude confirmar con la información disponible.",
}


def guardrail_in(state: TurnState) -> dict[str, Any]:
    with node_span("guardrail_in"):
        clean = redact(state["question"])  # run_turn already redacted; idempotent second line of defence
        return {
            "question": clean.text,
            "messages": [HumanMessage(clean.text)],
            "known_user": state["user_id"] in MERCHANTS,
            "notices": ["pii_redacted"] if clean.kinds or state.get("pii_kinds") else [],
            "route": None, "step": 0, "agents": [], "tools": [], "evidence": [], "sources": [],
            "case": None, "pending_action": None, "answer": "", "removed": [], "tokens": {"in": 0, "out": 0},
        }


def _value(token: str) -> Decimal | None:
    """'1.290,00', '1290.00' and '1,290.00' are the same figure."""
    if "," in token and "." in token:
        decimal_sep = "," if token.rfind(",") > token.rfind(".") else "."
        token = token.replace("." if decimal_sep == "," else ",", "").replace(",", ".")
    elif "," in token:
        whole, _, frac = token.rpartition(",")
        token = f"{whole.replace(',', '')}.{frac}" if len(frac) != 3 else token.replace(",", "")  # noqa: PLR2004
    elif token.count(".") == 1 and len(token.rpartition(".")[2]) == 3:  # noqa: PLR2004 — 1.290 thousands
        token = token.replace(".", "")
    try:
        return Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None


def figures(text: str) -> set[Decimal]:
    return {v for t in _NUMBER.findall(text) if (v := _value(t)) is not None}


def allowed_figures(state: TurnState) -> set[Decimal]:
    """Figures the reply may use: this turn's evidence, the conversation so far (already grounded), the question,
    today's and yesterday's dates, and small counts."""
    today = date.today()
    known = [state["question"], str(today), str(today - timedelta(days=1)), *(e.content for e in state["evidence"]),
             *(str(m.content) for m in state.get("messages", []) if m.type == "ai")]
    return figures(" ".join(known)) | {Decimal(i) for i in range(_SMALL_COUNT + 1)}


def grounding(state: TurnState) -> dict[str, Any]:
    with node_span("grounding"):
        allowed = allowed_figures(state)
        lines, removed = [], []
        for line in state["answer"].split("\n"):
            parts = re.split(r"(?<=[.!?])\s+", line)
            removed += [p for p in parts if figures(p) - allowed]
            lines.append(" ".join(p for p in parts if not figures(p) - allowed))
        if not removed:
            return {}
        GROUNDING_REMOVALS.inc(len(removed))
        route = state.get("route")
        language = route.language if route else "en"
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        return {"answer": f"{body}\n\n{_UNVERIFIED[language]}".strip(), "removed": removed,
                "notices": [*state["notices"], "figure_unverified"]}


def guardrail_out(state: TurnState) -> dict[str, Any]:
    with node_span("guardrail_out"):
        sentences = _SENTENCE.split(state["answer"])
        safe = [s for s in sentences if not any(c in s.lower() for c in COMPETITORS)]
        answer = " ".join(s.strip() for s in safe if s.strip()) if len(safe) < len(sentences) else state["answer"]
        notices = list(state["notices"])
        if sum(state.get("tokens", {}).values()) > get_settings().request_token_budget:
            notices.append("over_token_budget")  # soft budget: it stops new calls, it cannot cut one in flight
        return {"answer": answer, "messages": [AIMessage(answer)], "notices": notices}
