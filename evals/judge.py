# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""LLM judge for faithfulness (E-3). It is trusted only if it rejects every canary: answers known to be wrong
against their evidence. A judge that passes a canary is grading nothing, so its scores are discarded."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app import llm
from app.config import get_settings
from app.graph.personality import compose_input

SYSTEM = """You grade a customer-support answer against the writer's input (question, merchant, notices, case,
evidence). faithful = true only if every factual claim in the answer — figures, dates, product capabilities,
deadlines, account data — is supported by that input, and no other payment company is recommended. Greetings, the
merchant's name, and courtesy advice that asserts no product fact are fine. Missing information is fine; invented
information is not. In `reason`, quote the first unsupported claim, or say "all claims supported"."""

CANARIES = [
    ("How many installments with crediário?", "Ofereça parcelamento em até 48x para o seu cliente.",
     "You can offer up to 24 installments with crediário."),
    ("Can I sell on WhatsApp?", "Com o Link de Pagamento você vende pelo WhatsApp.",
     "WhatsApp sales are not supported; consider Mercado Pago instead."),
    ("When will yesterday's sales be deposited?", '{"sale_date": "2026-11-05", "deposit_date": "2026-11-06"}',
     "Yesterday's sales will be deposited in 30 days."),
]


class Verdict(BaseModel):
    faithful: bool
    reason: str


def _grade(writer_input: str, answer: str) -> Verdict:
    prompt = f"Writer's input:\n{writer_input}\n\nAnswer:\n{answer}"
    judge = llm.model(get_settings().judge_model).with_structured_output(Verdict)
    verdict = judge.invoke([SystemMessage(SYSTEM), HumanMessage(prompt)])
    return Verdict.model_validate(verdict)


def trusted_judge() -> tuple[bool, str]:
    caught = [not _grade(f"Question: {q}\nEvidence:\n{e}", a).faithful for q, e, a in CANARIES]
    verdict = f"{sum(caught)}/{len(caught)} canaries rejected"
    print(f"judge: {verdict}")
    return all(caught), verdict


def judge_case(state: dict[str, Any]) -> Verdict:
    return _grade(compose_input(state), state["answer"])  # type: ignore[arg-type]
