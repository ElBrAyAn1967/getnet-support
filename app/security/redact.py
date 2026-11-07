# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""THE PII redactor (G-1). Every writer — log, database, trace, case, response — goes through `redact`."""

import re
from collections.abc import Callable
from dataclasses import dataclass

_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_CNPJ = re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")
_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}-?\d{4}(?!\d)")


@dataclass(frozen=True)
class Redaction:
    text: str
    kinds: frozenset[str]


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2  # noqa: PLR2004 — the Luhn rule itself
        total += d
    return total % 10 == 0


def redact(text: str) -> Redaction:
    kinds: set[str] = set()

    def card(m: re.Match[str]) -> str:
        if _luhn_ok(re.sub(r"\D", "", m.group())):
            kinds.add("card")
            return "[CARD]"
        return m.group()

    def label(kind: str) -> Callable[[re.Match[str]], str]:
        def sub(_: re.Match[str]) -> str:
            kinds.add(kind)
            return f"[{kind.upper()}]"

        return sub

    out = _CARD.sub(card, text)
    for kind, pattern in (("cnpj", _CNPJ), ("cpf", _CPF), ("email", _EMAIL), ("phone", _PHONE)):
        out = pattern.sub(label(kind), out)
    return Redaction(out, frozenset(kinds))
