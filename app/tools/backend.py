# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Simulated Getnet merchant backend (the system of record the support tools read).

Every date is an offset from today, so "yesterday's sales" always exist no matter when the code runs.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Literal

Method = Literal["credit", "debit", "pix"]
SATURDAY = 5


@dataclass(frozen=True)
class Sale:
    day_offset: int
    amount: Decimal
    method: Method
    installments: int = 1


@dataclass(frozen=True)
class Decline:
    minutes_ago: int
    code: str
    reason: str
    brand: str
    amount: Decimal


@dataclass(frozen=True)
class Terminal:
    terminal_id: str
    model: str
    connection: Literal["wifi", "chip", "bluetooth"]
    signal_pct: int
    last_seen_minutes: int
    firmware: str


@dataclass(frozen=True)
class Merchant:
    user_id: str
    business: str
    owner_first_name: str
    settlement_business_days: dict[Method, int]
    terminals: list[Terminal]
    sales: list[Sale]
    declines: list[Decline] = field(default_factory=list)


def _sales(*rows: tuple[int, str, Method, int]) -> list[Sale]:
    return [Sale(d, Decimal(a), m, i) for d, a, m, i in rows]


MERCHANTS: dict[str, Merchant] = {
    "cliente1988": Merchant(
        user_id="cliente1988",
        business="Padaria Sol Nascente",
        owner_first_name="Ana",
        settlement_business_days={"debit": 1, "credit": 1, "pix": 0},
        terminals=[Terminal("GS-1988-01", "Get Smart", "wifi", 12, 340, "3.2.1")],
        sales=_sales(
            (-3, "412.80", "credit", 1), (-3, "96.50", "pix", 1),
            (-2, "1290.00", "credit", 3), (-2, "187.30", "debit", 1),
            (-1, "845.20", "credit", 1), (-1, "233.90", "debit", 1), (-1, "64.00", "pix", 1),
            (0, "152.40", "debit", 1),
        ),
        declines=[
            Decline(95, "51", "Saldo ou limite insuficiente no cartão do comprador", "Visa", Decimal("189.90")),
            Decline(42, "05", "Transação não autorizada pelo banco emissor", "Mastercard", Decimal("75.00")),
        ],
    ),
    "cliente2024": Merchant(
        user_id="cliente2024",
        business="Oficina Duas Rodas",
        owner_first_name="Bruno",
        settlement_business_days={"debit": 1, "credit": 30, "pix": 0},
        terminals=[Terminal("GC-2024-01", "Get Clássica", "chip", 78, 3, "2.8.0")],
        sales=_sales((-1, "560.00", "credit", 2), (-1, "120.00", "pix", 1)),
    ),
}


def add_business_days(start: date, days: int) -> date:
    current = start
    while days > 0 or current.weekday() >= SATURDAY:
        current += timedelta(days=1)
        if current.weekday() < SATURDAY:
            days -= 1
    return current


def settlements(merchant: Merchant, start_offset: int, end_offset: int, today: date) -> list[dict[str, str]]:
    rows = []
    for sale in merchant.sales:
        if start_offset <= sale.day_offset <= end_offset:
            sold = today + timedelta(days=sale.day_offset)
            deposit = add_business_days(sold, merchant.settlement_business_days[sale.method])
            rows.append({
                "sale_date": sold.isoformat(), "method": sale.method, "amount_brl": str(sale.amount),
                "installments": str(sale.installments), "deposit_date": deposit.isoformat(),
                "status": "deposited" if deposit <= today else "scheduled",
            })
    return rows


def terminal_status(merchant: Merchant) -> list[dict[str, str]]:
    return [
        {
            "terminal_id": t.terminal_id, "model": t.model, "connection": t.connection,
            "signal_pct": str(t.signal_pct), "last_seen_minutes_ago": str(t.last_seen_minutes),
            "status": "online" if t.last_seen_minutes <= 10 else "offline",  # noqa: PLR2004 — heartbeat window
            "firmware": t.firmware,
        }
        for t in merchant.terminals
    ]


def restart_terminal(merchant: Merchant, terminal_id: str) -> dict[str, str]:
    """Simulated remote restart command sent to the machine."""
    if terminal_id not in {t.terminal_id for t in merchant.terminals}:
        return {"terminal_id": terminal_id, "restart": "unknown terminal"}
    return {"terminal_id": terminal_id, "restart": "command sent", "expected_back_online_minutes": "3"}


def recent_declines(merchant: Merchant) -> list[dict[str, str]]:
    return [
        {"minutes_ago": str(d.minutes_ago), "code": d.code, "reason": d.reason, "brand": d.brand,
         "amount_brl": str(d.amount)}
        for d in merchant.declines
    ]
