# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Support agent tools (T-1..T-4). `user_id` is injected from the authenticated request: the model never sees
or chooses it, so a prompt injection cannot read another merchant's data."""

from datetime import date
from decimal import Decimal
from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, tool

from app.tools import backend

UserId = Annotated[str, InjectedToolArg]


def _merchant(user_id: str) -> backend.Merchant:
    return backend.MERCHANTS[user_id]  # the agent only binds these tools for known merchants (G-5)


@tool
def get_settlements(user_id: UserId, start_day_offset: int = -1, end_day_offset: int = -1) -> dict[str, Any]:
    """Sales of the merchant and when each one is deposited. Offsets are days relative to today
    (-1 = yesterday, 0 = today, -7..0 = last week)."""
    rows = backend.settlements(_merchant(user_id), start_day_offset, end_day_offset, date.today())
    by_day: dict[str, Decimal] = {}
    for r in rows:
        by_day[r["sale_date"]] = by_day.get(r["sale_date"], Decimal(0)) + Decimal(r["amount_brl"])
    totals = {status: str(sum((Decimal(r["amount_brl"]) for r in rows if r["status"] == status), Decimal(0)))
              for status in ("deposited", "scheduled")}
    return {  # totals are given so the reply never has to compute a figure (G-7)
        "today": date.today().isoformat(), "settlements": rows, "total_brl": str(sum(by_day.values(), Decimal(0))),
        "total_by_sale_date_brl": {d: str(v) for d, v in by_day.items()}, "total_by_status_brl": totals,
    }


@tool
def get_terminal_status(user_id: UserId) -> dict[str, Any]:
    """Connectivity status of the merchant's card machines: connection type, signal, last contact."""
    return {"terminals": backend.terminal_status(_merchant(user_id))}


@tool
def get_recent_declines(user_id: UserId) -> dict[str, Any]:
    """Recently declined transactions on the merchant's machines, with the issuer's response code and reason."""
    return {"declines": backend.recent_declines(_merchant(user_id))}


@tool
def restart_terminal(user_id: UserId, terminal_id: str) -> dict[str, Any]:
    """Ask for a remote restart of one of the merchant's machines. Needs human operator approval."""
    known = {t.terminal_id for t in _merchant(user_id).terminals}
    if terminal_id not in known:
        return {"error": f"terminal {terminal_id} does not belong to this merchant"}
    return {"terminal_id": terminal_id, "restart": "requested"}


SUPPORT_TOOLS = [get_settlements, get_terminal_status, get_recent_declines, restart_terminal]
WRITE_TOOLS = {"restart_terminal"}


def execute_approved(user_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, str]:
    """The one place an approved write action runs (HITL-2): called by the operator's approval, never by a model."""
    if tool_name == "restart_terminal":
        return backend.restart_terminal(_merchant(user_id), str(args["terminal_id"]))
    raise ValueError(f"no executor for {tool_name}")
