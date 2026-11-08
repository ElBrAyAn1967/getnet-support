# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Escalation agent tool (T-9). Identity and conversation come from the request, never from the model."""

from typing import Annotated, Any, Literal

from langchain_core.tools import InjectedToolArg, tool

from app import repo
from app.observability import ESCALATIONS
from app.security.redact import redact


@tool
def create_case(
    reason: Literal["user_request", "low_confidence", "no_source", "tool_failure"],
    summary: str,
    user_id: Annotated[str, InjectedToolArg],
    conversation_id: Annotated[str, InjectedToolArg],
) -> dict[str, Any]:
    """Open a case for a Getnet human agent with a short factual summary of what the merchant needs."""
    ESCALATIONS.labels(reason).inc()
    return repo.create_case(user_id, conversation_id, reason, redact(summary).text)


ESCALATION_TOOLS = [create_case]
