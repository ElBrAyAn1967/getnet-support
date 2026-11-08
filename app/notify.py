# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Optional push of new human cases to a Microsoft Teams channel (incoming webhook, TEAMS_WEBHOOK_URL).

The case already lives in the database and the operator API; this is a heads-up, so a failure is logged and the
merchant's turn is never affected. Only the case id, the reason and the redacted summary are sent."""

import httpx

from app.config import get_settings
from app.http import client
from app.observability import log


def case_opened(case_id: str, reason: str, summary: str) -> None:
    url = get_settings().teams_webhook_url
    if not url:
        return
    text = f"New Getnet support case **{case_id}** ({reason}): {summary}"
    try:
        client().post(url.get_secret_value(), json={"text": text}).raise_for_status()
    except httpx.HTTPError as exc:
        log("notify_failed", case=case_id, error=type(exc).__name__)
