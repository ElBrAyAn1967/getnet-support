# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""The one outbound HTTP client: same user agent, timeout and redirect policy for ingestion and tools."""

from functools import lru_cache

import httpx

from app.config import get_settings

USER_AGENT = "Mozilla/5.0 (compatible; getnet-support/1.0)"


@lru_cache
def client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=get_settings().http_timeout_s
    )
