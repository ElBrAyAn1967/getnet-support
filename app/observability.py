# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""O-1..O-2: one JSON line per request and per node (never the message), Prometheus metrics, trace ids."""

import hashlib
import hmac
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from prometheus_client import Counter, Histogram

from app.config import get_settings

trace_id: ContextVar[str] = ContextVar("trace_id", default="-")

REQUESTS = Counter("chat_requests_total", "Chat requests", ["agents", "outcome"])
LATENCY = Histogram("chat_latency_seconds", "End-to-end chat latency")
NODE_LATENCY = Histogram("node_latency_seconds", "Latency per graph node", ["node"])
ROUTER_FALLBACKS = Counter("router_fallback_total", "Router outputs that failed validation", ["result"])
ESCALATIONS = Counter("escalations_total", "Cases created", ["reason"])
GROUNDING_REMOVALS = Counter("grounding_removals_total", "Sentences removed for ungrounded figures")
TOKENS = Counter("llm_tokens_total", "LLM tokens", ["direction"])
COST = Counter("llm_cost_usd_total", "Estimated LLM cost in USD")

_log = logging.getLogger("getnet")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {"ts": round(record.created, 3), "level": record.levelname, "event": record.msg}
        line["trace_id"] = trace_id.get()
        line.update(getattr(record, "fields", {}))
        return json.dumps(line, ensure_ascii=False, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    _log.handlers[:] = [handler]
    _log.setLevel(logging.INFO)
    _log.propagate = False


def log(event: str, **fields: Any) -> None:
    _log.info(event, extra={"fields": fields})


def user_hash(user_id: str) -> str:
    """Keyed with the auth secret: ids like "cliente1988" are guessable, so a plain hash would be reversible."""
    secret = get_settings().auth_secret
    key = secret.get_secret_value().encode() if secret else b"-"
    return hmac.new(key, user_id.encode(), hashlib.sha256).hexdigest()[:12]


def new_trace() -> str:
    tid = f"t_{uuid.uuid4().hex[:16]}"
    trace_id.set(tid)
    return tid


@contextmanager
def node_span(node: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        NODE_LATENCY.labels(node).observe(elapsed)
        log("node", node=node, ms=round(elapsed * 1000))


def cost_usd(tokens_in: int, tokens_out: int) -> float:
    s = get_settings()
    return (tokens_in * s.price_per_mtok_in + tokens_out * s.price_per_mtok_out) / 1_000_000


def record_usage(tokens_in: int, tokens_out: int) -> float:
    cost = cost_usd(tokens_in, tokens_out)
    TOKENS.labels("in").inc(tokens_in)
    TOKENS.labels("out").inc(tokens_out)
    COST.inc(cost)
    return cost
