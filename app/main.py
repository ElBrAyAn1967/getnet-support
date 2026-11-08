# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""FastAPI app: routes only (H-1..H-8). Startup validates settings, migrates, and prepares demo identities."""

import secrets
from pathlib import Path
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres import PostgresSaver
from prometheus_client import generate_latest
from pydantic import BaseModel, Field, SecretStr

from app import db, repo
from app.config import Mode, Settings, get_settings
from app.graph.build import build, run_turn
from app.graph.state import SERDE
from app.observability import LATENCY, REQUESTS, configure_logging, cost_usd, log, new_trace, user_hash
from app.prompts import versions
from app.rag.index import count
from app.security.auth import (
    Principal,
    current_principal,
    hash_password,
    issue_token,
    login,
    require_operator,
    retry_after,
)
from app.security.redact import redact
from app.tools.account import execute_approved

DEMO_USERS: dict[str, Literal["customer", "operator"]] = {
    "cliente1988": "customer", "cliente2024": "customer", "operator1": "operator",
}


class LoginRequest(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=256)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(max_length=64)
    conversation_id: str | None = None


class FeedbackRequest(BaseModel):
    conversation_id: str
    turn: int = Field(ge=1)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


class OperatorReply(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class Decision(BaseModel):
    approve: bool


def _bootstrap(settings: Settings) -> None:
    """S-2: demo mode creates its secret and demo users once, shared by every replica through the database.
    Passwords are never logged: `python -m app.demo` prints them."""
    if settings.app_mode is not Mode.DEMO:
        return
    if settings.auth_secret is None:
        settings.auth_secret = SecretStr(repo.demo_value("auth_secret", lambda: secrets.token_urlsafe(48)))
    for user, role in DEMO_USERS.items():
        password = repo.demo_value(f"password:{user}", lambda: secrets.token_urlsafe(12))
        repo.upsert_user(user, hash_password(password), role, user)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    settings.validate_for_startup()
    db.migrate()
    _bootstrap(settings)
    checkpointer = PostgresSaver(db.pool(), serde=SERDE)
    checkpointer.setup()
    app.state.graph = build(checkpointer)
    log("startup", mode=settings.app_mode, llm=settings.llm_provider, corpus_chunks=count())
    yield


app = FastAPI(title="Getnet Multi-Agent Support", version="0.1.0", lifespan=lifespan)
WEB = Path(__file__).resolve().parent / "web" / "index.html"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """A minimal chat UI over /chat, added after the construction session (see README Time log)."""
    return FileResponse(WEB)


def _unavailable(user_id: str, conversation_id: str, trace: str, exc: Exception) -> dict[str, Any]:
    """Say the truth, hand the conversation to a human, keep the response contract."""
    log("chat_failed", user=user_hash(user_id), error=type(exc).__name__)
    REQUESTS.labels("none", "failed").inc()
    case = repo.create_case(user_id, conversation_id, "tool_failure",
                            f"The assistant could not answer ({type(exc).__name__}); please follow up.")
    answer = (f"I can't answer right now because the assistant is unavailable. I opened case {case['id']} so a "
              "Getnet agent can follow up with you.")
    return {"conversation_id": conversation_id, "answer": answer, "language": "en", "agents": [], "tools": [],
            "sources": [], "case": case, "notices": ["service_unavailable"], "trace_id": trace,
            "prompt_versions": versions()}


@app.post("/auth/login")
def auth_login(body: LoginRequest, request: Request, settings: Annotated[Settings, Depends(get_settings)]) -> Any:
    ip = request.client.host if request.client else "-"
    if wait := retry_after(ip, body.username, settings):
        return JSONResponse({"detail": "too many attempts"}, status.HTTP_429_TOO_MANY_REQUESTS,
                            headers={"Retry-After": str(wait)})
    principal = login(ip, body.username, body.password)
    if principal is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
    token = issue_token(principal.user_id, principal.role, settings)
    return {"access_token": token, "token_type": "bearer", "expires_in": settings.token_ttl_s}


@app.post("/chat")
def chat(body: ChatRequest, request: Request, principal: Annotated[Principal, Depends(current_principal)]) -> Any:
    trace, start = new_trace(), time.perf_counter()
    if principal.user_id != body.user_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "user_id does not match the token")
    if body.conversation_id is None:
        conversation_id = repo.create_conversation(principal.user_id)
    elif repo.owns_conversation(principal.user_id, body.conversation_id):
        conversation_id = body.conversation_id
    else:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    try:
        state = run_turn(request.app.state.graph, body.message, principal.user_id, conversation_id)
    except Exception as exc:  # noqa: BLE001 — a failing provider must never surface as a 500 (A-21)
        return _unavailable(principal.user_id, conversation_id, trace, exc)
    route = state.get("route")
    agents = state["agents"] or ["none"]
    outcome = "clarify" if "clarify" in state["notices"] else "escalated" if state.get("case") else "answered"
    REQUESTS.labels("+".join(agents), outcome).inc()
    LATENCY.observe(time.perf_counter() - start)
    log("chat", user=user_hash(principal.user_id), agents=agents, tools=state["tools"], tokens=state["tokens"],
        cost_usd=round(cost_usd(state["tokens"]["in"], state["tokens"]["out"]), 6), outcome=outcome,
        notices=state["notices"], ms=round((time.perf_counter() - start) * 1000))
    return {
        "conversation_id": conversation_id, "answer": state["answer"],
        "language": route.language if route else "en", "agents": state["agents"], "tools": state["tools"],
        "sources": [s.model_dump() for s in state["sources"]], "case": state.get("case"),
        "notices": state["notices"], "trace_id": trace, "prompt_versions": versions(),
    }


@app.get("/conversations/{conversation_id}")
def conversation(conversation_id: str, request: Request,
                 principal: Annotated[Principal, Depends(current_principal)]) -> Any:
    if not repo.owns_conversation(principal.user_id, conversation_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    snapshot = request.app.state.graph.get_state({"configurable": {"thread_id": conversation_id}})
    messages = snapshot.values.get("messages", [])
    return {"conversation_id": conversation_id,
            "messages": [{"role": m.type, "content": m.content} for m in messages]}


@app.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
def feedback(body: FeedbackRequest, principal: Annotated[Principal, Depends(current_principal)]) -> Response:
    if not repo.owns_conversation(principal.user_id, body.conversation_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
    comment = redact(body.comment).text if body.comment else None
    repo.add_feedback(body.conversation_id, body.turn, body.rating, comment)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/operator/cases")
def operator_cases(_: Annotated[Principal, Depends(require_operator)]) -> Any:
    return repo.open_cases()


@app.post("/operator/cases/{case_id}/reply", status_code=status.HTTP_204_NO_CONTENT)
def operator_reply(case_id: str, body: OperatorReply, request: Request,
                   _: Annotated[Principal, Depends(require_operator)]) -> Response:
    """HITL-1: the operator's answer is appended to the customer's conversation."""
    target = repo.reply(case_id, redact(body.message).text)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "case not found")
    request.app.state.graph.update_state(
        {"configurable": {"thread_id": target["conversation_id"]}},
        {"messages": [AIMessage(f"[Getnet agent] {redact(body.message).text}")]},
        as_node="guardrail_out",
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/operator/approvals/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
def operator_approval(case_id: str, body: Decision, request: Request,
                      _: Annotated[Principal, Depends(require_operator)]) -> Response:
    """HITL-2: the approval executes the action once. If the merchant's turn already answered "pending", the
    outcome is posted into their conversation."""
    case = repo.decide(case_id, body.approve)
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no pending approval with this id")
    action = case["action"]
    if body.approve:
        result = execute_approved(case["user_id"], action["tool"], action["args"])
        action = repo.merge_action(case_id, {"result": result})
        message = f"[Getnet agent] Your request {action['tool']} was approved: {result['restart']}."
    else:
        message = f"[Getnet agent] Your request {action['tool']} was not approved. A specialist will contact you."
    if action.get("turn_answered"):  # the merchant's turn already answered "pending": tell them here
        request.app.state.graph.update_state({"configurable": {"thread_id": case["conversation_id"]}},
                                             {"messages": [AIMessage(message)]}, as_node="guardrail_out")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    return {"status": "ok", "mode": settings.app_mode, "llm": settings.llm_provider,
            "hitl_sync_wait_s": settings.hitl_sync_wait_s,
            "retriever": "hybrid (pgvector + portuguese full-text)", "corpus_chunks": count()}


@app.get("/ready")
def ready() -> Any:
    try:
        chunks = count()
    except Exception as exc:  # noqa: BLE001 — readiness reports any dependency failure
        return JSONResponse({"ready": False, "reason": f"database: {type(exc).__name__}"}, 503)
    if chunks == 0:
        return JSONResponse({"ready": False, "reason": "knowledge index is empty: run the ingest service"}, 503)
    return {"ready": True, "corpus_chunks": chunks}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(_: Annotated[Principal, Depends(require_operator)]) -> bytes:
    return generate_latest()
