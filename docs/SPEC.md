# SPEC — Getnet Multi-Agent Support

Spec-driven: every module, test and README section traces back to an ID in this file.
If a behaviour is not here, it is not built; if code exists that no ID needs, it is deleted.

## 1. Goal

A single HTTP endpoint answers Getnet merchants. A router sends each message to the right agent (or a
sequence of agents); the answer is grounded, safe and in the user's language.

## 2. Contracts

### 2.1 HTTP

| ID | Endpoint | Auth | Request | Response |
|---|---|---|---|---|
| H-1 | `POST /auth/login` | — | `{username, password}` | `{access_token, token_type, expires_in}` · 401 bad credentials · 429 + `Retry-After` after 5 failures per (ip, username) in 15 min |
| H-2 | `POST /chat` | Bearer | `{message, user_id, conversation_id?}` | `ChatResponse` (2.2) · 401 no/invalid token · 403 `sub != user_id` · 404 unknown or foreign `conversation_id` |
| H-3 | `POST /feedback` | Bearer | `{conversation_id, turn, rating: up\|down, comment?}` | 204 · a `down` becomes an eval candidate (E-5) |
| H-4 | `GET /operator/cases` · `POST /operator/cases/{id}/reply` · `POST /operator/approvals/{id}` | Bearer, role `operator` | — / `{message}` / `{approve: bool}` | cases · 204 · 204 |
| H-5 | `GET /health` | — | — | `{status, mode, llm, retriever, corpus_chunks}` |
| H-6 | `GET /ready` | — | — | 200 when db + index + config ready, else 503 with reason |
| H-7 | `GET /metrics` | Bearer, role `operator` | — | Prometheus text |

### 2.2 ChatResponse

```json
{
  "conversation_id": "c_01J...",
  "answer": "string, user's language, numbered citations [1]",
  "language": "en|pt|es",
  "agents": ["support", "knowledge"],
  "tools": ["get_recent_declines", "search_getnet"],
  "sources": [{"n": 1, "title": "...", "url": "https://...", "fetched_at": "2026-11-06"}],
  "case": {"id": "case_01J...", "status": "open|pending_approval|answered"} | null,
  "notices": ["pii_redacted", "unofficial_source"],
  "trace_id": "t_...",
  "prompt_versions": {"router": "1.0.0", "knowledge": "1.0.0", "personality": "1.0.0"}
}
```

## 3. Graph

```
guardrail_in → router → [knowledge | support | escalation]{1..2} → personality → grounding → guardrail_out
```

| ID | Node | Behaviour |
|---|---|---|
| G-1 | guardrail_in | redact CPF, CNPJ, card (Luhn-valid only), email, phone → `notices += pii_redacted`; the raw text never leaves process memory |
| G-2 | router | structured output `RouteDecision{steps: list[Agent] (1..2), standalone_question, language, confidence, injection_suspected}`; today's date in the prompt; invalid output → 1 retry → clarify; `router_fallback_total++` |
| G-3 | router policy | `confidence < ROUTER_MIN_CONFIDENCE` → ask a clarifying question and offer a human; never guess silently |
| G-4 | knowledge | Getnet product questions → `search_getnet` only; weather → `get_weather`; FX → `get_exchange_rate`; other general → `web_search`; competitor domains blocked |
| G-5 | support | tools receive `user_id` from graph context, never from model arguments; unknown user → no account tools, honest answer |
| G-6 | escalation | creates a case (unique id) on explicit request or on consent after an offer; write tools require approval (HITL-2) |
| G-7 | grounding | every number in the draft (money, %, installments, days) must appear in a cited source or a tool output; otherwise it is removed and the answer says the figure could not be confirmed |
| G-8 | personality | Getnet voice, user's language, numbered citations, `unofficial_source` notice for weather/FX/web |
| G-9 | guardrail_out | no PII of any user other than the authenticated one; no competitor recommendation |

Multi-step: `steps=[support, knowledge]` passes support's tool outputs in state; knowledge explains them; personality merges into one answer.

## 4. Tools

| ID | Tool | Agent | Kind |
|---|---|---|---|
| T-1 | `get_settlements(start, end)` | support | read · simulated data with offsets from today, business days |
| T-2 | `get_terminal_status()` | support | read |
| T-3 | `get_recent_declines(days)` | support | read · decline code + reason |
| T-4 | `restart_terminal(terminal_id)` | support | **write** · requires approval (HITL-2) |
| T-5 | `search_getnet(query)` | knowledge | hybrid RAG |
| T-6 | `get_weather(city, date)` | knowledge | Open-Meteo, no key |
| T-7 | `get_exchange_rate(currency)` | knowledge | Banco Central do Brasil PTAX, BRL, dated, no key |
| T-8 | `web_search(query)` | knowledge | keyless provider, degrades with a notice |
| T-9 | `create_case(reason, summary)` | escalation | write · unique id |

## 5. RAG

| ID | Stage | Behaviour |
|---|---|---|
| R-1 | ingest | `data/sources.txt` → fetch (robots, delay, UA) → on failure use `data/snapshot/<hash>.txt` and log it → main-text extraction → section chunks with title, url, fetched_at |
| R-2 | store | one table: text, embedding (pgvector), `tsvector('portuguese')`, url, title, fetched_at, content hash; idempotent |
| R-3 | retrieve | vector top-k + full-text top-k → reciprocal rank fusion → threshold; below threshold → "not found" + offer human |
| R-4 | generate | model sees only numbered chunks; G-7 verifies |
| R-5 | corpus gate | every scenario topic covered by Brazil-domain chunks; boilerplate < 5 %; cited URLs answered 200 at ingest |

Embeddings run locally (ONNX), so RAG works without any API key.

## 6. Identity, memory, oversight

| ID | Behaviour |
|---|---|
| S-1 | JWT HS256, algorithm pinned server-side, `exp` required, `sub` = user id, `role` claim |
| S-2 | `APP_MODE=production` refuses to start without `AUTH_SECRET` (≥ 32 bytes); `APP_MODE=demo` generates the secret and demo passwords once (shared by replicas through the database), never logs them (`python -m app.demo` prints them), `/health.mode = demo` |
| S-3 | passwords hashed with scrypt |
| M-1 | `conversation_id` issued by the server, bound to the user |
| M-2 | LangGraph checkpoint in Postgres; history reaches router and personality only |
| HITL-1 | operator replies appear in the customer's conversation |
| HITL-2 | a write tool opens a `pending_approval` case; the operator's approval executes the action once; the turn waits up to `HITL_SYNC_WAIT_S` for it, otherwise answers pending and the outcome is posted into the conversation later |

## 7. Observability & evaluation

| ID | Behaviour |
|---|---|
| O-1 | one JSON log line per request and per node: trace_id, user hash, agents, tools, tokens, cost, latency, outcome — never the message |
| O-2 | Prometheus: requests by agent/status, latency, router fallbacks, escalations, grounding removals, cost |
| O-3 | timeouts, retries and token budget from settings |
| E-1 | dataset: official scenarios + paraphrases (en/pt/es) + multi-turn + adversarial |
| E-2 | deterministic metrics: route, tool, Recall@5, MRR@10, facts present, numbers grounded |
| E-3 | LLM judge for faithfulness, with canary cases it must fail |
| E-4 | thresholds gate CI (fake model); real-model results committed with prompt versions |
| E-5 | thumbs-down → stored, redacted, in the `feedback` table for human review |

## 8. Acceptance

A-01..A-30 — `scripts/acceptance.py` (black box over HTTP), `scripts/check_deploy.py`, `scripts/check_readme.py`, `scripts/quality.sh`, and the evaluation `evals/run.py`; each check prints its ID.
