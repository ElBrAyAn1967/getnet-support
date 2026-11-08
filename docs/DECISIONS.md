# Design decisions

Each decision names the alternative it rejected and why. IDs point to `docs/SPEC.md`.

| # | Decision | Rejected alternative | Why |
|---|---|---|---|
| D-1 | **LangGraph** state graph + LangChain models/tools | a hand-written loop; a single ReAct agent | the orchestration is explicit and inspectable (one file, `app/graph/build.py`), memory and checkpoints come from the framework, and each agent stays small |
| D-2 | **Router plans, agents choose their own tools, one node writes the reply** | every agent drafting its own answer | 2–4 model calls per turn instead of 5–7; the merchant gets one coherent voice; agents still reason about which of *their* tools to call and with which arguments |
| D-3 | The route comes from a **closed schema** (`RouteDecision`, enum agents) | parsing the model's free text | an unparseable plan is retried once and then the user is asked to clarify (G-3) — never a silent default |
| D-4 | `standalone_question` is the only text that reaches search and tools | passing the conversation history to the agents | history resolves references in the router; pasting it into a search query or a tool argument corrupts both |
| D-5 | **PostgreSQL + pgvector as the single store** (chunks, conversations, checkpoints, cases, users) | a vector DB + a cache + a relational DB | one dependency to run, back up and secure; the API is stateless, so scaling is `--scale api=N` (A-30) |
| D-6 | **Hybrid retrieval**: vectors + Portuguese full-text, OR-ed terms, fused by reciprocal rank | vectors only | the corpus is Portuguese and questions arrive in three languages; a product name ("crediário") must find its page even when the rest of the question is English |
| D-7 | **Local ONNX embeddings** baked into the image | a hosted embedding API | RAG works with no key and no network at runtime; one less account for whoever runs it |
| D-8 | Own HTML extractor: drop header/nav/footer, drop blocks repeated across pages, keep headings inline | a generic article extractor | FAQ answers live in accordions that generic extractors drop; the site menu repeats product claims on every page |
| D-9 | **Identity is enforced by the server**: JWT with the algorithm pinned, `sub` must equal `user_id`, account tools get `user_id` injected | trusting `user_id` from the body | the challenge contract keeps `user_id` in the body; the token decides whose data a turn can read, and the model never sees the id it could change |
| D-10 | Login throttling counts failures since the last success, per (ip, user), in the shared database | in-memory counters; deleting rows on success | correct across replicas; an attacker elsewhere cannot lock the owner out; history is kept, never deleted |
| D-11 | **PII is redacted on the way in** | filtering the output | nothing downstream — model, log, checkpoint, case, metric — can leak what it never received |
| D-12 | **Grounding** removes any sentence with a figure absent from the evidence | trusting the prompt alone | a model can still invent "24 installments"; a deterministic check costs nothing and is testable |
| D-13 | Human approval for write actions **inside the turn** (poll the case up to `HITL_SYNC_WAIT_S`) | pausing the graph with `interrupt` | the merchant gets a reply in the same request either way: approved, rejected, or pending with a case id |
| D-14 | A **deterministic fake model** behind the same factory | a separate demo mode | tests and the keyless demo run the real graph; no node knows which model it talks to |
| D-15 | Fail closed: production refuses to start without its secret or a real model; a provider failure answers with a case | defaults that "just work" | a missing secret is a security incident, not a convenience; a 500 tells the merchant nothing |
