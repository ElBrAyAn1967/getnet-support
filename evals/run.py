# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Evaluation (E-1..E-4): runs the real graph over evals/dataset.jsonl with whatever provider is configured.

    python -m evals.run                     # provider from settings; writes evals/results/<date>-<model>.json
    python -m evals.run --gate              # exit 1 when a metric is under its threshold (CI, fake model)

Deterministic metrics always; the LLM judge (evals/judge.py) only with a real model, and only if it fails all of
its canaries. Needs the database (search, cases): run it next to the API.
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from app import db, repo
from app.config import get_settings
from app.graph.build import build, run_turn
from app.graph.guardrails import allowed_figures, figures
from app.graph.state import SERDE
from app.prompts import versions
from app.rag.index import search
from evals.claude_code import install
from evals.judge import judge_case, trusted_judge

EVALS = Path(__file__).resolve().parent
# Quality bar for a real model. The fake model only stitches source text, so CI holds it to a regression floor
# (its measured baseline) instead: a drop means routing, tools or retrieval broke, not that the fake got worse.
THRESHOLDS = {"route": 0.9, "tools": 0.9, "facts": 0.8, "forbid": 1.0, "recall_at_5": 0.8, "faithful": 0.9}
FAKE_FLOOR = {"route": 0.9, "tools": 0.9, "facts": 0.6, "forbid": 1.0, "recall_at_5": 0.9}
TOPICS = {"classica": r"cl[aá]ssica", "smart": r"get smart", "pix": r"\bpix\b", "antecipacao": r"antecipa",
          "crediario": r"credi[aá]rio", "whatsapp": r"whatsapp", "conexao": r"wi-?fi|chip|sinal|conex"}


def _rows(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (EVALS / name).read_text().splitlines() if line]


def _score(case: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    exp, answer = case["expect"], state["answer"]
    low = answer.lower()
    row: dict[str, Any] = {"id": case["id"], "agents": state["agents"], "tools": state["tools"],
                           "notices": state["notices"], "answer": answer}
    if "agents" in exp or "agents_any" in exp:
        row["route"] = state["agents"] in exp.get("agents_any", [exp.get("agents")])
    if "tools" in exp or "tools_exact" in exp:
        row["tools_ok"] = (state["tools"] == exp["tools_exact"]) if "tools_exact" in exp else \
            set(exp["tools"]) <= set(state["tools"])
    if "facts" in exp:
        row["facts"] = all(any(alt.lower() in low for alt in group) for group in exp["facts"])
    if "forbid" in exp:
        row["forbid"] = not any(f.lower() in low for f in exp["forbid"])
    if "notices" in exp:
        row["notices_ok"] = set(exp["notices"]) <= set(state["notices"])
    if "case" in exp:
        row["case_ok"] = bool(state.get("case")) == exp["case"]
    before_reply = {**state, "messages": state["messages"][:-1]}  # the reply is not evidence for itself
    row["ungrounded"] = sorted(str(v) for v in figures(answer) - allowed_figures(before_reply))  # type: ignore[arg-type]
    row["grounding_removed"] = state.get("removed", [])
    row["tokens"] = state["tokens"]
    return row


def run_dataset(with_judge: bool) -> tuple[list[dict[str, Any]], str]:
    graph, rows = build(InMemorySaver(serde=SERDE)), []
    judge_ok, canaries = trusted_judge() if with_judge else (False, "judge not run")
    for case in _rows("dataset.jsonl"):
        if case.get("fake_only") and get_settings().llm_provider != "fake":
            continue  # harmful-request probes stay in the deterministic runs; hosted models' safety filters flag them
        thread = repo.create_conversation("cliente1988")
        for turn in case["turns"]:
            state = run_turn(graph, turn, case["user"], thread)
        row = _score(case, state)
        if judge_ok and state["evidence"]:
            verdict = judge_case(state)
            row["faithful"], row["judge_reason"] = verdict.faithful, verdict.reason
        rows.append(row)
        print(f"{case['id']:4} route={row.get('route', '-')!s:5} tools={row.get('tools_ok', '-')!s:5} "
              f"facts={row.get('facts', '-')!s:5} forbid={row.get('forbid', '-')!s:5} agents={state['agents']}")
    return rows, canaries


def run_retrieval() -> dict[str, float]:
    recalls, rranks = [], []
    for q in _rows("retrieval.jsonl"):
        hits = search(q["query"], k=10)
        # a hit is relevant when its TEXT carries the answer, not when its URL looks right
        ranks = [i for i, h in enumerate(hits, 1) if any(t.lower() in h.body.lower() for t in q["relevant_text"])]
        recalls.append(1.0 if ranks and ranks[0] <= 5 else 0.0)  # noqa: PLR2004 — Recall@5
        rranks.append(1 / ranks[0] if ranks else 0.0)
    return {"recall_at_5": mean(recalls), "mrr_at_10": mean(rranks), "queries": len(recalls)}


def run_corpus() -> dict[str, Any]:
    """A-22: every scenario topic covered by Brazil-domain chunks; site chrome repeated across pages stays rare."""
    with db.pool().connection() as conn:
        rows = conn.execute("SELECT url, body FROM chunks WHERE NOT stale").fetchall()
    brazil = [r for r in rows if "site.getnet.com.br" in r["url"]]
    coverage = {t: sum(bool(re.search(p, r["body"], re.I)) for r in brazil) for t, p in TOPICS.items()}
    first_lines = [r["body"].split("\n")[0] for r in rows]
    repeated = sum(first_lines.count(line) > 1 for line in first_lines) / max(len(rows), 1)
    return {"chunks": len(rows), "brazil_chunks": len(brazil), "coverage": coverage, "repeated_share": repeated}


def summarize(rows: list[dict[str, Any]], retrieval: dict[str, float]) -> dict[str, float]:
    def rate(key: str) -> float:
        values = [r[key] for r in rows if key in r]
        return mean(values) if values else 1.0

    summary = {"route": rate("route"), "tools": rate("tools_ok"), "facts": rate("facts"), "forbid": rate("forbid"),
               "notices": rate("notices_ok"), "case": rate("case_ok"),
               "ungrounded_answers": sum(bool(r["ungrounded"]) for r in rows),
               "grounding_interventions": sum(bool(r["grounding_removed"]) for r in rows),
               "recall_at_5": retrieval["recall_at_5"], "mrr_at_10": retrieval["mrr_at_10"],
               "tokens_per_case": mean(r["tokens"]["in"] + r["tokens"]["out"] for r in rows)}
    if any("faithful" in r for r in rows):
        summary["faithful"] = rate("faithful")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--via-claude-code", action="store_true", help="run the models through the Claude Code CLI")
    args = parser.parse_args()
    settings = get_settings()
    provider = install() if args.via_claude_code else settings.llm_provider
    db.migrate()
    (rows, canaries), retrieval, corpus = run_dataset(args.judge), run_retrieval(), run_corpus()
    summary = summarize(rows, retrieval)
    model = settings.llm_model if provider != "fake" else "fake"
    report = {"date": date.today().isoformat(), "provider": provider, "model": model,
              "prompt_versions": versions(), "judge": {"model": settings.judge_model, "canaries": canaries},
              "summary": summary, "retrieval": retrieval, "corpus": corpus,
              "cases": rows}
    out = EVALS / "results" / f"{report['date']}-{model}-{provider}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({"summary": summary, "corpus": corpus}, indent=1), f"\nwritten {out.relative_to(EVALS.parent)}")
    bar = FAKE_FLOOR if provider == "fake" else THRESHOLDS
    failed = {k: summary[k] for k, t in bar.items() if k in summary and summary[k] < t}  # faithful: judged runs only
    if args.gate and failed:
        raise SystemExit(f"under threshold: {failed}")


if __name__ == "__main__":
    main()
