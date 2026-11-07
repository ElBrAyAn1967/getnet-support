# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""R-2 store and R-3 retrieve: local embeddings, one pgvector table, hybrid search fused by reciprocal rank."""

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from app import db
from app.config import get_settings

RRF_K = 60
CANDIDATES = 20


@dataclass(frozen=True)
class Chunk:
    url: str
    title: str
    fetched_at: date
    body: str
    hash: str


@dataclass(frozen=True)
class Hit:
    url: str
    title: str
    fetched_at: date
    body: str
    similarity: float
    score: float


@lru_cache
def _model() -> TextEmbedding:
    return TextEmbedding(get_settings().embedding_model)


def embed(texts: list[str]) -> list[np.ndarray]:
    return list(_model().embed(texts))


def upsert(url: str, chunks: list[Chunk]) -> int:
    """Insert new chunks of `url`; mark its chunks that are no longer on the page as stale. Returns inserted.
    A page that produced no chunks changes nothing: an empty read must never empty the index."""
    if not chunks:
        return 0
    vectors = embed([f"{c.title}\n{c.body}" for c in chunks])
    with db.pool().connection() as conn:
        before = conn.execute("SELECT count(*) AS n FROM chunks WHERE url = %s AND NOT stale", (url,)).fetchone()
        for chunk, vector in zip(chunks, vectors, strict=True):
            conn.execute(
                "INSERT INTO chunks (url, title, fetched_at, body, hash, embedding) VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (hash) DO UPDATE SET stale = false",
                (chunk.url, chunk.title, chunk.fetched_at, chunk.body, chunk.hash, vector),
            )
        conn.execute(
            "UPDATE chunks SET stale = true WHERE url = %s AND NOT (hash = ANY(%s))", (url, [c.hash for c in chunks])
        )
        after = conn.execute("SELECT count(*) AS n FROM chunks WHERE url = %s AND NOT stale", (url,)).fetchone()
    return int(after["n"] if after else 0) - int(before["n"] if before else 0)


def count() -> int:
    with db.pool().connection() as conn:
        row = conn.execute("SELECT count(*) AS n FROM chunks WHERE NOT stale").fetchone()
    return int(row["n"]) if row else 0


def search(query: str, k: int | None = None) -> list[Hit]:
    """Vector and full-text candidates fused by reciprocal rank; a hit must clear the similarity floor or match
    the full-text query, so an unrelated question returns nothing instead of the least-bad chunk."""
    s = get_settings()
    (vector,) = embed([query])
    any_term = " or ".join(re.findall(r"\w{3,}", query))  # OR, ranked by how many terms match
    with db.pool().connection() as conn:
        rows = conn.execute(
            """
            WITH v AS (
              SELECT id, 1 - (embedding <=> %(vec)s) AS sim,
                     row_number() OVER (ORDER BY embedding <=> %(vec)s) AS r
              FROM chunks WHERE NOT stale ORDER BY embedding <=> %(vec)s LIMIT %(n)s
            ), t AS (
              SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
              FROM chunks, websearch_to_tsquery('portuguese', %(text)s) q
              WHERE NOT stale AND tsv @@ q ORDER BY ts_rank_cd(tsv, q) DESC LIMIT %(n)s
            )
            SELECT c.url, c.title, c.fetched_at, c.body,
                   coalesce(v.sim, 1 - (c.embedding <=> %(vec)s)) AS similarity,
                   coalesce(1.0 / (%(rrf)s + v.r), 0) + coalesce(1.0 / (%(rrf)s + t.r), 0) AS score
            FROM chunks c LEFT JOIN v ON v.id = c.id LEFT JOIN t ON t.id = c.id
            WHERE (v.id IS NOT NULL AND v.sim >= %(floor)s) OR t.id IS NOT NULL
            ORDER BY score DESC LIMIT %(k)s
            """,
            {"vec": vector, "text": any_term, "n": CANDIDATES, "rrf": RRF_K,
             "floor": s.retrieval_min_similarity, "k": k or s.retrieval_k},
        ).fetchall()
    return [Hit(**row) for row in rows]
