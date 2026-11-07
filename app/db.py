# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""The one database access: a connection pool and an idempotent schema applied at startup."""

from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
  id            text PRIMARY KEY,
  password_hash text NOT NULL,
  role          text NOT NULL CHECK (role IN ('customer', 'operator')),
  display_name  text NOT NULL
);

CREATE TABLE IF NOT EXISTS login_attempts (
  key text        NOT NULL,
  ok  boolean     NOT NULL,
  at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS login_attempts_key_at ON login_attempts (key, at);

CREATE TABLE IF NOT EXISTS demo_state (
  key   text PRIMARY KEY,
  value text NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
  id         text PRIMARY KEY,
  user_id    text NOT NULL REFERENCES users (id),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cases (
  id              text PRIMARY KEY,
  user_id         text NOT NULL,
  conversation_id text NOT NULL REFERENCES conversations (id),
  reason          text NOT NULL,
  summary         text NOT NULL,
  status          text NOT NULL CHECK (status IN ('open', 'pending_approval', 'answered', 'closed')),
  action          jsonb,
  operator_reply  text,
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback (
  conversation_id text NOT NULL REFERENCES conversations (id),
  turn            int  NOT NULL,
  rating          text NOT NULL CHECK (rating IN ('up', 'down')),
  comment         text,
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id         bigserial PRIMARY KEY,
  url        text NOT NULL,
  title      text NOT NULL,
  fetched_at date NOT NULL,
  body       text NOT NULL,
  hash       text NOT NULL UNIQUE,
  stale      boolean NOT NULL DEFAULT false,
  embedding  vector(%(dim)s) NOT NULL,
  tsv        tsvector GENERATED ALWAYS AS (to_tsvector('portuguese', title || ' ' || body)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
"""

_pool: ConnectionPool[Connection[DictRow]] | None = None


def migrate() -> None:
    """Runs before the pool exists: the pool registers the `vector` type, which needs the extension."""
    with Connection.connect(get_settings().database_url, autocommit=True) as conn:
        conn.execute(SCHEMA.replace("%(dim)s", str(get_settings().embedding_dim)))


def pool() -> ConnectionPool[Connection[DictRow]]:
    global _pool  # noqa: PLW0603 — one pool per process, created lazily
    if _pool is None:
        _pool = ConnectionPool(
            get_settings().database_url,
            kwargs={"row_factory": dict_row, "autocommit": True, "prepare_threshold": 0},
            connection_class=Connection[DictRow],
            configure=register_vector,
            open=True,
        )
    return _pool
