# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Domain records: conversations (M-1), cases (G-6, HITL), feedback (E-5), demo state (S-2).

Everything written here has already been redacted by the caller's graph node or API handler.
"""

import uuid
from typing import Any

from psycopg.types.json import Jsonb

from app import db, notify


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def upsert_user(user_id: str, password_hash: str, role: str, display_name: str) -> None:
    with db.pool().connection() as conn:
        conn.execute(
            "INSERT INTO users (id, password_hash, role, display_name) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET password_hash = EXCLUDED.password_hash, role = EXCLUDED.role",
            (user_id, password_hash, role, display_name),
        )


def create_conversation(user_id: str) -> str:
    cid = _id("c")
    with db.pool().connection() as conn:
        conn.execute("INSERT INTO conversations (id, user_id) VALUES (%s, %s)", (cid, user_id))
    return cid


def owns_conversation(user_id: str, conversation_id: str) -> bool:
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s", (conversation_id, user_id)
        ).fetchone()
    return row is not None


def create_case(
    user_id: str, conversation_id: str, reason: str, summary: str, action: dict[str, Any] | None = None
) -> dict[str, Any]:
    status = "pending_approval" if action else "open"
    case = {"id": _id("case"), "status": status}
    with db.pool().connection() as conn:
        conn.execute(
            "INSERT INTO cases (id, user_id, conversation_id, reason, summary, status, action) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (case["id"], user_id, conversation_id, reason, summary, status, Jsonb(action) if action else None),
        )
    notify.case_opened(case["id"], reason, summary)  # every case opens here, so every case is announced
    return case


def get_case(case_id: str) -> dict[str, Any] | None:
    with db.pool().connection() as conn:
        return conn.execute("SELECT * FROM cases WHERE id = %s", (case_id,)).fetchone()


def open_cases() -> list[dict[str, Any]]:
    with db.pool().connection() as conn:
        return conn.execute(
            "SELECT id, user_id, conversation_id, reason, summary, status, action, created_at FROM cases "
            "WHERE status IN ('open', 'pending_approval') ORDER BY created_at"
        ).fetchall()


def decide(case_id: str, approve: bool) -> dict[str, Any] | None:
    """Record the operator's decision once; returns the case, or None if there was no pending approval."""
    with db.pool().connection() as conn:
        return conn.execute(
            "UPDATE cases SET status = 'answered', action = action || %s WHERE id = %s AND status = 'pending_approval' "
            "RETURNING id, user_id, conversation_id, action, created_at",
            (Jsonb({"decision": "approved" if approve else "rejected"}), case_id),
        ).fetchone()


def merge_action(case_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    """Merge fields into a case's action and return the merged action (approval result, "turn answered" flag)."""
    with db.pool().connection() as conn:
        row = conn.execute("UPDATE cases SET action = action || %s WHERE id = %s RETURNING action",
                           (Jsonb(fields), case_id)).fetchone()
    return dict(row["action"]) if row else {}


def reply(case_id: str, message: str) -> dict[str, Any] | None:
    with db.pool().connection() as conn:
        return conn.execute(
            "UPDATE cases SET status = 'answered', operator_reply = %s WHERE id = %s AND status = 'open' "
            "RETURNING conversation_id, user_id",
            (message, case_id),
        ).fetchone()


def add_feedback(conversation_id: str, turn: int, rating: str, comment: str | None) -> None:
    with db.pool().connection() as conn:
        conn.execute(
            "INSERT INTO feedback (conversation_id, turn, rating, comment) VALUES (%s, %s, %s, %s)",
            (conversation_id, turn, rating, comment),
        )


def demo_value(key: str, factory: Any) -> str:
    """Read-or-create a demo-mode value shared by every replica (secret, demo passwords)."""
    with db.pool().connection() as conn:
        conn.execute(
            "INSERT INTO demo_state (key, value) VALUES (%s, %s) ON CONFLICT (key) DO NOTHING", (key, factory())
        )
        row = conn.execute("SELECT value FROM demo_state WHERE key = %s", (key,)).fetchone()
    assert row is not None  # noqa: S101 — inserted just above
    return str(row["value"])
