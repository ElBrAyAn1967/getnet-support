# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Identity (S-1..S-3, H-1): scrypt passwords, pinned-algorithm JWT, login throttling shared by all replicas."""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass
from typing import Annotated, Literal

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import db
from app.config import Settings, get_settings

ALGORITHM = "HS256"
Role = Literal["customer", "operator"]
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}
_bearer = HTTPBearer(auto_error=False)
_DUMMY_HASH = hashlib.scrypt(b"-", salt=b"0" * 16, **_SCRYPT).hex()  # unknown users take as long as known ones


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: Role


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    if "$" not in stored:
        stored = f"{'30' * 16}${stored}"  # the dummy hash: same cost, never matches a real password
    salt_hex, digest_hex = stored.split("$", 1)
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT)
    return hmac.compare_digest(digest.hex(), digest_hex)


def _secret(settings: Settings) -> str:
    if settings.auth_secret is None:
        raise RuntimeError("AUTH_SECRET is not set")  # demo startup generates it; production refuses to start
    return settings.auth_secret.get_secret_value()


def issue_token(user_id: str, role: Role, settings: Settings) -> str:
    now = int(time.time())
    claims = {"sub": user_id, "role": role, "iat": now, "exp": now + settings.token_ttl_s}
    return jwt.encode(claims, _secret(settings), algorithm=ALGORITHM)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"})


def current_principal(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Principal:
    if creds is None:
        raise _unauthorized("missing bearer token")
    try:
        claims = jwt.decode(
            creds.credentials, _secret(settings), algorithms=[ALGORITHM], options={"require": ["exp", "sub"]}
        )
    except jwt.PyJWTError as exc:
        raise _unauthorized("invalid token") from exc
    return Principal(user_id=claims["sub"], role=claims.get("role", "customer"))


def require_operator(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
    if principal.role != "operator":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "operator role required")
    return principal


def _throttle_key(ip: str, username: str) -> str:
    return hashlib.sha256(f"{ip}|{username}".encode()).hexdigest()


def retry_after(ip: str, username: str, settings: Settings) -> int:
    """Seconds to wait, or 0. Counts failures since the last success inside the window, per (ip, username),
    so an attacker on another IP cannot lock the owner out."""
    with db.pool().connection() as conn:
        row = conn.execute(
            """
            WITH last_ok AS (SELECT max(at) AS at FROM login_attempts WHERE key = %(k)s AND ok)
            SELECT count(*) AS n, min(a.at) AS first
            FROM login_attempts a, last_ok
            WHERE a.key = %(k)s AND NOT a.ok
              AND a.at > greatest(now() - make_interval(secs => %(w)s), coalesce(last_ok.at, '-infinity'))
            """,
            {"k": _throttle_key(ip, username), "w": settings.login_window_s},
        ).fetchone()
    if row is None or row["n"] < settings.login_max_failures:
        return 0
    return max(1, int(settings.login_window_s - (time.time() - row["first"].timestamp())))


def login(ip: str, username: str, password: str) -> Principal | None:
    with db.pool().connection() as conn:
        row = conn.execute("SELECT id, password_hash, role FROM users WHERE id = %s", (username,)).fetchone()
        ok = verify_password(password, row["password_hash"] if row else _DUMMY_HASH) and row is not None
        conn.execute("INSERT INTO login_attempts (key, ok) VALUES (%s, %s)", (_throttle_key(ip, username), ok))
    return Principal(user_id=row["id"], role=row["role"]) if ok and row else None
