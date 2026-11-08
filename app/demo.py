# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Print the demo credentials (APP_MODE=demo only): `docker compose exec api python -m app.demo`."""

from app import db
from app.config import Mode, get_settings


def main() -> None:
    if get_settings().app_mode is not Mode.DEMO:
        raise SystemExit("demo credentials exist only in APP_MODE=demo")
    with db.pool().connection() as conn:
        rows = conn.execute("SELECT key, value FROM demo_state WHERE key LIKE 'password:%' ORDER BY key").fetchall()
    for row in rows:
        print(f"{row['key'].removeprefix('password:'):12} {row['value']}")


if __name__ == "__main__":
    main()
