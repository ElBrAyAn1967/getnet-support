"""A-25: the deployment surface stays minimal — non-root image with a healthcheck, the API bound to localhost,
the database with no published port."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "compose.yaml").read_text()
    db_block = compose.split("\n  db:\n", 1)[1].split("\n  ingest:\n", 1)[0]
    checks = {
        "image runs as a non-root user": bool(re.search(r"^USER (?!root)\w+", dockerfile, re.M)),
        "image has a healthcheck": "HEALTHCHECK" in dockerfile,
        "api ports bound to 127.0.0.1": all(p.startswith('"127.0.0.1:') for p in
                                            re.findall(r'^\s+- ("[\d.:-]+")$', compose, re.M)),
        "database publishes no port": "ports:" not in db_block,
        "api waits for the ingest service": "service_completed_successfully" in compose,
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  A-25  {name}")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
