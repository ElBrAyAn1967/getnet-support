"""A-28: every path the README names in backticks exists, so the requirement map cannot point at nothing.
With --clock, also require the git tags that bracket the timed session."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH_LIKE = re.compile(r"`((?:app|tests|evals|scripts|prompts|data|docs)/[\w./-]+|[\w-]+\.(?:py|yaml|yml|md|toml))`")


def main() -> None:
    names = set(PATH_LIKE.findall((ROOT / "README.md").read_text()))
    missing = sorted(n for n in names if not (ROOT / n.split("#")[0]).exists())
    print(f"README paths: {len(names)} named, {len(missing)} missing {missing or ''}")
    if "--clock" in sys.argv:
        git = shutil.which("git") or "git"
        listing = subprocess.run([git, "tag", "-l", "clock-start", "clock-end"],  # noqa: S603 — fixed arguments
                                 cwd=ROOT, capture_output=True, text=True, check=True)
        tags = listing.stdout.split()
        print(f"clock tags: {tags}")
        missing += [t for t in ("clock-start", "clock-end") if t not in tags]
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
