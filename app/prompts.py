# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Versioned prompts: one Markdown file per prompt with `version:` front matter (reported in every response)."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    text: str


@cache
def load(name: str) -> Prompt:
    raw = (PROMPTS / f"{name}.md").read_text()
    _, front, body = raw.split("---", 2)
    meta = dict(line.split(":", 1) for line in front.strip().splitlines())
    return Prompt(name, meta["version"].strip(), body.strip())


def versions() -> dict[str, str]:
    return {p.stem: load(p.stem).version for p in sorted(PROMPTS.glob("*.md"))}
