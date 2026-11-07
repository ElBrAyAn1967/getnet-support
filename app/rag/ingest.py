# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""R-1 ingest: fetch → extract main text → drop site-wide boilerplate → chunk → embed/store (index.upsert).

Run once by the compose `ingest` service: `python -m app.rag.ingest`. Each source that fails to download falls
back to its last good snapshot in data/snapshot/, so the index never loses a source because a page is down.
"""

import hashlib
import io
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlsplit

import httpx
from lxml import html as lxml_html
from pypdf import PdfReader

from app import db
from app.config import get_settings
from app.http import USER_AGENT, client
from app.observability import configure_logging, log
from app.rag.index import Chunk, count, upsert

DATA = Path(__file__).resolve().parents[2] / "data"
CHUNK_CHARS = 900
MIN_CHUNK_CHARS = 80  # shorter fragments are labels and calls to action, not answers
BOILERPLATE_SHARE = 0.3
# Site chrome goes with its <header>/<nav>/<footer> tags and the cross-page repetition filter. Matching class names
# like "modal" or "header" also deletes content: this site keeps its FAQ answers inside `o-modal__content--faq`.
_NOISE = re.compile(r"cookie|newsletter|breadcrumb", re.I)
_BLOCKS = {"h1", "h2", "h3", "h4", "p", "li", "dt", "dd", "td", "summary"}
_HEADINGS = {"h1", "h2", "h3", "h4"}


@dataclass
class Page:
    url: str
    title: str
    fetched_at: str
    blocks: list[tuple[str, str]]  # (heading, text)


def read_sources(path: Path = DATA / "sources.txt") -> list[str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def _snapshot_path(url: str) -> Path:
    return DATA / "snapshot" / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.json"  # noqa: S324 — a filename


def _extract_html(url: str, content: bytes) -> Page:
    tree = lxml_html.fromstring(content)
    for el in tree.xpath("//script|//style|//noscript|//svg|//form|//header|//nav|//footer"):
        el.drop_tree()
    for el in tree.xpath("//*[@class or @id]"):
        if el.getparent() is not None and _NOISE.search(f"{el.get('class', '')} {el.get('id', '')}"):
            el.drop_tree()
    title = (tree.findtext(".//title") or url).split("|")[0].strip()
    heading, blocks = title, []
    for el in tree.iter(*_BLOCKS):
        text = re.sub(r"\s+", " ", el.text_content()).strip()
        if len(text) < 3 or "{{" in text:  # noqa: PLR2004 — symbols, empty cells, unrendered templates ({{x}})
            continue
        if el.tag in _HEADINGS:
            heading = text
        else:
            blocks.append((heading, text))
    return Page(url, title, date.today().isoformat(), blocks)


def _extract_pdf(url: str, content: bytes) -> Page:
    reader = PdfReader(io.BytesIO(content))
    title = Path(url).stem.replace("_", " ").replace("-", " ")
    blocks = [
        (f"{title} — p. {i}", re.sub(r"\s+", " ", page.extract_text() or "").strip())
        for i, page in enumerate(reader.pages, start=1)
    ]
    return Page(url, title, date.today().isoformat(), [b for b in blocks if b[1]])


def fetch(client: httpx.Client, url: str) -> Page | None:
    snapshot = _snapshot_path(url)
    try:
        resp = client.get(url)
        resp.raise_for_status()
        is_pdf = "pdf" in resp.headers.get("content-type", "") or url.lower().endswith(".pdf")
        page = (_extract_pdf if is_pdf else _extract_html)(str(resp.url), resp.content)
        if not page.blocks:
            raise ValueError("empty extraction")  # never index, or snapshot, a page we failed to read
    except (httpx.HTTPError, ValueError) as exc:
        data = json.loads(snapshot.read_text()) if snapshot.exists() else {"blocks": []}
        if data["blocks"]:  # an empty snapshot is no fallback: it would empty the index for this URL
            log("ingest_snapshot_fallback", url=url, error=type(exc).__name__)
            return Page(data["url"], data["title"], data["fetched_at"], [tuple(b) for b in data["blocks"]])
        log("ingest_source_failed", url=url, error=type(exc).__name__)
        return None
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps(asdict(page), ensure_ascii=False))
    return page


@lru_cache
def _robots(origin: str) -> robotparser.RobotFileParser:
    parser = robotparser.RobotFileParser()
    try:
        response = client().get(f"{origin}/robots.txt")
        parser.parse(response.text.splitlines() if response.status_code == 200 else [])  # noqa: PLR2004
    except httpx.HTTPError:
        parser.parse([])  # robots.txt unreachable: no rules to honour
    return parser


def allowed_by_robots(url: str) -> bool:
    parts = urlsplit(url)
    return _robots(f"{parts.scheme}://{parts.netloc}").can_fetch(USER_AGENT, url)


def drop_boilerplate(pages: list[Page]) -> list[Page]:
    """A block repeated on many HTML pages is site chrome (menus, banners), not the page's content."""
    html_pages = [p for p in pages if not p.url.lower().endswith(".pdf")]
    seen = Counter(text for p in html_pages for text in {t for _, t in p.blocks})
    limit = max(3, int(len(html_pages) * BOILERPLATE_SHARE))
    return [Page(p.url, p.title, p.fetched_at, [(h, t) for h, t in p.blocks if seen[t] < limit]) for p in pages]


def chunk(page: Page) -> list[Chunk]:
    """Pack a page's blocks into ~CHUNK_CHARS windows. Headings stay inline ("Ofereça parcelamento: em até 48x"), so
    a short fact keeps the words that say what it is about; repeated blocks within a page are kept once."""
    windows: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    last_heading, size = "", 0
    for heading, text in page.blocks:
        if text in seen:
            continue
        seen.add(text)
        line = text if heading in (last_heading, page.title) else f"{heading}: {text}"
        if not windows or size + len(line) > CHUNK_CHARS:
            windows.append((heading, []))
            size = 0
        windows[-1][1].append(line)
        size += len(line) + 1
        last_heading = heading
    chunks = []
    for heading, lines in windows:
        body = "\n".join(lines)
        if len(body) < MIN_CHUNK_CHARS:
            continue
        title = page.title if heading == page.title else f"{page.title} — {heading}"
        digest = hashlib.sha256(f"{page.url}\n{body}".encode()).hexdigest()
        chunks.append(Chunk(page.url, title, date.fromisoformat(page.fetched_at), body, digest))
    return chunks


def main() -> None:
    configure_logging()
    settings = get_settings()
    db.migrate()
    pages = []
    for url in read_sources():
        if not allowed_by_robots(url):
            log("ingest_robots_disallowed", url=url)
            continue
        if page := fetch(client(), url):
            pages.append(page)
        time.sleep(settings.ingest_delay_s)
    for page in drop_boilerplate(pages):
        added = upsert(page.url, chunk(page))
        log("ingest_page", url=page.url, blocks=len(page.blocks), added=added)
    log("ingest_done", sources=len(pages), chunks=count())


if __name__ == "__main__":
    main()
