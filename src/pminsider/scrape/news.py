"""News-article scraper for insider-trade callouts.

Not a full web crawler — we take a curated list of article URLs and fetch
each one. HTML is extracted to clean text with BeautifulSoup. Each article
becomes a single ``RawCallout`` (we don't split into paragraphs; the LLM
extractor sees the whole piece).

Curated seed URLs live in ``data/labels/news_seeds.json`` and are hand-
assembled via WebSearch queries for known allegations. Over time we can
expand by following references (article → cited article) but v1 is manual.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Iterator

from curl_cffi import requests as crequests

from ..extract.schema import RawCallout

log = logging.getLogger(__name__)


def _url_fingerprint(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


class NewsClient:
    def __init__(
        self,
        cache_dir: str | Path = "data/labels/_news_cache",
        min_delay_sec: float = 2.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_delay_sec = min_delay_sec
        self._last_call = 0.0
        self._session = crequests.Session(impersonate="chrome120")

    def _rate_limit(self) -> None:
        wait = self.min_delay_sec - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def fetch(self, url: str) -> str | None:
        """Return article text or None on failure. Cached on disk."""
        fp = _url_fingerprint(url)
        cache_path = self.cache_dir / f"{fp}.html"
        text_path = self.cache_dir / f"{fp}.txt"

        if text_path.exists():
            return text_path.read_text()

        self._rate_limit()
        try:
            r = self._session.get(url, timeout=30)
        except Exception as e:  # noqa: BLE001
            log.warning("fetch failed %s: %s", url, e)
            return None
        if r.status_code != 200:
            log.warning("HTTP %d for %s", r.status_code, url)
            return None

        html = r.text
        cache_path.write_text(html)
        text = _html_to_text(html)
        text_path.write_text(text)
        return text


def _html_to_text(html: str) -> str:
    """Minimal HTML-to-text converter. Prefers BeautifulSoup if installed."""
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError:
        # Fallback: strip tags with a regex — lossy but OK for extraction
        import re
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    soup = BeautifulSoup(html, "html.parser")
    for el in soup(["script", "style", "nav", "footer", "aside"]):
        el.decompose()
    text = soup.get_text(separator=" ")
    return " ".join(text.split())


def url_to_raw(url: str, text: str, *, published_ts: int | None = None, title: str | None = None) -> RawCallout:
    return RawCallout(
        source="news",
        source_id=f"news_{_url_fingerprint(url)}",
        source_url=url,
        author=None,
        created_utc=published_ts or 0,
        title=title,
        body=text,
        score=None,
        num_replies=None,
        parent_id=None,
        raw_metadata={},
    )


def scrape_seed_list(client: NewsClient, seeds: list[dict]) -> list[RawCallout]:
    """Given a list of {url, title?, published_ts?}, fetch and return RawCallouts."""
    out: list[RawCallout] = []
    for s in seeds:
        url = s["url"]
        text = client.fetch(url)
        if not text:
            continue
        out.append(url_to_raw(
            url=url,
            text=text,
            published_ts=s.get("published_ts"),
            title=s.get("title"),
        ))
    log.info("news seed scrape: %d / %d articles fetched", len(out), len(seeds))
    return out


def load_seed_list(seed_file: str | Path) -> list[dict]:
    with open(seed_file) as f:
        return json.load(f)
