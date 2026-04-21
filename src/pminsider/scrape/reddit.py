"""Reddit scraper using the public JSON API — no authentication required.

Reddit's public JSON endpoint serves a copy of the same data the website
uses, rate-limited to ~30 req/min per IP. Fine for our throughput. We pull
top posts per subreddit over configurable time windows, then pull each
post's comment tree.

Output: a list of ``RawCallout`` for the orchestrator to pass to the
LLM extractor.
"""
from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from curl_cffi import requests as crequests

from ..extract.schema import RawCallout

log = logging.getLogger(__name__)

USER_AGENT = "polymarket-insider-detection/0.1 by u/researcher"
# old.reddit.com responds to unauthenticated JSON requests; the www host
# started returning HTTP 403 to non-browser UAs in 2024.
BASE = "https://old.reddit.com"
# Sustained scraping over many minutes triggers Reddit's stricter
# rate-limiting — keep requests conservative.
RATE_LIMIT_SEC = 4.0   # ~15 req/min


# Subreddits likely to host insider-trade callouts.
DEFAULT_SUBS = [
    "Polymarket",
    "PredictionMarkets",
    "PredictIt",
    "CryptoCurrency",       # occasional Polymarket whale discussion
    "AskReddit",            # sometimes "has anyone seen insider trading" threads
]

# Time windows for the /top endpoint
TIME_FILTERS = ("year", "month", "week")


class RedditClient:
    def __init__(
        self,
        cache_dir: str | Path = "data/labels/_reddit_cache",
        user_agent: str = USER_AGENT,
        min_delay_sec: float = RATE_LIMIT_SEC,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.user_agent = user_agent
        self.min_delay_sec = min_delay_sec
        self._last_call = 0.0
        self._session = crequests.Session(impersonate="chrome120")

    def _rate_limit(self) -> None:
        wait = self.min_delay_sec - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _get(self, path: str) -> dict | list:
        self._rate_limit()
        url = f"{BASE}{path}"
        # Reddit's anti-bot returns 403 with cookies on first touch, then
        # subsequent requests on the same session succeed. We retry once.
        for attempt in range(3):
            r = self._session.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                log.warning("rate limited; backing off 30s")
                time.sleep(30)
                continue
            if r.status_code == 403:
                log.debug("403 on attempt %d (cookie handshake), retrying", attempt + 1)
                time.sleep(1.0)
                continue
            r.raise_for_status()
        raise RuntimeError(f"Reddit {path!s} still failing after 3 attempts")

    def iter_top_posts(
        self,
        subreddit: str,
        *,
        time_filter: str = "year",
        limit: int = 100,
    ) -> Iterator[dict]:
        """Yield raw post dicts from /r/{sub}/top."""
        after: str | None = None
        pulled = 0
        per_page = min(100, limit)
        while pulled < limit:
            qs = f"?t={time_filter}&limit={per_page}"
            if after:
                qs += f"&after={after}"
            try:
                data = self._get(f"/r/{subreddit}/top.json{qs}")
            except Exception as e:  # noqa: BLE001
                log.warning("subreddit fetch failed %s: %s", subreddit, e)
                return
            children = (data or {}).get("data", {}).get("children", []) or []
            if not children:
                return
            for c in children:
                yield c.get("data", {})
                pulled += 1
                if pulled >= limit:
                    return
            after = (data or {}).get("data", {}).get("after")
            if not after:
                return

    def iter_comments(
        self, subreddit: str, post_id: str, *, limit: int = 200
    ) -> Iterator[dict]:
        """Yield comment dicts for a specific post (flattened)."""
        try:
            data = self._get(f"/r/{subreddit}/comments/{post_id}.json?limit={limit}")
        except Exception as e:  # noqa: BLE001
            log.warning("comments fetch failed %s/%s: %s", subreddit, post_id, e)
            return
        # data is [post_listing, comments_listing]
        if not isinstance(data, list) or len(data) < 2:
            return
        comments_listing = data[1] or {}
        children = comments_listing.get("data", {}).get("children", []) or []
        for c in _walk_comments(children):
            yield c


def _walk_comments(children: list) -> Iterator[dict]:
    for c in children or []:
        if c.get("kind") != "t1":
            continue
        d = c.get("data", {})
        if d.get("body") and not d.get("body", "").startswith("[deleted]"):
            yield d
        replies = d.get("replies")
        if isinstance(replies, dict):
            sub = replies.get("data", {}).get("children", []) or []
            yield from _walk_comments(sub)


def _post_to_raw(sub: str, post: dict) -> RawCallout:
    return RawCallout(
        source="reddit",
        source_id=f"t3_{post.get('id')}",
        source_url=f"https://reddit.com{post.get('permalink','')}",
        author=post.get("author"),
        created_utc=int(post.get("created_utc") or 0),
        title=post.get("title"),
        body=post.get("selftext") or "",
        score=int(post.get("score") or 0),
        num_replies=int(post.get("num_comments") or 0),
        parent_id=None,
        raw_metadata={
            "subreddit": sub,
            "over_18": post.get("over_18"),
            "link_flair_text": post.get("link_flair_text"),
            "is_video": post.get("is_video"),
        },
    )


def _comment_to_raw(sub: str, post_id: str, comment: dict) -> RawCallout:
    return RawCallout(
        source="reddit",
        source_id=f"t1_{comment.get('id')}",
        source_url=f"https://reddit.com{comment.get('permalink','')}",
        author=comment.get("author"),
        created_utc=int(comment.get("created_utc") or 0),
        title=None,
        body=comment.get("body") or "",
        score=int(comment.get("score") or 0),
        num_replies=None,
        parent_id=comment.get("parent_id"),
        raw_metadata={"subreddit": sub, "post_id": post_id},
    )


def scrape_subreddit(
    client: RedditClient,
    subreddit: str,
    *,
    post_limit: int = 100,
    comments_per_post: int = 100,
    time_filter: str = "year",
    include_comments: bool = True,
) -> list[RawCallout]:
    """Collect top posts from a subreddit + their comments."""
    callouts: list[RawCallout] = []
    post_ids: list[str] = []
    for post in client.iter_top_posts(subreddit, time_filter=time_filter, limit=post_limit):
        pid = post.get("id")
        if not pid:
            continue
        post_ids.append(pid)
        callouts.append(_post_to_raw(subreddit, post))
    if include_comments:
        for pid in post_ids:
            for c in client.iter_comments(subreddit, pid, limit=comments_per_post):
                callouts.append(_comment_to_raw(subreddit, pid, c))
    log.info("scraped %s: %d items (%d posts + comments)", subreddit, len(callouts), len(post_ids))
    return callouts


def scrape_many(
    client: RedditClient,
    subs: list[str] | None = None,
    *,
    post_limit: int = 100,
    time_filters: tuple = ("year",),
    include_comments: bool = True,
    comments_per_post: int = 100,
) -> list[RawCallout]:
    subs = subs or DEFAULT_SUBS
    all_callouts: list[RawCallout] = []
    for sub in subs:
        for tf in time_filters:
            all_callouts.extend(scrape_subreddit(
                client, sub,
                post_limit=post_limit,
                comments_per_post=comments_per_post,
                time_filter=tf,
                include_comments=include_comments,
            ))
    # Dedupe by source_id
    seen: set[str] = set()
    out: list[RawCallout] = []
    for c in all_callouts:
        if c.source_id in seen:
            continue
        seen.add(c.source_id)
        out.append(c)
    log.info("total unique reddit items: %d", len(out))
    return out
