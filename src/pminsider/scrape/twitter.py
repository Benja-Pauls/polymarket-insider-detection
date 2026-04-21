"""Twitter/X scraper using the v2 API (Bearer token auth).

Free-tier quotas are stingy (~10K tweet reads / month), so we target a
curated list of Polymarket-whale-tracking accounts plus specific searches
for known high-profile cases rather than broad trawls.

Env vars required:
  TWITTER_BEARER_TOKEN  — app-only auth

Endpoints used:
  GET /2/users/by/username/{handle}       → resolve handle → user_id
  GET /2/users/{user_id}/tweets           → recent tweets from that user
  GET /2/tweets/search/recent             → keyword search (optional; limited)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from curl_cffi import requests as crequests

from ..extract.schema import RawCallout

log = logging.getLogger(__name__)

API_BASE = "https://api.x.com/2"

# Curated list of Polymarket / prediction-market watchers.
DEFAULT_HANDLES = [
    "PolymarketWhale",     # whale tracker
    "PolymarketNews",      # news aggregator
    "PolymarketIntel",     # ditto
    "PMCalls",             # trade callouts
    "PolymarketPicks",     # picks feed
    "PolymarketAlert",     # alerts
    "polymarket",          # official (low-signal for insider allegations but good for cross-refs)
    "Polymarket_HQ",       # alt official
    "LookonchainSpy",      # on-chain whale tracker
    "lookonchain",         # big on-chain tracker
    "web3_sleuth",
    "ZachXBT",             # crypto investigator, occasionally PM coverage
    "CryptoStakerNews",
    "predictionmkts",
    "polyseer",
    "polymarketbot",
    "pmnews_",
]


class TwitterClient:
    def __init__(
        self,
        bearer_token: str | None = None,
        cache_dir: str | Path = "data/labels/_twitter_cache",
        min_delay_sec: float = 1.5,
    ):
        self.bearer = bearer_token or os.environ.get("TWITTER_BEARER_TOKEN")
        if not self.bearer:
            raise RuntimeError("TWITTER_BEARER_TOKEN not set")
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

    def _get(self, path: str, params: dict | None = None) -> dict:
        self._rate_limit()
        params = params or {}
        r = self._session.get(
            f"{API_BASE}{path}",
            params=params,
            headers={"Authorization": f"Bearer {self.bearer}"},
            timeout=30,
        )
        if r.status_code == 429:
            # Twitter returns rate-limit reset time as Unix epoch in a header
            reset = r.headers.get("x-rate-limit-reset")
            try:
                reset_ts = int(reset) if reset else None
            except (TypeError, ValueError):
                reset_ts = None
            now = int(time.time())
            wait = max(5, (reset_ts - now + 1)) if reset_ts else 60
            wait = min(wait, 90 * 60)  # cap at 90 min
            log.warning("rate limited; sleeping %ds", wait)
            time.sleep(wait)
            return self._get(path, params)
        if r.status_code != 200:
            raise RuntimeError(f"Twitter HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    # --- resolution -------------------------------------------------------

    def resolve_handle(self, handle: str) -> str | None:
        cache_file = self.cache_dir / f"user_{handle}.json"
        if cache_file.exists():
            with cache_file.open() as f:
                cached = json.load(f)
            return cached.get("id")
        try:
            data = self._get(f"/users/by/username/{handle}")
        except Exception as e:  # noqa: BLE001
            log.warning("could not resolve @%s: %s", handle, e)
            with cache_file.open("w") as f:
                json.dump({"id": None, "error": str(e)}, f)
            return None
        user = (data or {}).get("data", {})
        uid = user.get("id")
        with cache_file.open("w") as f:
            json.dump({"id": uid, "username": handle}, f)
        return uid

    # --- timeline ---------------------------------------------------------

    def user_tweets(
        self,
        handle: str,
        *,
        max_tweets: int = 100,
        exclude_replies: bool = False,
        exclude_retweets: bool = True,
    ) -> list[dict]:
        user_id = self.resolve_handle(handle)
        if not user_id:
            return []
        cache_file = self.cache_dir / f"tweets_{handle}.json"
        if cache_file.exists():
            with cache_file.open() as f:
                return json.load(f)

        exclude = []
        if exclude_replies:
            exclude.append("replies")
        if exclude_retweets:
            exclude.append("retweets")
        params: dict = {
            "max_results": min(100, max_tweets),
            "tweet.fields": "id,text,created_at,author_id,public_metrics,conversation_id",
        }
        if exclude:
            params["exclude"] = ",".join(exclude)

        collected: list[dict] = []
        pagination_token: str | None = None
        while len(collected) < max_tweets:
            if pagination_token:
                params["pagination_token"] = pagination_token
            else:
                params.pop("pagination_token", None)
            try:
                data = self._get(f"/users/{user_id}/tweets", params)
            except Exception as e:  # noqa: BLE001
                log.warning("user_tweets failed for @%s: %s", handle, e)
                break
            items = data.get("data", []) or []
            collected.extend(items)
            pagination_token = (data.get("meta") or {}).get("next_token")
            if not pagination_token:
                break
        with cache_file.open("w") as f:
            json.dump(collected[:max_tweets], f)
        return collected[:max_tweets]

    # --- search (limited quota) -------------------------------------------

    def search_recent(self, query: str, *, max_results: int = 50) -> list[dict]:
        """Recent-search endpoint. Basic tier is limited to last 7 days."""
        cache_key = self.cache_dir / f"search_{_hash(query)}.json"
        if cache_key.exists():
            with cache_key.open() as f:
                return json.load(f)
        params = {
            "query": query,
            "max_results": min(100, max_results),
            "tweet.fields": "id,text,created_at,author_id,public_metrics,conversation_id",
        }
        try:
            data = self._get("/tweets/search/recent", params)
        except Exception as e:  # noqa: BLE001
            log.warning("search failed: %s", e)
            return []
        items = data.get("data", []) or []
        with cache_key.open("w") as f:
            json.dump(items, f)
        return items


def _hash(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# --- conversion to RawCallout --------------------------------------------

def _tweet_to_raw(handle: str, tweet: dict) -> RawCallout:
    tid = tweet.get("id")
    created_iso = tweet.get("created_at", "")
    try:
        import calendar, email.utils
        # created_at is ISO format "2024-01-01T12:34:56.000Z"
        from datetime import datetime, timezone
        created_utc = int(datetime.fromisoformat(created_iso.replace("Z", "+00:00")).timestamp())
    except Exception:  # noqa: BLE001
        created_utc = 0
    metrics = tweet.get("public_metrics") or {}
    return RawCallout(
        source="twitter",
        source_id=f"tw_{tid}",
        source_url=f"https://x.com/{handle}/status/{tid}",
        author=handle,
        created_utc=created_utc,
        title=None,
        body=tweet.get("text") or "",
        score=int(metrics.get("like_count", 0) or 0),
        num_replies=int(metrics.get("reply_count", 0) or 0),
        parent_id=tweet.get("conversation_id"),
        raw_metadata={
            "retweets": metrics.get("retweet_count"),
            "quotes": metrics.get("quote_count"),
            "impressions": metrics.get("impression_count"),
        },
    )


# --- high-level orchestration ---------------------------------------------

def scrape_handles(
    client: TwitterClient,
    handles: list[str] | None = None,
    *,
    max_tweets_per_handle: int = 100,
) -> list[RawCallout]:
    handles = handles or DEFAULT_HANDLES
    out: list[RawCallout] = []
    for h in handles:
        log.info("scraping @%s", h)
        tweets = client.user_tweets(h, max_tweets=max_tweets_per_handle)
        log.info("  @%s: %d tweets", h, len(tweets))
        for t in tweets:
            out.append(_tweet_to_raw(h, t))
    log.info("twitter total: %d tweets from %d handles", len(out), len(handles))
    return out


def scrape_searches(
    client: TwitterClient,
    queries: list[str],
    *,
    max_per_query: int = 50,
) -> list[RawCallout]:
    out: list[RawCallout] = []
    for q in queries:
        log.info("searching %r", q)
        tweets = client.search_recent(q, max_results=max_per_query)
        log.info("  %r: %d results", q, len(tweets))
        for t in tweets:
            author_id = t.get("author_id", "")
            # For search we don't know the handle — leave author as the author_id
            out.append(RawCallout(
                source="twitter",
                source_id=f"tw_{t.get('id')}",
                source_url=f"https://x.com/i/status/{t.get('id')}",
                author=f"id:{author_id}",
                created_utc=_parse_ts(t.get("created_at")),
                title=None,
                body=t.get("text") or "",
                score=int((t.get("public_metrics") or {}).get("like_count", 0)),
                num_replies=int((t.get("public_metrics") or {}).get("reply_count", 0)),
                raw_metadata={"search_query": q},
            ))
    return out


def _parse_ts(iso: str | None) -> int:
    if not iso:
        return 0
    try:
        from datetime import datetime
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except Exception:  # noqa: BLE001
        return 0
