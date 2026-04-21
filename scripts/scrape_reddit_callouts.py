"""Scrape r/Polymarket for posts describing suspected insider trades.

Uses the PRAW client configured in ``~/Documents/Personal/Stock_Portfolio/.env``
(REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT). Outputs a
parquet of candidate posts tagged with confidence-weighted keywords. Separately
we have to resolve the referenced market question text → condition_id, which
is done after GH Actions proxies the Gamma API (see
``fetch_gamma_metadata.py``).

This script only produces candidates; the merge happens in a downstream
script (``scripts/match_callouts_to_markets.py``).
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

# Load Stock_Portfolio/.env for Reddit credentials
try:
    from dotenv import load_dotenv
    load_dotenv("/Users/ben_paulson/Documents/Personal/Stock_Portfolio/.env")
except ImportError:  # pragma: no cover
    pass

INSIDER_KEYWORDS = {
    # Strong signals of a specific-trade callout
    "insider":        1.0,
    "front-ran":      0.9,
    "front ran":      0.9,
    "leaked":         0.9,
    "someone bet":    0.7,
    "someone knew":   0.9,
    "whale":          0.5,
    "suspicious":     0.6,
    "rigged":         0.7,
    "manipulation":   0.5,
    "information":    0.3,   # soft — common false positive
    "early":          0.3,
    "before the":     0.3,
    "minutes before": 0.8,
    "hours before":   0.8,
    "pump":           0.4,
    "dump":           0.4,
    "sniped":         0.7,
    "shark":          0.4,
    "non-public":     0.9,
    "knew about":     0.7,
}

# Subreddits known to discuss specific Polymarket bets
SUBREDDITS = ["Polymarket", "PredictionMarkets", "prediction_markets", "PredictIt"]


@dataclass
class RedditPost:
    id: str
    subreddit: str
    title: str
    selftext: str
    permalink: str
    score: int
    num_comments: int
    created_utc: float
    matched_keywords: str
    keyword_score: float


def _score(text: str) -> tuple[float, list[str]]:
    lowered = text.lower()
    hits: list[str] = []
    total = 0.0
    for kw, weight in INSIDER_KEYWORDS.items():
        if kw in lowered:
            hits.append(kw)
            total += weight
    return total, hits


def scrape(
    limit_per_sub: int = 1000,
    min_keyword_score: float = 0.7,
    include_comments: bool = True,
) -> list[RedditPost]:
    import praw  # heavy import deferred

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "pminsider-research"),
    )

    found: list[RedditPost] = []
    log = logging.getLogger("reddit")
    for sub_name in SUBREDDITS:
        try:
            sub = reddit.subreddit(sub_name)
            log.info("scanning r/%s (top %d)…", sub_name, limit_per_sub)
            seen = 0
            for post in sub.top(time_filter="all", limit=limit_per_sub):
                seen += 1
                text_all = f"{post.title}\n{post.selftext}"
                if include_comments:
                    try:
                        post.comments.replace_more(limit=0)
                        for c in post.comments.list():
                            text_all += "\n" + (c.body or "")
                    except Exception:  # noqa: BLE001
                        pass
                score, hits = _score(text_all)
                if score < min_keyword_score:
                    continue
                found.append(RedditPost(
                    id=post.id,
                    subreddit=sub_name,
                    title=post.title,
                    selftext=post.selftext or "",
                    permalink=f"https://reddit.com{post.permalink}",
                    score=int(post.score or 0),
                    num_comments=int(post.num_comments or 0),
                    created_utc=float(post.created_utc or 0),
                    matched_keywords=",".join(hits),
                    keyword_score=float(score),
                ))
                time.sleep(0.1)  # be polite
            log.info("  scanned %d, matched %d", seen, sum(1 for r in found if r.subreddit == sub_name))
        except Exception as e:  # noqa: BLE001
            log.warning("  r/%s failed: %s", sub_name, e)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--min-score", type=float, default=0.8)
    ap.add_argument("--no-comments", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("data/offchain/reddit_callouts.parquet"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")

    posts = scrape(
        limit_per_sub=args.limit,
        min_keyword_score=args.min_score,
        include_comments=not args.no_comments,
    )
    df = pd.DataFrame([asdict(p) for p in posts])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logging.info("wrote %d posts → %s", len(df), args.out)


if __name__ == "__main__":
    main()
