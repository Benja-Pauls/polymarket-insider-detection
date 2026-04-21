"""Reddit scrape + LLM extract for the labeling pipeline.

End-to-end run for checkpoint 1:
  1. Pull top posts + comments from a curated subreddit list.
  2. Run each raw item through Claude Haiku to extract structured callouts.
  3. Save raw + enriched parquets, plus a review-friendly CSV.

Usage:
    python scripts/scrape_reddit.py \\
        --out-dir data/labels/reddit \\
        --subs Polymarket,PredictionMarkets,PredictIt \\
        --post-limit 200 \\
        --log-level INFO
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Ensure Stock_Portfolio/.env is loaded BEFORE touching ANTHROPIC_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv("/Users/ben_paulson/Documents/Personal/Stock_Portfolio/.env")
except ImportError:  # pragma: no cover
    pass

from pminsider.extract.llm import LLMExtractor
from pminsider.extract.schema import RawCallout
from pminsider.scrape.reddit import (
    DEFAULT_SUBS, TIME_FILTERS, RedditClient, scrape_many,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/labels/reddit"))
    ap.add_argument(
        "--subs", type=str, default=",".join(DEFAULT_SUBS),
        help="comma-separated list of subreddits",
    )
    ap.add_argument(
        "--time-filters", type=str, default="year,month",
        help="comma-separated list of Reddit time filters (year/month/week/day/all)",
    )
    ap.add_argument("--post-limit", type=int, default=150)
    ap.add_argument("--comments-per-post", type=int, default=80)
    ap.add_argument("--no-comments", action="store_true")
    ap.add_argument("--max-extract", type=int, default=None,
                    help="cap on items sent to the LLM (useful for smoke testing)")
    ap.add_argument("--budget-usd", type=float, default=3.0,
                    help="extract budget; stop after this much spend")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("scrape_reddit")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Scrape raw callouts ------------------------------------------------
    subs = [s.strip() for s in args.subs.split(",") if s.strip()]
    time_filters = tuple(s.strip() for s in args.time_filters.split(",") if s.strip())
    log.info("scraping subs=%s time_filters=%s post_limit=%d", subs, time_filters, args.post_limit)
    client = RedditClient()
    raws = scrape_many(
        client,
        subs=subs,
        post_limit=args.post_limit,
        time_filters=time_filters,
        include_comments=not args.no_comments,
        comments_per_post=args.comments_per_post,
    )
    log.info("raw items: %d", len(raws))

    raw_df = pd.DataFrame([_raw_to_row(r) for r in raws])
    raw_df.to_parquet(args.out_dir / "raw.parquet", index=False)

    # --- 2. LLM-extract --------------------------------------------------------
    # Pre-filter: keep items that *plausibly* concern a trade, to save tokens.
    plausible = [r for r in raws if _plausibly_about_trading(r)]
    log.info("plausibly trade-related: %d / %d", len(plausible), len(raws))

    # Additional cap
    if args.max_extract is not None:
        plausible = plausible[: args.max_extract]

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set — cannot run extraction")

    extractor = LLMExtractor(cache_dir=args.out_dir / "_extract_cache")
    enriched = []
    for r in tqdm(plausible, desc="llm-extract"):
        if extractor.total_cost_usd >= args.budget_usd and extractor.cache_hits < extractor.total_calls:
            log.warning("budget cap $%.2f hit; stopping early", args.budget_usd)
            break
        enriched.append(extractor.extract(r))

    # --- 3. Dump parquets and a review CSV ------------------------------------
    enriched_rows = [e.to_flat_dict() for e in enriched]
    if enriched_rows:
        enriched_df = pd.DataFrame(enriched_rows)
        enriched_df.to_parquet(args.out_dir / "enriched.parquet", index=False)

        allegations = enriched_df[enriched_df["is_allegation"] == True].copy()  # noqa: E712
        log.info("allegations found: %d / %d extracted", len(allegations), len(enriched_df))

        review_cols = [
            "raw_source", "raw_source_url", "raw_created_utc", "raw_title",
            "raw_body", "raw_score", "is_allegation", "market_question",
            "ts_lower", "ts_upper", "size_usd_approx", "wallet_addr",
            "direction", "outcome_resolved", "confidence_tier",
            "quote", "reasoning",
        ]
        review_cols = [c for c in review_cols if c in enriched_df.columns]
        # Full review: every extracted item
        enriched_df[review_cols].to_csv(args.out_dir / "review_all.csv", index=False)
        # Allegations only
        if not allegations.empty:
            allegations[review_cols].to_csv(args.out_dir / "review_allegations.csv", index=False)

    log.info("extractor budget report: %s", extractor.budget_report())


def _raw_to_row(r: RawCallout) -> dict:
    import json
    d = {
        "source": r.source,
        "source_id": r.source_id,
        "source_url": r.source_url,
        "author": r.author,
        "created_utc": r.created_utc,
        "title": r.title,
        "body": r.body,
        "score": r.score,
        "num_replies": r.num_replies,
        "parent_id": r.parent_id,
        "raw_metadata_json": json.dumps(r.raw_metadata),
    }
    return d


_TRADING_KEYWORDS = {
    # Specific-allegation keywords
    "insider", "front-ran", "front ran", "frontrun", "leaked",
    "minutes before", "hours before", "before the announcement",
    "before the strike", "knew about", "non-public",
    # Activity / whale keywords
    "whale", "bought $", "bet $", "wager", "position", "suspicious",
    "rigged", "manipulation", "sniped", "pump", "dump", "0x",
    # Polymarket-specific
    "polymarket", "predictit", "kalshi", "prediction market",
    "yes token", "no token", "conditional",
    # Amounts (rough — cheap to filter)
    "$1m", "$10m", "$100k", "$400k", "$500k", "$200k", "$50k", "$1000000", "$400,000",
}


def _plausibly_about_trading(r: RawCallout) -> bool:
    """Quick keyword filter to avoid LLM-extracting obviously-irrelevant items."""
    haystack = f"{(r.title or '').lower()} {r.body.lower()}"
    if len(haystack) < 30:
        return False
    return any(kw in haystack for kw in _TRADING_KEYWORDS)


if __name__ == "__main__":
    main()
