"""Twitter/X scrape + LLM extract for the labeling pipeline.

Pulls recent tweets from a curated list of Polymarket/prediction-market
whale-tracking accounts. Uses only the user-timeline endpoint to stay
under free-tier quotas.

Usage:
    python scripts/scrape_twitter.py \\
        --handles PolymarketWhale,LookonchainSpy,ZachXBT \\
        --max-tweets-per-handle 100 \\
        --out-dir data/labels/twitter
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv("/Users/ben_paulson/Documents/Personal/Stock_Portfolio/.env")
except ImportError:  # pragma: no cover
    pass

from pminsider.extract.llm import LLMExtractor
from pminsider.extract.schema import RawCallout
from pminsider.scrape.twitter import DEFAULT_HANDLES, TwitterClient, scrape_handles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/labels/twitter"))
    ap.add_argument(
        "--handles", type=str, default=",".join(DEFAULT_HANDLES),
        help="comma-separated list of handles (no @)",
    )
    ap.add_argument("--max-tweets-per-handle", type=int, default=80)
    ap.add_argument("--budget-usd", type=float, default=3.0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("scrape_twitter")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    handles = [h.strip() for h in args.handles.split(",") if h.strip()]

    client = TwitterClient()
    raws = scrape_handles(client, handles=handles, max_tweets_per_handle=args.max_tweets_per_handle)
    log.info("raw tweets: %d", len(raws))

    raw_rows = [_raw_to_row(r) for r in raws]
    if raw_rows:
        pd.DataFrame(raw_rows).to_parquet(args.out_dir / "raw.parquet", index=False)

    plausible = [r for r in raws if _plausibly_about_trading(r)]
    log.info("plausibly relevant: %d / %d", len(plausible), len(raws))

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    extractor = LLMExtractor(cache_dir=args.out_dir / "_extract_cache")
    enriched = []
    for r in tqdm(plausible, desc="llm-extract"):
        if extractor.total_cost_usd >= args.budget_usd and extractor.cache_hits < extractor.total_calls:
            log.warning("budget cap $%.2f hit; stopping early", args.budget_usd)
            break
        enriched.append(extractor.extract(r))

    rows = [e.to_flat_dict() for e in enriched]
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(args.out_dir / "enriched.parquet", index=False)
        allegations = df[df["is_allegation"] == True]  # noqa: E712
        log.info("allegations: %d / %d extracted", len(allegations), len(df))
        review_cols = [
            "raw_source_url", "raw_author", "raw_score", "raw_created_utc",
            "raw_body", "is_allegation", "market_question",
            "ts_lower", "ts_upper", "size_usd_approx", "wallet_addr",
            "direction", "outcome_resolved", "confidence_tier",
            "quote", "reasoning",
        ]
        review_cols = [c for c in review_cols if c in df.columns]
        df[review_cols].to_csv(args.out_dir / "review_all.csv", index=False)
        if not allegations.empty:
            allegations[review_cols].to_csv(args.out_dir / "review_allegations.csv", index=False)

    log.info("extractor budget report: %s", extractor.budget_report())


def _raw_to_row(r: RawCallout) -> dict:
    return {
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


_TRADING_KEYWORDS = {
    "insider", "front-ran", "front ran", "frontrun", "leaked",
    "minutes before", "hours before", "before the", "knew about", "non-public",
    "whale", "bought $", "bet $", "position", "suspicious",
    "rigged", "manipulation", "sniped", "pump", "0x",
    "polymarket", "predictit", "kalshi", "prediction market",
    "yes token", "no token", "maduro", "iran", "strike",
    "$1m", "$100k", "$400k", "$500k", "$200k", "$50k",
    "burdensome", "magamyman",
}


def _plausibly_about_trading(r: RawCallout) -> bool:
    haystack = f"{(r.title or '').lower()} {r.body.lower()}"
    if len(haystack) < 30:
        return False
    return any(kw in haystack for kw in _TRADING_KEYWORDS)


if __name__ == "__main__":
    main()
