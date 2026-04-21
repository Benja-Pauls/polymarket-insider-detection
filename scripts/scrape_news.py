"""Fetch curated news articles + run LLM extraction.

Input: data/labels/news_seeds.json (list of {url, title?, published_ts?}).
Output: data/labels/news/raw.parquet + enriched.parquet + review CSVs.
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
from pminsider.scrape.news import NewsClient, scrape_seed_list


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=Path, default=Path("data/labels/news_seeds.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/labels/news"))
    ap.add_argument("--budget-usd", type=float, default=2.0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("scrape_news")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.seeds.open() as f:
        seeds = json.load(f)
    log.info("seed articles: %d", len(seeds))

    client = NewsClient()
    raws = scrape_seed_list(client, seeds)
    log.info("fetched articles: %d", len(raws))

    raw_rows = [_raw_to_row(r) for r in raws]
    pd.DataFrame(raw_rows).to_parquet(args.out_dir / "raw.parquet", index=False)

    if not raws:
        log.warning("no articles fetched; nothing to extract")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set")

    extractor = LLMExtractor(cache_dir=args.out_dir / "_extract_cache")
    enriched = []
    for r in tqdm(raws, desc="llm-extract"):
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
            "raw_source_url", "raw_title", "is_allegation",
            "market_question", "ts_lower", "ts_upper",
            "size_usd_approx", "wallet_addr", "direction",
            "outcome_resolved", "confidence_tier", "quote", "reasoning",
        ]
        review_cols = [c for c in review_cols if c in df.columns]
        df[review_cols].to_csv(args.out_dir / "review_all.csv", index=False)
        if not allegations.empty:
            allegations[review_cols].to_csv(args.out_dir / "review_allegations.csv", index=False)

    log.info("extractor budget report: %s", extractor.budget_report())


def _raw_to_row(r: RawCallout) -> dict:
    import json
    return {
        "source": r.source,
        "source_id": r.source_id,
        "source_url": r.source_url,
        "author": r.author,
        "created_utc": r.created_utc,
        "title": r.title,
        "body": r.body[:20000],  # cap for parquet sanity
        "score": r.score,
        "num_replies": r.num_replies,
        "parent_id": r.parent_id,
        "raw_metadata_json": json.dumps(r.raw_metadata),
    }


if __name__ == "__main__":
    main()
