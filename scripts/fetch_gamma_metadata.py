"""Fetch Polymarket Gamma API market metadata (question text + category).

Runs from GitHub Actions (Azure EU-West) where Polymarket's Cloudflare
geo-block does NOT apply. Should NOT be run from a US IP — it will fail
with TLS connection resets.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"


def fetch_markets(batch_size: int, resolved_only: bool) -> list[dict]:
    out: list[dict] = []
    offset = 0
    log = logging.getLogger("fetch_gamma")
    params_base = {
        "limit": batch_size,
        "order": "volumeNum",
        "ascending": "false",
    }
    if resolved_only:
        params_base["closed"] = "true"
    while True:
        params = dict(params_base, offset=offset)
        log.info("fetching offset=%d", offset)
        r = requests.get(f"{GAMMA_BASE}/markets", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        out.extend(data)
        offset += batch_size
        time.sleep(0.5)  # be polite
        if len(data) < batch_size:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--resolved", type=str, default="true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    resolved_only = args.resolved.lower() in ("true", "1", "yes")

    markets = fetch_markets(args.batch, resolved_only)
    logging.info("fetched %d markets", len(markets))

    # Keep a minimal, stable subset of columns for joining
    keep = []
    for m in markets:
        outcome_prices = m.get("outcomePrices")
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except (TypeError, ValueError):
                outcome_prices = None

        clob_token_ids = m.get("clobTokenIds")
        if isinstance(clob_token_ids, str):
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except (TypeError, ValueError):
                clob_token_ids = None

        keep.append({
            "id": m.get("id"),
            "conditionId": m.get("conditionId"),
            "questionId": m.get("questionID") or m.get("questionId"),
            "clobTokenIds": clob_token_ids,
            "question": m.get("question"),
            "slug": m.get("slug"),
            "category": m.get("category"),
            "outcomes": m.get("outcomes"),
            "outcomePrices": outcome_prices,
            "volume": m.get("volume") or m.get("volumeNum"),
            "liquidity": m.get("liquidity"),
            "active": m.get("active"),
            "closed": m.get("closed"),
            "archived": m.get("archived"),
            "startDate": m.get("startDate"),
            "endDate": m.get("endDate"),
            "createdAt": m.get("createdAt"),
            "updatedAt": m.get("updatedAt"),
            "umaEndDate": m.get("umaEndDate"),
            "resolvedBy": m.get("resolvedBy"),
        })

    df = pd.DataFrame(keep)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logging.info("wrote %d rows → %s", len(df), args.out)


if __name__ == "__main__":
    main()
