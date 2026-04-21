"""Fetch trades for a specific list of condition_ids.

Use this to backfill known-insider-trade markets (Maduro, AlphaRaccoon,
Nobel, etc.) that weren't in our original catalog because of volume caps.

Usage:
    python scripts/fetch_specific_markets.py --ids 0x580adc...,0xea17b1...

Reads metadata from data/offchain/markets_metadata.parquet for question
text + volume + outcomes, uses Goldsky for tokens + payouts + trades.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pminsider.collect import (
    MarketRecord, fetch_trades, lookup_resolution_historical,
    lookup_resolution_live, backfill_live_resolution_ts,
)
from pminsider.goldsky import GoldskyClient


def fetch_tokens_and_resolution(g: GoldskyClient, condition_id: str) -> dict:
    """Get the token_to_outcome mapping + payout data for a single condition_id."""
    # pnl subgraph has positionIds (ordered) + payoutNumerators
    q = f'{{ c: condition(id: "{condition_id}") {{ positionIds payoutNumerators payoutDenominator }} }}'
    res = g.query("pnl", q).data.get("c") or {}
    position_ids = [str(p) for p in (res.get("positionIds") or [])]
    payouts = [str(p) for p in (res.get("payoutNumerators") or [])]

    # orderbook_resync has questionId + resolutionTimestamp (pre-Jan-5-2026 only)
    q2 = f'{{ c: condition(id: "{condition_id}") {{ id questionId oracle resolutionTimestamp outcomeSlotCount }} }}'
    try:
        res2 = g.query("orderbook_resync", q2).data.get("c") or {}
    except Exception:  # noqa: BLE001
        res2 = {}

    return {
        "token_ids": position_ids,
        "token_to_outcome": {pid: i for i, pid in enumerate(position_ids)},
        "payouts": payouts,
        "question_id": res2.get("questionId"),
        "resolution_timestamp": int(res2["resolutionTimestamp"]) if res2.get("resolutionTimestamp") else None,
        "oracle": res2.get("oracle"),
        "outcome_slot_count": res2.get("outcomeSlotCount") or len(position_ids),
    }


def backfill_live_resolution_single(g: GoldskyClient, condition_id: str) -> int | None:
    """For markets resolved post-2026-01-05, use earliest Redemption timestamp."""
    q = f'{{ r: redemptions(first: 1, where: {{condition: "{condition_id}"}}, orderBy: timestamp, orderDirection: asc) {{ timestamp }} }}'
    for subgraph in ("activity", "orderbook_resync"):
        try:
            rows = g.query(subgraph, q).data.get("r") or []
            if rows:
                return int(rows[0]["timestamp"])
        except Exception:  # noqa: BLE001
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", type=str, required=True,
                    help="comma-separated condition_ids")
    ap.add_argument("--metadata", type=Path, default=Path("data/offchain/markets_metadata.parquet"))
    ap.add_argument("--catalog", type=Path, default=Path("data/raw/goldsky/markets.parquet"),
                    help="will be APPENDED with new rows if not already present")
    ap.add_argument("--trades-dir", type=Path, default=Path("data/raw/goldsky/trades"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/raw/goldsky/_cache"))
    ap.add_argument("--window-days", type=int, default=14)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("fetch_specific")

    ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    log.info("condition_ids to fetch: %d", len(ids))

    md = pd.read_parquet(args.metadata) if args.metadata.exists() else pd.DataFrame()
    md_by_id = {r["conditionId"]: r for _, r in md.iterrows()} if not md.empty else {}

    catalog = pd.read_parquet(args.catalog) if args.catalog.exists() else pd.DataFrame()
    existing_ids = set(catalog["condition_id"].tolist()) if not catalog.empty else set()

    g = GoldskyClient(cache_dir=args.cache_dir)
    args.trades_dir.mkdir(parents=True, exist_ok=True)

    new_records: list[MarketRecord] = []
    for cid in ids:
        if cid in existing_ids:
            log.info("  already in catalog: %s", cid[:20])
            continue
        log.info("fetching %s", cid[:20])
        info = fetch_tokens_and_resolution(g, cid)
        if not info["token_ids"]:
            log.warning("  no token_ids for %s", cid[:20])
            continue
        resolution_ts = info["resolution_timestamp"]
        if resolution_ts is None:
            resolution_ts = backfill_live_resolution_single(g, cid)
        # Fall back to Gamma endDate if both failed
        if resolution_ts is None:
            md_row = md_by_id.get(cid, {})
            end_date = md_row.get("endDate")
            if end_date:
                try:
                    from datetime import datetime
                    resolution_ts = int(datetime.fromisoformat(end_date.replace("Z", "+00:00")).timestamp())
                except Exception:  # noqa: BLE001
                    pass

        md_row = md_by_id.get(cid, {})
        total_vol = float(md_row.get("volume") or 0)
        source_tier = "historical" if (resolution_ts and resolution_ts < 1_736_121_600) else "live"

        rec = MarketRecord(
            condition_id=cid,
            question_id=info.get("question_id"),
            resolution_timestamp=resolution_ts,
            payouts=info.get("payouts"),
            outcome_slot_count=info.get("outcome_slot_count"),
            oracle=info.get("oracle"),
            token_ids=info["token_ids"],
            token_to_outcome=info["token_to_outcome"],
            total_volume_usd=total_vol,
            total_trades=0,  # unknown from Gamma without a trades count field
            source_tier=source_tier,
        )
        new_records.append(rec)

    if not new_records:
        log.warning("no new markets to fetch")
        return

    # Fetch trades per market
    log.info("fetching trades for %d markets (window=%d days)…", len(new_records), args.window_days)
    for rec in tqdm(new_records, desc="trades"):
        out = args.trades_dir / f"{rec.condition_id}.parquet"
        if out.exists() and out.stat().st_size > 0:
            log.info("  trades already cached: %s", rec.condition_id[:20])
            continue
        try:
            trades = fetch_trades(g, rec, window_days=args.window_days)
        except Exception as e:  # noqa: BLE001
            log.warning("  fetch failed for %s: %s", rec.condition_id, e)
            continue
        if not trades.empty:
            trades.to_parquet(out, index=False)
            log.info("  %s: %d trades", rec.condition_id[:20], len(trades))
        else:
            out.touch()
            log.info("  %s: 0 trades in window", rec.condition_id[:20])

    # Append to catalog
    new_df = pd.DataFrame([asdict(r) for r in new_records])
    if not catalog.empty:
        merged = pd.concat([catalog, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_parquet(args.catalog, index=False)
    log.info("catalog now has %d markets (was %d, +%d new)", len(merged), len(existing_ids), len(new_records))


if __name__ == "__main__":
    main()
