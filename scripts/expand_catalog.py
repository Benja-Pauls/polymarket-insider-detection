"""Expand the trade catalog by adding markets that match insider-tradeable patterns.

Workflow:
  1. Start from the 250K-market Gamma metadata snapshot.
  2. Filter by volume >= --min-volume and closed == True.
  3. Select markets matching:
     (a) known-case keyword patterns (Gemini, GPT, Nobel, Iran strikes, etc.)
     (b) optional: top-N by volume (for broad coverage)
  4. For each selected conditionId not already in the catalog, call
     fetch_specific_markets machinery to pull trades.
  5. Rate-limited: 1.5s between markets (Goldsky is generous but we've been
     bitten by deep-pagination statement timeouts before).

Output: appends to data/raw/goldsky/markets.parquet, writes new per-market
trade files into data/raw/goldsky/trades/.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pminsider.collect import MarketRecord, fetch_trades
from pminsider.goldsky import GoldskyClient


log = logging.getLogger(__name__)


# Keyword patterns that flag known insider-tradeable markets. Each pattern is
# a regex; matches against the Gamma `question` field. These capture the
# well-known cases journalists have reported on, plus categories of markets
# where insider information is plausibly available.
KNOWN_CASE_PATTERNS = [
    # --- Product releases (corporate insiders, engineering leaks) ---
    (r"\bgemini\s*3", "gemini_3_release"),
    (r"\bgpt[- ]?5", "gpt_5_release"),
    (r"\bgpt[- ]?6", "gpt_6_release"),
    (r"\bchatgpt\s+browser", "chatgpt_browser"),
    (r"\bclaude\s+[0-9]", "claude_release"),
    (r"\bopenai.*release", "openai_release"),
    (r"\bgrok\s+[0-9]", "grok_release"),
    (r"\bllama\s*[0-9]", "llama_release"),

    # --- Geopolitical strikes (military insiders) ---
    (r"\bstrike\s+(iran|tehran)", "iran_strike"),
    (r"\biran\s+strike", "iran_strike"),
    (r"\byemen", "yemen_strike"),
    (r"\bhouthi", "houthi_strike"),
    (r"\brussia.{0,20}ukraine.{0,20}(ceasefire|strike)", "russia_ukraine"),
    (r"\bceasefire", "ceasefire"),

    # --- Political / legal ---
    (r"\bmaduro", "maduro"),
    (r"\bnobel\s+peace", "nobel_peace"),
    (r"\bmachado", "machado"),
    (r"\bindicted", "indicted"),
    (r"\bimpeach", "impeachment"),
    (r"\bpardoned", "pardon"),

    # --- Corporate events / earnings / firings ---
    (r"\bearnings", "earnings"),
    (r"\b(fired|resign|out as|step down)", "executive_exit"),
    (r"\bacquisition", "acquisition"),
    (r"\bbuyout", "buyout"),
    (r"\bipo\b", "ipo"),

    # --- Awards / rankings (jury-determined privately) ---
    (r"\byear\s+in\s+search", "year_in_search"),
    (r"\bmost\s+searched", "most_searched_google"),
    (r"\boscar", "oscar"),
    (r"\bgolden\s+globe", "golden_globe"),
    (r"\bemmy", "emmy"),
    (r"\bgrammy", "grammy"),

    # --- Pre-taped reality TV (production-crew insiders) ---
    (r"\bsurvivor\b", "survivor"),
    (r"\bbachelor(ette)?\b", "bachelor"),
    (r"\bamazing\s+race", "amazing_race"),
    (r"\btop\s+chef", "top_chef"),
    (r"\bmasterchef", "masterchef"),

    # --- FDA / medical ---
    (r"\bfda.*approv", "fda_approval"),
    (r"\bclinical\s+trial", "clinical_trial"),

    # --- Scheduled-info Fed ---
    (r"\bfed.{0,20}(hike|cut|rate)", "fed_rate"),
    (r"\bfomc", "fomc"),
]


def match_known_cases(md: pd.DataFrame) -> pd.DataFrame:
    """Flag rows whose question matches any KNOWN_CASE pattern."""
    out = md.copy()
    out["case_tag"] = None
    for pattern, tag in KNOWN_CASE_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE)
        mask = out["question"].fillna("").apply(lambda q: bool(regex.search(q)))
        # Don't overwrite already-tagged rows
        out.loc[mask & out["case_tag"].isna(), "case_tag"] = tag
    return out[out["case_tag"].notna()]


def fetch_tokens_and_resolution(g: GoldskyClient, condition_id: str) -> dict:
    """Get positionIds + payouts for a condition."""
    q = f'{{ c: condition(id: "{condition_id}") {{ positionIds payoutNumerators payoutDenominator }} }}'
    try:
        res = g.query("pnl", q).data.get("c") or {}
    except Exception:  # noqa: BLE001
        res = {}
    position_ids = [str(p) for p in (res.get("positionIds") or [])]
    payouts = [str(p) for p in (res.get("payoutNumerators") or [])]

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
    q = f'{{ r: redemptions(first: 1, where: {{condition: "{condition_id}"}}, orderBy: timestamp, orderDirection: asc) {{ timestamp }} }}'
    for subgraph in ("activity", "orderbook_resync"):
        try:
            rows = g.query(subgraph, q).data.get("r") or []
            if rows:
                return int(rows[0]["timestamp"])
        except Exception:  # noqa: BLE001
            continue
    return None


def _parse_end_date(end_date_str: str | None) -> int | None:
    if not end_date_str:
        return None
    try:
        from datetime import datetime
        return int(datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).timestamp())
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", type=Path, default=Path("data/offchain/markets_metadata.parquet"))
    ap.add_argument("--catalog", type=Path, default=Path("data/raw/goldsky/markets.parquet"))
    ap.add_argument("--trades-dir", type=Path, default=Path("data/raw/goldsky/trades"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/raw/goldsky/_cache"))
    ap.add_argument("--min-volume", type=float, default=100_000,
                    help="only pull markets with Gamma volume >= this")
    ap.add_argument("--window-days", type=int, default=14)
    ap.add_argument("--max-markets", type=int, default=None,
                    help="cap on new markets to pull in this run")
    ap.add_argument("--inter-market-delay", type=float, default=1.0,
                    help="seconds to wait between markets (soft rate-limiting)")
    ap.add_argument("--strategy", choices=["known_cases", "top_volume", "both"],
                    default="known_cases",
                    help="which markets to include")
    ap.add_argument("--top-volume-n", type=int, default=2000,
                    help="if strategy includes top_volume, pull this many by descending volume")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be pulled but don't fetch")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")

    # Load metadata + existing catalog
    md = pd.read_parquet(args.metadata)
    md["vol_num"] = pd.to_numeric(md["volume"], errors="coerce").fillna(0)
    md = md[md["vol_num"] >= args.min_volume]
    md = md[md["closed"] == True]  # noqa: E712  resolved markets only
    log.info("candidate markets (closed, vol >= $%s): %d",
             f"{args.min_volume:,.0f}", len(md))

    existing_cat = (
        pd.read_parquet(args.catalog) if args.catalog.exists() else pd.DataFrame()
    )
    existing_cids = set(existing_cat["condition_id"].tolist()) if not existing_cat.empty else set()
    log.info("already in catalog: %d markets", len(existing_cids))

    # Build target set
    target_rows: list[dict] = []
    if args.strategy in ("known_cases", "both"):
        known = match_known_cases(md)
        log.info("known-case matches: %d", len(known))
        # Show tag distribution
        tag_counts = known["case_tag"].value_counts()
        for tag, c in tag_counts.items():
            log.info("  %s: %d", tag, c)
        for _, r in known.iterrows():
            target_rows.append({
                "conditionId": r["conditionId"], "question": r["question"],
                "volume": r["vol_num"], "endDate": r["endDate"], "tag": r["case_tag"],
            })
    if args.strategy in ("top_volume", "both"):
        top = md.nlargest(args.top_volume_n, "vol_num")
        for _, r in top.iterrows():
            target_rows.append({
                "conditionId": r["conditionId"], "question": r["question"],
                "volume": r["vol_num"], "endDate": r["endDate"], "tag": "top_volume",
            })

    # Dedupe + filter out those we already have
    seen = set()
    target_rows_unique: list[dict] = []
    for r in target_rows:
        cid = r["conditionId"]
        if cid in seen or cid in existing_cids:
            continue
        seen.add(cid)
        target_rows_unique.append(r)
    # Sort by volume descending so if we cap with --max-markets, we keep the
    # highest-volume (most likely insider-valuable) markets
    target_rows_unique.sort(key=lambda r: -float(r.get("volume") or 0))
    log.info("new markets to fetch: %d (after dedup + catalog overlap, sorted by volume)",
             len(target_rows_unique))

    if args.max_markets is not None:
        target_rows_unique = target_rows_unique[: args.max_markets]
        log.info("capped to %d markets", len(target_rows_unique))

    if args.dry_run:
        log.info("DRY RUN — listing targets and exiting")
        for r in target_rows_unique[:30]:
            log.info(
                "  $%s  tag=%s  q=%r",
                f"{r['volume']:,.0f}", r["tag"], r["question"][:70],
            )
        if len(target_rows_unique) > 30:
            log.info("  ... + %d more", len(target_rows_unique) - 30)
        return

    if not target_rows_unique:
        log.info("nothing new to fetch")
        return

    # Fetch trades for each
    g = GoldskyClient(cache_dir=args.cache_dir)
    args.trades_dir.mkdir(parents=True, exist_ok=True)
    new_records: list[MarketRecord] = []

    # Metadata lookup for volume/end date
    md_by_id = {r["conditionId"]: r for _, r in md.iterrows()}

    RESYNC_CUTOFF_TS = 1_736_121_600  # 2026-01-05
    failures = 0
    for r in tqdm(target_rows_unique, desc="expand catalog"):
        cid = r["conditionId"]
        try:
            info = fetch_tokens_and_resolution(g, cid)
            if not info["token_ids"]:
                log.warning("no tokens for %s (%s)", cid[:16], r["tag"])
                failures += 1
                continue
            rt = info["resolution_timestamp"]
            if rt is None:
                rt = backfill_live_resolution_single(g, cid)
            if rt is None:
                rt = _parse_end_date(r.get("endDate"))

            tier = "historical" if (rt and rt < RESYNC_CUTOFF_TS) else "live"
            rec = MarketRecord(
                condition_id=cid,
                question_id=info.get("question_id"),
                resolution_timestamp=rt,
                payouts=info.get("payouts"),
                outcome_slot_count=info.get("outcome_slot_count"),
                oracle=info.get("oracle"),
                token_ids=info["token_ids"],
                token_to_outcome=info["token_to_outcome"],
                total_volume_usd=float(r["volume"]),
                total_trades=0,
                source_tier=tier,
            )
            out = args.trades_dir / f"{cid}.parquet"
            if out.exists() and out.stat().st_size > 0:
                new_records.append(rec)
                continue
            try:
                trades = fetch_trades(g, rec, window_days=args.window_days)
            except Exception as e:  # noqa: BLE001
                log.warning("fetch_trades failed for %s: %s", cid[:16], e)
                failures += 1
                continue
            if not trades.empty:
                trades.to_parquet(out, index=False)
            else:
                out.touch()
            new_records.append(rec)
        finally:
            time.sleep(args.inter_market_delay)

    # Append to catalog
    if new_records:
        from dataclasses import asdict
        new_df = pd.DataFrame([asdict(x) for x in new_records])
        if not existing_cat.empty:
            merged = pd.concat([existing_cat, new_df], ignore_index=True)
        else:
            merged = new_df
        merged.to_parquet(args.catalog, index=False)
        log.info("catalog grew from %d → %d (+%d new, %d failures)",
                 len(existing_cids), len(merged), len(new_records), failures)


if __name__ == "__main__":
    main()
