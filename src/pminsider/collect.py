"""Data collection orchestration.

Collection strategy (pivoted 2026-04-21 after discovering FPMMs are empty and
``orderBy: scaledCollateralVolume`` triggers DB timeouts on The Graph):

  Phase A: Market catalog build
    1. Enumerate all Orderbook rows on the live subgraph, keyset paginating by id ASC.
    2. Filter in-memory to volume ≥ MIN_VOLUME.
    3. Group Orderbook (outcome-token) rows by their parent Condition via
       ``marketData``. Each binary market has exactly two outcome tokens.
    4. Look up resolution metadata for each condition:
         - orderbook_resync: questionId + resolutionTimestamp + payouts (pre-2026-01-05)
         - pnl: payoutNumerators + payoutDenominator (post-2026-01-05 fallback)
       Retain only resolved conditions.
    5. Rank by total market volume, keep top ``TOP_N``.
    6. Save as ``markets.parquet``.

  Phase B: Per-market trade collection
    For each resolved market, fetch trades in [resolution_ts - WINDOW_DAYS, resolution_ts]:
      - Historical markets → ``orderbook_resync.enrichedOrderFilleds`` (has Account entity)
      - Live markets → ``orderbook.orderFilledEvents`` (no Account — we join after)
    Filter ``market_in: [outcome_token_ids]`` so we only get trades for this market.
    Save as ``trades/{condition_id}.parquet``.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from .goldsky import GoldskyClient

log = logging.getLogger(__name__)

# Config
TRADE_WINDOW_DAYS = 14
MIN_VOLUME_USD = 10_000.0
TOP_N = 3_000
USDC_DECIMALS = 6
USDC_SCALE = 10 ** USDC_DECIMALS

# Resync subgraph's snapshot boundary — conditions resolved after this date
# won't be in the resync subgraph and need fallback metadata from pnl.
RESYNC_CUTOFF_TS = 1_736_121_600  # 2026-01-05 00:00 UTC


@dataclass(frozen=True)
class MarketRecord:
    condition_id: str
    question_id: str | None
    resolution_timestamp: int | None
    payouts: list[str] | None
    outcome_slot_count: int | None
    oracle: str | None
    token_ids: list[str]
    # Map token_id → outcome_index (0 for "Yes", 1 for "No" in a binary market).
    # Populated from MarketData.outcomeIndex. May have None values for tokens
    # whose outcome we couldn't resolve.
    token_to_outcome: dict | None
    total_volume_usd: float
    total_trades: int
    source_tier: str  # "historical" or "live"


# ----------------------------------------------------------------------
# Phase A helpers
# ----------------------------------------------------------------------

def iter_orderbooks(
    g: GoldskyClient,
    *,
    min_volume_usd: float = MIN_VOLUME_USD,
    max_rows: int | None = None,
) -> Iterable[dict]:
    """Stream all Orderbook rows, filtering volume in-memory (DB can't sort by it)."""
    seen = 0
    for ob in g.paginate(
        "orderbook",
        "orderbooks",
        "id tradesQuantity scaledCollateralVolume buysQuantity sellsQuantity",
        order_by="id",
        order_direction="asc",
        page_size=500,  # 1000 consistently triggers statement timeout on deep pages
    ):
        try:
            vol = float(ob.get("scaledCollateralVolume") or 0)
        except (TypeError, ValueError):
            continue
        if vol < min_volume_usd:
            continue
        yield ob
        seen += 1
        if max_rows is not None and seen >= max_rows:
            return


def lookup_conditions_for_tokens(
    g: GoldskyClient,
    token_ids: list[str],
    *,
    batch_size: int = 100,
) -> dict[str, dict]:
    """Return ``{token_id: {condition_id, outcome_index}}`` via ``marketDatas`` batch.

    Falls back to the orderbook_resync subgraph if outcomeIndex is null in the
    live one (which is common — live subgraph often has outcomeIndex=None).
    """
    out: dict[str, dict] = {}
    missing_outcome: list[str] = []
    for i in range(0, len(token_ids), batch_size):
        chunk = token_ids[i : i + batch_size]
        ids_gql = "[" + ", ".join(f'"{t}"' for t in chunk) + "]"
        q = f"""
        {{ marketDatas(first: {len(chunk)}, where: {{id_in: {ids_gql}}}) {{
            id condition outcomeIndex
          }} }}
        """
        res = g.query("orderbook", q)
        for m in res.data.get("marketDatas", []) or []:
            oi = m.get("outcomeIndex")
            if oi is None:
                missing_outcome.append(m["id"])
            out[m["id"]] = {
                "condition_id": m.get("condition"),
                "outcome_index": int(oi) if oi is not None else None,
            }

    # Fall back to resync for tokens missing outcomeIndex
    for i in range(0, len(missing_outcome), batch_size):
        chunk = missing_outcome[i : i + batch_size]
        ids_gql = "[" + ", ".join(f'"{t}"' for t in chunk) + "]"
        q = f"""
        {{ marketDatas(first: {len(chunk)}, where: {{id_in: {ids_gql}}}) {{
            id outcomeIndex
          }} }}
        """
        try:
            res = g.query("orderbook_resync", q)
        except Exception:  # noqa: BLE001
            continue
        for m in res.data.get("marketDatas", []) or []:
            oi = m.get("outcomeIndex")
            if oi is not None and m["id"] in out:
                out[m["id"]]["outcome_index"] = int(oi)
    return out


def lookup_resolution_historical(
    g: GoldskyClient, condition_ids: list[str], *, batch_size: int = 50
) -> dict[str, dict]:
    """Get resolution info from orderbook_resync for pre-Jan-2026 markets."""
    out: dict[str, dict] = {}
    for i in range(0, len(condition_ids), batch_size):
        chunk = condition_ids[i : i + batch_size]
        ids_gql = "[" + ", ".join(f'"{c}"' for c in chunk) + "]"
        q = f"""
        {{ conditions(first: {len(chunk)}, where: {{id_in: {ids_gql}}}) {{
            id questionId outcomeSlotCount oracle resolutionTimestamp payouts
          }} }}
        """
        try:
            res = g.query("orderbook_resync", q)
        except Exception:  # noqa: BLE001
            continue
        for c in res.data.get("conditions", []) or []:
            if c.get("resolutionTimestamp"):
                out[c["id"]] = {
                    "question_id": c.get("questionId"),
                    "resolution_timestamp": int(c["resolutionTimestamp"]),
                    "payouts": c.get("payouts"),
                    "outcome_slot_count": c.get("outcomeSlotCount"),
                    "oracle": c.get("oracle"),
                    "source_tier": "historical",
                }
    return out


def lookup_resolution_live(
    g: GoldskyClient, condition_ids: list[str], *, batch_size: int = 50
) -> dict[str, dict]:
    """Fall back to pnl subgraph for markets resolved after orderbook_resync froze."""
    out: dict[str, dict] = {}
    for i in range(0, len(condition_ids), batch_size):
        chunk = condition_ids[i : i + batch_size]
        ids_gql = "[" + ", ".join(f'"{c}"' for c in chunk) + "]"
        q = f"""
        {{ conditions(first: {len(chunk)}, where: {{id_in: {ids_gql}}}) {{
            id positionIds payoutNumerators payoutDenominator
          }} }}
        """
        try:
            res = g.query("pnl", q)
        except Exception:  # noqa: BLE001
            continue
        for c in res.data.get("conditions", []) or []:
            pn = c.get("payoutNumerators") or []
            if pn and any(int(p) > 0 for p in pn):
                out[c["id"]] = {
                    "question_id": None,
                    "resolution_timestamp": None,  # backfilled via Redemption
                    "payouts": pn,
                    "outcome_slot_count": len(c.get("positionIds") or []),
                    "oracle": None,
                    "source_tier": "live",
                }
    return out


def backfill_live_resolution_ts(
    g: GoldskyClient, condition_ids: list[str]
) -> dict[str, int]:
    """Use earliest Redemption timestamp as resolution-ts proxy for live markets."""
    out: dict[str, int] = {}
    for cond_id in tqdm(condition_ids, desc="backfill live resolution ts", disable=len(condition_ids) < 20):
        q = f'{{ r: redemptions(first: 1, where: {{condition: "{cond_id}"}}, orderBy: timestamp, orderDirection: asc) {{ timestamp }} }}'
        try:
            rows = g.query("orderbook_resync", q).data.get("r") or []
        except Exception:  # noqa: BLE001
            rows = []
        if not rows:
            # Try activity subgraph
            try:
                rows = g.query("activity", q).data.get("r") or []
            except Exception:  # noqa: BLE001
                rows = []
        if rows:
            out[cond_id] = int(rows[0]["timestamp"])
    return out


# ----------------------------------------------------------------------
# Market catalog build
# ----------------------------------------------------------------------

def build_market_catalog(
    g: GoldskyClient,
    *,
    min_volume_usd: float = MIN_VOLUME_USD,
    top_n: int = TOP_N,
    max_orderbooks: int | None = None,
) -> list[MarketRecord]:
    """End-to-end catalog build: enumerate orderbooks → group by condition → fetch resolution."""
    log.info("phase A: enumerating Orderbook entities (vol >= $%s)…", f"{min_volume_usd:,.0f}")
    orderbooks = list(tqdm(iter_orderbooks(g, min_volume_usd=min_volume_usd, max_rows=max_orderbooks),
                           desc="enumerate orderbooks"))
    log.info("  %d orderbooks with volume >= $%s", len(orderbooks), f"{min_volume_usd:,.0f}")

    if not orderbooks:
        return []

    log.info("phase B: mapping token ids → conditions via marketData…")
    token_ids = [ob["id"] for ob in orderbooks]
    token_to_cond = lookup_conditions_for_tokens(g, token_ids)
    log.info("  resolved %d / %d tokens to conditions", len(token_to_cond), len(token_ids))

    # Group orderbooks by condition
    by_cond: dict[str, dict] = defaultdict(lambda: {"tokens": [], "vol": 0.0, "trades": 0})
    for ob in orderbooks:
        tok = ob["id"]
        mapping = token_to_cond.get(tok)
        if not mapping:
            continue
        cid = mapping["condition_id"]
        if not cid:
            continue
        rec = by_cond[cid]
        rec["tokens"].append(tok)
        rec["vol"] += float(ob.get("scaledCollateralVolume") or 0)
        rec["trades"] += int(ob.get("tradesQuantity") or 0)
    log.info("  %d distinct conditions", len(by_cond))

    # Resolution lookups
    log.info("phase C: resolution metadata (historical subgraph first)…")
    all_cond_ids = list(by_cond.keys())
    hist_res = lookup_resolution_historical(g, all_cond_ids)
    log.info("  %d historical resolutions", len(hist_res))

    still_missing = [c for c in all_cond_ids if c not in hist_res]
    live_res = lookup_resolution_live(g, still_missing) if still_missing else {}
    log.info("  %d live resolutions (post-resync)", len(live_res))

    # Backfill resolution timestamps for live tier
    live_need_ts = [c for c, v in live_res.items() if v["resolution_timestamp"] is None]
    live_ts_map = backfill_live_resolution_ts(g, live_need_ts) if live_need_ts else {}
    for c, ts in live_ts_map.items():
        live_res[c]["resolution_timestamp"] = ts

    # Combine
    resolved: list[MarketRecord] = []
    for cid, agg in by_cond.items():
        meta = hist_res.get(cid) or live_res.get(cid)
        if not meta:
            continue  # unresolved
        # Build token → outcome_index map from the tokens we collected
        t2o: dict[str, int | None] = {}
        for tok in agg["tokens"]:
            mapping = token_to_cond.get(tok) or {}
            t2o[tok] = mapping.get("outcome_index")

        rec = MarketRecord(
            condition_id=cid,
            question_id=meta.get("question_id"),
            resolution_timestamp=meta.get("resolution_timestamp"),
            payouts=meta.get("payouts"),
            outcome_slot_count=meta.get("outcome_slot_count"),
            oracle=meta.get("oracle"),
            token_ids=sorted(agg["tokens"]),
            token_to_outcome=t2o,
            total_volume_usd=agg["vol"],
            total_trades=agg["trades"],
            source_tier=meta.get("source_tier", "unknown"),
        )
        resolved.append(rec)
    log.info("  %d resolved markets in catalog", len(resolved))

    resolved.sort(key=lambda r: r.total_volume_usd, reverse=True)
    return resolved[:top_n]


def save_catalog(records: list[MarketRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in records])
    df.to_parquet(out_path, index=False)
    log.info("wrote %d markets → %s", len(df), out_path)


# ----------------------------------------------------------------------
# Phase B: per-market trades
# ----------------------------------------------------------------------

# Field sets per subgraph — maker/taker differ (String on live, Account entity on resync)
_TRADE_FIELDS_LIVE = """id transactionHash timestamp orderHash
    maker taker makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled fee"""

_TRADE_FIELDS_RESYNC = """id transactionHash timestamp orderHash
    maker { id } taker { id } makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled fee"""


def fetch_trades(
    g: GoldskyClient,
    market: MarketRecord,
    *,
    window_days: int = TRADE_WINDOW_DAYS,
) -> pd.DataFrame:
    """Pull trades in [resolution_ts - window, resolution_ts] for a single market.

    Uses ``orderFilledEvents`` on both tiers (resync for historical, orderbook for
    live). The subgraph indexes ``makerAssetId`` / ``takerAssetId`` but NOT
    ``market`` on the enriched entity — so we run two paginated queries (one for
    each side of the book) and union by trade id.
    """
    if market.resolution_timestamp is None or not market.token_ids:
        return pd.DataFrame()
    cutoff = market.resolution_timestamp - window_days * 86400
    subgraph = "orderbook_resync" if market.source_tier == "historical" else "orderbook"
    fields = _TRADE_FIELDS_RESYNC if subgraph == "orderbook_resync" else _TRADE_FIELDS_LIVE

    rows_by_id: dict[str, dict] = {}

    for side_filter in ("makerAssetId_in", "takerAssetId_in"):
        for tr in g.paginate(
            subgraph,
            "orderFilledEvents",
            fields,
            where={
                "timestamp_gte": str(cutoff),
                "timestamp_lte": str(market.resolution_timestamp),
                side_filter: market.token_ids,
            },
            order_by="timestamp",
            order_direction="asc",
            page_size=500,
        ):
            row = _normalize_trade(tr, market)
            rows_by_id[row["trade_id"]] = row  # dedupe

    if not rows_by_id:
        return pd.DataFrame()
    return pd.DataFrame(rows_by_id.values()).sort_values("timestamp").reset_index(drop=True)


def _normalize_trade(tr: dict, market: MarketRecord) -> dict:
    """Convert a raw orderFilledEvent into a normalized trade row.

    The Polymarket CLOB trade convention: for each trade one side is USDC (the
    collateral token) and the other side is an outcome token (positionId).

      * ``makerAssetId == "0"`` (collateral) and ``takerAssetId ∈ token_ids``
        → maker provided USDC, taker got outcome tokens → **buy** of that token
      * ``makerAssetId ∈ token_ids`` and ``takerAssetId == "0"``
        → maker provided outcome tokens, taker got USDC → **sell** of that token

    USDC has 6 decimals on Polygon. Outcome tokens also have 6 decimals
    (conditional-tokens framework mirrors the collateral).
    """
    maker_asset = tr["makerAssetId"]
    taker_asset = tr["takerAssetId"]
    maker_amt = int(tr["makerAmountFilled"])
    taker_amt = int(tr["takerAmountFilled"])

    if maker_asset == "0" or maker_asset not in market.token_ids:
        # Maker gave USDC, taker received outcome tokens → BUY
        side = "BUY"
        token_id = taker_asset
        size_outcome = taker_amt
        usd_spent = maker_amt
    else:
        # Maker gave outcome tokens, taker received USDC → SELL
        side = "SELL"
        token_id = maker_asset
        size_outcome = maker_amt
        usd_spent = taker_amt

    price = (usd_spent / size_outcome) if size_outcome > 0 else None

    return {
        "trade_id": tr["id"],
        "tx_hash": tr["transactionHash"],
        "timestamp": int(tr["timestamp"]),
        "side": side,
        "token_id": token_id,
        "size_outcome_raw": size_outcome,
        "usd_spent_raw": usd_spent,
        "size_outcome_usdc": size_outcome / USDC_SCALE,
        "usd_spent_usdc": usd_spent / USDC_SCALE,
        "price": price,
        "maker": tr["maker"] if isinstance(tr["maker"], str) else (tr["maker"] or {}).get("id"),
        "taker": tr["taker"] if isinstance(tr["taker"], str) else (tr["taker"] or {}).get("id"),
        "fee_raw": int(tr["fee"]),
        "fee_usdc": int(tr["fee"]) / USDC_SCALE,
        "condition_id": market.condition_id,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Collect Polymarket data for insider detection.")
    ap.add_argument("--phase", choices=["catalog", "trades", "all"], default="all")
    ap.add_argument("--min-volume", type=float, default=MIN_VOLUME_USD)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--max-orderbooks", type=int, default=None, help="limit for smoke-testing")
    ap.add_argument("--window-days", type=int, default=TRADE_WINDOW_DAYS)
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw/goldsky"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    g = GoldskyClient(cache_dir=args.out_dir / "_cache")

    catalog_path = args.out_dir / "markets.parquet"
    if args.phase in ("catalog", "all"):
        records = build_market_catalog(
            g,
            min_volume_usd=args.min_volume,
            top_n=args.top_n,
            max_orderbooks=args.max_orderbooks,
        )
        save_catalog(records, catalog_path)

    if args.phase in ("trades", "all"):
        if not catalog_path.exists():
            raise SystemExit(f"catalog not found at {catalog_path}; run --phase catalog first")
        df = pd.read_parquet(catalog_path)
        log.info("phase B: fetching trades for %d markets (window=%d days)…", len(df), args.window_days)
        trades_dir = args.out_dir / "trades"
        trades_dir.mkdir(parents=True, exist_ok=True)
        ok = failed = skipped = 0
        for row in tqdm(df.to_dict("records"), desc="per-market trades"):
            rec = _record_from_row(row)
            out = trades_dir / f"{rec.condition_id}.parquet"
            if out.exists():
                skipped += 1
                continue
            try:
                trades = fetch_trades(g, rec, window_days=args.window_days)
            except Exception as e:  # noqa: BLE001
                log.warning("  fetch failed for %s: %s", rec.condition_id, e)
                failed += 1
                continue
            if not trades.empty:
                trades.to_parquet(out, index=False)
                ok += 1
            else:
                # Still write an empty marker so we don't re-try
                out.touch()
                ok += 1
        log.info("phase B complete: ok=%d skipped=%d failed=%d", ok, skipped, failed)


def _record_from_row(row: dict) -> MarketRecord:
    """Normalize a pandas-produced row dict into a MarketRecord."""
    def _coerce_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return list(v)
        return list(v)

    def _coerce_dict(v):
        if v is None:
            return {}
        if isinstance(v, dict):
            return dict(v)
        # pandas sometimes stores dicts as numpy object arrays
        try:
            return dict(v)
        except (TypeError, ValueError):
            return {}

    rt = row.get("resolution_timestamp")
    return MarketRecord(
        condition_id=str(row["condition_id"]),
        question_id=(
            str(row["question_id"]) if row.get("question_id") not in (None, "")
            else None
        ),
        resolution_timestamp=int(rt) if rt is not None and not pd.isna(rt) else None,
        payouts=(
            list(row["payouts"]) if row.get("payouts") is not None
            and not isinstance(row["payouts"], float) else None
        ),
        outcome_slot_count=(
            int(row["outcome_slot_count"])
            if row.get("outcome_slot_count") is not None
            and not pd.isna(row["outcome_slot_count"])
            else None
        ),
        oracle=(
            str(row["oracle"]) if row.get("oracle") not in (None, "")
            else None
        ),
        token_ids=_coerce_list(row.get("token_ids")),
        token_to_outcome=_coerce_dict(row.get("token_to_outcome")),
        total_volume_usd=float(row.get("total_volume_usd") or 0),
        total_trades=int(row.get("total_trades") or 0),
        source_tier=str(row.get("source_tier") or "unknown"),
    )


if __name__ == "__main__":
    main()
