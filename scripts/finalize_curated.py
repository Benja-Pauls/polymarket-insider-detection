"""Post-process curated candidates: apply merges, recompute ranking.

The Opus verdict is the primary signal, but we refine with:

  1. **(wallet, condition_id) merge** — when both a callout row and an on-chain
     row cover the same wallet+market, merge them. Confidence tier of the
     merged row = max of the inputs. Evidence combined.

  2. **Profit-based re-rank within each market** — my on-chain miner's
     `flag_win_aligned_top` is noisy because it conflates:
       (a) wallet bought winning outcome at LOW entry price (true insider signal)
       (b) wallet bought winning outcome at NEAR-CERTAINTY price (late arb)
     Opus sometimes labeled (b) as "confirmed" and (a) as "suspected/ambiguous"
     because (b) has cleaner all-6-flags signatures. We re-score within each
     market by estimated profit = size_usd × (1 - entry_vwap) for YES-held
     and size_usd × (entry_vwap) for NO-held-lost-inverse. Top profit earners
     in a suspicious market get a score boost.

Output: data/labels/curated_final.csv with one row per (canonical_wallet,
canonical_market) and a rank-order column.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_TIER_ORDER = {"T1": 0, "T2": 1, "T3": 2}
_VERDICT_ORDER = {"confirmed": 0, "suspected": 1, "ambiguous": 2, "rejected": 3}


def _better(a: str | None, b: str | None, order: dict) -> str:
    a_rank = order.get(a, 999)
    b_rank = order.get(b, 999)
    return a if a_rank <= b_rank else b


def merge_wallet_market_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows that share (wallet, condition_id)."""
    key_cols = ["wallet", "condition_id"]
    # Only merge rows where BOTH wallet and condition_id are present
    has_both = df["wallet"].notna() & df["condition_id"].notna()
    mergable = df[has_both].copy()
    standalone = df[~has_both].copy()
    log.info("rows with wallet+market: %d  standalone: %d", len(mergable), len(standalone))

    if mergable.empty:
        return df

    grouped = mergable.groupby(key_cols, dropna=False)
    merged_rows = []
    merges_applied = 0
    for (wallet, cond), g in grouped:
        if len(g) == 1:
            merged_rows.append(g.iloc[0].to_dict())
            continue
        merges_applied += 1
        # Pick best verdict + tier
        verdict = min(g["final_verdict"].dropna().tolist(),
                      key=lambda v: _VERDICT_ORDER.get(v, 999), default=None)
        tier = min(g["confidence_tier_final"].dropna().tolist(),
                   key=lambda t: _TIER_ORDER.get(t, 999), default=None)
        # Combine sources + reasoning
        srcs = sorted(set(g["source"].dropna().tolist()))
        reasonings = g["final_reasoning"].fillna("").tolist()
        combined_reason = " || ".join(r[:220] for r in reasonings if r)
        # Take the largest size as the canonical
        sz = g["size_usd_approx"].max()
        # Callout evidence wins if available
        cr = g[g["source"] == "callout"]
        evidence = cr["evidence_urls"].iloc[0] if not cr.empty else g["evidence_urls"].iloc[0]
        # Build merged row
        row = g.iloc[0].to_dict()
        row["final_verdict"] = verdict
        row["confidence_tier_final"] = tier
        row["final_reasoning"] = combined_reason
        row["source"] = ";".join(srcs)
        row["n_supporting_sources"] = int(g["n_supporting_sources"].fillna(0).sum())
        row["merged_candidate_ids"] = json.dumps(g["candidate_id"].tolist())
        row["size_usd_approx"] = float(sz) if pd.notna(sz) else None
        row["evidence_urls"] = evidence
        merged_rows.append(row)

    log.info("merges applied: %d", merges_applied)
    merged_df = pd.DataFrame(merged_rows)
    return pd.concat([merged_df, standalone], ignore_index=True)


def add_profit_score(df: pd.DataFrame, trades_dir: Path, catalog_path: Path) -> pd.DataFrame:
    """Compute estimated profit per (wallet, market) and add as a ranking column.

    profit_estimate = wallet's net position × (resolution_price - entry_vwap)
    For YES-held-to-res in a YES-winning market: size_outcome × (1 - entry_price)
    We approximate using total USD and VWAP.
    """
    catalog = pd.read_parquet(catalog_path)
    out_df = df.copy()
    profits = []
    for _, r in out_df.iterrows():
        wallet = r.get("wallet")
        cid = r.get("condition_id")
        if not wallet or not cid:
            profits.append(None)
            continue
        tp = trades_dir / f"{cid}.parquet"
        if not tp.exists() or tp.stat().st_size < 100:
            profits.append(None)
            continue
        try:
            trades = pd.read_parquet(tp)
        except Exception:  # noqa: BLE001
            profits.append(None)
            continue
        w = str(wallet).lower()
        wt = trades[
            (trades["maker"].fillna("").str.lower() == w) |
            (trades["taker"].fillna("").str.lower() == w)
        ]
        if wt.empty:
            profits.append(None)
            continue
        # Determine winner
        m = catalog[catalog["condition_id"] == cid]
        if m.empty:
            profits.append(None)
            continue
        payouts = m["payouts"].iloc[0]
        if payouts is None or len(payouts) < 2:
            profits.append(None)
            continue
        winner_idx = [int(p) for p in payouts].index(max(int(p) for p in payouts))

        # Work out per-token outcome mapping if available
        t2o = {}
        try:
            t2o = dict(m["token_to_outcome"].iloc[0])
        except (TypeError, ValueError):
            t2o = {}

        # Compute estimated profit.
        # For each trade, if the wallet bought a token whose outcome won,
        # they profit (1 - price) × size. If bought a loser, they lose price × size.
        profit = 0.0
        for _, t in wt.iterrows():
            tok = t.get("token_id")
            oi = t2o.get(tok)
            if oi is None:
                continue
            side = t.get("side")
            size_usdc = float(t.get("size_outcome_usdc") or 0)
            price = float(t.get("price") or 0)
            if side == "BUY":
                # Bought this token at `price`. If its outcome won, token pays 1.
                if oi == winner_idx:
                    profit += size_usdc * (1 - price)
                else:
                    profit -= size_usdc * price
            elif side == "SELL":
                # Sold this token at `price`. Opposite accounting.
                if oi == winner_idx:
                    profit -= size_usdc * (1 - price)
                else:
                    profit += size_usdc * price
        profits.append(float(profit))

    out_df["estimated_profit_usd"] = profits
    return out_df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curated", type=Path, default=Path("data/labels/curated_candidates.parquet"))
    ap.add_argument("--trades-dir", type=Path, default=Path("data/raw/goldsky/trades"))
    ap.add_argument("--catalog", type=Path, default=Path("data/raw/goldsky/markets.parquet"))
    ap.add_argument("--out-parquet", type=Path, default=Path("data/labels/curated_final.parquet"))
    ap.add_argument("--out-csv", type=Path, default=Path("data/labels/curated_final.csv"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")

    df = pd.read_parquet(args.curated)
    log.info("input rows: %d", len(df))

    merged = merge_wallet_market_duplicates(df)
    log.info("after (wallet, market) merge: %d rows", len(merged))

    with_profit = add_profit_score(merged, args.trades_dir, args.catalog)
    log.info("rows with profit estimate: %d", with_profit["estimated_profit_usd"].notna().sum())

    # Compose a refined ranking score:
    #  - Opus verdict tier as primary
    #  - Estimated profit as secondary (big profits in suspicious markets surface)
    def _score(r):
        verdict = r.get("final_verdict")
        tier = r.get("confidence_tier_final")
        base = {"confirmed": 1000, "suspected": 500, "ambiguous": 200, "rejected": 0}.get(verdict, 0)
        tier_bonus = {"T1": 100, "T2": 50, "T3": 20}.get(tier, 0) if verdict != "rejected" else 0
        profit = r.get("estimated_profit_usd") or 0
        profit_bonus = min(500, max(0, profit / 1000))  # cap at $500K → 500 points
        return base + tier_bonus + profit_bonus

    with_profit["final_score"] = with_profit.apply(_score, axis=1)
    with_profit = with_profit.sort_values("final_score", ascending=False).reset_index(drop=True)
    with_profit["final_rank"] = with_profit.index + 1

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    with_profit.to_parquet(args.out_parquet, index=False)

    # Human review CSV with most-relevant columns
    csv_cols = [
        "final_rank", "final_score", "final_verdict", "confidence_tier_final",
        "source", "wallet", "condition_id", "market_question",
        "size_usd_approx", "estimated_profit_usd", "direction", "outcome_resolved",
        "model_used", "final_reasoning",
        "onchain_flags_count", "onchain_wallet_concentration",
        "onchain_size_percentile", "suspicion_score",
        "evidence_urls",
    ]
    csv_cols = [c for c in csv_cols if c in with_profit.columns]
    with_profit[csv_cols].to_csv(args.out_csv, index=False)
    log.info("wrote %s and %s", args.out_parquet, args.out_csv)

    # Summary
    by_verdict = with_profit["final_verdict"].value_counts().to_dict()
    log.info("final verdict distribution: %s", by_verdict)
    non_rej = with_profit[with_profit["final_verdict"] != "rejected"]
    log.info("non-rejected rows: %d", len(non_rej))
    if not non_rej.empty and "estimated_profit_usd" in non_rej.columns:
        total_profit = non_rej["estimated_profit_usd"].fillna(0).sum()
        log.info("total estimated profit across non-rejected: $%.0f", total_profit)


if __name__ == "__main__":
    main()
