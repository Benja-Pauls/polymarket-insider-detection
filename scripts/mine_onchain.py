"""Run the on-chain miner on our 264-market catalog.

Usage:
    python scripts/mine_onchain.py \\
        --catalog data/raw/goldsky/markets.parquet \\
        --trades-dir data/raw/goldsky/trades \\
        --out data/labels/onchain_candidates.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pminsider.labels.onchain_miner import mine_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=Path("data/raw/goldsky/markets.parquet"))
    ap.add_argument("--trades-dir", type=Path, default=Path("data/raw/goldsky/trades"))
    ap.add_argument("--out", type=Path, default=Path("data/labels/onchain_candidates.parquet"))
    ap.add_argument("--min-market-volume", type=float, default=50_000)
    ap.add_argument("--top-review", type=int, default=200,
                    help="also write a review CSV of the top-N candidates by score")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")

    df = mine_all(
        catalog_path=args.catalog,
        trades_dir=args.trades_dir,
        out_path=args.out,
        min_market_volume=args.min_market_volume,
    )

    # Top-N review CSV
    if not df.empty:
        review_cols = [
            "wallet", "condition_id", "suspicion_score", "flags_count",
            "wallet_usd_in_market", "wallet_total_usd_across_all_markets",
            "wallet_market_count", "wallet_concentration_in_this_market",
            "wallet_size_percentile_in_market",
            "wallet_flow_direction",
            "winner_outcome_index", "realized_profit_usd", "profit_ratio",
            "entry_vwap_on_winner",
            "flag_fresh_wallet", "flag_single_market", "flag_large_first_position",
            "flag_realized_profit", "flag_informed_entry", "flag_early_timing",
            "resolution_timestamp", "market_total_volume_usd",
        ]
        review = df.head(args.top_review)[review_cols]
        review_path = args.out.parent / "onchain_candidates_top.csv"
        review.to_csv(review_path, index=False)
        print(f"wrote top-{len(review)} review CSV → {review_path}")


if __name__ == "__main__":
    main()
