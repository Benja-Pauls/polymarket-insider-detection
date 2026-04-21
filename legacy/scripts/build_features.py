"""Build the features parquet from the collected trade files and market catalog.

Usage:
    python scripts/build_features.py \
        --catalog data/raw/goldsky/markets.parquet \
        --trades-dir data/raw/goldsky/trades \
        --out data/processed/features.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pminsider.features import compute_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=Path("data/raw/goldsky/markets.parquet"))
    ap.add_argument("--trades-dir", type=Path, default=Path("data/raw/goldsky/trades"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/features.parquet"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("build_features")

    catalog = pd.read_parquet(args.catalog)
    log.info("catalog has %d markets", len(catalog))

    rows: list[dict] = []
    missing = 0
    for _, m in tqdm(catalog.iterrows(), total=len(catalog), desc="compute features"):
        cid = m["condition_id"]
        tp = args.trades_dir / f"{cid}.parquet"
        if not tp.exists() or tp.stat().st_size == 0:
            missing += 1
            trades = pd.DataFrame()
        else:
            try:
                trades = pd.read_parquet(tp)
            except Exception:  # noqa: BLE001
                missing += 1
                trades = pd.DataFrame()

        token_map = None
        if "token_to_outcome" in m and m["token_to_outcome"] is not None:
            # Handle pandas-level dict/numpy types
            try:
                token_map = dict(m["token_to_outcome"])
            except (TypeError, ValueError):
                token_map = None
        feats = compute_features(
            trades,
            resolution_timestamp=int(m["resolution_timestamp"]),
            payouts=list(m["payouts"]) if m["payouts"] is not None else None,
            total_volume_usd=float(m["total_volume_usd"]),
            total_trades=int(m["total_trades"]),
            outcome_slot_count=int(m["outcome_slot_count"]) if m["outcome_slot_count"] else 2,
            token_to_outcome=token_map,
        )
        feats["condition_id"] = cid
        feats["source_tier"] = m["source_tier"]
        rows.append(feats)

    log.info("missing trades for %d / %d markets", missing, len(catalog))
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("wrote %d rows → %s (%d cols)", len(df), args.out, len(df.columns))


if __name__ == "__main__":
    main()
