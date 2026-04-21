"""Refill the trades directory for markets whose catalog token list grew.

The initial catalog only captured the higher-volume outcome token for each
binary market (the one whose Orderbook entity exceeded the volume floor). We
later enrich the catalog with the *second* outcome token via
``pnl.Condition.positionIds``. This script detects markets where the token
list grew, deletes the stale trade file, and re-fetches with the full token
list so that per-market directional/price features aggregate across BOTH
outcomes.

Does NOT touch markets whose token list is unchanged — those files are still
correct.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pminsider.collect import fetch_trades, _record_from_row
from pminsider.goldsky import GoldskyClient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-catalog", type=Path, required=True,
                    help="pre-enrich catalog (for comparing token lists)")
    ap.add_argument("--new-catalog", type=Path, required=True,
                    help="enriched catalog (with token_to_outcome + extra tokens)")
    ap.add_argument("--trades-dir", type=Path, default=Path("data/raw/goldsky/trades"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/raw/goldsky/_cache"))
    ap.add_argument("--window-days", type=int, default=14)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("refill")

    old = pd.read_parquet(args.old_catalog).set_index("condition_id")
    new = pd.read_parquet(args.new_catalog).set_index("condition_id")
    g = GoldskyClient(cache_dir=args.cache_dir)

    def _set(v):
        if v is None:
            return set()
        try:
            return set(v)
        except TypeError:
            return set()

    # Find markets whose token list grew
    to_refill: list[str] = []
    for cid in new.index:
        new_tokens = _set(new.loc[cid, "token_ids"])
        old_tokens = _set(old.loc[cid, "token_ids"]) if cid in old.index else set()
        if new_tokens - old_tokens:
            to_refill.append(cid)

    log.info("markets needing refill: %d / %d", len(to_refill), len(new))

    args.trades_dir.mkdir(parents=True, exist_ok=True)
    ok = failed = 0
    for cid in tqdm(to_refill, desc="refill trades"):
        row = new.loc[cid].to_dict()
        row["condition_id"] = cid
        rec = _record_from_row(row)
        out_path = args.trades_dir / f"{cid}.parquet"
        if out_path.exists():
            out_path.unlink()
        try:
            trades = fetch_trades(g, rec, window_days=args.window_days)
        except Exception as e:  # noqa: BLE001
            log.warning("  failed %s: %s", cid, e)
            failed += 1
            continue
        if not trades.empty:
            trades.to_parquet(out_path, index=False)
            ok += 1
        else:
            out_path.touch()
            ok += 1
    log.info("refill complete: ok=%d failed=%d", ok, failed)


if __name__ == "__main__":
    main()
