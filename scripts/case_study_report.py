"""Generate a human-readable ranking of the top insider-candidate markets.

For each market, we compute a composite "insider score":

    score = log(1 + vol_24h_spike_ratio)
          * (1 + 2 * abs(price_24h_move_zscore))
          * (1 + 3 * max(0, winner_24h_hit))
          * (1 + wallet_24h_top1_share)

This blends volume magnitude, price acceleration, outcome-alignment, and
wallet concentration. The weights are purely heuristic and intended to surface
manually-inspectable candidates for the paper's case studies; they do NOT
constitute the model's prediction.

The output markdown includes each candidate's condition id, resolution
timestamp, payouts, volume, key features, and a link we'd eventually resolve
to the Polymarket question text (through the GH Actions metadata proxy).
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def compute_insider_score(df: pd.DataFrame) -> pd.Series:
    spike = np.log1p(df["vol_24h_spike_ratio"].fillna(0).clip(lower=0))
    z = df["price_24h_move_zscore"].abs().fillna(0)
    top1 = df["wallet_24h_top1_share"].fillna(0)
    winner_hit = df.get("winner_24h_hit", pd.Series(0, index=df.index)).fillna(0)
    score = spike * (1 + 2 * z) * (1 + 3 * winner_hit) * (1 + top1)
    return score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    ap.add_argument("--catalog",  type=Path, default=Path("data/raw/goldsky/markets.parquet"))
    ap.add_argument("--metadata", type=Path, default=Path("data/offchain/markets_metadata.parquet"),
                    help="optional off-chain question-text snapshot from GH Actions")
    ap.add_argument("--out",      type=Path, default=Path("results/case_study_candidates.md"))
    ap.add_argument("--top-k",    type=int,  default=25)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("case_studies")

    feats = pd.read_parquet(args.features)
    catalog = pd.read_parquet(args.catalog)
    meta = (
        pd.read_parquet(args.metadata)
        if args.metadata.exists() else pd.DataFrame()
    )

    catalog_cols = ["condition_id", "resolution_timestamp", "payouts", "total_volume_usd", "question_id"]
    if "source_tier" in catalog.columns:
        catalog_cols.append("source_tier")
    # Drop source_tier from feats if present to avoid merge suffix collision
    feats = feats.drop(columns=[c for c in ("source_tier",) if c in feats.columns])
    df = feats.merge(
        catalog[catalog_cols],
        on="condition_id", how="left",
    )
    if "source_tier" not in df.columns:
        df["source_tier"] = "unknown"
    df["insider_score"] = compute_insider_score(df)
    df = df[df["meta_window_trade_count"].fillna(0) >= 20]
    df = df.sort_values("insider_score", ascending=False).head(args.top_k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(f"# Case Study Candidates — Top {len(df)} by Insider Score\n\n")
        f.write(f"Generated {datetime.now(tz=timezone.utc).isoformat()}\n\n")
        f.write("> These are the highest-ranked candidates on the blended heuristic\n")
        f.write("> (volume spike × price acceleration × wallet concentration ×\n")
        f.write("> winner-aligned flow). They are NOT confirmed insider trades;\n")
        f.write("> they are suggestions for manual review against news archives.\n\n")

        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            ts = int(row["resolution_timestamp"]) if pd.notna(row["resolution_timestamp"]) else 0
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else "?"
            payouts = row["payouts"]
            if payouts is not None and hasattr(payouts, "__iter__"):
                payouts = list(payouts)
            winner_idx = None
            if payouts:
                try:
                    ints = [int(p) for p in payouts]
                    if max(ints) > 0:
                        winner_idx = ints.index(max(ints))
                except (TypeError, ValueError):
                    pass

            # Try to attach question text if metadata is present
            question_text = ""
            slug = ""
            if not meta.empty and pd.notna(row.get("question_id")):
                match = meta[meta["questionId"] == row["question_id"]]
                if not match.empty:
                    question_text = str(match.iloc[0].get("question") or "")
                    slug = str(match.iloc[0].get("slug") or "")

            f.write(f"## #{rank}   score={row['insider_score']:.2f}\n")
            f.write(f"- **condition_id**: `{row['condition_id']}`\n")
            if row["source_tier"] == "historical" and row.get("question_id"):
                f.write(f"- **question_id**: `{row['question_id']}`\n")
            if question_text:
                f.write(f"- **question**: {question_text}\n")
                if slug:
                    f.write(f"- **polymarket**: https://polymarket.com/event/{slug}\n")
            f.write(f"- **source_tier**: {row['source_tier']}\n")
            f.write(f"- **resolved**: {dt} (ts={ts})\n")
            f.write(f"- **payouts**: {payouts}  → winner = outcome {winner_idx}\n")
            f.write(f"- **total_volume_usd**: ${row['total_volume_usd']:,.0f}\n")
            f.write(f"- **vol_24h_usd**: ${row['vol_24h_usd']:,.0f}   "
                    f"**spike×**: {row['vol_24h_spike_ratio']:.2f}\n")
            f.write(f"- **dir_24h_signed_imbalance**: {row['dir_24h_signed_imbalance']:+.3f}\n")
            if "winner_24h_net_usd" in row.index:
                f.write(f"- **winner_24h_net_usd**: ${row['winner_24h_net_usd']:,.0f}   "
                        f"**hit**: {int(row.get('winner_24h_hit', 0))}\n")
            f.write(f"- **price_24h_move_zscore**: {row['price_24h_move_zscore']:+.2f}\n")
            f.write(f"- **wallet_24h_top1_share**: {row['wallet_24h_top1_share']:.2%}   "
                    f"**unique wallets 24h**: {int(row.get('wallet_24h_unique', 0))}\n")
            f.write("\n")

    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
