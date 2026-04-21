"""Deep-dive the Andy Byron Astronomer CEO insider-trading cluster.

Produces:
  * a timeline of every flagged wallet's first-trade entry
  * estimated profit per wallet (size × (1 - entry_price) for YES buyers)
  * USDC funding source per wallet (who transferred USDC first to this addr)
  * a human-readable markdown case study at results/byron_cluster_case_study.md
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pminsider.polygon_rpc import PolygonRPC

log = logging.getLogger(__name__)

USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174".lower()  # Polygon USDC.e
USDC_NATIVE = "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359".lower()  # Polygon native USDC


def resolve_funding(rpc: PolygonRPC, wallet: str, *, max_back_blocks: int = 50_000) -> dict:
    """Look for the first incoming USDC Transfer to this wallet.

    eth_getLogs on Transfer(address,address,uint256) with topic[2] = padded wallet.
    We scan in ~5K-block chunks going backwards from latest.
    """
    # Transfer(address indexed from, address indexed to, uint256 value)
    TRANSFER_SIG = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
    wallet_topic = "0x" + wallet.lower().removeprefix("0x").rjust(64, "0")

    latest = rpc.block_number()
    tip = latest
    found = None
    for _ in range(10):  # scan up to ~50K blocks (~3-4 days on Polygon)
        from_block = max(0, tip - 5_000)
        for token in (USDC_E, USDC_NATIVE):
            try:
                logs = rpc.call("eth_getLogs", [{
                    "fromBlock": hex(from_block),
                    "toBlock": hex(tip),
                    "address": token,
                    "topics": [TRANSFER_SIG, None, wallet_topic],
                }])
            except Exception:  # noqa: BLE001
                continue
            if logs:
                # Pick the earliest
                logs.sort(key=lambda l: (int(l["blockNumber"], 16), int(l["logIndex"], 16)))
                l = logs[0]
                from_topic = l["topics"][1]
                sender = "0x" + from_topic[-40:]
                amount = int(l["data"], 16) / 1e6
                block = int(l["blockNumber"], 16)
                found = {"funder": sender.lower(), "token": token, "amount_usdc": amount, "block": block}
                return found
        tip = from_block
        if tip == 0:
            break
    return {"funder": None, "token": None, "amount_usdc": None, "block": None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curated", type=Path, default=Path("data/labels/curated_candidates.parquet"))
    ap.add_argument("--trades-dir", type=Path, default=Path("data/raw/goldsky/trades"))
    ap.add_argument("--catalog", type=Path, default=Path("data/raw/goldsky/markets.parquet"))
    ap.add_argument("--out-md", type=Path, default=Path("results/byron_cluster_case_study.md"))
    ap.add_argument("--out-csv", type=Path, default=Path("results/byron_cluster_wallets.csv"))
    ap.add_argument("--byron-condition", type=str,
                    default="0x7590f51f00a87c7bc60cf740f43b1ca4a97c5e9ad583cf4cf94379508d0a067b")
    ap.add_argument("--skip-funding", action="store_true",
                    help="skip the (slow) USDC-funding RPC lookup")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")

    cur = pd.read_parquet(args.curated)
    byron = cur[
        cur["market_question"].fillna("").str.contains("Andy Byron", case=False, na=False)
    ].copy()
    log.info("Byron-related curated rows: %d", len(byron))
    flagged = byron[byron["final_verdict"].isin(["confirmed", "suspected", "ambiguous"])]
    log.info("flagged (confirmed/suspected/ambiguous): %d", len(flagged))

    # Pull Byron trades
    trades = pd.read_parquet(args.trades_dir / f"{args.byron_condition}.parquet")
    log.info("total trades in Byron market: %d", len(trades))

    catalog = pd.read_parquet(args.catalog)
    byron_meta = catalog[catalog["condition_id"] == args.byron_condition].iloc[0]
    resolution_ts = int(byron_meta["resolution_timestamp"])
    payouts = list(byron_meta["payouts"])
    winner_idx = payouts.index("1")
    log.info("resolution ts=%s  payouts=%s  winner_outcome_idx=%d",
             datetime.fromtimestamp(resolution_ts, tz=timezone.utc).isoformat(),
             payouts, winner_idx)

    # Per-wallet analysis
    rpc = PolygonRPC() if not args.skip_funding else None
    rows = []
    for _, r in flagged.iterrows():
        w = str(r["wallet"]).lower()
        wt = trades[
            (trades["maker"].fillna("").str.lower() == w) |
            (trades["taker"].fillna("").str.lower() == w)
        ].copy()
        if wt.empty:
            continue

        # First-trade meta
        wt_sorted = wt.sort_values("timestamp")
        first_ts = int(wt_sorted["timestamp"].iloc[0])
        first_price = float(wt_sorted["price"].iloc[0])
        hrs_before = (resolution_ts - first_ts) / 3600

        # Total USD, sides
        total_usd = float(wt["usd_spent_usdc"].sum())
        n_trades = len(wt)

        # Simple profit estimate for a YES buyer who held to resolution
        # (this over-estimates — some positions are sold back before resolution,
        # and real profit depends on exact token holdings at settlement. This is
        # a rough upper bound for YES-held-to-resolution.)
        buy_usd = float(wt[wt["side"] == "BUY"]["usd_spent_usdc"].sum())
        buy_vwap_price = 0.0
        if buy_usd > 0:
            # VWAP based on USDC/size ratio
            buys = wt[wt["side"] == "BUY"]
            buy_vwap_price = float((buys["usd_spent_usdc"] / buys["size_outcome_usdc"]).mean())
        # If they bought YES (which resolved to 1), profit ≈ (1 - entry_price) * size_outcome_usdc
        est_profit = 0.0
        if buy_vwap_price > 0:
            est_profit = float(wt[wt["side"] == "BUY"]["size_outcome_usdc"].sum()) * (1 - buy_vwap_price)

        # Funding source
        funding = {"funder": None, "amount_usdc": None} if args.skip_funding else None
        if rpc:
            try:
                funding = resolve_funding(rpc, w)
            except Exception:  # noqa: BLE001
                funding = {"funder": None, "amount_usdc": None}
            time.sleep(0.1)

        rows.append({
            "wallet": w,
            "verdict": r["final_verdict"],
            "tier": r["confidence_tier_final"],
            "first_trade_iso": datetime.fromtimestamp(first_ts, tz=timezone.utc).isoformat(),
            "hours_before_resolution": round(hrs_before, 1),
            "first_buy_price": round(first_price, 4),
            "buy_vwap_price": round(buy_vwap_price, 4),
            "total_usd": round(total_usd, 2),
            "n_trades": n_trades,
            "est_profit_usd_if_yes_held": round(est_profit, 2),
            "funder": funding.get("funder"),
            "funder_amount_usdc": funding.get("amount_usdc"),
        })

    out_df = pd.DataFrame(rows).sort_values("first_trade_iso").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    log.info("wrote %s", args.out_csv)

    # Funder cluster analysis
    funder_counts = {}
    if "funder" in out_df.columns:
        fs = out_df["funder"].dropna().tolist()
        for f in fs:
            funder_counts[f] = funder_counts.get(f, 0) + 1

    # Markdown report
    md: list[str] = []
    md.append("# Andy Byron Astronomer CEO Insider-Trading Cluster — Case Study\n")
    md.append(f"**Market:** `Andy Byron out as Astronomer CEO by next Friday?`")
    md.append(f"**Condition ID:** `{args.byron_condition}`")
    md.append(f"**Resolution:** {datetime.fromtimestamp(resolution_ts, tz=timezone.utc).isoformat()}")
    md.append(f"**Payouts:** `{payouts}` (winner = outcome {winner_idx}, YES)")
    md.append(f"**Total market trades:** {len(trades):,}\n")
    md.append(f"**Flagged wallets in this market:** {len(out_df)}\n")

    md.append("## Summary statistics\n")
    md.append(f"- Total flagged USDC deployed: **${out_df['total_usd'].sum():,.0f}**")
    md.append(f"- Top wallet bet: **${out_df['total_usd'].max():,.0f}**")
    md.append(f"- Estimated combined profit if YES-held: **${out_df['est_profit_usd_if_yes_held'].sum():,.0f}**")
    md.append(f"- Median first-trade entry price: **{out_df['first_buy_price'].median():.3f}**")
    md.append(f"- Median hours before resolution at first trade: **{out_df['hours_before_resolution'].median():.0f}h**\n")

    md.append("## Timeline of entries\n")
    md.append("| first trade (UTC) | hrs pre-res | entry price | VWAP | total USDC | est profit (YES held) | verdict | wallet |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in out_df.iterrows():
        md.append(
            f"| {r['first_trade_iso']} | {r['hours_before_resolution']:.0f}h | {r['first_buy_price']:.3f} | "
            f"{r['buy_vwap_price']:.3f} | ${r['total_usd']:,.0f} | ${r['est_profit_usd_if_yes_held']:,.0f} | "
            f"{r['verdict']} ({r['tier']}) | `{r['wallet'][:14]}…` |"
        )

    if funder_counts:
        md.append("\n## Funding sources (USDC transfers IN)\n")
        md.append("| funder address | # wallets funded |")
        md.append("|---|---|")
        for f, c in sorted(funder_counts.items(), key=lambda x: -x[1]):
            md.append(f"| `{f}` | {c} |")
        multi = {f: c for f, c in funder_counts.items() if c >= 2}
        if multi:
            md.append(f"\n**Cluster signal:** {len(multi)} funders fund ≥2 of these wallets. "
                      f"Largest: {max(multi.values())} wallets funded by one address.\n")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md))
    log.info("wrote %s", args.out_md)

    print(out_df.to_string(index=False, max_colwidth=60))


if __name__ == "__main__":
    main()
