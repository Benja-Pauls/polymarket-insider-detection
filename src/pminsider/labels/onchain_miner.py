"""Mine per-market trades for wallets that look like informed actors.

For each (wallet, market) pair in our on-chain data, compute signals:

    fresh_wallet           wallet's first-ever Polymarket trade (across any market)
                           happened within 14 days of their first trade in THIS market
    single_market          this market accounts for >= 80% of the wallet's total USDC volume
    large_first_position   the first position this wallet ever opened was >= $10,000 USDC
    realized_profit        wallet's realized P&L from their buy/sell history in this market
                           exceeds $5,000 AND profit / usd_in_market > 15%. Computed from
                           per-trade accounting against the resolved outcome.
    informed_entry         wallet's VWAP entry price on the winning outcome's token is
                           below 0.5 (market was uncertain) AND they were in the top
                           10% of market size — "bought cheap, won big" pattern.
    early_timing           wallet's first trade happened >= 24 hours before resolution
                           (i.e. not a last-minute reaction that couldn't have been informed)

A wallet is a CANDIDATE for a market if it scores >= 2 signals. Each
candidate gets a composite score in [0, 1] that we use to rank globally.

Produces one row per (wallet, condition_id) with the full scoring breakdown.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

FRESH_WALLET_DAYS = 14
SINGLE_MARKET_THRESHOLD = 0.80
LARGE_FIRST_POSITION_USD = 10_000
SIZE_PERCENTILE_THRESHOLD = 90
EARLY_HOURS_BEFORE_RESOLUTION = 24

# Realized-profit flag thresholds
REALIZED_PROFIT_MIN_USD = 5_000.0
REALIZED_PROFIT_MIN_RATIO = 0.15

# Informed-entry flag thresholds
INFORMED_ENTRY_MAX_VWAP = 0.5

MIN_FLAGS_FOR_CANDIDATE = 2


@dataclass
class WalletCandidate:
    wallet: str
    condition_id: str
    resolution_timestamp: int | None
    market_total_volume_usd: float

    # Per-wallet-in-this-market aggregates
    wallet_usd_in_market: float
    wallet_trade_count_in_market: int
    wallet_first_trade_ts: int
    wallet_last_trade_ts: int
    wallet_flow_direction: str | None
    wallet_size_percentile_in_market: float

    # Cross-market context
    wallet_total_usd_across_all_markets: float
    wallet_market_count: int
    wallet_first_trade_across_all_ts: int
    wallet_concentration_in_this_market: float  # this-market / all-markets

    # Winner + profit context
    winner_outcome_index: int | None
    realized_profit_usd: float | None
    profit_ratio: float | None
    entry_vwap_on_winner: float | None

    # Boolean flags (each contributes to the composite score)
    flag_fresh_wallet: bool
    flag_single_market: bool
    flag_large_first_position: bool
    flag_realized_profit: bool
    flag_informed_entry: bool
    flag_early_timing: bool

    suspicion_score: float  # weighted sum, [0, 1]
    flags_count: int


def _winning_outcome_index(payouts) -> int | None:
    if payouts is None:
        return None
    try:
        # pandas may give a numpy array; handle both
        if hasattr(payouts, "tolist"):
            payouts = payouts.tolist()
        ints = [int(p) for p in payouts]
    except (TypeError, ValueError):
        return None
    if not ints or max(ints) == 0:
        return None
    return ints.index(max(ints))


def _coerce_token_to_outcome(raw) -> dict:
    """Normalize the per-market token_to_outcome mapping into {token_id: int_index}."""
    if raw is None:
        return {}
    try:
        items = dict(raw).items()
    except (TypeError, ValueError):
        return {}
    out: dict = {}
    for tok, idx in items:
        if tok is None or idx is None:
            continue
        try:
            out[str(tok)] = int(idx)
        except (TypeError, ValueError):
            continue
    return out


def mine_all(
    catalog_path: Path,
    trades_dir: Path,
    out_path: Path,
    *,
    min_market_volume: float = 50_000,
) -> pd.DataFrame:
    """Walk every market, compute per-wallet signals, emit candidates."""
    catalog = pd.read_parquet(catalog_path)
    log.info("catalog markets: %d", len(catalog))

    # First pass: build a cross-market wallet summary so we know each wallet's
    # total activity and first-ever trade timestamp. We also collect per-market
    # per-wallet aggregates used for flag computation.
    wallet_totals: dict[str, dict] = defaultdict(
        lambda: {"total_usd": 0.0, "markets": set(), "first_ts": None}
    )

    log.info("first pass: aggregating wallet-level totals across all markets…")
    # market_wallet_volumes[cid][wallet] = {
    #   usd_in_market, trades, first_ts, last_ts, buy_count, sell_count,
    #   realized_profit, winner_buy_usdc, winner_buy_notional,   # winner_buy_notional = size × price
    # }
    market_wallet_volumes: dict[str, dict[str, dict]] = {}

    for _, m in tqdm(catalog.iterrows(), total=len(catalog), desc="pass-1"):
        cid = m["condition_id"]
        tp = trades_dir / f"{cid}.parquet"
        if not tp.exists() or tp.stat().st_size < 100:
            continue
        try:
            trades = pd.read_parquet(tp)
        except Exception:  # noqa: BLE001
            continue
        if trades.empty:
            continue

        # Market-level context used for realized-profit + VWAP calculations.
        payouts = m.get("payouts")
        winner_idx = _winning_outcome_index(payouts)
        t2o = _coerce_token_to_outcome(m.get("token_to_outcome"))

        per_wallet: dict[str, dict] = {}
        for _, t in trades.iterrows():
            ts = int(t["timestamp"])
            usd = float(t.get("usd_spent_usdc") or 0)
            side = t.get("side")
            tok = t.get("token_id")
            tok_key = str(tok) if tok is not None else None
            size_outcome = float(t.get("size_outcome_usdc") or 0)
            price = float(t.get("price") or 0)
            outcome_idx = t2o.get(tok_key) if tok_key else None

            # Per-trade realized profit contribution (only computable when we
            # know both the winner and the outcome mapping for this token).
            trade_profit = None
            if winner_idx is not None and outcome_idx is not None:
                if side == "BUY":
                    if outcome_idx == winner_idx:
                        trade_profit = size_outcome * (1 - price)
                    else:
                        trade_profit = -size_outcome * price
                elif side == "SELL":
                    if outcome_idx == winner_idx:
                        trade_profit = -size_outcome * (1 - price)
                    else:
                        trade_profit = size_outcome * price

            for addr in (t.get("maker"), t.get("taker")):
                if not addr or not isinstance(addr, str):
                    continue
                w = wallet_totals[addr]
                w["total_usd"] += usd
                w["markets"].add(cid)
                if w["first_ts"] is None or ts < w["first_ts"]:
                    w["first_ts"] = ts

                pm = per_wallet.setdefault(addr, {
                    "usd_in_market": 0.0, "trades": 0,
                    "first_ts": ts, "last_ts": ts,
                    "buy_count": 0, "sell_count": 0,
                    "realized_profit": 0.0,
                    "has_profit_data": False,
                    "winner_buy_size": 0.0,      # Σ size_outcome_usdc on winner token BUYs
                    "winner_buy_notional": 0.0,  # Σ size × price on winner token BUYs
                })
                pm["usd_in_market"] += usd
                pm["trades"] += 1
                pm["first_ts"] = min(pm["first_ts"], ts)
                pm["last_ts"] = max(pm["last_ts"], ts)
                if side == "BUY":
                    pm["buy_count"] += 1
                elif side == "SELL":
                    pm["sell_count"] += 1

                if trade_profit is not None:
                    pm["realized_profit"] += trade_profit
                    pm["has_profit_data"] = True

                # Track the wallet's VWAP *entry* price on the winning token
                # (BUYs only; they're what establishes the entry cost basis).
                if (
                    winner_idx is not None
                    and outcome_idx is not None
                    and outcome_idx == winner_idx
                    and side == "BUY"
                    and size_outcome > 0
                ):
                    pm["winner_buy_size"] += size_outcome
                    pm["winner_buy_notional"] += size_outcome * price

        market_wallet_volumes[cid] = per_wallet

    log.info("wallets tracked across all markets: %d", len(wallet_totals))

    # Second pass: compute flags per (wallet, market)
    log.info("second pass: scoring per (wallet, market) pair…")
    candidates: list[WalletCandidate] = []
    for _, m in tqdm(catalog.iterrows(), total=len(catalog), desc="pass-2"):
        cid = m["condition_id"]
        market_vol = float(m.get("total_volume_usd") or 0)
        if market_vol < min_market_volume:
            continue
        payouts = m.get("payouts")
        winner_idx = _winning_outcome_index(payouts)
        resolution_ts = int(m["resolution_timestamp"]) if m.get("resolution_timestamp") else None

        per_wallet = market_wallet_volumes.get(cid, {})
        if not per_wallet:
            continue

        # Compute size percentile thresholds for this market
        sizes = np.array([pw["usd_in_market"] for pw in per_wallet.values()])
        if len(sizes) == 0:
            continue

        for addr, pw in per_wallet.items():
            wallet_total = wallet_totals[addr]["total_usd"]
            wallet_markets = wallet_totals[addr]["markets"]
            concentration = pw["usd_in_market"] / wallet_total if wallet_total > 0 else 0
            first_ever_ts = wallet_totals[addr]["first_ts"]
            days_between_first_trades = (
                (pw["first_ts"] - first_ever_ts) / 86400 if first_ever_ts else 0
            )

            # Flag 1: fresh wallet — first-ever trade was within N days of this market's first trade
            fresh_wallet = days_between_first_trades <= FRESH_WALLET_DAYS

            # Flag 2: single-market concentration
            single_market = concentration >= SINGLE_MARKET_THRESHOLD

            # Flag 3: large first position — only if the wallet's first-ever
            # trade lives in this market and the size meets the threshold.
            if first_ever_ts is not None and first_ever_ts == pw["first_ts"]:
                large_first_position = pw["usd_in_market"] >= LARGE_FIRST_POSITION_USD
            else:
                large_first_position = False

            # Size percentile (rank of this wallet's market spend)
            size_pct = (
                (np.sum(sizes <= pw["usd_in_market"]) / len(sizes)) * 100
                if len(sizes) > 0 else 0.0
            )

            # Flag 4: realized profit — trade-level accounting against winner.
            realized_profit: float | None = None
            profit_ratio: float | None = None
            realized_profit_flag = False
            if pw["has_profit_data"] and winner_idx is not None:
                realized_profit = float(pw["realized_profit"])
                if pw["usd_in_market"] > 0:
                    profit_ratio = realized_profit / pw["usd_in_market"]
                else:
                    profit_ratio = 0.0
                realized_profit_flag = (
                    realized_profit > REALIZED_PROFIT_MIN_USD
                    and profit_ratio > REALIZED_PROFIT_MIN_RATIO
                )

            # Flag 5: informed entry — bought the winning outcome cheap, big.
            entry_vwap_on_winner: float | None = None
            informed_entry_flag = False
            if winner_idx is not None and pw["winner_buy_size"] > 0:
                entry_vwap_on_winner = pw["winner_buy_notional"] / pw["winner_buy_size"]
                informed_entry_flag = (
                    entry_vwap_on_winner < INFORMED_ENTRY_MAX_VWAP
                    and size_pct >= SIZE_PERCENTILE_THRESHOLD
                )

            # Flag 6: early timing — first trade well before resolution
            early_timing = False
            if resolution_ts:
                hours_before = (resolution_ts - pw["first_ts"]) / 3600
                early_timing = hours_before >= EARLY_HOURS_BEFORE_RESOLUTION

            flags = [
                fresh_wallet,
                single_market,
                large_first_position,
                realized_profit_flag,
                informed_entry_flag,
                early_timing,
            ]
            flag_count = sum(flags)
            if flag_count < MIN_FLAGS_FOR_CANDIDATE:
                continue

            # Composite score
            score = (
                0.25 * fresh_wallet
                + 0.20 * single_market
                + 0.15 * large_first_position
                + 0.25 * realized_profit_flag
                + 0.10 * informed_entry_flag
                + 0.05 * early_timing
            )

            # Majority direction label
            flow_dir = None
            if pw["buy_count"] > pw["sell_count"]:
                flow_dir = "BUY"
            elif pw["sell_count"] > pw["buy_count"]:
                flow_dir = "SELL"

            candidates.append(WalletCandidate(
                wallet=addr,
                condition_id=cid,
                resolution_timestamp=resolution_ts,
                market_total_volume_usd=market_vol,
                wallet_usd_in_market=pw["usd_in_market"],
                wallet_trade_count_in_market=pw["trades"],
                wallet_first_trade_ts=pw["first_ts"],
                wallet_last_trade_ts=pw["last_ts"],
                wallet_flow_direction=flow_dir,
                wallet_size_percentile_in_market=float(size_pct),
                wallet_total_usd_across_all_markets=wallet_total,
                wallet_market_count=len(wallet_markets),
                wallet_first_trade_across_all_ts=first_ever_ts or 0,
                wallet_concentration_in_this_market=concentration,
                winner_outcome_index=winner_idx,
                realized_profit_usd=realized_profit,
                profit_ratio=profit_ratio,
                entry_vwap_on_winner=entry_vwap_on_winner,
                flag_fresh_wallet=fresh_wallet,
                flag_single_market=single_market,
                flag_large_first_position=large_first_position,
                flag_realized_profit=realized_profit_flag,
                flag_informed_entry=informed_entry_flag,
                flag_early_timing=early_timing,
                suspicion_score=score,
                flags_count=flag_count,
            ))

    log.info("raw candidates: %d", len(candidates))

    df = pd.DataFrame([asdict(c) for c in candidates])
    df = df.sort_values("suspicion_score", ascending=False).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("wrote %d candidates → %s", len(df), out_path)

    return df
