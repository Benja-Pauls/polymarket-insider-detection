"""Mine per-market trades for wallets that look like informed actors.

For each (wallet, market) pair in our on-chain data, compute signals:

    fresh_wallet           wallet's first-ever Polymarket trade (across any market)
                           happened within 14 days of their first trade in THIS market
    single_market          this market accounts for >= 80% of the wallet's total USDC volume
    large_first_position   the first position this wallet ever opened was >= $10,000 USDC
    win_aligned            the wallet's net flow direction agreed with the winner AND
                           their size percentile in this market is >= 90
    directional            the wallet's trades are >= 80% on one side (no round-tripping)
    early_timing           wallet's first trade happened >= 24 hours before resolution
                           (i.e. not a last-minute reaction that couldn't have been informed)

A wallet is a CANDIDATE for a market if it scores >= 2 signals. Each
candidate gets a composite score in [0, 1] that we use to rank globally.

Produces one row per (wallet, condition_id) with the full scoring breakdown.
"""
from __future__ import annotations

import json
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
DIRECTIONAL_THRESHOLD = 0.80
EARLY_HOURS_BEFORE_RESOLUTION = 24

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
    wallet_directional_pct: float
    wallet_size_percentile_in_market: float

    # Cross-market context
    wallet_total_usd_across_all_markets: float
    wallet_market_count: int
    wallet_first_trade_across_all_ts: int
    wallet_concentration_in_this_market: float  # this-market / all-markets

    # Winner alignment
    winner_outcome_index: int | None
    wallet_net_toward_winner_usd: float
    wallet_win_aligned: bool

    # Boolean flags (each contributes to the composite score)
    flag_fresh_wallet: bool
    flag_single_market: bool
    flag_large_first_position: bool
    flag_win_aligned_top: bool
    flag_directional: bool
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


def _decide_side_sign(row, token_to_outcome: dict) -> int:
    """+1 if this trade moves the market TOWARD outcome 0, -1 if toward outcome 1."""
    tok = row.get("token_id")
    oi = token_to_outcome.get(tok) if tok else None
    if oi is None:
        return 0
    side = row.get("side")
    # BUY of outcome-0 token pushes outcome-0 price up ⇒ +1
    # SELL of outcome-0 pushes outcome-0 down ⇒ -1
    # BUY of outcome-1 pushes outcome-1 up = outcome-0 down ⇒ -1
    if oi == 0:
        return +1 if side == "BUY" else -1
    if oi == 1:
        return -1 if side == "BUY" else +1
    return 0


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
    # total activity and first-ever trade timestamp.
    wallet_totals: dict[str, dict] = defaultdict(
        lambda: {"total_usd": 0.0, "markets": set(), "first_ts": None,
                 "first_market_usd": 0.0}
    )

    log.info("first pass: aggregating wallet-level totals across all markets…")
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
        # Per-market per-wallet
        per_wallet: dict[str, dict] = {}
        for _, t in trades.iterrows():
            ts = int(t["timestamp"])
            usd = float(t.get("usd_spent_usdc") or 0)
            side = t.get("side")
            for role, addr in (("maker", t.get("maker")), ("taker", t.get("taker"))):
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
                    "signed_toward_0": 0.0,
                })
                pm["usd_in_market"] += usd
                pm["trades"] += 1
                pm["first_ts"] = min(pm["first_ts"], ts)
                pm["last_ts"] = max(pm["last_ts"], ts)
                if side == "BUY":
                    pm["buy_count"] += 1
                elif side == "SELL":
                    pm["sell_count"] += 1
                # Directional toward outcome-0
                t2o = {}
                if "token_to_outcome" in m.index and m["token_to_outcome"] is not None:
                    try:
                        t2o = dict(m["token_to_outcome"])
                    except (TypeError, ValueError):
                        pass
                sign = _decide_side_sign(t, t2o)
                pm["signed_toward_0"] += sign * usd
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
        p_threshold = float(np.percentile(sizes, SIZE_PERCENTILE_THRESHOLD))

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

            # Flag 3: large first position (approximated as large first-market spend)
            large_first_position = wallet_totals[addr]["first_market_usd"] >= LARGE_FIRST_POSITION_USD
            # Actually we haven't populated first_market_usd — approximate with size
            # in FIRST market. For simplicity: first-ever trade amount proxy via
            # this market if the wallet's first-ever ts falls in this market.
            if first_ever_ts == pw["first_ts"]:
                large_first_position = pw["usd_in_market"] >= LARGE_FIRST_POSITION_USD
            else:
                large_first_position = False

            # Flag 4: directional concentration
            total_sides = pw["buy_count"] + pw["sell_count"]
            if total_sides > 0:
                directional_pct = max(pw["buy_count"], pw["sell_count"]) / total_sides
            else:
                directional_pct = 0.0
            directional = directional_pct >= DIRECTIONAL_THRESHOLD

            # Flag 5: winner-aligned AND top-tier size
            size_pct = (
                (np.sum(sizes <= pw["usd_in_market"]) / len(sizes)) * 100
                if len(sizes) > 0 else 0
            )
            win_aligned = False
            net_toward_winner = 0.0
            if winner_idx is not None:
                signed = pw["signed_toward_0"]
                net_toward_winner = signed if winner_idx == 0 else -signed
                win_aligned = (net_toward_winner > 0) and (size_pct >= SIZE_PERCENTILE_THRESHOLD)

            # Flag 6: early timing — first trade well before resolution
            early_timing = False
            if resolution_ts:
                hours_before = (resolution_ts - pw["first_ts"]) / 3600
                early_timing = hours_before >= EARLY_HOURS_BEFORE_RESOLUTION

            flags = [
                fresh_wallet, single_market, large_first_position,
                win_aligned, directional, early_timing,
            ]
            flag_count = sum(flags)
            if flag_count < MIN_FLAGS_FOR_CANDIDATE:
                continue

            # Composite score: weighted sum, fresh+single+first are strongest
            score = (
                0.25 * fresh_wallet
                + 0.20 * single_market
                + 0.20 * large_first_position
                + 0.20 * win_aligned
                + 0.10 * directional
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
                wallet_directional_pct=directional_pct,
                wallet_size_percentile_in_market=float(size_pct),
                wallet_total_usd_across_all_markets=wallet_total,
                wallet_market_count=len(wallet_markets),
                wallet_first_trade_across_all_ts=first_ever_ts or 0,
                wallet_concentration_in_this_market=concentration,
                winner_outcome_index=winner_idx,
                wallet_net_toward_winner_usd=net_toward_winner,
                wallet_win_aligned=win_aligned,
                flag_fresh_wallet=fresh_wallet,
                flag_single_market=single_market,
                flag_large_first_position=large_first_position,
                flag_win_aligned_top=win_aligned,
                flag_directional=directional,
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
