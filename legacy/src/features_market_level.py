"""Feature engineering per market for the insider-detection classifier.

Given a market's trades DataFrame (output of ``collect.fetch_trades``) and the
market record (metadata), produce a single row of engineered features. The
research hypothesis is that insider-driven trading leaves a distinctive
fingerprint in the joint distribution of volume, direction, price acceleration,
and wallet concentration in the final hours before resolution.

Feature groups:
  * ``vol_*``      volume-spike ratios at multiple horizons
  * ``dir_*``      directional imbalance of flow
  * ``price_*``    price-acceleration signals
  * ``wallet_*``   concentration / novelty of the trading counterparty set
  * ``meta_*``     market-level metadata used as controls (age, total vol, etc.)

All horizons are counted backwards from ``resolution_timestamp``. For example
``vol_24h`` is the trade volume in ``[resolution_ts - 24h, resolution_ts]``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

HORIZONS_HOURS = (2, 6, 12, 24, 48, 72, 168)  # 2h → 1 week

# Default baseline window for computing "per-day average volume" when computing
# spike ratios: we use everything earlier than the longest short-window so we
# don't trivially pollute the baseline with the tail we're trying to measure.
BASELINE_START_HOURS = 168  # 7 days before resolution onward backwards is "tail"


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def compute_features(
    trades: pd.DataFrame,
    *,
    resolution_timestamp: int,
    payouts: list[str] | None,
    total_volume_usd: float | None = None,
    total_trades: int | None = None,
    outcome_slot_count: int | None = None,
    market_creation_ts: int | None = None,
    token_to_outcome: dict | None = None,
) -> dict:
    """Produce a flat dict of features for one market.

    ``trades`` must contain columns: ``timestamp``, ``side``, ``price``,
    ``usd_spent_usdc``, ``size_outcome_usdc``, ``maker``, ``taker``,
    ``token_id``.

    ``payouts`` is the raw on-chain payout vector from ``Condition.payouts``
    (e.g. ``['1','0']`` means outcome index 0 won). Used to determine the
    "winning direction" for ``dir_flow_agrees_with_resolution``.

    ``token_to_outcome``: optional map {token_id: outcome_index}. When provided
    we normalize the trade directions so that ``dir_*_toward_winner`` features
    correctly reflect net USDC flow into the winning outcome (disambiguating
    Yes-buys from No-sells).
    """
    feats: dict = {
        "meta_total_volume_usd": float(total_volume_usd or 0),
        "meta_total_trades": int(total_trades or 0),
        "meta_outcome_slot_count": int(outcome_slot_count or 2),
        "meta_market_age_days": (
            (resolution_timestamp - market_creation_ts) / 86400
            if market_creation_ts else None
        ),
    }

    if trades is None or trades.empty:
        feats["meta_window_trade_count"] = 0
        return feats

    df = trades.sort_values("timestamp").reset_index(drop=True)
    feats["meta_window_trade_count"] = len(df)

    # For binary markets, we pick the "winning outcome" and measure direction toward it
    winning_outcome_index = _winning_outcome_index(payouts)
    feats["meta_winning_outcome_index"] = winning_outcome_index
    feats["meta_resolution_is_binary"] = (
        outcome_slot_count == 2 if outcome_slot_count else None
    )

    # Add per-trade signed USDC flow toward outcome-0 (if we have the mapping)
    if token_to_outcome:
        df = df.assign(
            signed_usd=df.apply(
                lambda r: _signed_flow_toward_outcome0(r, token_to_outcome),
                axis=1,
            )
        )
    else:
        # Fallback: positive for BUY, negative for SELL (ignores token/outcome)
        df = df.assign(
            signed_usd=df.apply(
                lambda r: r["usd_spent_usdc"] if r["side"] == "BUY" else -r["usd_spent_usdc"],
                axis=1,
            )
        )

    # Establish the time window bounds
    t_max = resolution_timestamp
    earliest_in_df = int(df["timestamp"].min())
    t_min = min(earliest_in_df, t_max - BASELINE_START_HOURS * 3600)

    # Volume features
    feats.update(_volume_features(df, t_max, total_window_seconds=(t_max - t_min)))
    # Direction features
    feats.update(_direction_features(df, t_max, winning_outcome_index=winning_outcome_index))
    # Price features
    feats.update(_price_features(df, t_max, token_to_outcome=token_to_outcome))
    # Wallet features
    feats.update(_wallet_features(df, t_max))
    # Outcome-normalized winner-direction features (requires token_to_outcome)
    if token_to_outcome and winning_outcome_index is not None:
        feats.update(_winner_aligned_features(df, t_max, winning_outcome_index=winning_outcome_index))

    return feats


def _signed_flow_toward_outcome0(row: pd.Series, token_to_outcome: dict) -> float:
    """USDC flow is positive if it pushes probability of outcome 0 UP.

    BUY of outcome-0 token    → +usd (bullish outcome 0)
    SELL of outcome-0 token   → -usd
    BUY of outcome-1 token    → -usd (bullish outcome 1 = bearish outcome 0)
    SELL of outcome-1 token   → +usd
    Unknown outcome → 0 (drop from directional signal).
    """
    tok = row.get("token_id")
    oi = token_to_outcome.get(tok) if tok else None
    if oi is None:
        return 0.0
    usd = float(row.get("usd_spent_usdc") or 0)
    if oi == 0:
        return usd if row["side"] == "BUY" else -usd
    if oi == 1:
        return -usd if row["side"] == "BUY" else usd
    return 0.0


def _winner_aligned_features(df: pd.DataFrame, t_max: int, *, winning_outcome_index: int) -> dict:
    """Features expressing the net flow in the direction of the actual winner.

    Positive value = flow pointed toward the winner (informed-trading-consistent).
    """
    feats: dict = {}
    winner_sign = +1 if winning_outcome_index == 0 else -1
    df = df.assign(toward_winner=df["signed_usd"] * winner_sign)
    for h in HORIZONS_HOURS:
        mask = df["timestamp"] >= (t_max - h * 3600)
        win = df.loc[mask]
        feats[f"winner_{h}h_net_usd"] = float(win["toward_winner"].sum())
        feats[f"winner_{h}h_hit"] = int(win["toward_winner"].sum() > 0)
    return feats


def compute_features_batch(
    catalog: pd.DataFrame,
    trades_for_condition: "dict[str, pd.DataFrame]",
) -> pd.DataFrame:
    """Compute features for every market in ``catalog`` given a mapping of
    condition_id → trades DataFrame."""
    rows: list[dict] = []
    for _, m in catalog.iterrows():
        cid = m["condition_id"]
        trades = trades_for_condition.get(cid, pd.DataFrame())
        feats = compute_features(
            trades,
            resolution_timestamp=int(m["resolution_timestamp"]),
            payouts=list(m["payouts"]) if m["payouts"] is not None else None,
            total_volume_usd=float(m["total_volume_usd"]),
            total_trades=int(m["total_trades"]),
            outcome_slot_count=int(m["outcome_slot_count"]) if m["outcome_slot_count"] else 2,
        )
        feats["condition_id"] = cid
        feats["source_tier"] = m["source_tier"]
        rows.append(feats)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Individual feature groups
# ----------------------------------------------------------------------

def _winning_outcome_index(payouts: list[str] | None) -> int | None:
    if not payouts:
        return None
    try:
        ints = [int(p) for p in payouts]
    except (TypeError, ValueError):
        return None
    if not ints:
        return None
    mx = max(ints)
    if mx == 0:
        return None
    for i, p in enumerate(ints):
        if p == mx:
            return i
    return None


def _volume_features(df: pd.DataFrame, t_max: int, total_window_seconds: int) -> dict:
    feats: dict = {}
    # Total notional over the whole window
    total_usd = float(df["usd_spent_usdc"].sum())
    total_seconds = max(1, total_window_seconds)
    daily_avg = total_usd * 86400 / total_seconds
    feats["vol_total_window_usd"] = total_usd
    feats["vol_daily_avg_usd"] = daily_avg

    for h in HORIZONS_HOURS:
        mask = df["timestamp"] >= (t_max - h * 3600)
        win_usd = float(df.loc[mask, "usd_spent_usdc"].sum())
        win_trades = int(mask.sum())
        feats[f"vol_{h}h_usd"] = win_usd
        feats[f"vol_{h}h_trades"] = win_trades
        feats[f"vol_{h}h_pct_total"] = win_usd / total_usd if total_usd > 0 else 0.0
        # Spike ratio: last N hours vs daily baseline scaled to N hours
        expected_h = daily_avg * h / 24
        feats[f"vol_{h}h_spike_ratio"] = (
            win_usd / expected_h if expected_h > 0 else math.nan
        )
    return feats


def _direction_features(df: pd.DataFrame, t_max: int, *, winning_outcome_index: int | None) -> dict:
    feats: dict = {}

    for h in HORIZONS_HOURS:
        mask = df["timestamp"] >= (t_max - h * 3600)
        w = df.loc[mask]
        buy_usd = float(w.loc[w["side"] == "BUY", "usd_spent_usdc"].sum())
        sell_usd = float(w.loc[w["side"] == "SELL", "usd_spent_usdc"].sum())
        total = buy_usd + sell_usd
        feats[f"dir_{h}h_buy_usd"] = buy_usd
        feats[f"dir_{h}h_sell_usd"] = sell_usd
        feats[f"dir_{h}h_net_usd"] = buy_usd - sell_usd
        feats[f"dir_{h}h_abs_imbalance"] = abs(buy_usd - sell_usd) / total if total > 0 else 0.0
        feats[f"dir_{h}h_signed_imbalance"] = (buy_usd - sell_usd) / total if total > 0 else 0.0

    # Agreement: does the last-24h net-flow direction point at the winning outcome?
    # (For binary markets: BUY of outcome=0 token → up-move of outcome 0.
    # We don't currently split by outcome token; that's a v2 upgrade.)
    # For v1 we report the sign of the net flow and let the model combine with
    # payouts one-hot.
    feats["dir_resolution_outcome_index"] = (
        winning_outcome_index if winning_outcome_index is not None else -1
    )
    return feats


def _price_features(df: pd.DataFrame, t_max: int, *, token_to_outcome: dict | None = None) -> dict:
    feats: dict = {}
    # Normalize price to "implied probability of outcome 0" when we know
    # which outcome each trade's token represents. Without mapping we fall
    # back to the raw price (mixed Yes/No series, more noisy).
    if token_to_outcome:
        df = df.assign(
            norm_price=df.apply(
                lambda r: (
                    float(r["price"])
                    if token_to_outcome.get(r["token_id"]) == 0
                    else 1.0 - float(r["price"])
                    if token_to_outcome.get(r["token_id"]) == 1
                    else float(r["price"])
                ),
                axis=1,
            )
        )
        prices = df["norm_price"].astype(float)
    else:
        prices = df["price"].astype(float)

    # Last trade price (nearest resolution)
    last_price = float(prices.iloc[-1]) if len(prices) else math.nan
    feats["price_last"] = last_price

    # Price as of (t_max - Nh): nearest trade before that cutoff
    prior = {}
    for h in HORIZONS_HOURS:
        cutoff = t_max - h * 3600
        before = df[df["timestamp"] <= cutoff]
        if len(before):
            p = float(before["price"].iloc[-1])
        else:
            p = math.nan
        prior[h] = p
        feats[f"price_as_of_{h}h_ago"] = p
        feats[f"price_change_{h}h"] = last_price - p if not math.isnan(p) else math.nan

    # Acceleration: Δprice(last 2h) / Δprice(2–6h) (higher = price moved fast only in final window)
    dp_tail = feats.get("price_change_2h")
    dp_prior = feats.get("price_change_6h")
    if dp_tail is not None and dp_prior and not math.isnan(dp_prior) and dp_prior != 0:
        feats["price_accel_2h_vs_6h"] = dp_tail / dp_prior
    else:
        feats["price_accel_2h_vs_6h"] = math.nan

    # Volatility: stdev of trade-to-trade price changes across the whole window
    feats["price_stdev"] = float(prices.diff().std()) if len(prices) > 1 else 0.0
    # Price range (max - min)
    feats["price_range"] = float(prices.max() - prices.min()) if len(prices) else 0.0

    # Z-score of the last-24h move vs historical stdev
    dp_24h = feats.get("price_change_24h")
    if feats["price_stdev"] > 0 and dp_24h is not None and not math.isnan(dp_24h):
        feats["price_24h_move_zscore"] = dp_24h / feats["price_stdev"]
    else:
        feats["price_24h_move_zscore"] = 0.0

    return feats


def _wallet_features(df: pd.DataFrame, t_max: int) -> dict:
    feats: dict = {}

    # Aggregate all unique wallets that touched this market across the window
    all_wallets = pd.concat([df["maker"], df["taker"]]).dropna().unique()
    feats["wallet_unique_total"] = int(len(all_wallets))

    for h in HORIZONS_HOURS:
        mask = df["timestamp"] >= (t_max - h * 3600)
        w = df.loc[mask]
        # Wallet concentration: pool maker + taker with their USDC footprint
        maker_vol = w.groupby("maker")["usd_spent_usdc"].sum()
        taker_vol = w.groupby("taker")["usd_spent_usdc"].sum()
        combined = maker_vol.add(taker_vol, fill_value=0)
        total = float(combined.sum())

        feats[f"wallet_{h}h_unique"] = int(combined.shape[0])

        if total > 0:
            shares = combined / total
            feats[f"wallet_{h}h_herfindahl"] = float((shares ** 2).sum())
            feats[f"wallet_{h}h_top1_share"] = float(shares.max())
            top5 = shares.nlargest(5)
            feats[f"wallet_{h}h_top5_share"] = float(top5.sum())
        else:
            feats[f"wallet_{h}h_herfindahl"] = math.nan
            feats[f"wallet_{h}h_top1_share"] = math.nan
            feats[f"wallet_{h}h_top5_share"] = math.nan

        # New-wallet share: wallets appearing in this window that did NOT
        # appear in [t_max - 7d, t_max - h] (the "historical prior"). Only
        # meaningful for short horizons.
        if h <= 24:
            prior_mask = (
                (df["timestamp"] >= (t_max - 168 * 3600)) &
                (df["timestamp"] < (t_max - h * 3600))
            )
            prior_wallets = set(
                pd.concat([df.loc[prior_mask, "maker"], df.loc[prior_mask, "taker"]])
                .dropna()
                .unique()
            )
            new_wallet_vol = 0.0
            for wlt, v in combined.items():
                if wlt and wlt not in prior_wallets:
                    new_wallet_vol += v
            feats[f"wallet_{h}h_new_usd"] = float(new_wallet_vol)
            feats[f"wallet_{h}h_new_share"] = (
                new_wallet_vol / total if total > 0 else math.nan
            )

    return feats


# ----------------------------------------------------------------------
# Weak labels (for PU learning / pre-label calibration)
# ----------------------------------------------------------------------

def weak_label(feats: dict) -> int:
    """Simple heuristic weak label: 1 if this market shows an insider-like pattern, else 0.

    Heuristic: volume-spike AND directional concentration AND sharp price move.
    """
    vol_spike = feats.get("vol_24h_spike_ratio", 0) or 0
    top1_share = feats.get("wallet_24h_top1_share", 0) or 0
    z_move = feats.get("price_24h_move_zscore", 0) or 0

    return int(
        vol_spike >= 3.0
        and top1_share >= 0.15
        and abs(z_move) >= 2.0
    )
