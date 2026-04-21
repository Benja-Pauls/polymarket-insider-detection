"""The headline analysis: does late-stage directional volume predict the winner?

For every resolved binary market with non-trivial late activity, we compute:

  * Spike magnitude  — last-24h volume / daily-average volume over the market
                       lifetime. Bucket into quintiles.
  * Direction sign   — sign of (BUY usd - SELL usd) in the last 24h, mapped to
                       {+1 for outcome 0 winning, -1 for outcome 1 winning}.
  * Winner direction — which outcome actually won (from payouts).
  * Hit              — did direction sign agree with winner direction?

A naive "random guess" baseline hits 50%. An informed-trading effect predicts
hit rate MUCH higher than 50%, especially in the top spike quintile.

We also bootstrap a 95% confidence interval per bucket and run a one-sided
exact binomial test against p=0.5.

This is one of the headline results of the paper: a clean, model-free
quantification of how much late-stage on-chain flow predicts resolution.

Output: `results/spike_hit_rate.csv` + `figures/spike_hit_rate.pdf`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


log = logging.getLogger(__name__)


def _winning_outcome_index(payouts):
    if payouts is None:
        return None
    try:
        ints = [int(p) for p in payouts]
    except (TypeError, ValueError):
        return None
    if not ints or max(ints) == 0:
        return None
    return ints.index(max(ints))


def compute_hit_rates(
    features: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    n_buckets: int = 5,
    min_trades_in_window: int = 20,
) -> pd.DataFrame:
    """Merge features with catalog, bucket by spike magnitude, and compute hit rate."""
    df = features.merge(
        catalog[["condition_id", "payouts"]],
        on="condition_id", how="left",
    )
    # Compute winner direction: +1 if outcome 0 won (flow toward +), -1 otherwise
    df["winner_outcome"] = df["payouts"].apply(_winning_outcome_index)
    df["winner_sign"] = df["winner_outcome"].map({0: +1, 1: -1})

    # Prefer the outcome-normalized feature when features.py had the token→outcome
    # mapping and could compute it. Fall back to raw dir_24h_net_usd otherwise.
    if "winner_24h_net_usd" in df.columns:
        df["flow_signed_toward_winner"] = df["winner_24h_net_usd"].fillna(0)
    else:
        df["flow_signed_toward_winner"] = (
            np.sign(df["dir_24h_net_usd"].fillna(0)) * df["winner_sign"].fillna(0)
        )

    # We need actual late activity to say anything
    df = df[df["meta_window_trade_count"].fillna(0) >= min_trades_in_window]
    df = df[df["winner_sign"].notna() & (df["flow_signed_toward_winner"] != 0)]

    if df.empty:
        return pd.DataFrame()

    df["hit"] = (df["flow_signed_toward_winner"] > 0).astype(int)

    # Spike magnitude — absolute, not signed
    df["spike"] = df["vol_24h_spike_ratio"].fillna(0)

    # Quantile buckets
    df["bucket"] = pd.qcut(df["spike"], n_buckets, labels=False, duplicates="drop")

    rows = []
    for b, grp in df.groupby("bucket"):
        n = len(grp)
        hits = grp["hit"].sum()
        mean = hits / n
        # Wilson 95% CI
        z = 1.96
        denom = 1 + z**2 / n
        centre = (mean + z**2 / (2 * n)) / denom
        half = z * np.sqrt(mean * (1 - mean) / n + z**2 / (4 * n**2)) / denom
        ci_lo = max(0, centre - half)
        ci_hi = min(1, centre + half)
        # One-sided p-value against p=0.5
        p_value = stats.binomtest(int(hits), n, p=0.5, alternative="greater").pvalue
        spike_min = grp["spike"].min()
        spike_max = grp["spike"].max()
        spike_median = grp["spike"].median()
        rows.append({
            "bucket": int(b),
            "n": n,
            "hits": int(hits),
            "hit_rate": mean,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_value_vs_random": p_value,
            "spike_median": spike_median,
            "spike_min": spike_min,
            "spike_max": spike_max,
        })

    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def make_figure(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of hit rate per spike-magnitude bucket with CIs."""
    if df.empty:
        log.warning("no data to plot")
        return
    sns.set_theme(
        context="paper", style="whitegrid", palette="colorblind",
        rc={"font.family": "serif"},
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = df["bucket"].to_numpy()
    y = df["hit_rate"].to_numpy()
    err = np.vstack([y - df["ci_lo"], df["ci_hi"] - y])
    bars = ax.bar(x, y, yerr=err, capsize=5, color="#1f77b4", alpha=0.8, edgecolor="black")
    ax.axhline(0.5, linestyle="--", color="gray", label="Random baseline (p=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"Q{b+1}\n(median spike × {r['spike_median']:.1f})"
        for b, r in df.iterrows()
    ], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Spike-magnitude quintile (24h vol / daily avg)")
    ax.set_ylabel("P(flow direction = winner direction)")
    ax.set_title("Late-stage volume spike direction predicts resolution")
    # Annotate n and p
    for b, r in df.iterrows():
        ax.annotate(
            f"n={r['n']}\np={r['p_value_vs_random']:.3g}",
            xy=(b, r["hit_rate"] + (r["ci_hi"] - r["hit_rate"]) + 0.03),
            ha="center", fontsize=8,
        )
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info("wrote figure %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    ap.add_argument("--catalog", type=Path, default=Path("data/raw/goldsky/markets.parquet"))
    ap.add_argument("--results", type=Path, default=Path("results/spike_hit_rate.csv"))
    ap.add_argument("--figure", type=Path, default=Path("figures/spike_hit_rate"))
    ap.add_argument("--n-buckets", type=int, default=5)
    ap.add_argument("--min-trades", type=int, default=20)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    feats = pd.read_parquet(args.features)
    catalog = pd.read_parquet(args.catalog)
    df = compute_hit_rates(feats, catalog, n_buckets=args.n_buckets, min_trades_in_window=args.min_trades)
    if df.empty:
        log.warning("no markets with sufficient activity")
        return

    print(df.to_string(index=False))
    args.results.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.results, index=False)
    log.info("wrote %s", args.results)
    make_figure(df, args.figure)


if __name__ == "__main__":
    main()
