"""Publication figures for the insider-detection paper.

Every figure is emitted as a PDF (vector, crisp in LaTeX) and a PNG (preview,
for the README). Matplotlib style uses the Tableau-inspired ``tableau-colorblind10``
palette for accessibility.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_recall_curve, roc_curve

log = logging.getLogger(__name__)

sns.set_theme(
    context="paper",
    style="whitegrid",
    palette="colorblind",
    rc={
        "figure.figsize": (6.5, 4.0),
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    },
)


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s (.pdf, .png)", out_path)


# ----------------------------------------------------------------------
# Figure 1: price + volume timeline for a single market (anchor case)
# ----------------------------------------------------------------------

def fig_market_timeline(
    trades: pd.DataFrame,
    *,
    resolution_timestamp: int,
    title: str,
    out_path: Path,
) -> None:
    """Dual-axis timeline: price (top) + 1h bucketed USDC volume (bottom)."""
    if trades.empty:
        log.warning("  skip %s: no trades", title)
        return
    df = trades.sort_values("timestamp").copy()
    df["hours_to_resolution"] = -(resolution_timestamp - df["timestamp"]) / 3600.0

    buckets = df.assign(
        h_bucket=np.floor(df["hours_to_resolution"]).astype(int)
    )
    bucket_vol = buckets.groupby(["h_bucket", "side"])["usd_spent_usdc"].sum().unstack(fill_value=0)

    fig, (ax_p, ax_v) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})

    # Price (trade-level)
    ax_p.plot(df["hours_to_resolution"], df["price"], linewidth=0.8, alpha=0.7, color="#1f77b4")
    ax_p.scatter(df["hours_to_resolution"], df["price"], s=3, alpha=0.3, color="#1f77b4")
    ax_p.set_ylabel("Implied probability")
    ax_p.set_ylim(0, 1)
    ax_p.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_p.axvline(0, linestyle="--", color="gray", alpha=0.6)
    ax_p.set_title(title, fontsize=11)

    # Volume (hourly, stacked by side)
    if "BUY" in bucket_vol.columns:
        ax_v.bar(bucket_vol.index, bucket_vol["BUY"], width=0.9, color="#2ca02c", label="BUY")
    if "SELL" in bucket_vol.columns:
        bottom = bucket_vol.get("BUY", pd.Series(0, index=bucket_vol.index))
        ax_v.bar(bucket_vol.index, bucket_vol["SELL"], bottom=bottom, width=0.9, color="#d62728", label="SELL")
    ax_v.set_xlabel("Hours to resolution")
    ax_v.set_ylabel("Hourly USDC volume")
    ax_v.axvline(0, linestyle="--", color="gray", alpha=0.6)
    ax_v.legend(loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout()
    _save(fig, out_path)


# ----------------------------------------------------------------------
# Figure 2: feature distributions — positives vs negatives
# ----------------------------------------------------------------------

def fig_feature_distributions(
    features_with_labels: pd.DataFrame,
    features_to_plot: list[str],
    out_path: Path,
) -> None:
    """One small panel per feature with positive/negative kernel-density."""
    positives_mask = features_with_labels["label"].isin(["weak_positive", "strong_positive"])
    n = len(features_to_plot)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.0 * rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(features_to_plot):
        ax = axes[i]
        pos = features_with_labels.loc[positives_mask, col].dropna()
        neg = features_with_labels.loc[~positives_mask, col].dropna()
        if len(pos) >= 2:
            sns.kdeplot(pos, ax=ax, label="positive", color="#d62728", fill=True, alpha=0.3)
        if len(neg) >= 2:
            sns.kdeplot(neg, ax=ax, label="negative", color="#1f77b4", fill=True, alpha=0.3)
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    for ax in axes[len(features_to_plot):]:
        ax.axis("off")

    fig.tight_layout()
    _save(fig, out_path)


# ----------------------------------------------------------------------
# Figure 3: ROC + PR curves across all models
# ----------------------------------------------------------------------

def fig_roc_pr_comparison(
    runs: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    out_path: Path,
) -> None:
    """``runs``: {model_name: (y_true, y_score)} on the test set."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4.2))

    for name, (y, s) in runs.items():
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        from sklearn.metrics import average_precision_score, roc_auc_score
        roc_auc = roc_auc_score(y, s)
        pr_auc = average_precision_score(y, s)
        ax_roc.plot(fpr, tpr, label=f"{name}  AUC={roc_auc:.3f}")
        ax_pr.plot(rec, prec, label=f"{name}  AP={pr_auc:.3f}")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    ax_roc.set_xlabel("False-positive rate")
    ax_roc.set_ylabel("True-positive rate")
    ax_roc.set_title("ROC curves (test set)")
    ax_roc.legend(fontsize=8)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-recall (test set)")
    ax_pr.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out_path)


# ----------------------------------------------------------------------
# Figure 4: feature-importance bar chart (top K from best model)
# ----------------------------------------------------------------------

def fig_feature_importance(
    importance: pd.Series,
    top_k: int = 20,
    title: str = "Feature importance",
    out_path: Path | None = None,
) -> None:
    top = importance.head(top_k)
    fig, ax = plt.subplots(figsize=(7, 0.35 * top_k + 1))
    sns.barplot(x=top.values, y=top.index, ax=ax, orient="h", color="#1f77b4")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    if out_path is not None:
        _save(fig, out_path)


# ----------------------------------------------------------------------
# Figure 5: calibration reliability diagram
# ----------------------------------------------------------------------

def fig_calibration(
    runs: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    n_bins: int = 10,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, s) in runs.items():
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(s, bins) - 1
        xs, ys = [], []
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() < 3:
                continue
            xs.append(s[mask].mean())
            ys.append(y[mask].mean())
        ax.plot(xs, ys, "o-", label=name, markersize=5)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("Calibration (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)
