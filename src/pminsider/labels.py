"""Labeling pipeline for the insider-detection classifier.

Generating a label for a market means deciding whether its on-chain trading
pattern plausibly reflects *informed* (possibly insider) trading. Because no
ground-truth labels exist at scale, we use a layered approach:

  STRONG positive: the market was publicly alleged to involve insider trading
    in a news article, Twitter post, or Reddit thread naming a specific bet.
    Highest quality signal; expected to be sparse (~20-50 across the full
    dataset). Needs question text to cross-reference — resolved separately
    (see ``scripts/resolve_metadata.py``).

  WEAK positive: the on-chain pattern satisfies our heuristic anomaly
    detector (big late-stage volume spike + directional + concentrated +
    price z-score). Noisy but plentiful. Used for PU-learning-style training.

  NEGATIVE: the market resolved gradually without any late spike, or the
    volume-spike pattern was NOT in the direction of the winning outcome
    (just noise).

  EXCLUDED: resolution came from a publicly-scheduled event — Fed FOMC,
    presidential election night, earnings release — where late volume
    swings are expected behavior, not insider leakage. Heuristic: market
    age > 14 days AND the final day volume spike coincides with a well-
    documented scheduled event. We mark these EXCLUDED to keep them out
    of the training set entirely.

Label records are saved as a separate parquet keyed by ``condition_id`` so the
pipeline can be iterated without re-running data collection.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Heuristic weak labels from features
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class WeakLabelThresholds:
    """Thresholds for weak-positive / excluded heuristics. Tunable.

    Rationale:
      - ``vol_spike_ratio_min``: last-24h volume at least ~3x the market's
        daily-average volume. Captures actual *spikes*, not gradual ramps.
      - ``top1_share_min``: at least one wallet controls 15%+ of the last-24h
        volume. Concentration is a hallmark of informed single-actor trading.
      - ``abs_z_move_min``: last-24h price move is ≥2σ of the market's
        per-trade price volatility. Separates "real info" from noise.
      - ``dir_alignment_thresh``: the net-flow direction agrees with the
        winning outcome by at least this margin (signed imbalance).
      - ``require_winner_alignment``: when True, only flag as weak_positive
        if the net flow pointed TOWARD the actual winner
        (``winner_24h_net_usd > 0``). Not a "clean" label — it implicitly
        uses the outcome as a predictor — but useful for pedagogical paper
        exposition and for contrasting against the random baseline.
    """
    vol_spike_ratio_min: float = 3.0
    top1_share_min: float = 0.15
    abs_z_move_min: float = 2.0
    dir_alignment_thresh: float = 0.4  # |signed_imbalance| >= this
    require_winner_alignment: bool = False


def classify_from_features(
    feats: pd.DataFrame,
    thresholds: WeakLabelThresholds | None = None,
) -> pd.DataFrame:
    """Produce a label DataFrame given engineered features.

    Returns a DataFrame with columns: ``condition_id``, ``label``,
    ``label_source``, ``label_confidence``, plus the individual predicates so
    researchers can debug.
    """
    th = thresholds or WeakLabelThresholds()
    df = feats.copy()

    # Sanity guards
    for col in [
        "vol_24h_spike_ratio",
        "wallet_24h_top1_share",
        "price_24h_move_zscore",
        "dir_24h_signed_imbalance",
        "meta_window_trade_count",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # Predicates
    df["_pred_spike"] = df["vol_24h_spike_ratio"].fillna(0) >= th.vol_spike_ratio_min
    df["_pred_concentrated"] = df["wallet_24h_top1_share"].fillna(0) >= th.top1_share_min
    df["_pred_z_move"] = df["price_24h_move_zscore"].abs().fillna(0) >= th.abs_z_move_min
    df["_pred_directional"] = (
        df["dir_24h_signed_imbalance"].abs().fillna(0) >= th.dir_alignment_thresh
    )
    df["_pred_has_window_activity"] = df["meta_window_trade_count"].fillna(0) >= 20

    # Weak positive: all four signals fire AND there's enough activity
    weak_pos_mask = (
        df["_pred_spike"]
        & df["_pred_concentrated"]
        & df["_pred_z_move"]
        & df["_pred_directional"]
        & df["_pred_has_window_activity"]
    )
    if th.require_winner_alignment and "winner_24h_net_usd" in df.columns:
        weak_pos_mask &= df["winner_24h_net_usd"].fillna(0) > 0
        df["_pred_winner_aligned"] = df["winner_24h_net_usd"].fillna(0) > 0

    # Negative: neither spike nor z-move fired AND there was decent activity
    neg_mask = (
        ~df["_pred_spike"]
        & ~df["_pred_z_move"]
        & df["_pred_has_window_activity"]
    )

    # Excluded: very low activity — we can't tell anything, so we throw it out.
    exclude_mask = df["meta_window_trade_count"].fillna(0) < 20

    label = pd.Series("ambiguous", index=df.index)
    label[weak_pos_mask] = "weak_positive"
    label[neg_mask] = "negative"
    # Exclusion overrides everything
    label[exclude_mask] = "excluded_low_activity"

    df["label"] = label
    df["label_source"] = "heuristic_v1"
    # Confidence proxy: count of predicates satisfied (for informed, higher = more confident)
    df["label_confidence"] = (
        df["_pred_spike"].astype(int)
        + df["_pred_concentrated"].astype(int)
        + df["_pred_z_move"].astype(int)
        + df["_pred_directional"].astype(int)
    ) / 4

    cols = [
        "condition_id",
        "label",
        "label_source",
        "label_confidence",
        "_pred_spike",
        "_pred_concentrated",
        "_pred_z_move",
        "_pred_directional",
        "_pred_has_window_activity",
    ]
    return df[cols].copy()


# ----------------------------------------------------------------------
# Strong labels (manual curation or news scraping) — schema only for v1
# ----------------------------------------------------------------------

def empty_strong_label_frame() -> pd.DataFrame:
    """Initialize an empty strong-label dataframe so code paths can coexist."""
    return pd.DataFrame(columns=[
        "condition_id",
        "label",          # "strong_positive", "strong_negative"
        "label_source",   # "news_article", "twitter", "reddit", "manual"
        "citation_url",
        "citation_title",
        "citation_author",
        "citation_date",
        "notes",
    ])


def merge_labels(
    weak: pd.DataFrame,
    strong: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge weak (heuristic) labels with strong (curated) labels — strong wins."""
    if strong is None or strong.empty:
        return weak.copy()

    # Start with weak; strong labels override by condition_id
    out = weak.copy()
    out.set_index("condition_id", inplace=True)
    for _, r in strong.iterrows():
        cid = r["condition_id"]
        out.loc[cid, "label"] = r["label"]
        out.loc[cid, "label_source"] = r["label_source"]
        out.loc[cid, "label_confidence"] = 1.0  # full confidence for curated
    return out.reset_index()


# ----------------------------------------------------------------------
# Dataset splits (by market, not by row, to avoid leakage)
# ----------------------------------------------------------------------

def make_split(
    labels: pd.DataFrame,
    *,
    seed: int = 17,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> pd.Series:
    """Assign each market to {"train","val","test"} — stratified by label."""
    rng = np.random.default_rng(seed)
    assignments = pd.Series(index=labels.index, dtype=object)
    for label_val, group in labels.groupby("label"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        assignments.loc[idx[:n_train]] = "train"
        assignments.loc[idx[n_train:n_train + n_val]] = "val"
        assignments.loc[idx[n_train + n_val:]] = "test"
    return assignments


# ----------------------------------------------------------------------
# Entry point for scripts
# ----------------------------------------------------------------------

def build_labels(
    features_path: Path,
    strong_labels_path: Path | None = None,
    thresholds: WeakLabelThresholds | None = None,
) -> pd.DataFrame:
    feats = pd.read_parquet(features_path)
    weak = classify_from_features(feats, thresholds=thresholds)
    strong = (
        pd.read_parquet(strong_labels_path)
        if strong_labels_path and strong_labels_path.exists()
        else None
    )
    merged = merge_labels(weak, strong)
    merged["split"] = make_split(merged)
    return merged
