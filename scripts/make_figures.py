"""Regenerate every figure used in the paper.

Usage:
    python scripts/make_figures.py \
        --features data/processed/features.parquet \
        --labels data/processed/labels.parquet \
        --models-dir models \
        --trades-dir data/raw/goldsky/trades \
        --catalog data/raw/goldsky/markets.parquet \
        --figures-dir figures
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pminsider.models import make_xy, POSITIVE_LABELS
from pminsider.viz import (
    fig_calibration,
    fig_feature_distributions,
    fig_feature_importance,
    fig_market_timeline,
    fig_roc_pr_comparison,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    ap.add_argument("--labels", type=Path, default=Path("data/processed/labels.parquet"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--trades-dir", type=Path, default=Path("data/raw/goldsky/trades"))
    ap.add_argument("--catalog", type=Path, default=Path("data/raw/goldsky/markets.parquet"))
    ap.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("make_figures")

    feats = pd.read_parquet(args.features)
    labels = pd.read_parquet(args.labels)
    catalog = pd.read_parquet(args.catalog)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure: timelines for the top 3 highest-scoring markets -----
    X, y, split, cid = make_xy(feats, labels)
    # Find flagged-positive markets with trades — pick top 3 by label_confidence
    pos_markets = labels[labels["label"].isin(POSITIVE_LABELS)].nlargest(3, "label_confidence")
    for _, r in pos_markets.iterrows():
        tp = args.trades_dir / f"{r['condition_id']}.parquet"
        if not tp.exists() or tp.stat().st_size == 0:
            continue
        trades = pd.read_parquet(tp)
        match = catalog[catalog["condition_id"] == r["condition_id"]]
        if match.empty:
            continue
        rt = int(match["resolution_timestamp"].iloc[0])
        fig_market_timeline(
            trades,
            resolution_timestamp=rt,
            title=f"Flagged market {r['condition_id'][:10]}… (confidence={r['label_confidence']:.2f})",
            out_path=args.figures_dir / f"timeline_{r['condition_id'][:12]}",
        )

    # ---- Figure: feature distributions (key features) ----------------
    key_features = [
        "vol_24h_spike_ratio",
        "vol_6h_spike_ratio",
        "dir_24h_signed_imbalance",
        "wallet_24h_top1_share",
        "wallet_24h_new_share",
        "price_24h_move_zscore",
    ]
    key_features = [c for c in key_features if c in feats.columns]
    merged = feats.merge(labels[["condition_id", "label"]], on="condition_id", how="inner")
    if len(key_features):
        fig_feature_distributions(
            merged, features_to_plot=key_features,
            out_path=args.figures_dir / "feature_distributions",
        )

    # ---- Figure: ROC / PR comparison + calibration -------------------
    runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mp in sorted(args.models_dir.glob("*.joblib")):
        name = mp.stem
        est = joblib.load(mp)
        X_test = X[split == "test"]
        y_test = y[split == "test"].values
        if hasattr(est, "predict_proba"):
            s = est.predict_proba(X_test)[:, 1]
        elif hasattr(est, "decision_function"):
            raw = est.decision_function(X_test)
            rng = raw.max() - raw.min()
            s = (raw - raw.min()) / rng if rng > 0 else np.zeros_like(raw)
        else:
            s = est.predict(X_test).astype(float)
        runs[name] = (y_test, s)

    if runs:
        fig_roc_pr_comparison(runs, out_path=args.figures_dir / "roc_pr_comparison")
        fig_calibration(runs, out_path=args.figures_dir / "calibration")

    # ---- Figure: feature importance for best model -------------------
    # Pick the first tree model that has a CSV dump
    for candidate in ["lightgbm", "xgboost", "randforest", "logreg"]:
        fi_path = args.models_dir / f"{candidate}__feature_importance.csv"
        if fi_path.exists():
            fi = pd.read_csv(fi_path, index_col=0).iloc[:, 0]
            fig_feature_importance(
                fi, top_k=20, title=f"{candidate}: top 20 features",
                out_path=args.figures_dir / f"feature_importance_{candidate}",
            )
            break

    log.info("wrote figures to %s", args.figures_dir)


if __name__ == "__main__":
    main()
