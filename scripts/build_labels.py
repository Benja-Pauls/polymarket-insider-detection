"""Build the labels parquet from engineered features (and optionally curated labels).

Usage:
    python scripts/build_labels.py \
        --features data/processed/features.parquet \
        --out data/processed/labels.parquet \
        [--strong-labels data/strong_labels.parquet]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pminsider.labels import build_labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/labels.parquet"))
    ap.add_argument("--strong-labels", type=Path, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("build_labels")

    labels = build_labels(args.features, strong_labels_path=args.strong_labels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.out, index=False)
    log.info("wrote %d rows → %s", len(labels), args.out)
    log.info("label distribution:")
    for label, n in labels["label"].value_counts().items():
        log.info("  %-25s %d", label, n)
    log.info("split distribution:")
    for s, n in labels["split"].value_counts().items():
        log.info("  %-10s %d", s, n)


if __name__ == "__main__":
    main()
