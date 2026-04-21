"""Train and evaluate all models defined in pminsider.models.

Usage:
    python scripts/train_models.py \
        --features data/processed/features.parquet \
        --labels data/processed/labels.parquet \
        --models-dir models \
        --results results/model_results.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pminsider.models import run_all_models


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.parquet"))
    ap.add_argument("--labels", type=Path, default=Path("data/processed/labels.parquet"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--results", type=Path, default=Path("results/model_results.csv"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    summary = run_all_models(
        features_path=args.features,
        labels_path=args.labels,
        models_dir=args.models_dir,
        results_path=args.results,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
