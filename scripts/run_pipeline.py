"""End-to-end pipeline: catalog → trades → enrich → features → labels → models → figures.

This is the reproducibility entry point. Running it with a freshly-cloned repo
produces every artifact required to compile the paper, subject to network
access to api.goldsky.com and the public Polygon RPCs.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    log = logging.getLogger("pipeline")
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-catalog",  action="store_true")
    ap.add_argument("--skip-trades",   action="store_true")
    ap.add_argument("--skip-enrich",   action="store_true")
    ap.add_argument("--skip-features", action="store_true")
    ap.add_argument("--skip-labels",   action="store_true")
    ap.add_argument("--skip-models",   action="store_true")
    ap.add_argument("--skip-figures",  action="store_true")
    ap.add_argument("--min-volume",    type=float, default=50_000)
    ap.add_argument("--top-n",         type=int,   default=3_000)
    ap.add_argument("--window-days",   type=int,   default=14)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    root = Path(__file__).resolve().parent.parent
    py = sys.executable

    if not args.skip_catalog:
        run([py, "-m", "pminsider.collect", "--phase", "catalog",
             "--min-volume", str(args.min_volume),
             "--top-n", str(args.top_n)])

    if not args.skip_trades:
        run([py, "-m", "pminsider.collect", "--phase", "trades",
             "--window-days", str(args.window_days)])

    if not args.skip_enrich:
        run([py, str(root / "scripts" / "enrich_catalog.py")])

    if not args.skip_features:
        run([py, str(root / "scripts" / "build_features.py")])

    if not args.skip_labels:
        run([py, str(root / "scripts" / "build_labels.py")])

    if not args.skip_models:
        run([py, str(root / "scripts" / "train_models.py")])
        run([py, str(root / "scripts" / "spike_hit_rate_analysis.py")])

    if not args.skip_figures:
        run([py, str(root / "scripts" / "make_figures.py")])

    log = logging.getLogger("pipeline")
    log.info("pipeline complete — artifacts in data/processed, models/, results/, figures/")


if __name__ == "__main__":
    main()
