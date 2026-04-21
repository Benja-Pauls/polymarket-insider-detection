"""Merge per-source enriched callouts into canonical incidents."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pminsider.labels.canonicalize import (
    canonicalize,
    load_allegations_from_sources,
    save_incidents,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", type=Path, nargs="+", default=[
        Path("data/labels/news/enriched.parquet"),
        Path("data/labels/reddit/enriched.parquet"),
        Path("data/labels/twitter/enriched.parquet"),
    ])
    ap.add_argument("--out-dir", type=Path, default=Path("data/labels/incidents"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("canonicalize")

    allegations = load_allegations_from_sources(args.sources)
    log.info("total allegations to canonicalize: %d", len(allegations))
    if allegations.empty:
        log.warning("no allegations found")
        return

    incidents = canonicalize(allegations)
    log.info("canonical incidents: %d (from %d sources)", len(incidents), len(allegations))
    save_incidents(incidents, args.out_dir)
    log.info("wrote %s", args.out_dir)

    # Summary
    from collections import Counter
    tiers = Counter(i.confidence_tier for i in incidents)
    log.info("tier distribution: %s", dict(tiers))
    multi = sum(1 for i in incidents if i.n_sources >= 2)
    log.info("incidents with multiple sources: %d", multi)


if __name__ == "__main__":
    main()
