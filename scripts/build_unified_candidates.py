"""Merge all label-candidate sources into a single ranked table.

Inputs:
    data/labels/incidents/incidents.parquet     (canonical callout incidents)
    data/labels/onchain_candidates.parquet      (on-chain miner output)
    data/labels/whale_trackers/*.parquet        (optional, when scraped)

Output:
    data/labels/unified_candidates.parquet
    data/labels/unified_candidates_review.csv   (human-readable top-N for manual curation)

Each unified row is a (wallet?, market?, time_window) tuple with:
  - candidate_id   stable hash
  - source         news | reddit | manual | onchain | whale_tracker
  - suspicion_score in [0, 1]
  - flags          list of human-readable reasons
  - evidence_urls  list of source URLs
  - known_handle   optional Polymarket handle (Burdensome-Mix, etc.)
  - known_wallet   optional 0x address
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _stable_id(*parts) -> str:
    key = json.dumps([str(p) for p in parts], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _incidents_to_candidates(incidents_path: Path) -> pd.DataFrame:
    """Load either incidents.parquet OR the preferred incidents_matched.parquet."""
    # Prefer the matched version (has matched_condition_id populated)
    matched_path = incidents_path.parent / "incidents_matched.parquet"
    path = matched_path if matched_path.exists() else incidents_path
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    has_match_col = "matched_condition_id" in df.columns
    rows = []
    for _, r in df.iterrows():
        wallets = []
        try:
            wallets = json.loads(r["wallet_addrs"]) if r.get("wallet_addrs") else []
        except Exception:  # noqa: BLE001
            wallets = []
        citations = []
        try:
            citations = json.loads(r["citations"]) if r.get("citations") else []
        except Exception:  # noqa: BLE001
            citations = []
        urls = [c.get("source_url", "") for c in citations if c.get("source_url")]

        # Use matched condition_id when available
        cid = r.get("matched_condition_id") if has_match_col else None
        if isinstance(cid, float):
            cid = None  # NaN

        rows.append({
            "candidate_id": _stable_id(
                "incident", r["incident_id"], ",".join(wallets), r.get("market_question", ""),
            ),
            "source": "callout",
            "sub_source": ";".join({c.get("source", "") for c in citations if c.get("source")}),
            "wallet": wallets[0] if wallets else None,
            "all_wallets": ";".join(wallets),
            "condition_id": cid,
            "market_question": r.get("market_question", ""),
            "ts_lower": r.get("ts_lower"),
            "ts_upper": r.get("ts_upper"),
            "size_usd_approx": r.get("size_usd_approx"),
            "direction": r.get("direction"),
            "outcome_resolved": r.get("outcome_resolved"),
            "confidence_tier": r.get("confidence_tier"),
            "n_supporting_sources": r.get("n_sources"),
            "suspicion_score": _tier_to_score(r.get("confidence_tier", "T3")),
            "flags": f"tier={r.get('confidence_tier')};n_sources={r.get('n_sources')}",
            "evidence_urls": ";".join(urls[:5]),
            "incident_id": r.get("incident_id"),
            "matched_question": r.get("matched_question") if has_match_col else None,
            "onchain_usd_in_market": None,
            "onchain_wallet_concentration": None,
            "onchain_size_percentile": None,
            "onchain_win_aligned": None,
            "onchain_flags_count": None,
        })
    return pd.DataFrame(rows)


def _tier_to_score(tier: str) -> float:
    return {"T1": 0.95, "T2": 0.75, "T3": 0.55}.get(tier or "T3", 0.5)


def _onchain_to_candidates(onchain_path: Path, min_flags: int = 4) -> pd.DataFrame:
    if not onchain_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(onchain_path)
    df = df[df["flags_count"] >= min_flags].copy()
    rows = []
    for _, r in df.iterrows():
        flags_str = ";".join([
            f
            for f, name in [
                (r["flag_fresh_wallet"], "fresh_wallet"),
                (r["flag_single_market"], "single_market"),
                (r["flag_large_first_position"], "large_first_position"),
                (r["flag_win_aligned_top"], "win_aligned_top"),
                (r["flag_directional"], "directional"),
                (r["flag_early_timing"], "early_timing"),
            ]
            if f
            for f in [name]  # trick to use `name` as the value when True
        ])
        rows.append({
            "candidate_id": _stable_id("onchain", r["wallet"], r["condition_id"]),
            "source": "onchain",
            "sub_source": "miner",
            "wallet": r["wallet"],
            "all_wallets": r["wallet"],
            "condition_id": r["condition_id"],
            "market_question": None,
            "ts_lower": None,
            "ts_upper": None,
            "size_usd_approx": float(r["wallet_usd_in_market"]),
            "direction": r.get("wallet_flow_direction"),
            "outcome_resolved": None,
            "confidence_tier": None,
            "n_supporting_sources": 1,
            "suspicion_score": float(r["suspicion_score"]),
            "flags": flags_str,
            "evidence_urls": "",
            "incident_id": None,
            "onchain_usd_in_market": float(r["wallet_usd_in_market"]),
            "onchain_wallet_concentration": float(r["wallet_concentration_in_this_market"]),
            "onchain_size_percentile": float(r["wallet_size_percentile_in_market"]),
            "onchain_win_aligned": bool(r["wallet_win_aligned"]),
            "onchain_flags_count": int(r["flags_count"]),
        })
    return pd.DataFrame(rows)


def _whale_trackers_to_candidates(trackers_dir: Path) -> pd.DataFrame:
    """Placeholder — filled once whale_trackers scraper lands."""
    if not trackers_dir.exists():
        return pd.DataFrame()
    frames = []
    for p in trackers_dir.glob("*.parquet"):
        try:
            frames.append(pd.read_parquet(p))
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--incidents", type=Path, default=Path("data/labels/incidents/incidents.parquet"))
    ap.add_argument("--onchain", type=Path, default=Path("data/labels/onchain_candidates.parquet"))
    ap.add_argument("--trackers-dir", type=Path, default=Path("data/labels/whale_trackers"))
    ap.add_argument("--metadata", type=Path, default=Path("data/offchain/markets_metadata.parquet"),
                    help="GH-Actions-proxied Polymarket Gamma API snapshot")
    ap.add_argument("--out", type=Path, default=Path("data/labels/unified_candidates.parquet"))
    ap.add_argument("--review-out", type=Path, default=Path("data/labels/unified_candidates_review.csv"))
    ap.add_argument("--onchain-min-flags", type=int, default=5)
    ap.add_argument("--top-review", type=int, default=500)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("unify")

    frames = []
    a = _incidents_to_candidates(args.incidents)
    if not a.empty:
        log.info("callout-based candidates: %d", len(a))
        frames.append(a)
    b = _onchain_to_candidates(args.onchain, min_flags=args.onchain_min_flags)
    if not b.empty:
        log.info("on-chain candidates (flags ≥ %d): %d", args.onchain_min_flags, len(b))
        frames.append(b)
    c = _whale_trackers_to_candidates(args.trackers_dir)
    if not c.empty:
        log.info("whale-tracker candidates: %d", len(c))
        frames.append(c)

    if not frames:
        log.warning("no candidate sources available")
        return

    df = pd.concat(frames, ignore_index=True)

    # Enrich on-chain rows with market_question via metadata
    if args.metadata.exists():
        md = pd.read_parquet(args.metadata)
        md_slim = md[["conditionId", "question", "slug", "category", "volume", "endDate"]].rename(
            columns={"conditionId": "condition_id", "question": "md_question",
                     "slug": "md_slug", "category": "md_category",
                     "volume": "md_volume", "endDate": "md_endDate"}
        )
        df = df.merge(md_slim, on="condition_id", how="left")
        # Fill market_question from metadata when on-chain rows don't have it
        df["market_question"] = df["market_question"].fillna(df.get("md_question"))
        log.info("joined metadata — markets with question text: %d / %d",
                 df["market_question"].notna().sum(), len(df))

    df = df.sort_values("suspicion_score", ascending=False).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("wrote %d unified candidates → %s", len(df), args.out)

    # Source breakdown
    log.info("by source: %s", df["source"].value_counts().to_dict())
    log.info("score summary: mean=%.3f median=%.3f", df["suspicion_score"].mean(), df["suspicion_score"].median())

    # Human review CSV — a manageable top-N with the key columns
    review_cols = [
        "source", "suspicion_score", "confidence_tier", "n_supporting_sources",
        "wallet", "condition_id", "market_question",
        "size_usd_approx", "direction", "outcome_resolved",
        "onchain_flags_count", "onchain_wallet_concentration", "onchain_size_percentile",
        "onchain_win_aligned", "flags", "ts_lower", "ts_upper",
        "evidence_urls",
    ]
    review_cols = [c for c in review_cols if c in df.columns]
    df.head(args.top_review)[review_cols].to_csv(args.review_out, index=False)
    log.info("wrote top-%d review CSV → %s", args.top_review, args.review_out)


if __name__ == "__main__":
    main()
