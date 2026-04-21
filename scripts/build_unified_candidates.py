"""Merge all label-candidate sources into a single ranked table.

Inputs:
    data/labels/incidents/incidents_matched.parquet  (callouts + fuzzy-matched condition_id)
    data/labels/onchain_candidates.parquet           (on-chain miner output)
    data/labels/market_tradability.parquet           (Haiku insider-tradeability classifier)
    data/labels/whale_trackers/*.parquet             (optional, when scraped)

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

Pipeline:
  1. Load callouts (prefer incidents_matched.parquet for matched_condition_id)
  2. Load on-chain miner candidates, filter by --onchain-min-flags
  3. Drop on-chain candidates whose market was classified as non-insider-tradeable
     (sports, crypto-price, elections). Keep unknown/ambiguous for triage.
  4. Enrich via markets_metadata (question, slug, category, volume, endDate)
  5. Collapse (wallet, condition_id) duplicates — a callout row merged with
     its on-chain row keeps callout evidence URLs but gains trade-level context.
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
            "onchain_realized_profit_usd": None,
            "onchain_profit_ratio": None,
            "onchain_entry_vwap_on_winner": None,
        })
    return pd.DataFrame(rows)


def _tier_to_score(tier: str) -> float:
    return {"T1": 0.95, "T2": 0.75, "T3": 0.55}.get(tier or "T3", 0.5)


def _onchain_to_candidates(onchain_path: Path, min_flags: int = 4) -> pd.DataFrame:
    """Load on-chain miner output. Flags include realized_profit + informed_entry."""
    if not onchain_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(onchain_path)
    df = df[df["flags_count"] >= min_flags].copy()
    rows = []
    for _, r in df.iterrows():
        flag_pairs = [
            (r.get("flag_fresh_wallet"), "fresh_wallet"),
            (r.get("flag_single_market"), "single_market"),
            (r.get("flag_large_first_position"), "large_first_position"),
            (r.get("flag_realized_profit"), "realized_profit"),
            (r.get("flag_informed_entry"), "informed_entry"),
            (r.get("flag_early_timing"), "early_timing"),
        ]
        flags_str = ";".join(name for v, name in flag_pairs if bool(v))
        # Win-aligned: proxy from informed_entry OR realized_profit (both imply
        # the wallet held the eventual winner). Preserves the column name the
        # downstream curator and finalizer read.
        win_aligned = bool(r.get("flag_informed_entry")) or bool(r.get("flag_realized_profit"))
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
            "matched_question": None,
            "onchain_usd_in_market": float(r["wallet_usd_in_market"]),
            "onchain_wallet_concentration": float(r["wallet_concentration_in_this_market"]),
            "onchain_size_percentile": float(r["wallet_size_percentile_in_market"]),
            "onchain_win_aligned": win_aligned,
            "onchain_flags_count": int(r["flags_count"]),
            "onchain_realized_profit_usd": (
                float(r["realized_profit_usd"]) if pd.notna(r.get("realized_profit_usd")) else None
            ),
            "onchain_profit_ratio": (
                float(r["profit_ratio"]) if pd.notna(r.get("profit_ratio")) else None
            ),
            "onchain_entry_vwap_on_winner": (
                float(r["entry_vwap_on_winner"]) if pd.notna(r.get("entry_vwap_on_winner")) else None
            ),
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


def _apply_tradability_filter(
    df: pd.DataFrame,
    tradability_path: Path,
    *,
    log=log,
) -> pd.DataFrame:
    """Drop on-chain candidates whose market is not insider-tradeable.

    Keeps:
      - all callout rows (they already cleared a news-based filter)
      - on-chain rows whose market is in {tradeable_*, unknown, ambiguous}
      - on-chain rows whose condition_id is not in the tradability catalog
        at all (no classification → cannot reject)

    Drops:
      - on-chain rows where is_insider_tradeable=False AND category starts
        with ``not_tradeable_`` (sports, crypto-price, elections, weather, social).
    """
    if not tradability_path.exists():
        log.warning("tradability catalog missing: %s — skipping filter", tradability_path)
        return df
    td = pd.read_parquet(tradability_path)
    log.info("loaded tradability catalog: %d markets", len(td))
    # Build the set of condition_ids to DROP
    drop_mask = (td["is_insider_tradeable"] == False) & (  # noqa: E712
        td["category_tradability"].fillna("").str.startswith("not_tradeable_")
    )
    drop_cids = set(td.loc[drop_mask, "condition_id"].tolist())
    log.info("tradability catalog: %d markets flagged not_tradeable_*", len(drop_cids))

    before = len(df)
    # Only filter on-chain (callouts we keep regardless)
    is_callout = df["source"].eq("callout") if "source" in df.columns else False
    in_drop = df["condition_id"].isin(drop_cids)
    keep = ~in_drop | is_callout
    out = df[keep].copy()
    dropped = before - len(out)
    log.info("tradability filter: %d → %d rows  (dropped %d non-tradeable)",
             before, len(out), dropped)
    return out


_TIER_ORDER = {"T1": 0, "T2": 1, "T3": 2}


def _merge_pre_curation_duplicates(df: pd.DataFrame, *, log=log) -> pd.DataFrame:
    """Collapse rows that share (wallet, condition_id) BEFORE the LLM pass.

    When a callout row and an on-chain row point at the same wallet+market:
      - Keep callout evidence_urls and incident_id
      - Gain on-chain trade-level context (flags_count, concentration, profit)
      - Mark source as "callout;onchain"
      - Take max suspicion_score, best (T1 > T2 > T3) confidence_tier
      - Sum n_supporting_sources
      - Use callout candidate_id (more stable for cache reuse on callout side)
    """
    if df.empty or "wallet" not in df.columns or "condition_id" not in df.columns:
        return df

    has_pair = df["wallet"].notna() & df["condition_id"].notna()
    pair_df = df[has_pair].copy()
    standalone = df[~has_pair].copy()
    if pair_df.empty:
        return df

    # Normalize wallet case so the same wallet (e.g. EIP-55 mixed case vs lowercased
    # from Goldsky) collapses into a single group. We only use the lowercase form
    # for grouping — we preserve the original wallet field in the output row.
    pair_df["_wallet_key"] = pair_df["wallet"].astype(str).str.lower()

    merged_rows = []
    merges_applied = 0
    for (wallet_key, cid), g in pair_df.groupby(["_wallet_key", "condition_id"], dropna=False):
        if len(g) == 1:
            merged_rows.append(g.iloc[0].to_dict())
            continue
        merges_applied += 1
        # Sort so callout rows come first — they win on identity-bearing fields
        callout_rows = g[g["source"] == "callout"]
        onchain_rows = g[g["source"] == "onchain"]
        base = callout_rows.iloc[0].to_dict() if not callout_rows.empty else g.iloc[0].to_dict()
        row = dict(base)

        # Combine sources
        srcs = sorted({s for s in g["source"].dropna().tolist() if s})
        row["source"] = ";".join(srcs)
        row["sub_source"] = ";".join(sorted({
            s for s in g["sub_source"].dropna().tolist() if s
        }))

        # Best suspicion score + tier
        row["suspicion_score"] = float(g["suspicion_score"].max())
        tiers = [t for t in g["confidence_tier"].dropna().tolist() if t]
        if tiers:
            row["confidence_tier"] = min(tiers, key=lambda t: _TIER_ORDER.get(t, 999))

        # Sum supporting sources
        row["n_supporting_sources"] = int(g["n_supporting_sources"].fillna(0).sum())

        # Callout-side evidence wins
        if not callout_rows.empty:
            ev = callout_rows["evidence_urls"].iloc[0]
            if isinstance(ev, str) and ev:
                row["evidence_urls"] = ev
            iid = callout_rows["incident_id"].iloc[0]
            if pd.notna(iid):
                row["incident_id"] = iid
            mq = callout_rows["market_question"].iloc[0]
            if isinstance(mq, str) and mq:
                row["market_question"] = mq
            # Preserve callout time-window
            for c in ("ts_lower", "ts_upper", "direction", "outcome_resolved"):
                v = callout_rows[c].iloc[0] if c in callout_rows.columns else None
                if pd.notna(v):
                    row[c] = v

        # On-chain trade-level fields win when callout was NULL
        if not onchain_rows.empty:
            oc = onchain_rows.iloc[0]
            for col in (
                "onchain_usd_in_market", "onchain_wallet_concentration",
                "onchain_size_percentile", "onchain_win_aligned",
                "onchain_flags_count", "onchain_realized_profit_usd",
                "onchain_profit_ratio", "onchain_entry_vwap_on_winner",
            ):
                if col in onchain_rows.columns:
                    v = oc.get(col)
                    if pd.notna(v):
                        row[col] = v
            # Inherit the richer flag string if callout was empty
            if (not row.get("flags")) or row.get("flags") == "":
                row["flags"] = oc.get("flags", row.get("flags"))
            else:
                # Concatenate: "tier=T1;n_sources=5 | fresh_wallet;realized_profit;..."
                row["flags"] = f"{row.get('flags','')} | {oc.get('flags','')}"
            # Use the largest observed size
            sz_max = g["size_usd_approx"].dropna().max() if "size_usd_approx" in g.columns else None
            if pd.notna(sz_max):
                row["size_usd_approx"] = float(sz_max)

        # Record the merge
        row["merged_candidate_ids"] = json.dumps(g["candidate_id"].tolist())
        # Normalize wallet to lowercase so downstream trade lookups / merges match
        row["wallet"] = wallet_key
        # Drop the grouping helper before writing
        row.pop("_wallet_key", None)
        merged_rows.append(row)

    log.info("pre-curation merges applied: %d", merges_applied)
    merged_df = pd.DataFrame(merged_rows)
    if "_wallet_key" in merged_df.columns:
        merged_df = merged_df.drop(columns=["_wallet_key"])
    out = pd.concat([merged_df, standalone], ignore_index=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--incidents", type=Path, default=Path("data/labels/incidents/incidents.parquet"))
    ap.add_argument("--onchain", type=Path, default=Path("data/labels/onchain_candidates.parquet"))
    ap.add_argument("--trackers-dir", type=Path, default=Path("data/labels/whale_trackers"))
    ap.add_argument("--metadata", type=Path, default=Path("data/offchain/markets_metadata.parquet"),
                    help="GH-Actions-proxied Polymarket Gamma API snapshot")
    ap.add_argument("--tradability", type=Path,
                    default=Path("data/labels/market_tradability.parquet"),
                    help="Haiku insider-tradeability classifier output")
    ap.add_argument("--out", type=Path, default=Path("data/labels/unified_candidates.parquet"))
    ap.add_argument("--review-out", type=Path, default=Path("data/labels/unified_candidates_review.csv"))
    ap.add_argument("--onchain-min-flags", type=int, default=4,
                    help="Minimum number of on-chain flags required for a candidate")
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
    log.info("concat: %d rows  (by source: %s)",
             len(df), df["source"].value_counts().to_dict())

    # ---- TRADABILITY FILTER -----
    # Drop on-chain rows whose market is not insider-tradeable (sports, crypto-price, elections)
    df = _apply_tradability_filter(df, args.tradability, log=log)

    # Enrich with markets_metadata (question, slug, category, volume, endDate)
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

    # ---- PRE-CURATION MERGE -----
    # Collapse (wallet, condition_id) pairs that appear in both callout + on-chain
    df = _merge_pre_curation_duplicates(df, log=log)

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
        "onchain_win_aligned", "onchain_realized_profit_usd", "onchain_profit_ratio",
        "flags", "ts_lower", "ts_upper", "merged_candidate_ids",
        "evidence_urls",
    ]
    review_cols = [c for c in review_cols if c in df.columns]
    df.head(args.top_review)[review_cols].to_csv(args.review_out, index=False)
    log.info("wrote top-%d review CSV → %s", args.top_review, args.review_out)


if __name__ == "__main__":
    main()
