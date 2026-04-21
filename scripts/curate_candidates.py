"""Curate the unified-candidates parquet with LLM review.

Pipeline:
  1. Load unified_candidates.parquet
  2. For each row, build a dossier (context bundle)
  3. Pass 1 — Sonnet: verdict every row
  4. Pass 2 — Opus: re-review anything Sonnet marked {confirmed, suspected, ambiguous}
     (Opus skips rows Sonnet rejected, saving cost)
  5. Resolve merges: when two rows reference each other, collapse them
  6. Write curated.parquet + review CSV

Concurrency: ThreadPoolExecutor with configurable workers. The Anthropic
SDK is thread-safe; we use 8-16 concurrent requests by default.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv("/Users/ben_paulson/Documents/Personal/Stock_Portfolio/.env")
except ImportError:  # pragma: no cover
    pass

from pminsider.curate.dossier import DossierContext, build_dossier
from pminsider.curate.reviewer import Curator, SONNET_MODEL, OPUS_MODEL, Verdict


log = logging.getLogger(__name__)


def _review_one(curator: Curator, candidate: dict, ctx: DossierContext, model: str) -> Verdict:
    dossier = build_dossier(candidate, ctx)
    return curator.review(
        candidate_id=candidate["candidate_id"],
        dossier=dossier,
        model=model,
    )


def review_all(
    candidates: list[dict],
    curator: Curator,
    ctx: DossierContext,
    *,
    model: str,
    workers: int = 8,
    desc: str = "review",
) -> list[Verdict]:
    out: list[Verdict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_review_one, curator, c, ctx, model): c["candidate_id"]
            for c in candidates
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                out.append(fut.result())
            except Exception as e:  # noqa: BLE001
                log.warning("review failed for %s: %s", futures[fut], e)
    return out


def resolve_merges(verdicts: list[Verdict]) -> dict[str, str]:
    """Build a union-find over merge claims and return {candidate_id: canonical_id}."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Always pick the lexicographically smaller id as the root for determinism
        root, child = (ra, rb) if ra < rb else (rb, ra)
        parent[child] = root

    for v in verdicts:
        parent.setdefault(v.candidate_id, v.candidate_id)
        for other in v.merge_with_candidate_ids:
            if not isinstance(other, str) or not other:
                continue
            parent.setdefault(other, other)
            union(v.candidate_id, other)

    return {cid: find(cid) for cid in parent}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, default=Path("data/labels/unified_candidates.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/labels/curated_candidates.parquet"))
    ap.add_argument("--review-csv", type=Path, default=Path("data/labels/curated_review.csv"))
    ap.add_argument("--catalog", type=Path, default=Path("data/raw/goldsky/markets.parquet"))
    ap.add_argument("--metadata", type=Path, default=Path("data/offchain/markets_metadata.parquet"))
    ap.add_argument("--incidents", type=Path, default=Path("data/labels/incidents/incidents.parquet"))
    ap.add_argument("--trades-dir", type=Path, default=Path("data/raw/goldsky/trades"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-usd", type=float, default=200.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="only curate the top-N by suspicion_score")
    ap.add_argument("--skip-opus", action="store_true",
                    help="do the Sonnet pass only (cheaper)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")

    # Load candidates
    df = pd.read_parquet(args.candidates)
    df = df.sort_values("suspicion_score", ascending=False).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    log.info("candidates to curate: %d", len(df))

    # Load context
    log.info("building dossier context (catalog/metadata/incidents/trades)…")
    ctx = DossierContext.load(
        catalog_path=args.catalog,
        metadata_path=args.metadata,
        incidents_path=args.incidents,
        trades_dir=args.trades_dir,
    )
    log.info("context ready: %d catalog markets, %d metadata rows, %d incidents",
             len(ctx.catalog), len(ctx.metadata), len(ctx.incidents))

    curator = Curator()

    # ------ PASS 1: Sonnet ------
    log.info("pass 1 (Sonnet) on %d rows…", len(df))
    sonnet_verdicts = review_all(
        df.to_dict("records"), curator, ctx,
        model=SONNET_MODEL, workers=args.workers, desc="sonnet",
    )
    by_id = {v.candidate_id: v for v in sonnet_verdicts}
    sonnet_cost = curator.cost.cost_usd
    log.info("pass 1 complete: cost=$%.2f  calls=%d  cache_hits=%d",
             sonnet_cost, curator.cost.calls, curator.cost.cache_hits)

    pass1_counts = {}
    for v in sonnet_verdicts:
        pass1_counts[v.verdict] = pass1_counts.get(v.verdict, 0) + 1
    log.info("pass 1 verdict distribution: %s", pass1_counts)

    # ------ PASS 2: Opus escalation ------
    final_verdicts: dict[str, Verdict] = {v.candidate_id: v for v in sonnet_verdicts}
    if not args.skip_opus:
        escalate = [
            c for c in df.to_dict("records")
            if by_id.get(c["candidate_id"]) and
               by_id[c["candidate_id"]].verdict in ("confirmed", "suspected", "ambiguous")
        ]
        if curator.cost.cost_usd >= args.budget_usd:
            log.warning("budget reached after Sonnet; skipping Opus pass")
        else:
            log.info("pass 2 (Opus) on %d escalated rows…", len(escalate))
            opus_verdicts = review_all(
                escalate, curator, ctx,
                model=OPUS_MODEL, workers=max(1, args.workers // 2),
                desc="opus",
            )
            for v in opus_verdicts:
                final_verdicts[v.candidate_id] = v
            opus_cost = curator.cost.cost_usd - sonnet_cost
            log.info("pass 2 complete: cost=$%.2f", opus_cost)
            if curator.cost.cost_usd >= args.budget_usd:
                log.warning("budget cap $%.2f reached", args.budget_usd)

    # ------ MERGE RESOLUTION ------
    all_verdicts = list(final_verdicts.values())
    merge_map = resolve_merges(all_verdicts)

    # ------ WRITE OUTPUT ------
    log.info("writing outputs…")
    verdict_rows = []
    for v in all_verdicts:
        verdict_rows.append({
            "candidate_id": v.candidate_id,
            "canonical_candidate_id": merge_map.get(v.candidate_id, v.candidate_id),
            "final_verdict": v.verdict,
            "confidence_tier_final": v.confidence_tier_final,
            "final_reasoning": v.reasoning,
            "merge_with_candidate_ids": json.dumps(v.merge_with_candidate_ids),
            "coordinated_with_candidate_ids": json.dumps(v.coordinated_with_candidate_ids),
            "strongest_evidence": json.dumps(v.strongest_evidence),
            "concerns": json.dumps(v.concerns),
            "model_used": v.model,
        })
    verdict_df = pd.DataFrame(verdict_rows)

    # Join back to the original candidates data
    merged_df = df.merge(verdict_df, on="candidate_id", how="left")
    merged_df.to_parquet(args.out, index=False)
    log.info("wrote %d curated rows → %s", len(merged_df), args.out)

    # Human-review CSV: keep only the columns a reviewer wants to see
    csv_cols = [
        "candidate_id", "canonical_candidate_id", "source",
        "final_verdict", "confidence_tier_final",
        "model_used", "final_reasoning",
        "wallet", "condition_id", "market_question",
        "size_usd_approx", "direction", "outcome_resolved",
        "onchain_flags_count", "onchain_wallet_concentration",
        "onchain_size_percentile", "onchain_win_aligned",
        "suspicion_score", "strongest_evidence", "concerns",
        "merge_with_candidate_ids", "coordinated_with_candidate_ids",
    ]
    csv_cols = [c for c in csv_cols if c in merged_df.columns]

    args.review_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df[csv_cols].to_csv(args.review_csv, index=False)
    log.info("wrote review CSV → %s", args.review_csv)

    # Final distribution
    final_counts = merged_df["final_verdict"].value_counts().to_dict()
    log.info("final verdict distribution: %s", final_counts)
    log.info("TOTAL curator cost: $%.2f  (Sonnet $%.2f, Opus $%.2f)",
             curator.cost.cost_usd,
             curator.cost.by_model.get(SONNET_MODEL, 0),
             curator.cost.by_model.get(OPUS_MODEL, 0))


if __name__ == "__main__":
    main()
