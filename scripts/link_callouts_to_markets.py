"""Fuzzy-match callout incidents to on-chain condition_ids by question text.

Each callout has a natural-language market question extracted by the LLM.
Each Gamma metadata row has a canonical question. We do a simple token-overlap
Jaccard match and pick the best candidate above a threshold.

The output enriches the incidents parquet with `matched_condition_id` so
downstream unified_candidates can join. Also writes
data/labels/incidents/incidents_matched.parquet.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


_STOPWORDS = {
    "will", "the", "a", "an", "by", "in", "on", "of", "to", "for", "and", "or",
    "be", "is", "are", "would", "should", "could", "can", "do", "does", "did",
    "has", "have", "had", "with", "at", "as", "that", "this", "it", "its",
    "what", "which", "who", "when", "where", "why", "how",
}


def _tokenize(q: str) -> set:
    if not q:
        return set()
    q = q.lower()
    tokens = set(re.findall(r"[a-z0-9]+", q))
    tokens -= _STOPWORDS
    return {t for t in tokens if len(t) >= 3}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def match_one(question: str, md_df: pd.DataFrame, top_k: int = 3, threshold: float = 0.3) -> list[dict]:
    """Return the top-K best market matches above threshold."""
    q_tokens = _tokenize(question)
    if not q_tokens:
        return []
    scores = []
    for _, m in md_df.iterrows():
        mq = m.get("question") or ""
        score = _jaccard(q_tokens, _tokenize(mq))
        if score >= threshold:
            scores.append({
                "conditionId": m.get("conditionId"),
                "question": mq,
                "volume": m.get("volume"),
                "endDate": m.get("endDate"),
                "score": score,
            })
    scores.sort(key=lambda x: -x["score"])
    return scores[:top_k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--incidents", type=Path, default=Path("data/labels/incidents/incidents.parquet"))
    ap.add_argument("--metadata", type=Path, default=Path("data/offchain/markets_metadata.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/labels/incidents/incidents_matched.parquet"))
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")

    incidents = pd.read_parquet(args.incidents)
    md = pd.read_parquet(args.metadata)
    log.info("incidents: %d   metadata rows: %d", len(incidents), len(md))

    md_slim = md[["conditionId", "question", "volume", "endDate"]]
    md_slim = md_slim[md_slim["question"].notna()].reset_index(drop=True)

    out_rows = []
    for _, inc in tqdm(incidents.iterrows(), total=len(incidents), desc="matching"):
        matches = match_one(str(inc.get("market_question", "") or ""), md_slim,
                            top_k=args.top_k, threshold=args.threshold)
        best = matches[0] if matches else None
        r = inc.to_dict()
        r["matched_condition_id"] = best["conditionId"] if best else None
        r["matched_question"] = best["question"] if best else None
        r["matched_score"] = best["score"] if best else None
        r["alt_matches_json"] = json.dumps(matches[1:]) if len(matches) > 1 else ""
        out_rows.append(r)

    out = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    matched = out["matched_condition_id"].notna().sum()
    log.info("wrote %d incidents, %d with matched condition_id (%.1f%%)",
             len(out), matched, 100 * matched / max(1, len(out)))


if __name__ == "__main__":
    main()
