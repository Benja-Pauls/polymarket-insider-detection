"""Classify Polymarket markets by whether they are *possible* to insider-trade.

Two modes:
  --mode catalog   classify the 269 on-chain markets in
                   data/raw/goldsky/markets.parquet (join to offchain metadata
                   for the question text).
                   Output: data/labels/market_tradability.parquet

  --mode full      classify the top N (default 5000) offchain markets in
                   data/offchain/markets_metadata.parquet filtered by
                   volume > $10K. Output:
                   data/labels/market_tradability_full.parquet

Uses Claude Haiku 4.5 with prompt caching, batch size 20, disk cache.

Sanity check:
  --sanity         run a fixed set of five known-answer questions and print
                   the classifications (no parquet write).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv

    # Look up the Stock_Portfolio .env first (per environment convention) then
    # fall back to a local .env if present.
    load_dotenv("/Users/ben_paulson/Documents/Personal/Stock_Portfolio/.env")
    load_dotenv()  # also look in cwd / parents for any repo-local overrides
except ImportError:  # pragma: no cover
    pass

# Make src/ importable when running via `python scripts/...`.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from pminsider.curate.market_tradability import (  # noqa: E402
    TradabilityClassifier,
    classify_questions,
)

log = logging.getLogger(__name__)

DEFAULT_GOLDSKY = _REPO / "data" / "raw" / "goldsky" / "markets.parquet"
DEFAULT_METADATA = _REPO / "data" / "offchain" / "markets_metadata.parquet"
DEFAULT_CATALOG_OUT = _REPO / "data" / "labels" / "market_tradability.parquet"
DEFAULT_FULL_OUT = _REPO / "data" / "labels" / "market_tradability_full.parquet"
DEFAULT_CACHE_DIR = _REPO / "data" / "labels" / "_tradability_cache"


# --- data loaders ----------------------------------------------------------


def load_catalog_markets(
    goldsky_path: Path, metadata_path: Path
) -> pd.DataFrame:
    """Join goldsky markets with offchain metadata to get question text."""
    g = pd.read_parquet(goldsky_path, columns=["condition_id", "total_volume_usd"])
    m = pd.read_parquet(
        metadata_path, columns=["conditionId", "question", "volume"]
    ).rename(columns={"conditionId": "condition_id"})

    # De-dup metadata: some condition_ids appear multiple times; prefer the row
    # with the highest offchain volume (most complete record).
    m["volume_num"] = pd.to_numeric(m["volume"], errors="coerce").fillna(0.0)
    m = m.sort_values("volume_num", ascending=False).drop_duplicates(
        subset=["condition_id"], keep="first"
    )

    merged = g.merge(
        m[["condition_id", "question"]], on="condition_id", how="left"
    )

    # Normalize blank questions to NaN so we can detect "missing".
    merged["question"] = merged["question"].where(
        merged["question"].astype(str).str.strip() != "", other=pd.NA
    )
    return merged


def load_full_markets(metadata_path: Path, top_n: int, min_volume: float) -> pd.DataFrame:
    m = pd.read_parquet(
        metadata_path, columns=["conditionId", "question", "volume"]
    ).rename(columns={"conditionId": "condition_id"})
    m["volume_num"] = pd.to_numeric(m["volume"], errors="coerce").fillna(0.0)
    m = m[m["volume_num"] > min_volume]
    m = m.dropna(subset=["question"])
    m = m[m["question"].astype(str).str.strip() != ""]
    m = m.sort_values("volume_num", ascending=False)
    # Collapse duplicate condition_ids — keep highest-volume row.
    m = m.drop_duplicates(subset=["condition_id"], keep="first")
    m = m.head(top_n).reset_index(drop=True)
    return m[["condition_id", "question", "volume_num"]]


# --- run pipeline ----------------------------------------------------------


def run_classification(
    df: pd.DataFrame,
    classifier: TradabilityClassifier,
    *,
    batch_size: int,
) -> pd.DataFrame:
    """Classify rows in `df`. Rows with missing `question` get `unknown`.

    `df` must have columns `condition_id` and `question` (question may be NA).
    Returns a DataFrame with the required output schema.
    """
    has_q = df["question"].notna()
    to_classify = df[has_q].reset_index(drop=True)
    no_q = df[~has_q].reset_index(drop=True)

    if len(to_classify):
        questions = to_classify["question"].astype(str).tolist()
        results = classify_questions(
            questions, batch_size=batch_size, classifier=classifier
        )
        assert len(results) == len(questions), (
            f"classifier returned {len(results)} results for "
            f"{len(questions)} questions"
        )
        classified = to_classify.copy()
        classified["category_tradability"] = [r.category_tradability for r in results]
        classified["is_insider_tradeable"] = [r.is_insider_tradeable for r in results]
        classified["confidence"] = [r.confidence for r in results]
        classified["reasoning"] = [r.reasoning for r in results]
        classified["llm_model"] = [r.model for r in results]
    else:
        classified = to_classify.copy()
        for col in [
            "category_tradability",
            "is_insider_tradeable",
            "confidence",
            "reasoning",
            "llm_model",
        ]:
            classified[col] = []

    # Fill the missing-question rows with the `unknown` placeholder.
    if len(no_q):
        no_q = no_q.copy()
        no_q["question"] = no_q["question"].fillna("").astype(str)
        no_q["category_tradability"] = "unknown"
        no_q["is_insider_tradeable"] = False
        no_q["confidence"] = 0.0
        no_q["reasoning"] = "No question text available in metadata."
        no_q["llm_model"] = "n/a"

    out = pd.concat([classified, no_q], ignore_index=True)
    # Keep only the schema columns in the expected order.
    keep = [
        "condition_id",
        "question",
        "category_tradability",
        "is_insider_tradeable",
        "confidence",
        "reasoning",
        "llm_model",
    ]
    return out[keep]


# --- sanity check ----------------------------------------------------------

SANITY_CASES = [
    ("Maduro out by January 31, 2026?", "tradeable_geopolitical"),
    ("Andy Byron out as Astronomer CEO by next Friday?", "tradeable_corporate"),
    (
        "Will d4vd be the #1 searched person on Google this year?",
        "tradeable_awards",
    ),
    ("Will the San Jose Sharks win the 2025 Stanley Cup?", "not_tradeable_sports"),
    (
        "Will the price of Bitcoin be above $96,000 on December 2?",
        "not_tradeable_price",
    ),
]


def run_sanity(classifier: TradabilityClassifier) -> bool:
    questions = [q for q, _ in SANITY_CASES]
    results = classify_questions(questions, batch_size=20, classifier=classifier, progress=False)
    all_ok = True
    print("\nSANITY CHECK")
    print("=" * 100)
    for (q, expected), r in zip(SANITY_CASES, results):
        ok = r.category_tradability == expected
        tag = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(
            f"[{tag}] expected={expected:<35s} got={r.category_tradability:<35s} "
            f"conf={r.confidence:.2f}"
        )
        print(f"       Q: {q}")
        print(f"       R: {r.reasoning}")
    print("=" * 100)
    return all_ok


# --- distribution helper ---------------------------------------------------


def print_distribution(df: pd.DataFrame, title: str) -> Counter:
    ctr = Counter(df["category_tradability"].tolist())
    total = sum(ctr.values())
    print(f"\n{title} — {total} markets classified")
    print("-" * 60)
    for cat, n in sorted(ctr.items(), key=lambda x: (-x[1], x[0])):
        pct = 100.0 * n / total if total else 0.0
        print(f"  {cat:<38s} {n:>5d}  ({pct:5.1f}%)")
    n_tradeable = int(df["is_insider_tradeable"].sum())
    print(f"\n  insider-tradeable: {n_tradeable} / {total} "
          f"({100.0 * n_tradeable / total:.1f}%)")
    return ctr


# --- main ------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["catalog", "full", "sanity"],
        default="catalog",
    )
    p.add_argument("--top-n", type=int, default=5000)
    p.add_argument("--min-volume", type=float, default=10_000.0)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--goldsky", type=Path, default=DEFAULT_GOLDSKY)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--catalog-out", type=Path, default=DEFAULT_CATALOG_OUT)
    p.add_argument("--full-out", type=Path, default=DEFAULT_FULL_OUT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument(
        "--max-cost",
        type=float,
        default=5.0,
        help="Abort (in full mode) if running estimate would exceed this many USD.",
    )
    p.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip the sanity check even in catalog/full mode.",
    )
    args = p.parse_args()

    clf = TradabilityClassifier(cache_dir=args.cache_dir)

    # Sanity first — cheap and quick, validates the prompt.
    if args.mode == "sanity" or not args.skip_sanity:
        ok = run_sanity(clf)
        if args.mode == "sanity":
            print(f"\nTotal cost so far: ${clf.cost.cost_usd:.4f} across {clf.cost.calls} call(s) "
                  f"(cache hits: {clf.cost.cache_hits})")
            return 0 if ok else 1

    if args.mode == "catalog":
        print("\nLoading 269 catalog markets from goldsky + offchain metadata...")
        df = load_catalog_markets(args.goldsky, args.metadata)
        print(f"  {len(df)} markets loaded; "
              f"{df['question'].isna().sum()} missing question text")
        out = run_classification(df, clf, batch_size=args.batch_size)
        args.catalog_out.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.catalog_out, index=False)
        print(f"\nWrote {args.catalog_out}")
        print_distribution(out, "Catalog markets distribution")

    elif args.mode == "full":
        print(
            f"\nLoading top-{args.top_n} offchain markets with "
            f"volume > ${args.min_volume:,.0f}..."
        )
        full = load_full_markets(args.metadata, args.top_n, args.min_volume)
        print(f"  {len(full)} markets loaded.")
        n_batches = (len(full) + args.batch_size - 1) // args.batch_size
        # Rough estimate: ~$0.02/batch at ~800 input + ~1500 output tokens per batch
        # after the first cached call. Conservative.
        est = 0.02 * n_batches
        print(
            f"  {n_batches} batches of {args.batch_size} — rough cost estimate "
            f"${est:.2f} (cap ${args.max_cost:.2f})"
        )
        if est > args.max_cost:
            print(
                f"ERROR: estimated cost ${est:.2f} exceeds --max-cost "
                f"${args.max_cost:.2f}. Lower --top-n or raise --max-cost."
            )
            return 2

        df = full[["condition_id", "question"]].copy()
        out = run_classification(df, clf, batch_size=args.batch_size)
        args.full_out.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.full_out, index=False)
        print(f"\nWrote {args.full_out}")
        print_distribution(out, "Full markets distribution")

    usage = clf.cost
    print(
        f"\nLLM usage — calls={usage.calls} cache_hits={usage.cache_hits} "
        f"questions={usage.questions_classified}"
    )
    print(
        f"  tokens: input={usage.input_tokens} output={usage.output_tokens} "
        f"cache_read={usage.cache_read_tokens} "
        f"cache_creation={usage.cache_creation_tokens}"
    )
    print(f"  total cost: ${usage.cost_usd:.4f}")
    for m, c in usage.by_model.items():
        print(f"    {m}: ${c:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
