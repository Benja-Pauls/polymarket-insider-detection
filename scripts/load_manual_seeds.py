"""Convert the hand-curated manual_seed_cases.json into the same enriched-callout
schema that the LLM extractor produces, so canonicalize_callouts.py can merge
them seamlessly with news / reddit / twitter extractions.

Each manual case becomes one synthetic EnrichedCallout per evidence URL, with
is_allegation=True and confidence_tier from the manual record.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=Path, default=Path("data/labels/manual_seed_cases.json"))
    ap.add_argument("--out", type=Path, default=Path("data/labels/manual/enriched.parquet"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("manual_seeds")

    with args.seeds.open() as f:
        payload = json.load(f)
    cases = payload.get("cases", [])
    log.info("manual seed cases: %d", len(cases))

    rows = []
    for c in cases:
        # Pick a canonical direction; strip annotations like "YES (Maria Corina Machado)"
        direction = c.get("direction")
        if direction and direction.upper().startswith("YES"):
            direction = "YES"
        elif direction and direction.upper().startswith("NO"):
            direction = "NO"
        else:
            direction = None

        wallet_candidates = c.get("wallet_candidates") or []
        primary_wallet = wallet_candidates[0] if wallet_candidates else None

        urls = c.get("evidence_urls") or [""]
        if not urls:
            urls = [""]

        # One row per evidence URL so canonicalizer sees multi-source corroboration
        for url in urls:
            rows.append({
                "raw_source": "news",
                "raw_source_id": f"manual::{c['case_id']}::{url[:40]}",
                "raw_source_url": url,
                "raw_author": None,
                "raw_created_utc": 0,
                "raw_title": c.get("short_name") or c.get("case_id"),
                "raw_body": (c.get("notes") or "") + "\n\nManual-curated seed case.",
                "raw_score": 0,
                "raw_num_replies": 0,
                "raw_parent_id": None,
                "raw_raw_metadata": "",
                # Extracted fields — same schema as LLM output
                "is_allegation": True,
                "market_question": c["market_question"],
                "ts_lower": c.get("ts_lower_iso"),
                "ts_upper": c.get("ts_upper_iso"),
                "size_usd_approx": c.get("size_usd_invested") or c.get("size_usd_payout"),
                "wallet_addr": primary_wallet,
                "direction": direction,
                "outcome_resolved": c.get("outcome_resolved"),
                "confidence_tier": c.get("confidence_tier", "T1"),
                "quote": f"[Manual seed: {c.get('short_name','?')}] handle={c.get('handle','?')}, size_invested=${c.get('size_usd_invested','?')}, size_payout=${c.get('size_usd_payout','?')}; event={c.get('event_description','?')}",
                "reasoning": c.get("notes", ""),
                "extractor_model": "manual-curation",
                "extractor_timestamp_iso": payload.get("last_updated_iso", ""),
                "extraction_cost_usd": 0.0,
                # Pass-through manual-only fields
                "manual_case_id": c["case_id"],
                "manual_handle": c.get("handle"),
                "manual_wallet_candidates": ";".join(wallet_candidates),
                "manual_size_usd_invested": c.get("size_usd_invested"),
                "manual_size_usd_payout": c.get("size_usd_payout"),
                "manual_event_public_ts_iso": c.get("event_public_ts_iso"),
            })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    log.info("wrote %d synthetic enriched callouts to %s", len(df), args.out)

    # Also write a review CSV for easy human check
    review_cols = [
        "manual_case_id", "market_question", "ts_lower", "ts_upper",
        "size_usd_approx", "wallet_addr", "direction", "outcome_resolved",
        "confidence_tier", "manual_handle", "raw_source_url",
    ]
    df[review_cols].drop_duplicates(subset=["manual_case_id", "raw_source_url"]).to_csv(
        args.out.parent / "review_manual_seeds.csv", index=False,
    )
    log.info("wrote review CSV to %s", args.out.parent / "review_manual_seeds.csv")


if __name__ == "__main__":
    main()
