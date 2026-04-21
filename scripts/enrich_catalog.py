"""Enrich the markets catalog with per-token outcome indices.

The first version of ``collect.py`` saved ``token_ids`` without mapping each
token to its ``outcome_index`` (0 = Yes, 1 = No in binary markets). This script
fills in the mapping post-hoc via ``orderbook.marketData`` (live subgraph)
with an ``orderbook_resync`` fallback, and writes an enriched parquet.

For each condition we also fetch BOTH outcome tokens (in case the catalog only
stored the higher-volume side), so downstream analysis sees the full picture.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pminsider.goldsky import GoldskyClient


def fetch_tokens_for_condition(g: GoldskyClient, condition_id: str) -> dict[str, int | None]:
    """Return {token_id: outcome_index} for all outcome tokens of this condition."""
    out: dict[str, int | None] = {}
    # Live subgraph first
    q = f"""
    {{ marketDatas(first: 10, where: {{condition: "{condition_id}"}}) {{
        id condition outcomeIndex
      }} }}
    """
    try:
        res = g.query("orderbook", q).data.get("marketDatas", []) or []
    except Exception:  # noqa: BLE001
        res = []
    for m in res:
        oi = m.get("outcomeIndex")
        out[m["id"]] = int(oi) if oi is not None else None

    # Resync fallback for outcome index if null
    missing = [t for t, oi in out.items() if oi is None]
    if missing:
        try:
            ids = "[" + ",".join(f'"{t}"' for t in missing) + "]"
            q2 = f"""
            {{ marketDatas(first: {len(missing)}, where: {{id_in: {ids}}}) {{
                id outcomeIndex
              }} }}
            """
            res2 = g.query("orderbook_resync", q2).data.get("marketDatas", []) or []
            for m in res2:
                oi = m.get("outcomeIndex")
                if oi is not None:
                    out[m["id"]] = int(oi)
        except Exception:  # noqa: BLE001
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", type=Path, default=Path("data/raw/goldsky/markets.parquet"))
    ap.add_argument("--out", type=Path, default=None, help="default: overwrite --catalog")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/raw/goldsky/_cache"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(message)s")
    log = logging.getLogger("enrich_catalog")

    df = pd.read_parquet(args.catalog)
    log.info("loaded %d markets", len(df))
    g = GoldskyClient(cache_dir=args.cache_dir)

    enriched_tokens_list: list[list[str]] = []
    token_to_outcome_list: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="enrich catalog"):
        cid = row["condition_id"]
        mapping = fetch_tokens_for_condition(g, cid)
        if not mapping:
            # Keep original tokens; mapping empty
            mapping = {t: None for t in (row["token_ids"] or [])}
        enriched_tokens_list.append(sorted(mapping.keys()))
        token_to_outcome_list.append(mapping)

    df["token_ids"] = enriched_tokens_list
    df["token_to_outcome"] = token_to_outcome_list

    out_path = args.out or args.catalog
    df.to_parquet(out_path, index=False)
    log.info("wrote %d rows → %s", len(df), out_path)

    with_both_tokens = df["token_ids"].apply(len).ge(2).sum()
    with_outcome_map = df["token_to_outcome"].apply(
        lambda d: any(v is not None for v in (d or {}).values())
    ).sum()
    log.info("markets with ≥2 tokens resolved: %d", with_both_tokens)
    log.info("markets with ≥1 outcome_index resolved: %d", with_outcome_map)


if __name__ == "__main__":
    main()
