"""Dossier builder: compile per-candidate context for LLM review.

A dossier packs everything an LLM needs to verdict a single candidate row:
  * the candidate's structured fields and scoring flags
  * market metadata (question, category, resolution, payouts)
  * the wallet's top trades in THIS market (timestamps, prices, sizes)
  * the wallet's cross-market footprint (other markets, total activity)
  * any callouts that match by wallet address OR by market question

The dossier is rendered as a single text blob (markdown-ish) that we feed
to Claude. All expensive joins (wallet → trades, market → metadata) are
pre-computed once in the ``DossierContext``; ``build_dossier`` is cheap per call.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _isofmt(ts: int | float | None) -> str:
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return "?"
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, OSError, OverflowError):
        return "?"


def _safe_float(v) -> float:
    try:
        if v is None:
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class DossierContext:
    catalog: pd.DataFrame           # data/raw/goldsky/markets.parquet
    metadata: pd.DataFrame          # data/offchain/markets_metadata.parquet (if present)
    incidents: pd.DataFrame         # data/labels/incidents/incidents.parquet
    trades_dir: Path
    # Pre-computed
    _market_metadata_by_cid: dict[str, dict] = field(default_factory=dict)
    _incidents_by_wallet: dict[str, list[dict]] = field(default_factory=dict)
    _trades_cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        catalog_path: Path = Path("data/raw/goldsky/markets.parquet"),
        metadata_path: Path = Path("data/offchain/markets_metadata.parquet"),
        incidents_path: Path = Path("data/labels/incidents/incidents.parquet"),
        trades_dir: Path = Path("data/raw/goldsky/trades"),
    ) -> "DossierContext":
        catalog = pd.read_parquet(catalog_path) if catalog_path.exists() else pd.DataFrame()
        metadata = pd.read_parquet(metadata_path) if metadata_path.exists() else pd.DataFrame()
        incidents = pd.read_parquet(incidents_path) if incidents_path.exists() else pd.DataFrame()
        ctx = cls(
            catalog=catalog, metadata=metadata, incidents=incidents,
            trades_dir=Path(trades_dir),
        )
        ctx._build_indices()
        return ctx

    def _build_indices(self) -> None:
        # Market metadata by condition_id
        if not self.metadata.empty:
            for r in self.metadata[["conditionId", "question", "slug", "category",
                                    "volume", "endDate", "startDate", "outcomes",
                                    "outcomePrices"]].to_dict("records"):
                cid = r.get("conditionId")
                if cid:
                    self._market_metadata_by_cid[cid] = r
        # Callouts by wallet address (lowercase-normalized)
        if not self.incidents.empty:
            for r in self.incidents.to_dict("records"):
                wallets = []
                try:
                    raw = r.get("wallet_addrs") or ""
                    wallets = json.loads(raw) if isinstance(raw, str) and raw else []
                except Exception:  # noqa: BLE001
                    wallets = []
                for w in wallets:
                    if isinstance(w, str) and w.lower().startswith("0x"):
                        self._incidents_by_wallet.setdefault(w.lower(), []).append(r)

    def trades_for(self, condition_id: str) -> pd.DataFrame:
        if condition_id in self._trades_cache:
            return self._trades_cache[condition_id]
        path = self.trades_dir / f"{condition_id}.parquet"
        if not path.exists() or path.stat().st_size < 100:
            self._trades_cache[condition_id] = pd.DataFrame()
            return self._trades_cache[condition_id]
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            df = pd.DataFrame()
        self._trades_cache[condition_id] = df
        return df


def build_dossier(candidate: dict, ctx: DossierContext) -> str:
    """Render a markdown-style dossier for one candidate."""
    lines: list[str] = []

    cid = candidate.get("condition_id")
    if isinstance(cid, float) and np.isnan(cid):
        cid = None
    wallet = candidate.get("wallet")
    if isinstance(wallet, float) and np.isnan(wallet):
        wallet = None
    source = candidate.get("source", "?")

    # -------- HEADER --------
    lines.append(f"# Candidate {candidate.get('candidate_id','?')}")
    lines.append(f"**Source:** {source}")
    lines.append(f"**Heuristic suspicion score:** {candidate.get('suspicion_score','?')}")
    if source == "onchain":
        lines.append(f"**On-chain flags fired:** {candidate.get('flags','') or '(none)'}")
        lines.append(f"**Flags count:** {candidate.get('onchain_flags_count','?')} of 6")
    if source == "callout":
        lines.append(f"**Confidence tier (callout):** {candidate.get('confidence_tier','?')}")
        lines.append(f"**Supporting sources:** {candidate.get('n_supporting_sources','?')}")
        urls = candidate.get("evidence_urls", "") or ""
        if urls:
            lines.append(f"**Evidence URLs:** {urls}")

    lines.append("")
    lines.append("## Candidate fields")
    for k in [
        "wallet", "condition_id", "market_question",
        "size_usd_approx", "direction", "outcome_resolved",
        "ts_lower", "ts_upper",
        "onchain_usd_in_market", "onchain_wallet_concentration",
        "onchain_size_percentile", "onchain_win_aligned",
    ]:
        v = candidate.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            lines.append(f"- `{k}`: {v}")

    # -------- MARKET METADATA --------
    md = ctx._market_metadata_by_cid.get(cid) if cid else None
    if md:
        lines.append("\n## Market metadata")
        lines.append(f"- **Question (Polymarket, verbatim):** {md.get('question','?')}")
        lines.append(f"- **Category:** {md.get('category','?')}")
        lines.append(f"- **Slug:** `{md.get('slug','?')}`")
        lines.append(f"- **Volume (Gamma):** ${_safe_float(md.get('volume')):,.0f}")
        lines.append(f"- **Start:** {md.get('startDate','?')}")
        lines.append(f"- **End:** {md.get('endDate','?')}")
        lines.append(f"- **Outcomes:** {md.get('outcomes','?')}")
        lines.append(f"- **Final prices:** {md.get('outcomePrices','?')}")

    # Also grab from our catalog for resolution timestamp and payouts
    if cid and not ctx.catalog.empty:
        m = ctx.catalog[ctx.catalog["condition_id"] == cid]
        if not m.empty:
            m = m.iloc[0]
            resolution_ts = m.get("resolution_timestamp")
            lines.append(f"- **Resolution (on-chain):** {_isofmt(resolution_ts)}")
            lines.append(f"- **Payouts:** {list(m['payouts']) if m.get('payouts') is not None else '?'}")
            lines.append(f"- **Outcome slot count:** {m.get('outcome_slot_count','?')}")
            lines.append(f"- **Catalog total volume:** ${_safe_float(m.get('total_volume_usd')):,.0f}")
            lines.append(f"- **Catalog total trades:** {m.get('total_trades','?')}")
            lines.append(f"- **Source tier (collection):** {m.get('source_tier','?')}")

    # -------- TRADES DETAIL --------
    if wallet and cid:
        trades = ctx.trades_for(cid)
        if not trades.empty:
            w = wallet.lower()
            wt = trades[
                (trades["maker"].fillna("").str.lower() == w) |
                (trades["taker"].fillna("").str.lower() == w)
            ].copy()
            if not wt.empty:
                wt["role"] = np.where(
                    wt["maker"].fillna("").str.lower() == w,
                    "maker", "taker",
                )
                wt = wt.sort_values("usd_spent_usdc", ascending=False).head(10)
                lines.append(f"\n## Top 10 trades by this wallet in this market ({len(trades)} total trades in market)")
                for _, t in wt.iterrows():
                    lines.append(
                        f"- {_isofmt(t.get('timestamp'))}  role={t.get('role')}  "
                        f"side={t.get('side')}  "
                        f"size={_safe_float(t.get('size_outcome_usdc')):,.1f}  "
                        f"price={_safe_float(t.get('price')):.4f}  "
                        f"usd=${_safe_float(t.get('usd_spent_usdc')):,.0f}  "
                        f"token={str(t.get('token_id','?'))[:14]}.."
                    )

    # -------- CROSS-MARKET FOOTPRINT --------
    if wallet and not ctx.catalog.empty:
        w = wallet.lower()
        other_markets = []
        for cond in ctx.catalog["condition_id"]:
            if cond == cid:
                continue
            other_trades = ctx.trades_for(cond)
            if other_trades.empty:
                continue
            wtx = other_trades[
                (other_trades["maker"].fillna("").str.lower() == w) |
                (other_trades["taker"].fillna("").str.lower() == w)
            ]
            if not wtx.empty:
                other_markets.append({
                    "condition_id": cond,
                    "n_trades": len(wtx),
                    "usd": float(wtx["usd_spent_usdc"].sum()),
                })
                if len(other_markets) >= 50:
                    break
        if other_markets:
            other_markets.sort(key=lambda x: x["usd"], reverse=True)
            lines.append(f"\n## Wallet's other markets ({len(other_markets)}+ detected)")
            for om in other_markets[:8]:
                om_q = ctx._market_metadata_by_cid.get(om["condition_id"], {}).get("question", "?")
                lines.append(f"- `{om['condition_id'][:18]}..` {om_q[:60]!r}  "
                             f"${om['usd']:,.0f} across {om['n_trades']} trades")
        else:
            lines.append("\n## Wallet's other markets: NONE in our 264-market catalog.")
            lines.append("(Signal: this wallet is single-purpose within our sampled markets.)")

    # -------- CALLOUTS BY WALLET --------
    if wallet:
        matched = ctx._incidents_by_wallet.get(wallet.lower(), [])
        if matched:
            lines.append(f"\n## Callouts mentioning this wallet ({len(matched)})")
            for inc in matched[:5]:
                lines.append(f"- tier={inc.get('confidence_tier')} "
                             f"n_sources={inc.get('n_sources')} "
                             f"market={inc.get('market_question','?')[:70]!r}")

    # -------- CALLOUTS BY MARKET (question fuzzy match) --------
    if cid and not ctx.incidents.empty and candidate.get("market_question"):
        q_lower = str(candidate.get("market_question") or "").lower()
        matches = []
        for _, r in ctx.incidents.iterrows():
            callout_q = str(r.get("market_question") or "").lower()
            if not callout_q:
                continue
            # simple keyword overlap
            q_words = set(w for w in q_lower.split() if len(w) >= 4)
            c_words = set(w for w in callout_q.split() if len(w) >= 4)
            if q_words and c_words and len(q_words & c_words) >= 3:
                matches.append(r.to_dict())
        if matches:
            lines.append(f"\n## Callouts possibly matching this market ({len(matches)})")
            for r in matches[:5]:
                lines.append(f"- tier={r.get('confidence_tier')} "
                             f"n_sources={r.get('n_sources')} "
                             f"market={str(r.get('market_question',''))[:70]!r}")

    # -------- CLOSING --------
    lines.append("")
    return "\n".join(lines)
