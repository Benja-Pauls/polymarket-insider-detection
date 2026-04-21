"""Merge multiple source extractions into canonical insider-trade incidents.

Each raw source (news article, Reddit comment, tweet) may reference the
same underlying incident. We group them into canonical incidents using a
heuristic: same market_question fuzzy-match AND overlapping time windows.
Each incident collects all supporting citations with their confidence tiers.

An incident has:
  - incident_id: hash of canonical fields
  - market_question (best paraphrase)
  - ts_lower / ts_upper (intersection of source windows)
  - size_usd_approx (mean where available)
  - wallet_addrs (set from sources)
  - direction
  - outcome_resolved
  - max confidence tier observed
  - list of source citations
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class IncidentCitation:
    source: str
    source_url: str
    source_id: str
    quote: str | None
    confidence_tier: str | None
    extracted_ts_lower: str | None
    extracted_ts_upper: str | None
    extracted_size_usd: float | None
    extracted_wallet_addr: str | None


@dataclass
class Incident:
    incident_id: str
    market_question: str
    ts_lower: str | None
    ts_upper: str | None
    size_usd_approx: float | None
    wallet_addrs: list[str]
    direction: str | None
    outcome_resolved: str | None
    confidence_tier: str                # strongest observed
    citations: list[IncidentCitation] = field(default_factory=list)
    n_sources: int = 0


# --- Canonicalization helpers --------------------------------------------

def _norm_question(q: str | None) -> str:
    if not q:
        return ""
    q = q.lower()
    # Strip articles, keep named entities
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\b(will|the|a|an|by|in|on|of|to|for)\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


# Key named entities that anchor the match. If both rows share any of these,
# they're likely the same incident (subject to time-window overlap).
ANCHOR_TOKENS = [
    "maduro", "venezuela", "caracas",
    "iran", "strike", "strikes",
    "gpt", "openai",
    "google", "search ranking",
    "maga", "trump", "election",
    "ceasefire", "ukraine", "russia",
    "axiom",
    "tv series", "aliens", "survivor", "bachelorette",
    "yemen", "houthi",
    "fed", "fomc",
]


def _anchor_set(text: str) -> set[str]:
    t = (text or "").lower()
    return {a for a in ANCHOR_TOKENS if a in t}


def _windows_overlap(a_lo, a_hi, b_lo, b_hi, *, slack_hours: int = 72) -> bool:
    """Do two [lo, hi] windows overlap (with tolerance)?"""
    if not (a_lo and a_hi and b_lo and b_hi):
        return True  # can't disprove — allow
    try:
        from datetime import datetime, timedelta
        def _p(s):
            return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        slack = timedelta(hours=slack_hours)
        alo, ahi = _p(a_lo) - slack, _p(a_hi) + slack
        blo, bhi = _p(b_lo) - slack, _p(b_hi) + slack
        return alo <= bhi and blo <= ahi
    except Exception:  # noqa: BLE001
        return True


def _same_incident(a: dict, b: dict) -> bool:
    # If explicit wallet matches, always merge
    wa = (a.get("wallet_addr") or "").lower()
    wb = (b.get("wallet_addr") or "").lower()
    if wa and wb and wa == wb:
        return True

    # Anchor-word intersection on market_question + quote
    atext = f"{a.get('market_question','')} {a.get('quote','')}"
    btext = f"{b.get('market_question','')} {b.get('quote','')}"
    a_anchors = _anchor_set(atext)
    b_anchors = _anchor_set(btext)
    if not (a_anchors & b_anchors):
        return False

    # Plus overlapping time windows (with slack)
    if not _windows_overlap(
        a.get("ts_lower"), a.get("ts_upper"),
        b.get("ts_lower"), b.get("ts_upper"),
    ):
        return False

    # And reasonable size agreement (within 10x or both null)
    sa = a.get("size_usd_approx")
    sb = b.get("size_usd_approx")
    if sa and sb and (max(sa, sb) / max(1, min(sa, sb))) > 10:
        return False

    return True


_TIER_ORDER = {"T1": 0, "T2": 1, "T3": 2, None: 3}


def _tier_max(t1, t2):
    """Return the stronger of two tier strings (T1 > T2 > T3)."""
    if _TIER_ORDER.get(t1, 3) <= _TIER_ORDER.get(t2, 3):
        return t1
    return t2


def canonicalize(allegations_df: pd.DataFrame) -> list[Incident]:
    """Group rows from (possibly many) source extractions into incidents."""
    rows = allegations_df.to_dict("records")
    groups: list[list[dict]] = []
    for row in rows:
        merged_into = None
        for i, g in enumerate(groups):
            if any(_same_incident(row, r) for r in g):
                g.append(row)
                merged_into = i
                break
        if merged_into is None:
            groups.append([row])

    incidents: list[Incident] = []
    for g in groups:
        # Pick the canonical values — take the most-cited / most-T1 source's wording
        g_sorted = sorted(g, key=lambda r: _TIER_ORDER.get(r.get("confidence_tier"), 3))
        best = g_sorted[0]

        # Collect wallet addresses across all citations
        wallets: list[str] = []
        for r in g:
            w = r.get("wallet_addr")
            if w and isinstance(w, str) and w.lower().startswith("0x"):
                if w.lower() not in [x.lower() for x in wallets]:
                    wallets.append(w)

        # Size: mean of non-null values
        sizes = [float(r.get("size_usd_approx")) for r in g if r.get("size_usd_approx") is not None]
        size_mean = sum(sizes) / len(sizes) if sizes else None

        # Window: intersection (max lower, min upper)
        lowers = [r.get("ts_lower") for r in g if r.get("ts_lower")]
        uppers = [r.get("ts_upper") for r in g if r.get("ts_upper")]
        ts_lower = max(lowers) if lowers else None
        ts_upper = min(uppers) if uppers else None

        # Citations
        citations = [IncidentCitation(
            source=r.get("raw_source", "") or "",
            source_url=r.get("raw_source_url", "") or "",
            source_id=r.get("raw_source_id", "") or "",
            quote=r.get("quote"),
            confidence_tier=r.get("confidence_tier"),
            extracted_ts_lower=r.get("ts_lower"),
            extracted_ts_upper=r.get("ts_upper"),
            extracted_size_usd=r.get("size_usd_approx"),
            extracted_wallet_addr=r.get("wallet_addr"),
        ) for r in g]

        best_tier = None
        for r in g:
            best_tier = _tier_max(best_tier, r.get("confidence_tier"))

        inc = Incident(
            incident_id=_incident_id(best.get("market_question", ""), ts_lower, ts_upper, wallets, size_mean),
            market_question=best.get("market_question", ""),
            ts_lower=ts_lower,
            ts_upper=ts_upper,
            size_usd_approx=size_mean,
            wallet_addrs=wallets,
            direction=best.get("direction"),
            outcome_resolved=best.get("outcome_resolved"),
            confidence_tier=best_tier or "T3",
            citations=citations,
            n_sources=len(g),
        )
        incidents.append(inc)

    incidents.sort(key=lambda i: (-i.n_sources, _TIER_ORDER.get(i.confidence_tier, 3)))
    return incidents


def _incident_id(question: str, ts_lower, ts_upper, wallets: list[str], size_mean) -> str:
    blob = {
        "q": _norm_question(question),
        "lo": str(ts_lower or ""),
        "hi": str(ts_upper or ""),
        "w": sorted(w.lower() for w in wallets),
    }
    return hashlib.sha256(
        json.dumps(blob, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


# --- IO -------------------------------------------------------------------

def load_allegations_from_sources(source_paths: list[Path]) -> pd.DataFrame:
    """Concat all enriched parquets and keep only is_allegation=True rows."""
    frames = []
    for p in source_paths:
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if "is_allegation" not in df.columns:
            continue
        frames.append(df[df["is_allegation"] == True])  # noqa: E712
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_incidents(incidents: list[Incident], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Flat incidents parquet
    rows = []
    for inc in incidents:
        d = asdict(inc)
        d["citations"] = json.dumps(d["citations"])
        d["wallet_addrs"] = json.dumps(d["wallet_addrs"])
        rows.append(d)
    pd.DataFrame(rows).to_parquet(out_dir / "incidents.parquet", index=False)

    # Review CSV — one row per incident, with top citation preview
    review_rows = []
    for inc in incidents:
        top_cit = inc.citations[0] if inc.citations else None
        review_rows.append({
            "incident_id": inc.incident_id,
            "market_question": inc.market_question,
            "ts_lower": inc.ts_lower,
            "ts_upper": inc.ts_upper,
            "size_usd_approx": inc.size_usd_approx,
            "wallet_addrs": ";".join(inc.wallet_addrs),
            "direction": inc.direction,
            "outcome_resolved": inc.outcome_resolved,
            "confidence_tier": inc.confidence_tier,
            "n_sources": inc.n_sources,
            "top_source_url": top_cit.source_url if top_cit else "",
            "top_quote": (top_cit.quote[:400] if (top_cit and top_cit.quote) else ""),
        })
    pd.DataFrame(review_rows).to_csv(out_dir / "incidents_review.csv", index=False)
