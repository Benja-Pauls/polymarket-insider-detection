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
    "maduro", "venezuela", "caracas", "operation absolute resolve",
    "iran", "iranian", "tehran",
    "strike on iran", "strikes on iran", "iran strike",
    "ceasefire", "cease-fire", "cease fire",
    "gpt", "openai", "gpt-5", "chatgpt",
    "gemini", "google", "year in search", "alpharaccoon",
    "maga", "trump", "election 2024", "presidential election", "french whale", "theo", "fredi",
    "ukraine", "russia",
    "axiom",
    "survivor", "bachelorette", "aliens show",
    "yemen", "houthi", "houthis",
    "fed", "fomc", "federal reserve",
    "nobel", "peace prize", "machado", "maria corina",
    "burdensome-mix", "burdensome", "magamyman", "dirtycup", "alpharaccoon",
    "supreme court", "scotus",
    "hungary", "orban", "magyar", "tisza",
    "barron", "barron trump",
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


_HANDLE_RE = re.compile(r'"([A-Za-z][A-Za-z0-9_-]{3,40})"')


def _extract_handles(text) -> set[str]:
    """Extract quoted Polymarket handles from text (e.g., \"Burdensome-Mix\")."""
    text = _str_or_empty(text)
    if not text:
        return set()
    matches = _HANDLE_RE.findall(text)
    # Filter out things that are obviously not handles
    blacklist = {"Yes", "No", "YES", "NO", "yes", "no"}
    return {m for m in matches if m not in blacklist and len(m) >= 4}


def _str_or_empty(v) -> str:
    """Safe string-coercion that treats NaN/None/float-nan as empty."""
    if v is None:
        return ""
    if isinstance(v, float):
        # NaN and inf are not strings
        import math
        if math.isnan(v) or math.isinf(v):
            return ""
    if not isinstance(v, str):
        return ""
    return v


def _same_incident(a: dict, b: dict) -> bool:
    """Return True if two extractions are likely the same underlying incident.

    Conservative: we only merge when there is a HIGH-CONFIDENCE signal —
    matching wallet, matching trader handle, or (matching anchors AND very
    close time windows AND size agreement). The design goal: prefer to
    over-split rather than over-merge. Over-splitting is easy to fix with
    a manual merge step; over-merging silently collapses distinct events.
    """
    # 1. Explicit wallet-address match — strongest possible signal
    wa = _str_or_empty(a.get("wallet_addr")).lower()
    wb = _str_or_empty(b.get("wallet_addr")).lower()
    if wa and wb and wa == wb:
        return True

    # 2. Polymarket-handle match (quoted string like "Burdensome-Mix")
    # STRONGER if both sources quote a distinctive handle the other also quotes.
    a_handles = _extract_handles(a.get("quote")) | _extract_handles(a.get("reasoning"))
    b_handles = _extract_handles(b.get("quote")) | _extract_handles(b.get("reasoning"))
    shared_handles = a_handles & b_handles
    if shared_handles:
        # Filter out very generic ones like short common English words
        distinctive = {h for h in shared_handles if len(h) >= 5 and not h.isalpha() or h.count("-") > 0}
        if distinctive or any(len(h) >= 8 for h in shared_handles):
            return True

    # 3. Manual seed case_id match — when both rows come from the same
    # manual_case_id, they're definitionally the same incident.
    if a.get("manual_case_id") and a.get("manual_case_id") == b.get("manual_case_id"):
        return True

    # 4. Otherwise: require STRONG anchor match (≥2 anchors) AND tight time
    # overlap (within 48h, not the default 72h) AND plausible size agreement.
    atext = f"{_str_or_empty(a.get('market_question'))} {_str_or_empty(a.get('quote'))}"
    btext = f"{_str_or_empty(b.get('market_question'))} {_str_or_empty(b.get('quote'))}"
    a_anchors = _anchor_set(atext)
    b_anchors = _anchor_set(btext)
    common = a_anchors & b_anchors
    if len(common) < 2:
        return False

    # Multi-anchor overlap + tight window
    if not _windows_overlap(
        a.get("ts_lower"), a.get("ts_upper"),
        b.get("ts_lower"), b.get("ts_upper"),
        slack_hours=48,
    ):
        return False

    # Size plausibility
    sa = a.get("size_usd_approx")
    sb = b.get("size_usd_approx")
    if sa and sb:
        ratio = max(sa, sb) / max(1, min(sa, sb))
        if ratio > 20:   # very different — likely unrelated
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

        # Window: intersection (max lower, min upper) — filter NaN/None
        lowers = [_str_or_empty(r.get("ts_lower")) for r in g]
        uppers = [_str_or_empty(r.get("ts_upper")) for r in g]
        lowers = [x for x in lowers if x]
        uppers = [x for x in uppers if x]
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
