"""Claude-based extraction of insider-trading allegations from raw posts.

Uses Claude Haiku 4.5 with prompt caching on the system prompt + ephemeral
on-disk response cache. One API call per post. Target cost per 1K posts:
~$1.50, well within the $10 full-pipeline budget.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import anthropic
from anthropic.types import TextBlock

from .schema import EnrichedCallout, ExtractedCallout, RawCallout, now_iso

log = logging.getLogger(__name__)

MODEL_ID = "claude-sonnet-4-6"

# Approximate pricing per million tokens. Update if Anthropic changes them.
# Sonnet 4.6 is our default for labeling quality. Haiku 4.5 available for
# bulk first-pass classification.
_PRICING = {
    "claude-haiku-4-5-20251001": (0.80, 0.08, 4.00),
    "claude-sonnet-4-6":         (3.00, 0.30, 15.00),
    "claude-sonnet-4-6-20250929": (3.00, 0.30, 15.00),
    "claude-opus-4-7":            (15.00, 1.50, 75.00),
}
# Fallback for unknown models — use sonnet pricing
_DEFAULT_PRICING = (3.00, 0.30, 15.00)

SYSTEM_PROMPT = """You extract structured insider-trading allegations from prediction-market community posts.

Given a post (Reddit comment or thread, tweet, news excerpt) about Polymarket or other prediction markets, determine whether the post contains a SPECIFIC allegation that a trader had non-public information — and if so, extract structured fields.

A GENUINE ALLEGATION has at least one of:
  (a) A concrete suspicious trade with at least TWO of {size, market, approximate timing, wallet}
  (b) A named trader/wallet the author claims acted on non-public information
  (c) A documented pattern where a prediction market moved substantially before a public event materialized, framed as suspicious

NOT an allegation:
  - General market-efficiency commentary
  - Complaints about "manipulation" without specifics
  - Speculation about *who might* be betting
  - Jokes, hypotheticals, or conspiracy memes lacking specifics
  - Descriptions of the trader's own bets ("I bought…")

For each genuine allegation, extract:
  - market_question: the prediction market's question (paraphrase if verbatim unavailable; preserve proper nouns)
  - ts_lower / ts_upper: ISO 8601 (UTC). earliest/latest time the suspicious trade COULD have happened.
      If the post says "the day before X on YYYY-MM-DD", set [YYYY-MM-(DD-1) 00:00, YYYY-MM-DD 00:00].
      If just "hours before X" with X's datetime known, set [X - 12h, X].
  - size_usd_approx: USD size. "$400K" -> 400000. null if not specified.
  - wallet_addr: 0x-prefixed Polygon address if mentioned. null otherwise.
  - direction: "YES" | "NO" | null — which outcome the trader bet on.
  - outcome_resolved: "YES" | "NO" | null — what actually resolved, if mentioned.
  - confidence_tier:
      "T1" if a journalist or other high-credibility outlet explicitly identifies a specific trade or wallet
      "T2" if community-reported with multi-source corroboration (e.g. multiple commenters or a screenshot of on-chain data)
      "T3" if single-source speculation without corroboration
  - quote: verbatim excerpt (<= 300 chars) that supports the extraction
  - reasoning: 1-2 sentences explaining your determination

OUTPUT STRICT JSON, no code fences, following EXACTLY this schema:

{"is_allegation": bool,
 "market_question": str | null,
 "ts_lower": str | null,
 "ts_upper": str | null,
 "size_usd_approx": number | null,
 "wallet_addr": str | null,
 "direction": "YES" | "NO" | null,
 "outcome_resolved": "YES" | "NO" | null,
 "confidence_tier": "T1" | "T2" | "T3" | null,
 "quote": str | null,
 "reasoning": str}

If not an allegation, still include "reasoning" (1 sentence) and set every other field to null/false."""


class LLMExtractor:
    """Thin wrapper around the Anthropic SDK with disk caching and cost accounting."""

    def __init__(
        self,
        cache_dir: str | Path = "data/labels/_extract_cache",
        model: str = MODEL_ID,
        api_key: str | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)
        self.total_cost_usd = 0.0
        self.total_calls = 0
        self.cache_hits = 0

    def extract(self, raw: RawCallout) -> EnrichedCallout:
        cache_path = self._cache_path(raw)
        if cache_path.exists():
            self.cache_hits += 1
            with cache_path.open() as f:
                return _from_cached(json.load(f))

        user_msg = self._format_post(raw)
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as e:  # noqa: BLE001
            log.warning("API failure on %s: %s", raw.fingerprint(), e)
            return EnrichedCallout(
                raw=raw,
                extracted=ExtractedCallout(is_allegation=False, reasoning=f"API error: {e}"),
                extractor_model=self.model,
                extractor_timestamp_iso=now_iso(),
                extraction_cost_usd=0.0,
            )

        # Pricing
        usage = resp.usage
        in_price, cached_price, out_price = _PRICING.get(self.model, _DEFAULT_PRICING)
        input_cost = (
            (usage.input_tokens or 0) * in_price
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * cached_price
        ) / 1_000_000
        output_cost = (usage.output_tokens or 0) * out_price / 1_000_000
        cost = input_cost + output_cost
        self.total_cost_usd += cost
        self.total_calls += 1

        # Parse text block → JSON
        text = "".join(
            b.text for b in resp.content if isinstance(b, TextBlock)
        ).strip()
        extracted = _parse_json(text)

        enriched = EnrichedCallout(
            raw=raw,
            extracted=extracted,
            extractor_model=self.model,
            extractor_timestamp_iso=now_iso(),
            extraction_cost_usd=cost,
        )
        # Persist cache
        with cache_path.open("w") as f:
            json.dump({
                "raw": _raw_to_json(raw),
                "extracted": _extracted_to_json(extracted),
                "extractor_model": enriched.extractor_model,
                "extractor_timestamp_iso": enriched.extractor_timestamp_iso,
                "extraction_cost_usd": cost,
            }, f)
        return enriched

    def extract_many(self, raws: list[RawCallout]) -> list[EnrichedCallout]:
        out = []
        for r in raws:
            out.append(self.extract(r))
        return out

    def budget_report(self) -> dict:
        return {
            "calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "total_cost_usd": round(self.total_cost_usd, 4),
        }

    # --- helpers ---

    def _format_post(self, raw: RawCallout) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(raw.created_utc))
        header = [
            f"[source: {raw.source}]",
            f"[posted: {ts}]",
        ]
        if raw.author:
            header.append(f"[author: {raw.author}]")
        if raw.title:
            header.append(f"[title: {raw.title}]")
        if raw.source_url:
            header.append(f"[url: {raw.source_url}]")
        return "\n".join(header) + "\n\n" + raw.body[:6000]  # cap to keep input bounded

    def _cache_path(self, raw: RawCallout) -> Path:
        key = hashlib.sha256(
            json.dumps(
                {"m": self.model, "fp": raw.fingerprint(), "b": raw.body},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return self.cache_dir / f"{key}.json"


def _parse_json(text: str) -> ExtractedCallout:
    """Lenient parse — tolerate Markdown code fences and extra whitespace."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    # Find first { and last } to tolerate surrounding chatter
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        t = t[start : end + 1]
    try:
        d = json.loads(t)
    except json.JSONDecodeError:
        return ExtractedCallout(is_allegation=False, reasoning=f"unparseable LLM output: {text[:200]}")
    try:
        return ExtractedCallout(
            is_allegation=bool(d.get("is_allegation", False)),
            market_question=d.get("market_question"),
            ts_lower=d.get("ts_lower"),
            ts_upper=d.get("ts_upper"),
            size_usd_approx=(float(d["size_usd_approx"]) if d.get("size_usd_approx") is not None else None),
            wallet_addr=d.get("wallet_addr"),
            direction=d.get("direction"),
            outcome_resolved=d.get("outcome_resolved"),
            confidence_tier=d.get("confidence_tier"),
            quote=d.get("quote"),
            reasoning=d.get("reasoning"),
        )
    except (TypeError, ValueError) as e:
        return ExtractedCallout(is_allegation=False, reasoning=f"schema mismatch: {e}; raw={text[:200]}")


def _raw_to_json(raw: RawCallout) -> dict:
    from dataclasses import asdict
    return asdict(raw)


def _extracted_to_json(ex: ExtractedCallout) -> dict:
    from dataclasses import asdict
    return asdict(ex)


def _from_cached(payload: dict) -> EnrichedCallout:
    r = payload["raw"]
    x = payload["extracted"]
    raw = RawCallout(
        source=r["source"],
        source_id=r["source_id"],
        source_url=r["source_url"],
        author=r.get("author"),
        created_utc=int(r["created_utc"]),
        title=r.get("title"),
        body=r["body"],
        score=r.get("score"),
        num_replies=r.get("num_replies"),
        parent_id=r.get("parent_id"),
        raw_metadata=r.get("raw_metadata") or {},
    )
    extracted = ExtractedCallout(
        is_allegation=bool(x["is_allegation"]),
        market_question=x.get("market_question"),
        ts_lower=x.get("ts_lower"),
        ts_upper=x.get("ts_upper"),
        size_usd_approx=x.get("size_usd_approx"),
        wallet_addr=x.get("wallet_addr"),
        direction=x.get("direction"),
        outcome_resolved=x.get("outcome_resolved"),
        confidence_tier=x.get("confidence_tier"),
        quote=x.get("quote"),
        reasoning=x.get("reasoning"),
    )
    return EnrichedCallout(
        raw=raw,
        extracted=extracted,
        extractor_model=payload["extractor_model"],
        extractor_timestamp_iso=payload["extractor_timestamp_iso"],
        extraction_cost_usd=payload.get("extraction_cost_usd", 0.0),
    )
