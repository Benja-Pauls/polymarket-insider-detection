"""Dataclasses for the labeling pipeline.

``RawCallout`` is platform-agnostic raw content (a Reddit comment, a tweet,
a news paragraph). ``ExtractedCallout`` is the LLM's structured take on
whether that content contains a concrete insider-trading allegation, and
if so, what the alleged trade looks like. ``EnrichedCallout`` is the union.

Designed as plain dataclasses (no pydantic) to minimize dependencies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal

ConfidenceTier = Literal["T1", "T2", "T3"]
Direction = Literal["YES", "NO"]
Source = Literal["reddit", "twitter", "news", "blog"]


@dataclass(frozen=True)
class RawCallout:
    source: Source
    source_id: str              # platform-unique id (reddit post/comment id, tweet id, URL hash)
    source_url: str
    author: str | None
    created_utc: int            # epoch seconds
    title: str | None
    body: str
    score: int | None = None    # upvotes / likes
    num_replies: int | None = None
    parent_id: str | None = None  # for Reddit comments, etc.
    raw_metadata: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        return f"{self.source}:{self.source_id}"


@dataclass(frozen=True)
class ExtractedCallout:
    is_allegation: bool
    # All fields below are None when is_allegation is False
    market_question: str | None = None
    ts_lower: str | None = None        # ISO 8601
    ts_upper: str | None = None        # ISO 8601
    size_usd_approx: float | None = None
    wallet_addr: str | None = None
    direction: Direction | None = None
    outcome_resolved: Direction | None = None
    confidence_tier: ConfidenceTier | None = None
    quote: str | None = None
    reasoning: str | None = None


@dataclass(frozen=True)
class EnrichedCallout:
    raw: RawCallout
    extracted: ExtractedCallout
    extractor_model: str
    extractor_timestamp_iso: str
    extraction_cost_usd: float

    def to_flat_dict(self) -> dict:
        """Flattened representation for pandas / parquet."""
        d = {}
        r = asdict(self.raw)
        for k, v in r.items():
            if k == "raw_metadata":
                # Collapse to JSON string to avoid struct-union weirdness
                import json
                d[f"raw_{k}"] = json.dumps(v) if v else ""
            else:
                d[f"raw_{k}"] = v
        x = asdict(self.extracted)
        for k, v in x.items():
            d[k] = v
        d["extractor_model"] = self.extractor_model
        d["extractor_timestamp_iso"] = self.extractor_timestamp_iso
        d["extraction_cost_usd"] = self.extraction_cost_usd
        return d


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
