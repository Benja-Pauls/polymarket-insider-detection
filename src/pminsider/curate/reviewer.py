"""LLM-driven per-candidate verdict.

Two-stage review:
  1. Sonnet first pass — cheap, reliably rejects obvious non-insider rows
     (sports markets, BTC/ETH speculation, arb, market-making, etc.).
  2. Opus escalation — anything Sonnet didn't confidently reject is re-reviewed
     with deeper reasoning.

Each verdict is a structured JSON:
  verdict: confirmed | suspected | rejected | ambiguous
  confidence_tier_final: T1 | T2 | T3 | null
  reasoning: 1-4 sentences
  merge_with_candidate_ids: [..]
  coordinated_with_candidate_ids: [..]
  strongest_evidence: [..]
  concerns: [..]

Responses cached on disk by content hash so reruns are free.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from anthropic.types import TextBlock

log = logging.getLogger(__name__)

SONNET_MODEL = "claude-sonnet-4-6"
OPUS_MODEL = "claude-opus-4-7"

# Approximate per-million-token pricing (2026).
_PRICING = {
    "claude-sonnet-4-6":          (3.00, 0.30, 15.00),
    "claude-opus-4-7":             (15.00, 1.50, 75.00),
    "claude-haiku-4-5-20251001":   (0.80, 0.08, 4.00),
}
_DEFAULT_PRICING = (3.00, 0.30, 15.00)


SYSTEM_PROMPT = """You are curating a dataset of suspected insider trades on Polymarket, a decentralized prediction market on Polygon.

Each candidate below was surfaced by our pipeline as a possible insider trade: either
  (a) a public news/Reddit callout that our LLM extractor identified as a specific trade allegation, OR
  (b) an on-chain pattern from our miner (a wallet-market pair where ≥5 of 6 heuristics fired).

Your job: verdict each candidate and identify merges.

VERDICT TAXONOMY
================

- **confirmed** — the evidence is overwhelming. Either:
  * A credible journalist identifies this specific wallet/handle for this specific market, AND the on-chain pattern corroborates the allegation; OR
  * The on-chain pattern is extreme: ALL of
      · wallet's activity >=95% concentrated in this single market
      · size in market is top 1% (in our provided percentile)
      · wallet's first-ever Polygon position was >=$10K in this market
      · flow direction agrees with the winning outcome
      · first trade was >=24 hours before resolution (not a reaction)
      · wallet has few or no other markets in our coverage

- **suspected** — strong on-chain pattern with ONE of the confirmed criteria missing (e.g. wallet has one other market, or size is top-3% instead of top-1%). OR community-reported with multi-source corroboration. Flag the missing criterion in `concerns`.

- **rejected** — the pattern is clearly NOT insider. Examples:
  * The market is a sports match / event and the trader is just a large bettor — concentration isn't suspicious on a sports market.
  * The market is a BTC/ETH price prediction — speculation, not insider info.
  * The wallet placed two-sided inventory (market-making).
  * Clear arbitrage of mispriced contracts — profit from mispricing, not info asymmetry.
  * The wallet's flow did NOT align with the winning outcome (they lost or hedged).
  * The wallet has normal, broad trading history — just happened to be big in this market.

- **ambiguous** — genuinely unclear. Evidence partly supports insider, partly doesn't. Explain both sides.

MERGING RULES
=============

- If another candidate in the provided context (or the same batch) has the same wallet AND same market, propose merging by putting its id in `merge_with_candidate_ids`.
- If a news-sourced callout row matches an on-chain row by (market similarity + wallet if known + similar size): propose merging. This upgrades the callout from generic to specific-on-chain.
- If multiple wallets are trading the same market with very similar size/timing/direction (suggesting Sybil coordination): put their ids in `coordinated_with_candidate_ids`, but do NOT merge (we want trade-level labels per wallet).

CONFIDENCE TIERS (for output)
=============================

- T1: journalist-named specific trade, OR wallet explicitly named by credible news sources, OR all six on-chain flags fire AND flow is strongly winner-aligned
- T2: strong on-chain pattern OR community-reported with corroboration
- T3: single-source speculation or weaker on-chain pattern

OUTPUT FORMAT
=============

Return STRICT JSON, no code fences:
{
  "candidate_id": "...",
  "verdict": "confirmed" | "suspected" | "rejected" | "ambiguous",
  "confidence_tier_final": "T1" | "T2" | "T3" | null,
  "reasoning": "1-4 sentences explaining your decision, citing specific evidence from the dossier",
  "merge_with_candidate_ids": ["..."],
  "coordinated_with_candidate_ids": ["..."],
  "strongest_evidence": ["bullet1", "bullet2"],
  "concerns": ["what could make this NOT be insider"]
}
"""


@dataclass
class CuratorCost:
    calls: int = 0
    cache_hits: int = 0
    cost_usd: float = 0.0
    by_model: dict = field(default_factory=dict)


@dataclass
class Verdict:
    candidate_id: str
    verdict: str
    confidence_tier_final: str | None
    reasoning: str
    merge_with_candidate_ids: list[str]
    coordinated_with_candidate_ids: list[str]
    strongest_evidence: list[str]
    concerns: list[str]
    model: str


class Curator:
    def __init__(
        self,
        cache_dir: str | Path = "data/labels/_curate_cache",
        api_key: str | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)
        self.cost = CuratorCost()

    def review(self, candidate_id: str, dossier: str, *, model: str) -> Verdict:
        cache_path = self._cache_path(candidate_id, dossier, model)
        if cache_path.exists():
            self.cost.cache_hits += 1
            with cache_path.open() as f:
                return _verdict_from_json(json.load(f))

        try:
            resp = self.client.messages.create(
                model=model,
                max_tokens=1024,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": f"CANDIDATE DOSSIER\n\n{dossier}\n\nReturn ONLY the JSON verdict.",
                }],
            )
        except Exception as e:  # noqa: BLE001
            log.warning("API failure for %s (%s): %s", candidate_id, model, e)
            return Verdict(
                candidate_id=candidate_id,
                verdict="ambiguous",
                confidence_tier_final=None,
                reasoning=f"API error: {e}",
                merge_with_candidate_ids=[],
                coordinated_with_candidate_ids=[],
                strongest_evidence=[],
                concerns=[f"api_error: {e}"],
                model=model,
            )

        usage = resp.usage
        pricing = _PRICING.get(model, _DEFAULT_PRICING)
        cost = _compute_cost(usage, pricing)
        self.cost.calls += 1
        self.cost.cost_usd += cost
        self.cost.by_model[model] = self.cost.by_model.get(model, 0.0) + cost

        text = "".join(b.text for b in resp.content if isinstance(b, TextBlock)).strip()
        parsed = _parse_verdict_json(text, candidate_id, model)

        with cache_path.open("w") as f:
            json.dump({
                "candidate_id": parsed.candidate_id,
                "verdict": parsed.verdict,
                "confidence_tier_final": parsed.confidence_tier_final,
                "reasoning": parsed.reasoning,
                "merge_with_candidate_ids": parsed.merge_with_candidate_ids,
                "coordinated_with_candidate_ids": parsed.coordinated_with_candidate_ids,
                "strongest_evidence": parsed.strongest_evidence,
                "concerns": parsed.concerns,
                "model": parsed.model,
            }, f)
        return parsed

    def _cache_path(self, candidate_id: str, dossier: str, model: str) -> Path:
        h = hashlib.sha256(
            json.dumps({"id": candidate_id, "d": dossier, "m": model}, sort_keys=True).encode()
        ).hexdigest()[:40]
        return self.cache_dir / f"{h}.json"


def _compute_cost(usage, pricing: tuple) -> float:
    """Anthropic usage accounting:
      - input_tokens          INCLUDES cache_creation_input_tokens but NOT cache_read_input_tokens
      - cache_read_input_tokens  priced at the cached rate (typically 0.1x)
    So cost = input_tokens * full_rate + cache_read * cached_rate + output_tokens * out_rate.
    """
    in_price, cached_price, out_price = pricing
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    input_cost = (
        (usage.input_tokens or 0) * in_price
        + cache_read * cached_price
    ) / 1_000_000
    output_cost = (usage.output_tokens or 0) * out_price / 1_000_000
    return input_cost + output_cost


def _parse_verdict_json(text: str, candidate_id: str, model: str) -> Verdict:
    """Lenient JSON parse."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    # Tolerate chatter before/after the JSON
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        t = t[start : end + 1]
    try:
        d = json.loads(t)
    except json.JSONDecodeError as e:
        log.warning("bad JSON from %s (%s): %s; raw=%r", candidate_id, model, e, text[:200])
        return Verdict(
            candidate_id=candidate_id,
            verdict="ambiguous",
            confidence_tier_final=None,
            reasoning=f"LLM returned unparseable JSON: {text[:200]}",
            merge_with_candidate_ids=[],
            coordinated_with_candidate_ids=[],
            strongest_evidence=[],
            concerns=["parse_error"],
            model=model,
        )
    return Verdict(
        candidate_id=d.get("candidate_id") or candidate_id,
        verdict=str(d.get("verdict", "ambiguous")),
        confidence_tier_final=d.get("confidence_tier_final"),
        reasoning=str(d.get("reasoning", "")),
        merge_with_candidate_ids=list(d.get("merge_with_candidate_ids") or []),
        coordinated_with_candidate_ids=list(d.get("coordinated_with_candidate_ids") or []),
        strongest_evidence=list(d.get("strongest_evidence") or []),
        concerns=list(d.get("concerns") or []),
        model=model,
    )


def _verdict_from_json(d: dict) -> Verdict:
    return Verdict(
        candidate_id=d["candidate_id"],
        verdict=d.get("verdict", "ambiguous"),
        confidence_tier_final=d.get("confidence_tier_final"),
        reasoning=d.get("reasoning", ""),
        merge_with_candidate_ids=d.get("merge_with_candidate_ids") or [],
        coordinated_with_candidate_ids=d.get("coordinated_with_candidate_ids") or [],
        strongest_evidence=d.get("strongest_evidence") or [],
        concerns=d.get("concerns") or [],
        model=d.get("model", "unknown"),
    )
