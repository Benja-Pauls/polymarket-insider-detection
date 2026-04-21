"""Market-tradability classifier.

Given a Polymarket market question, decide whether the outcome is *possible* to
insider-trade (i.e. whether there exists a class of people — board members,
government officials, production crew, Nobel jurors, etc. — who know the
outcome before the public), or whether it is pure speculation / live
competition where no one has private information.

We run Claude Haiku 4.5 in batches of 20 questions per call. The system prompt
carries the taxonomy + few-shot examples and is cached via
`cache_control: {"type": "ephemeral"}`. Responses are cached on disk keyed by
(model, system-prompt-hash, batch-question-hash) so reruns are free.
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

HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Per-million-token pricing (USD).
# (input, cache_read, output)
_PRICING: dict[str, tuple[float, float, float]] = {
    "claude-haiku-4-5-20251001": (0.80, 0.08, 4.00),
}
_DEFAULT_PRICING = (0.80, 0.08, 4.00)

# --- Taxonomy --------------------------------------------------------------

INSIDER_TRADEABLE_CATEGORIES = {
    "tradeable_geopolitical",
    "tradeable_corporate",
    "tradeable_political",
    "tradeable_medical",
    "tradeable_entertainment_scripted",
    "tradeable_awards",
    "tradeable_other",
}
NOT_TRADEABLE_CATEGORIES = {
    "not_tradeable_sports",
    "not_tradeable_price",
    "not_tradeable_election",
    "not_tradeable_social",
    "not_tradeable_weather",
}
FLAG_CATEGORIES = {"ambiguous", "unknown"}
ALL_CATEGORIES = (
    INSIDER_TRADEABLE_CATEGORIES | NOT_TRADEABLE_CATEGORIES | FLAG_CATEGORIES
)


SYSTEM_PROMPT = """You classify Polymarket prediction-market questions by whether INSIDER TRADING is even *possible* given the nature of the question.

A question is insider-tradeable when the outcome is determined in advance by a small, identifiable class of humans (corporate officers, government officials, juries, production crews, doctors running a trial) whose private knowledge of the outcome could give them an informational edge when trading the market.

A question is NOT insider-tradeable when the outcome is decided live (sports games), is dominated by public/efficient information (asset prices, public polls), or is driven by broad social behavior (tweet counts, weather).

TAXONOMY (you MUST pick exactly one `category_tradability` per question)
========================================================================

INSIDER-TRADEABLE CATEGORIES
----------------------------
- tradeable_geopolitical: wars, military strikes, diplomacy, treaties, sanctions, regime change, hostage releases, government actions abroad. Governments and military officials know plans before the public.
  Examples: "Will the US strike Iran by X?", "Will Maduro be ousted?", "Will Russia agree to a ceasefire by X?", "Will a hostage deal be signed?"

- tradeable_corporate: earnings beats/misses, CEO firings/hirings, M&A deals, product launches, layoffs, bankruptcies, token listings, share prints. Executives, board members, employees know before the public.
  Examples: "Will GPT-5 release by X?", "Will Andy Byron be out as CEO?", "Will Company X acquire Y?", "Will Apple launch VR headset by X?"

- tradeable_political: policy announcements, executive appointments, indictments, cabinet picks, SCOTUS rulings, federal court decisions, pardons, bill signings. Staff, judges, DOJ insiders know before the public.
  Examples: "Will Trump sign X bill?", "Will the SCOTUS rule X?", "Will X be indicted this week?", "Will X be nominated Secretary of State?"
  NOT this category: public elections decided by voters — those go to not_tradeable_election.

- tradeable_medical: FDA approvals, clinical trial results, drug rejections, device approvals, surgical outcomes for named patients, disease-X declarations. Company scientists, FDA reviewers, doctors know before the public.
  Examples: "Will the FDA approve X by date?", "Will X drug trial succeed?", "Will the WHO declare a pandemic?"

- tradeable_entertainment_scripted: pre-taped or pre-produced reality / competition TV where the production crew know the outcome weeks in advance.
  Examples: "Will X win Survivor season Y?", "Will X get a rose on the Bachelor?", "Next Bachelor contestant to go home?"

- tradeable_awards: Nobel Prize, Oscars, Emmys, Grammys, Google Year-in-Search, MVPs picked by panels, Time Person of the Year, Pulitzer. A small jury or committee decides privately.
  Examples: "Will X win Best Picture?", "Nobel Peace Prize winner?", "Will d4vd be the #1 searched person on Google Year-in-Search?", "Time Person of the Year?"

- tradeable_other: fits the insider-tradeable bar (outcome determined by a small identifiable group of humans with private knowledge) but doesn't cleanly fit any category above. Use sparingly.

NOT-TRADEABLE CATEGORIES
------------------------
- not_tradeable_sports: live athletic / e-sports competition, game outcomes, match winners, game props (total points, player performance), playoff series, championship winners decided on the field. Outcome is decided live — no insider has it.
  Examples: "Will the Sharks win the Stanley Cup?", "Ravens vs Chiefs", "Man City vs Arsenal", "Will Mahomes throw 3 TDs?", UFC / boxing / LoL / Valorant match winners.

- not_tradeable_price: price predictions on public liquid assets (BTC, ETH, any crypto, stocks, FX, commodities). The underlying spot market is efficient; no one has edge from non-public info that isn't already illegal on the underlying exchange.
  Examples: "Bitcoin above $118,000 on X?", "Will ETH close above $4k?", "Will TSLA hit $300?", "Will SOL flippen BTC?"

- not_tradeable_election: publicly-held elections where voters decide (presidential, senate, house, governor, mayor, party primaries, parliamentary elections). Polls dominate and there's no meaningful insider edge on the final tally.
  Examples: "Will Trump win 2024?", "Will Democrat win NC?", "Next UK PM?", "Will X win the senate primary?"
  Exception: if the question is about BEING NOMINATED / APPOINTED / FIRED (not elected), that's tradeable_political.

- not_tradeable_social: social-media metrics (tweet counts, follower counts, YouTube views), viral trends, public internet engagement. Too diffuse; no discrete insider.
  Examples: "Will Elon tweet X times this week?", "YouTube views on X video?", "TikTok most-liked?"

- not_tradeable_weather: weather, temperature, rainfall, storm paths. Forecasting, not insider info.

FLAG CATEGORIES
---------------
- ambiguous: genuinely could be either side and a human reviewer should look. Use when the question is borderline (e.g. a corporate announcement that is also kind of a price target, or a mixed political/sports question).

- unknown: you cannot classify from the question text alone (too short, in another language, obviously truncated, or nonsensical).

DECISION HEURISTICS
===================
1. Is there a small, named class of humans (officials, officers, jurors, doctors, producers) whose private knowledge determines the outcome? → insider-tradeable.
2. Is the outcome decided live, in public, by a crowd, by a market, or by the weather? → not tradeable.
3. When a question mixes categories (e.g. "Will Company X stock be above $Y after earnings?"), the dominant driver wins. If the question is fundamentally about the *stock price* → not_tradeable_price. If the question is about *whether earnings beat* → tradeable_corporate.
4. Horse racing, dog racing, Formula 1 race winners → not_tradeable_sports (decided live on the track).
5. "Will X song be #1 on Spotify?" / "Will X movie gross $Y?" → not_tradeable_social (aggregate public behavior), UNLESS it's an awards jury pick.

OUTPUT FORMAT
=============
For each question in the numbered list I send you, return exactly one JSON object with these fields:
  - index: int (matching the question number in my prompt, 1-indexed)
  - category_tradability: one of the categories above (use the exact snake_case string)
  - is_insider_tradeable: true if category starts with "tradeable_", false otherwise
  - confidence: float between 0.0 and 1.0 (your confidence in the classification)
  - reasoning: ONE sentence explaining why, referencing the actual question

Return ONLY a JSON array of these objects, in the same order as my input, no prose, no code fences, no commentary. Length of the array MUST equal the number of questions I sent.
"""


# --- Data classes ----------------------------------------------------------


@dataclass
class Classification:
    index: int
    category_tradability: str
    is_insider_tradeable: bool
    confidence: float
    reasoning: str
    model: str


@dataclass
class ClassifierCost:
    calls: int = 0
    cache_hits: int = 0
    questions_classified: int = 0
    cost_usd: float = 0.0
    by_model: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


# --- Classifier ------------------------------------------------------------


class TradabilityClassifier:
    """Batch classifier for Polymarket market questions.

    Usage:
        clf = TradabilityClassifier()
        results = clf.classify_batch(["Will X happen by Y?", "Ravens vs Chiefs"])
        # results: list[Classification] same length as input
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/labels/_tradability_cache",
        api_key: str | None = None,
        model: str = HAIKU_MODEL,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.cost = ClassifierCost()
        self._system_hash = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16]

    # -- single batch ------------------------------------------------------

    def classify_batch(self, questions: list[str]) -> list[Classification]:
        """Classify a batch of questions in a single API call. Cache-aware."""
        if not questions:
            return []

        cache_path = self._cache_path(questions)
        if cache_path.exists():
            self.cost.cache_hits += 1
            with cache_path.open() as f:
                data = json.load(f)
            return [_classification_from_json(d) for d in data]

        user_msg = _build_user_message(questions)

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as e:  # noqa: BLE001
            log.warning("API failure on batch of %d: %s", len(questions), e)
            return [
                Classification(
                    index=i + 1,
                    category_tradability="unknown",
                    is_insider_tradeable=False,
                    confidence=0.0,
                    reasoning=f"API error: {e}",
                    model=self.model,
                )
                for i in range(len(questions))
            ]

        self._account(resp.usage)

        text = "".join(b.text for b in resp.content if isinstance(b, TextBlock)).strip()
        parsed = _parse_batch_json(text, len(questions), self.model)

        self.cost.questions_classified += len(parsed)

        with cache_path.open("w") as f:
            json.dump(
                [
                    {
                        "index": c.index,
                        "category_tradability": c.category_tradability,
                        "is_insider_tradeable": c.is_insider_tradeable,
                        "confidence": c.confidence,
                        "reasoning": c.reasoning,
                        "model": c.model,
                    }
                    for c in parsed
                ],
                f,
            )
        return parsed

    # -- internals ---------------------------------------------------------

    def _cache_path(self, questions: list[str]) -> Path:
        payload = json.dumps(
            {"m": self.model, "s": self._system_hash, "q": questions}, sort_keys=True
        )
        h = hashlib.sha256(payload.encode()).hexdigest()[:40]
        return self.cache_dir / f"{h}.json"

    def _account(self, usage) -> None:
        pricing = _PRICING.get(self.model, _DEFAULT_PRICING)
        cost = _compute_cost(usage, pricing)
        self.cost.calls += 1
        self.cost.cost_usd += cost
        self.cost.by_model[self.model] = self.cost.by_model.get(self.model, 0.0) + cost
        self.cost.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.cost.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        self.cost.cache_read_tokens += int(
            getattr(usage, "cache_read_input_tokens", 0) or 0
        )
        self.cost.cache_creation_tokens += int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )


# --- helpers ---------------------------------------------------------------


def _build_user_message(questions: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    return (
        f"Classify each of the following {len(questions)} Polymarket questions. "
        f"Return a JSON array of exactly {len(questions)} objects in the same "
        "order (indexed 1.." + str(len(questions)) + "), each with fields "
        "{index, category_tradability, is_insider_tradeable, confidence, "
        "reasoning}. Output ONLY the JSON array, no prose.\n\n"
        "QUESTIONS:\n" + numbered
    )


def _compute_cost(usage, pricing: tuple) -> float:
    """Anthropic usage accounting, matching curate/reviewer.py."""
    in_price, cached_price, out_price = pricing
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    input_cost = (
        (usage.input_tokens or 0) * in_price + cache_read * cached_price
    ) / 1_000_000
    output_cost = (usage.output_tokens or 0) * out_price / 1_000_000
    return input_cost + output_cost


def _parse_batch_json(text: str, expected: int, model: str) -> list[Classification]:
    """Lenient JSON array parse. Pads / truncates to `expected` length."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    # Tolerate chatter before/after the JSON array.
    start = t.find("[")
    end = t.rfind("]")
    if start >= 0 and end > start:
        t = t[start : end + 1]
    try:
        data = json.loads(t)
    except json.JSONDecodeError as e:
        log.warning("bad JSON from batch (%s): %s; raw=%r", model, e, text[:400])
        return [
            Classification(
                index=i + 1,
                category_tradability="unknown",
                is_insider_tradeable=False,
                confidence=0.0,
                reasoning=f"LLM returned unparseable JSON: {text[:160]}",
                model=model,
            )
            for i in range(expected)
        ]

    if not isinstance(data, list):
        return [
            Classification(
                index=i + 1,
                category_tradability="unknown",
                is_insider_tradeable=False,
                confidence=0.0,
                reasoning="LLM returned non-array JSON",
                model=model,
            )
            for i in range(expected)
        ]

    # Keep results in supplied order by `index`; fall back to positional order
    # if indices are missing / wrong.
    by_index: dict[int, dict] = {}
    positional: list[dict] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        idx = d.get("index")
        if isinstance(idx, int) and 1 <= idx <= expected:
            by_index[idx] = d
        else:
            positional.append(d)

    out: list[Classification] = []
    for i in range(1, expected + 1):
        d = by_index.get(i)
        if d is None:
            # positional fallback
            pos_i = i - 1 - len(by_index)
            if 0 <= pos_i < len(positional):
                d = positional[pos_i]
        if d is None:
            out.append(
                Classification(
                    index=i,
                    category_tradability="unknown",
                    is_insider_tradeable=False,
                    confidence=0.0,
                    reasoning="LLM omitted this question from the batch response",
                    model=model,
                )
            )
            continue
        out.append(_classification_from_llm_dict(d, i, model))
    return out


def _classification_from_llm_dict(d: dict, fallback_index: int, model: str) -> Classification:
    cat = str(d.get("category_tradability", "unknown")).strip()
    if cat not in ALL_CATEGORIES:
        # Try to salvage common misses (lenient mapping).
        low = cat.lower()
        if low in ALL_CATEGORIES:
            cat = low
        else:
            cat = "unknown"

    if cat in INSIDER_TRADEABLE_CATEGORIES:
        derived_bool = True
    elif cat in NOT_TRADEABLE_CATEGORIES:
        derived_bool = False
    else:
        derived_bool = False

    # Honor the LLM's stated bool if it matches; otherwise prefer the derived bool.
    raw_bool = d.get("is_insider_tradeable")
    if isinstance(raw_bool, bool):
        is_insider = raw_bool if (raw_bool == derived_bool) else derived_bool
    else:
        is_insider = derived_bool

    try:
        conf = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    idx = d.get("index")
    if not isinstance(idx, int):
        idx = fallback_index

    reasoning = str(d.get("reasoning", "")).strip() or "(no reasoning returned)"

    return Classification(
        index=idx,
        category_tradability=cat,
        is_insider_tradeable=is_insider,
        confidence=conf,
        reasoning=reasoning,
        model=model,
    )


def _classification_from_json(d: dict) -> Classification:
    return Classification(
        index=int(d["index"]),
        category_tradability=str(d["category_tradability"]),
        is_insider_tradeable=bool(d["is_insider_tradeable"]),
        confidence=float(d["confidence"]),
        reasoning=str(d["reasoning"]),
        model=str(d.get("model", "unknown")),
    )


# --- public batching helper ------------------------------------------------


def classify_questions(
    questions: list[str],
    *,
    batch_size: int = 20,
    classifier: TradabilityClassifier | None = None,
    progress: bool = True,
) -> list[Classification]:
    """Classify a flat list of questions. Preserves input order."""
    clf = classifier or TradabilityClassifier()
    out: list[Classification] = []

    batches = [
        questions[i : i + batch_size] for i in range(0, len(questions), batch_size)
    ]

    iterator = batches
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(batches, desc="tradability", unit="batch")
        except ImportError:
            pass

    for b in iterator:
        results = clf.classify_batch(b)
        out.extend(results)
    return out
