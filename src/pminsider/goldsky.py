"""GraphQL client for Polymarket's Goldsky-hosted subgraphs.

Polymarket's REST APIs (gamma-api, clob) are Cloudflare geo-blocked from US IPs,
but their on-chain data is public and indexed into five Goldsky subgraphs that
are reachable. This module wraps those subgraphs in a typed, paginated, cached
client.

Subgraphs:
    orderbook (live)       per-trade OrderFilledEvent / OrdersMatchedEvent
    orderbook_resync       enriched historical trades with Account links (frozen Jan 2026)
    activity               Condition + FPMM entities (market resolution, payouts)
    pnl                    per-wallet aggregate statistics
    oi                     market-level open interest

All queries go through curl_cffi with Chrome TLS fingerprinting — Goldsky itself
doesn't require it, but we keep it consistent so the same code can fall back
through any Cloudflare-fronted endpoint without surprises.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from curl_cffi import requests as crequests

PROJECT_ID = "project_cl6mb8i9h0003e201j6li0diw"
_BASE = f"https://api.goldsky.com/api/public/{PROJECT_ID}/subgraphs"

SUBGRAPHS: dict[str, str] = {
    "orderbook":        f"{_BASE}/orderbook-subgraph/0.0.1/gn",
    "orderbook_resync": f"{_BASE}/polymarket-orderbook-resync/prod/gn",
    "activity":         f"{_BASE}/activity-subgraph/0.0.4/gn",
    "pnl":              f"{_BASE}/pnl-subgraph/0.0.14/gn",
    "oi":               f"{_BASE}/oi-subgraph/0.0.6/gn",
    "positions":        f"{_BASE}/positions-subgraph/0.0.7/gn",
}

DEFAULT_PAGE_SIZE = 1000  # The Graph's hard limit per query
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 1.0

# Used as the ``_lt`` seed for desc iteration — larger than any realistic
# tokenId / volume / timestamp / hex-id. 2**256 - 1 as decimal.
_MAX_SENTINEL = str((1 << 256) - 1)


class GoldskyError(RuntimeError):
    pass


class GoldskyTimeout(GoldskyError):
    """Subgraph reported a statement timeout on the backing DB — caller should try a smaller page."""


@dataclass
class QueryResult:
    data: Any
    from_cache: bool
    elapsed_sec: float


class GoldskyClient:
    """Thin GraphQL client over Polymarket's Goldsky subgraphs with on-disk caching.

    Caching is keyed by (subgraph, query_hash, variables_hash). Responses are
    stored as gzipped JSON in ``cache_dir``. Use ``force_refresh=True`` to bypass.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/raw/goldsky",
        impersonate: str = "chrome120",
        timeout_sec: int = 30,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.impersonate = impersonate
        self.timeout_sec = timeout_sec
        self._session = crequests.Session(impersonate=impersonate)

    # --- core query ---------------------------------------------------------

    def query(
        self,
        subgraph: str,
        query: str,
        variables: dict | None = None,
        *,
        force_refresh: bool = False,
    ) -> QueryResult:
        if subgraph not in SUBGRAPHS:
            raise GoldskyError(f"unknown subgraph {subgraph!r}; known: {list(SUBGRAPHS)}")
        url = SUBGRAPHS[subgraph]
        variables = variables or {}

        cache_path = self._cache_path(subgraph, query, variables)
        if cache_path.exists() and not force_refresh:
            with cache_path.open() as f:
                return QueryResult(data=json.load(f), from_cache=True, elapsed_sec=0.0)

        payload = {"query": query, "variables": variables}
        backoff = INITIAL_BACKOFF_SEC
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start = time.monotonic()
                r = self._session.post(url, json=payload, timeout=self.timeout_sec)
                elapsed = time.monotonic() - start
                if r.status_code == 429:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                if r.status_code >= 500:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                if r.status_code != 200:
                    raise GoldskyError(f"HTTP {r.status_code} on {subgraph}: {r.text[:200]}")
                body = r.json()
                if "errors" in body:
                    msgs = "; ".join(e.get("message", "") for e in body["errors"])
                    low = msgs.lower()
                    if (
                        "statement timeout" in low
                        or "canceling statement" in low
                        or "query timed out" in low
                        or "timeout" in low
                    ):
                        raise GoldskyTimeout(f"timeout on {subgraph}: {msgs[:120]}")
                    raise GoldskyError(f"GraphQL errors on {subgraph}: {body['errors']}")
                data = body.get("data")
                if data is None:
                    raise GoldskyError(f"empty data on {subgraph}: {body}")
                with cache_path.open("w") as f:
                    json.dump(data, f)
                return QueryResult(data=data, from_cache=False, elapsed_sec=elapsed)
            except (crequests.exceptions.RequestException, json.JSONDecodeError) as e:
                if attempt >= MAX_RETRIES:
                    raise GoldskyError(f"network failure on {subgraph} after {MAX_RETRIES}: {e}") from e
                time.sleep(backoff)
                backoff *= 2

        raise GoldskyError(f"unreachable: exhausted retries on {subgraph}")

    # --- pagination --------------------------------------------------------

    def paginate(
        self,
        subgraph: str,
        entity: str,
        fields: str,
        *,
        where: dict | None = None,
        order_by: str = "id",
        order_direction: str = "asc",
        page_size: int = DEFAULT_PAGE_SIZE,
        max_pages: int | None = None,
        force_refresh: bool = False,
    ) -> Iterator[dict]:
        """Iterate over all rows of an entity using keyset pagination on ``order_by``.

        The Graph caps ``first`` at 1000 and ``skip`` at 5000, so we do keyset
        pagination: after each page we filter ``{order_by}_{gt|lt}: last_value``
        instead of using ``skip``.

        IMPORTANT: The Graph's hosted database (Postgres) indexes every entity on
        ``block_range``, and queries WITHOUT an explicit ``{key}_gt`` (or ``_lt``)
        where-clause force a scan across the full block_range index → statement
        timeout. To prevent this we always inject a seed ``{order_by}_{gt|lt}``
        filter on the first page (0 for ascending, a max sentinel for descending).
        """
        where = dict(where or {})
        op = "_gt" if order_direction == "asc" else "_lt"
        # Seed values that are always outside any real ``id`` / timestamp / volume:
        seed = "0" if order_direction == "asc" else _MAX_SENTINEL
        last_value: Any = seed
        pages_yielded = 0
        total_rows = 0
        cur_page = page_size
        while True:
            w = dict(where)
            # Always include the keyset filter (seed on page 1, real last_value after)
            w[f"{order_by}{op}"] = last_value
            where_str = _where_to_gql(w)
            query = (
                f"query Page($first: Int!) {{\n"
                f"  items: {entity}(first: $first, orderBy: {order_by}, orderDirection: {order_direction}"
                f"{', where: ' + where_str if where_str else ''}) {{\n"
                f"    {fields}\n"
                f"  }}\n"
                f"}}"
            )
            try:
                res = self.query(
                    subgraph,
                    query,
                    variables={"first": cur_page},
                    force_refresh=force_refresh,
                )
            except GoldskyTimeout:
                # Shrink the page and retry; don't advance the cursor
                if cur_page <= 50:
                    raise
                cur_page = max(50, cur_page // 2)
                continue
            # Recover page size after a successful query
            cur_page = min(page_size, cur_page * 2)
            rows = res.data.get("items", [])
            if not rows:
                return
            for row in rows:
                yield row
            total_rows += len(rows)
            pages_yielded += 1
            if len(rows) < page_size:
                return
            if max_pages is not None and pages_yielded >= max_pages:
                return
            # Advance keyset
            last_row = rows[-1]
            last_value = last_row.get(order_by)
            if last_value is None:
                raise GoldskyError(
                    f"cannot paginate: row missing {order_by!r}: {last_row}"
                )

    # --- meta / health -----------------------------------------------------

    def meta(self, subgraph: str) -> dict:
        """Return subgraph indexing metadata (block number, timestamp of last indexed block)."""
        q = "{ _meta { block { number timestamp hash } deployment hasIndexingErrors } }"
        res = self.query(subgraph, q)
        return res.data.get("_meta", {})

    # --- helpers -----------------------------------------------------------

    def _cache_path(self, subgraph: str, query: str, variables: dict) -> Path:
        key = hashlib.sha256(
            json.dumps(
                {"q": query.strip(), "v": _json_canonical(variables)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return self.cache_dir / f"{subgraph}__{key}.json"


# ----------------------------------------------------------------------
# GraphQL where-clause encoding
# ----------------------------------------------------------------------

def _where_to_gql(where: dict) -> str:
    """Convert a Python dict into GraphQL object-literal syntax (not valid JSON)."""
    if not where:
        return ""
    parts = []
    for k, v in where.items():
        parts.append(f"{k}: {_gql_value(v)}")
    return "{" + ", ".join(parts) + "}"


def _gql_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_gql_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return _where_to_gql(v)
    return json.dumps(str(v))


def _json_canonical(obj: Any) -> Any:
    """Make a deterministic representation of a dict for cache-keying."""
    if isinstance(obj, dict):
        return {k: _json_canonical(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [_json_canonical(x) for x in obj]
    return obj
