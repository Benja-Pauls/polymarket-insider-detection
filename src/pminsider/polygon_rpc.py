"""Polygon JSON-RPC client with failover across free public endpoints.

We only need read-only operations: ``eth_blockNumber``, ``eth_getBlockByNumber``,
``eth_getTransactionReceipt``. Used for (a) resolving absolute timestamps for
blocks referenced in subgraph data, (b) cross-validating a sample of trades
against the blockchain directly, (c) backfilling any subgraph gaps.

Public endpoints rotate through a pool; rate-limit / 5xx triggers failover.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from curl_cffi import requests as crequests

PUBLIC_RPCS: list[str] = [
    "https://polygon-bor-rpc.publicnode.com",
    "https://polygon.drpc.org",
    "https://polygon.api.onfinality.io/public",
    "https://1rpc.io/matic",
]


@dataclass
class _EndpointState:
    url: str
    failures: int = 0
    last_failure_ts: float = 0.0
    cooldown_sec: float = 0.0


class PolygonRPC:
    def __init__(self, endpoints: list[str] | None = None, *, timeout_sec: int = 10):
        ep_list = endpoints or PUBLIC_RPCS
        self._endpoints: list[_EndpointState] = [_EndpointState(url=u) for u in ep_list]
        self.timeout_sec = timeout_sec
        self._session = crequests.Session(impersonate="chrome120")

    # --- low-level ---------------------------------------------------------

    def call(self, method: str, params: list | None = None) -> dict:
        params = params or []
        body = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
        last_err: Exception | None = None
        attempts = 0
        # Rotate through healthy endpoints, respecting cooldowns
        for ep in self._shuffled_endpoints():
            now = time.monotonic()
            if ep.last_failure_ts + ep.cooldown_sec > now:
                continue
            attempts += 1
            try:
                r = self._session.post(ep.url, json=body, timeout=self.timeout_sec)
                if r.status_code == 429:
                    ep.failures += 1
                    ep.last_failure_ts = now
                    ep.cooldown_sec = min(60.0, 2 ** ep.failures)
                    continue
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:100]}")
                data = r.json()
                if "error" in data:
                    msg = (data.get("error") or {}).get("message", "")
                    # Unauthorized and method-not-allowed — bad endpoint for this method
                    if "unauthorized" in msg.lower() or "disabled" in msg.lower():
                        ep.failures += 1
                        ep.last_failure_ts = now
                        ep.cooldown_sec = min(60.0, 2 ** ep.failures)
                        continue
                    raise RuntimeError(f"RPC error: {msg}")
                ep.failures = max(0, ep.failures - 1)
                return data.get("result")
            except Exception as e:  # noqa: BLE001
                last_err = e
                ep.failures += 1
                ep.last_failure_ts = now
                ep.cooldown_sec = min(60.0, 2 ** ep.failures)
                continue
        raise RuntimeError(
            f"all {attempts} polygon RPC endpoints failed; last: {last_err}"
        )

    # --- sugar -------------------------------------------------------------

    def block_number(self) -> int:
        return int(self.call("eth_blockNumber"), 16)

    def get_block(self, number_or_hash: int | str, *, full: bool = False) -> dict:
        if isinstance(number_or_hash, int):
            tag = hex(number_or_hash)
        else:
            tag = number_or_hash
        return self.call("eth_getBlockByNumber", [tag, full]) or {}

    def block_timestamp(self, number: int) -> int:
        blk = self.get_block(number, full=False)
        return int(blk.get("timestamp", "0x0"), 16)

    def transaction_receipt(self, tx_hash: str) -> dict:
        return self.call("eth_getTransactionReceipt", [tx_hash]) or {}

    # --- internals ---------------------------------------------------------

    def _shuffled_endpoints(self) -> list[_EndpointState]:
        pool = list(self._endpoints)
        random.shuffle(pool)
        # Prefer healthy endpoints first
        pool.sort(key=lambda ep: ep.failures)
        return pool
