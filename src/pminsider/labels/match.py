"""Link canonical callout-incidents to on-chain trade episodes.

Given:
  - A catalog of resolved markets (data/raw/goldsky/markets.parquet)
  - Per-market trade files (data/raw/goldsky/trades/*.parquet)
  - Optional market metadata from the GH Actions proxy
    (data/offchain/markets_metadata.parquet)
  - A list of canonical Incidents (data/labels/incidents/incidents.parquet)

Produce for each incident:
  1. Best-match condition_id by fuzzy question-text matching or
     resolution-timestamp + category match.
  2. Within that market, the trade cluster matching the incident's
     (ts_lower, ts_upper, size, wallet) signature. A cluster is a set of
     trades from the SAME wallet inside the time window, aggregated.

Matching is scored 0-1; ambiguous / low-score matches are flagged for
manual review rather than auto-confirmed.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class TradeCluster:
    wallet: str
    first_trade_ts: int
    last_trade_ts: int
    n_trades: int
    total_usd_in: float     # USDC paid into the market by this wallet
    total_usd_out: float    # USDC received from the market by this wallet
    net_usd: float          # total_usd_out - total_usd_in (profit if positive)
    direction: str | None   # BUY / SELL majority
    trades_ids: list[str] = field(default_factory=list)


@dataclass
class IncidentMatch:
    incident_id: str
    condition_id: str | None
    market_question: str | None
    question_match_score: float        # 0-1
    cluster: TradeCluster | None
    cluster_match_score: float          # 0-1 (timing / size / wallet agreement)
    overall_score: float                # product of the two
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Question-text → condition_id matching
# ----------------------------------------------------------------------

def _tokenize(q: str) -> set[str]:
    tokens = re.findall(r"\w+", (q or "").lower())
    return {t for t in tokens if len(t) >= 3}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_market(
    incident_q: str,
    catalog: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> tuple[str | None, str | None, float]:
    """Return (condition_id, matched_question_text, score) best-matching this callout."""
    if metadata is None or metadata.empty:
        return None, None, 0.0

    q_tokens = _tokenize(incident_q)
    best_score = 0.0
    best_qid = None
    best_text = None
    for _, m in metadata.iterrows():
        txt = m.get("question") or ""
        score = _jaccard(q_tokens, _tokenize(txt))
        if score > best_score:
            best_score = score
            best_qid = m.get("conditionId")
            best_text = txt
    if not best_qid:
        return None, None, 0.0
    # Verify conditionId is actually in our catalog (has trades)
    if best_qid in set(catalog["condition_id"]):
        return best_qid, best_text, best_score
    # Near-miss: catalog has historical tier condition-ids; try case-insensitive match
    lower = {c.lower() for c in catalog["condition_id"]}
    if best_qid.lower() in lower:
        return best_qid, best_text, best_score
    return None, best_text, best_score


# ----------------------------------------------------------------------
# Trade-cluster matching within a market
# ----------------------------------------------------------------------

def _parse_ts(s: str | None) -> int | None:
    if not s:
        return None
    try:
        return int(datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp())
    except Exception:  # noqa: BLE001
        return None


def cluster_wallet_episodes(
    trades: pd.DataFrame,
    ts_lower: int | None,
    ts_upper: int | None,
) -> list[TradeCluster]:
    """Group trades by wallet within the window and compute net position change.

    For CLOB Polymarket trades in our normalized format:
      - a BUY means the wallet paid USDC and received outcome tokens
      - a SELL means the wallet sent outcome tokens and received USDC
    We roll up per wallet over the window.
    """
    if trades.empty:
        return []
    df = trades.copy()
    if ts_lower is not None:
        df = df[df["timestamp"] >= ts_lower]
    if ts_upper is not None:
        df = df[df["timestamp"] <= ts_upper]
    if df.empty:
        return []

    clusters: dict[str, dict] = {}

    def _touch(wallet, ts, usd_spent, side, trade_id, is_maker):
        if not wallet:
            return
        c = clusters.setdefault(wallet, {
            "wallet": wallet,
            "first_ts": ts, "last_ts": ts,
            "n": 0, "usd_in": 0.0, "usd_out": 0.0,
            "buy_count": 0, "sell_count": 0,
            "trade_ids": [],
        })
        c["first_ts"] = min(c["first_ts"], ts)
        c["last_ts"] = max(c["last_ts"], ts)
        c["n"] += 1
        c["trade_ids"].append(trade_id)
        # In our normalized schema, `side` is BUY/SELL from the perspective of
        # the market's outcome token. We need to figure out who paid USDC.
        #   BUY: someone bought outcome tokens → the one who paid USDC is the one
        #        whose assetId was "0" (the collateral). In our trade record,
        #        the maker paid USDC if maker_asset == "0" (i.e. the maker in
        #        our unnormalized trade had makerAssetId = "0"). But our
        #        `_normalize_trade` loses that distinction and just tags
        #        side=BUY/SELL.
        # Simplification: attribute usd_spent to whichever side we're seeing;
        # maker vs taker roll-up is best-effort. For matching, what matters
        # is that the WALLET is active during the window and the USD magnitude
        # is on the right order.
        if is_maker:
            # Maker's perspective: opposite side economically for the outcome token.
            if side == "BUY":
                c["usd_out"] += float(usd_spent)
                c["sell_count"] += 1
            else:
                c["usd_in"] += float(usd_spent)
                c["buy_count"] += 1
        else:
            if side == "BUY":
                c["usd_in"] += float(usd_spent)
                c["buy_count"] += 1
            else:
                c["usd_out"] += float(usd_spent)
                c["sell_count"] += 1

    for _, r in df.iterrows():
        ts = int(r["timestamp"])
        usd = float(r.get("usd_spent_usdc") or 0)
        side = r.get("side")
        tid = str(r.get("trade_id") or "")
        _touch(r.get("maker"), ts, usd, side, tid, True)
        _touch(r.get("taker"), ts, usd, side, tid, False)

    out: list[TradeCluster] = []
    for w, c in clusters.items():
        buy = c["buy_count"]
        sell = c["sell_count"]
        direction = "BUY" if buy > sell else "SELL" if sell > buy else None
        out.append(TradeCluster(
            wallet=w,
            first_trade_ts=int(c["first_ts"]),
            last_trade_ts=int(c["last_ts"]),
            n_trades=int(c["n"]),
            total_usd_in=float(c["usd_in"]),
            total_usd_out=float(c["usd_out"]),
            net_usd=float(c["usd_out"] - c["usd_in"]),
            direction=direction,
            trades_ids=list(c["trade_ids"]),
        ))

    out.sort(key=lambda c: c.total_usd_in + c.total_usd_out, reverse=True)
    return out


def score_cluster_match(
    cluster: TradeCluster,
    incident: dict,
) -> tuple[float, list[str]]:
    """Heuristic [0,1] score for how well a cluster matches an incident."""
    notes: list[str] = []
    score = 0.0

    # Wallet match (if we have it)
    inc_wallets = []
    if incident.get("wallet_addrs"):
        try:
            inc_wallets = json.loads(incident["wallet_addrs"]) if isinstance(incident["wallet_addrs"], str) else incident["wallet_addrs"]
        except Exception:  # noqa: BLE001
            pass
    if inc_wallets:
        wl = [w.lower() for w in inc_wallets]
        if cluster.wallet.lower() in wl:
            score += 0.5
            notes.append("wallet_match_explicit")

    # Size agreement
    inc_size = incident.get("size_usd_approx")
    if inc_size:
        size_ratio = min(cluster.total_usd_in, inc_size) / max(cluster.total_usd_in, inc_size) if cluster.total_usd_in > 0 else 0
        if size_ratio >= 0.5:
            score += 0.25
            notes.append(f"size_match ({cluster.total_usd_in:.0f} vs alleged {inc_size:.0f})")
        elif size_ratio >= 0.1:
            score += 0.10
            notes.append(f"size_partial ({cluster.total_usd_in:.0f} vs alleged {inc_size:.0f})")

    # Direction agreement
    if incident.get("direction") and cluster.direction:
        if incident["direction"] == cluster.direction:
            score += 0.10
            notes.append("direction_match")

    # Activity level — need at least 5 trades to be a real cluster
    if cluster.n_trades >= 5:
        score += 0.15
        notes.append(f"active_cluster ({cluster.n_trades} trades)")

    return min(1.0, score), notes


# ----------------------------------------------------------------------
# Top-level match for one incident
# ----------------------------------------------------------------------

def match_incident(
    incident_row: dict,
    catalog: pd.DataFrame,
    metadata: pd.DataFrame | None,
    trades_dir: Path,
    *,
    top_k_clusters: int = 5,
) -> IncidentMatch:
    cond_id, matched_q, q_score = find_market(
        incident_row.get("market_question", ""),
        catalog,
        metadata,
    )
    if not cond_id:
        return IncidentMatch(
            incident_id=incident_row["incident_id"],
            condition_id=None,
            market_question=matched_q,
            question_match_score=q_score,
            cluster=None,
            cluster_match_score=0.0,
            overall_score=0.0,
            notes=["no_market_match"],
        )

    tp = trades_dir / f"{cond_id}.parquet"
    if not tp.exists() or tp.stat().st_size < 100:
        return IncidentMatch(
            incident_id=incident_row["incident_id"],
            condition_id=cond_id,
            market_question=matched_q,
            question_match_score=q_score,
            cluster=None,
            cluster_match_score=0.0,
            overall_score=0.0,
            notes=["market_matched_no_trades_file"],
        )
    try:
        trades = pd.read_parquet(tp)
    except Exception:  # noqa: BLE001
        return IncidentMatch(
            incident_id=incident_row["incident_id"],
            condition_id=cond_id,
            market_question=matched_q,
            question_match_score=q_score,
            cluster=None,
            cluster_match_score=0.0,
            overall_score=0.0,
            notes=["trades_file_unreadable"],
        )

    clusters = cluster_wallet_episodes(
        trades,
        ts_lower=_parse_ts(incident_row.get("ts_lower")),
        ts_upper=_parse_ts(incident_row.get("ts_upper")),
    )

    # Score each cluster, keep best
    best: TradeCluster | None = None
    best_score = 0.0
    best_notes: list[str] = []
    for c in clusters[:top_k_clusters]:
        s, notes = score_cluster_match(c, incident_row)
        if s > best_score:
            best_score = s
            best = c
            best_notes = notes

    overall = q_score * (0.5 + 0.5 * best_score)  # weighted
    return IncidentMatch(
        incident_id=incident_row["incident_id"],
        condition_id=cond_id,
        market_question=matched_q,
        question_match_score=q_score,
        cluster=best,
        cluster_match_score=best_score,
        overall_score=overall,
        notes=best_notes,
    )


def match_all(
    incidents: pd.DataFrame,
    catalog: pd.DataFrame,
    metadata: pd.DataFrame | None,
    trades_dir: Path,
) -> pd.DataFrame:
    rows = []
    for _, inc in incidents.iterrows():
        m = match_incident(inc.to_dict(), catalog, metadata, trades_dir)
        row = {
            "incident_id": m.incident_id,
            "condition_id": m.condition_id,
            "market_question": m.market_question,
            "question_match_score": m.question_match_score,
            "cluster_match_score": m.cluster_match_score,
            "overall_score": m.overall_score,
            "notes": "; ".join(m.notes),
        }
        if m.cluster:
            row.update({
                "cluster_wallet": m.cluster.wallet,
                "cluster_first_ts": m.cluster.first_trade_ts,
                "cluster_last_ts": m.cluster.last_trade_ts,
                "cluster_n_trades": m.cluster.n_trades,
                "cluster_usd_in": m.cluster.total_usd_in,
                "cluster_usd_out": m.cluster.total_usd_out,
                "cluster_net_usd": m.cluster.net_usd,
                "cluster_direction": m.cluster.direction,
            })
        rows.append(row)
    return pd.DataFrame(rows)
