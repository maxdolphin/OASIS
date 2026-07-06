"""Stage 2 (pure): turn stored Gmail rows into a weighted flow matrix.

Hybrid weighting per directed node pair (a -> b):
    volume(a,b)  = sum_i exp(-ln2/half_life * (now - t_i))   # recency decay
    sustain(a,b) = 1 + beta * ln(1 + A)                      # A = # distinct active 7-day epoch-aligned windows
    weight(a,b)  = volume(a,b) * sustain(a,b)

No wall clock is read here: `now_utc` is an explicit argument.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple

try:
    from network_ingestion import build_flow_matrix_from_edges, ParseResult
except ImportError:  # pragma: no cover - import path when run as a package
    from src.network_ingestion import build_flow_matrix_from_edges, ParseResult

_WEEK = 7 * 86400


def _dept(orgunit: str) -> str:
    """Leaf of an orgUnitPath: '/Sales/EMEA' -> 'EMEA'; '/' or '' -> 'Root'."""
    if not orgunit:
        return "Root"
    leaf = orgunit.rstrip("/").split("/")[-1]
    return leaf or "Root"


def build_flow_matrix(
    rows: List[Dict],
    org_users: Set[str],
    now_utc: int,
    window_seconds: int,
    half_life_seconds: int,
    beta: float,
    granularity: str = "individual",
) -> Tuple[ParseResult, int]:
    """Build a weighted flow matrix from raw interaction rows.

    Args:
        rows: raw interaction dicts (see GmailInteractionStore).
        org_users: lower-cased set of known internal email addresses.
        now_utc: reference time (epoch s) for decay — supplied, never clock-read.
        window_seconds: only messages with ts_utc >= now_utc - window_seconds count.
        half_life_seconds: recency-decay half life.
        beta: sustained-engagement coefficient (>= 0).
        granularity: 'individual' (node=email) or 'department' (node=orgUnit leaf).

    Messages with ``ts_utc > now_utc`` (future timestamps / clock skew) are
    excluded, as are rows older than ``now_utc - window_seconds``.

    Returns:
        (ParseResult, dropped_external_count).
    """
    if half_life_seconds <= 0:
        raise ValueError(f"half_life_seconds must be > 0, got {half_life_seconds!r}")
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta!r}")
    if window_seconds < 0:
        raise ValueError(f"window_seconds must be >= 0, got {window_seconds!r}")

    lam = math.log(2) / float(half_life_seconds)
    cutoff = now_utc - window_seconds

    # Accumulators keyed by (src_node, dst_node).
    volume: Dict[Tuple[str, str], float] = defaultdict(float)
    weeks: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    dropped_external = 0

    for r in rows:
        ts = int(r["ts_utc"])
        if ts < cutoff or ts > now_utc:
            continue
        src_email = str(r["src_email"]).strip().lower()
        dst_email = str(r["dst_email"]).strip().lower()
        # External filtering: both endpoints must be known org users.
        if src_email not in org_users or dst_email not in org_users:
            dropped_external += 1
            continue
        if granularity == "department":
            src_node = _dept(r.get("src_orgunit"))
            dst_node = _dept(r.get("dst_orgunit"))
        else:
            src_node = src_email
            dst_node = dst_email
        if src_node == dst_node:
            continue  # ignore intra-node self-flow
        key = (src_node, dst_node)
        volume[key] += math.exp(-lam * (now_utc - ts))
        weeks[key].add(ts // _WEEK)

    edges = []
    for key, vol in volume.items():
        active_weeks = len(weeks[key])
        sustain = 1.0 + beta * math.log(1 + active_weeks)
        edges.append((key[0], key[1], vol * sustain))

    parsed = build_flow_matrix_from_edges(edges)
    return parsed, dropped_external
