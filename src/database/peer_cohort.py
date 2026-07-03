"""
Peer-cohort benchmarking scaffold — percentile-vs-peers, with an HONEST
insufficient-cohort fallback.

WHY THIS EXISTS
---------------
OASIS today compares an organization against *theoretical* thresholds (the
Window of Viability) and *ecological* reference anchors. Neither is a peer
benchmark. This module adds the mechanism to score a metric (primarily alpha =
relative ascendency) as a PERCENTILE within a size/sector-matched cohort of
OTHER analyzed organizations — but ONLY when a real cohort of at least
``MIN_COHORT_SIZE`` peers exists.

HONESTY CONTRACT (non-negotiable)
---------------------------------
- The cohort is drawn EXCLUSIVELY from networks already persisted in the store
  (see :mod:`src.database.db_manager`). Nothing is invented.
- If the matched cohort has fewer than ``MIN_COHORT_SIZE`` peers, this module
  returns ``{'status': 'insufficient_cohort', ...}`` and NEVER a percentile.
  Percentiles are not extrapolated from a tiny sample.
- ``insufficient_cohort`` is the EXPECTED DEFAULT today: the store does not yet
  hold >=10 sector-matched peers. Growing a real cohort is a separate data
  acquisition task — see :func:`ingest_directory`.

This module contains NO scientific/OASIS formula. It reads already-computed,
already-persisted metrics and computes a standard statistical percentile rank.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Minimum peers (excluding the org itself) required to report a percentile.
# Below this the mechanism falls back to the indicative theoretical reference.
MIN_COHORT_SIZE = 10

# Tier under which full profiles are persisted (mirrors precompute_pipeline).
FULL_PROFILE_TIER = 3
# Fallback tier: plain vectorized metrics (carry relative_ascendency/robustness).
VECTORIZED_TIER = 2

# Size buckets derived from node_count. Boundaries chosen to separate
# micro / small / mid / large organizations by structural scale.
#   micro : < 10 nodes
#   small : 10 - 49
#   mid   : 50 - 249
#   large : >= 250
_SIZE_BUCKET_BOUNDS = (
    ('micro', 10),   # n < 10
    ('small', 50),   # 10 <= n < 50
    ('mid', 250),    # 50 <= n < 250
)
_LARGE_BUCKET = 'large'  # n >= 250


# ---------------------------------------------------------------------------
# Size-bucket derivation
# ---------------------------------------------------------------------------

def size_bucket_from_node_count(node_count: int) -> str:
    """Map a node count to a coarse size bucket.

    micro (<10), small (10-49), mid (50-249), large (250+).
    """
    n = int(node_count or 0)
    for label, upper in _SIZE_BUCKET_BOUNDS:
        if n < upper:
            return label
    return _LARGE_BUCKET


# ---------------------------------------------------------------------------
# Percentile computation (standard, no min-cohort gate)
# ---------------------------------------------------------------------------

def compute_peer_percentile(value: float,
                            cohort_values: Sequence[float]) -> Dict[str, Any]:
    """Percentile rank of ``value`` within ``cohort_values`` + cohort stats.

    Uses the "mean rank" (midpoint) definition of percentile rank::

        percentile = 100 * (count_below + 0.5 * count_equal) / n

    which places a value equal to the cohort median at exactly the 50th
    percentile. Also returns the cohort size and its median / quartiles for
    context.

    Args:
        value: The organization's metric value.
        cohort_values: Peer metric values (should EXCLUDE the org itself).

    Returns:
        {'percentile', 'n', 'median', 'q1', 'q3', 'min', 'max'}.

    Raises:
        ValueError: if ``cohort_values`` is empty (callers should gate on size
            via :func:`peer_benchmark` first).
    """
    arr = np.asarray([float(v) for v in cohort_values], dtype=float)
    n = int(arr.size)
    if n == 0:
        raise ValueError("compute_peer_percentile requires a non-empty cohort")

    v = float(value)
    count_below = int(np.sum(arr < v))
    count_equal = int(np.sum(arr == v))
    percentile = 100.0 * (count_below + 0.5 * count_equal) / n

    return {
        'percentile': float(percentile),
        'n': n,
        'median': float(np.median(arr)),
        'q1': float(np.percentile(arr, 25)),
        'q3': float(np.percentile(arr, 75)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


def peer_benchmark(value: float,
                   cohort_values: Sequence[float],
                   min_cohort_size: int = MIN_COHORT_SIZE) -> Dict[str, Any]:
    """Gate the percentile on cohort size (the HONESTY guard).

    Returns ``{'status': 'insufficient_cohort', 'n': k, 'min': min_cohort_size}``
    when there are fewer than ``min_cohort_size`` peers — and NO percentile.
    Otherwise returns ``{'status': 'ok', **compute_peer_percentile(...)}``.
    """
    values = list(cohort_values)
    n = len(values)
    if n < min_cohort_size:
        return {'status': 'insufficient_cohort', 'n': n, 'min': min_cohort_size}
    result = compute_peer_percentile(value, values)
    result['status'] = 'ok'
    return result


# ---------------------------------------------------------------------------
# Cohort query against the store
# ---------------------------------------------------------------------------

def _extract_member_metrics(blob: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the key cohort metrics out of a stored profile blob.

    Handles both the tier-3 nested full profile ({'core': {...}, 'oasis': {...}})
    and a flat tier-2 vectorized-metrics blob. Missing values are returned as
    None (and are excluded later when collecting metric values).
    """
    if not isinstance(blob, dict):
        return {'relative_ascendency': None, 'robustness': None,
                'oasis_overall': None, 'oasis_dimensions': None}

    core = blob.get('core') if isinstance(blob.get('core'), dict) else blob
    oasis = blob.get('oasis') if isinstance(blob.get('oasis'), dict) else {}

    def _num(d, key):
        val = d.get(key)
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    dims = oasis.get('dimension_scores')
    return {
        'relative_ascendency': _num(core, 'relative_ascendency'),
        'robustness': _num(core, 'robustness'),
        'oasis_overall': _num(oasis, 'overall_score'),
        'oasis_dimensions': dims if isinstance(dims, dict) else None,
    }


def query_cohort(db,
                 size_bucket: Optional[str] = None,
                 sector: Optional[str] = None,
                 exclude_network_id: Optional[int] = None,
                 require_sector_tag: bool = True) -> List[Dict[str, Any]]:
    """Return stored networks (with key metrics) matching the given filters.

    The cohort is the set of persisted networks that:
      - are NOT ``exclude_network_id`` (the org benchmarking itself),
      - are ELIGIBLE peers: when ``require_sector_tag`` is True (the default) a
        network must carry a non-null ``sector`` tag to count. This is the core
        HONESTY safeguard — an untagged network in the store may be an ecological
        reference sample or a synthetic test fixture, NOT a vetted peer
        organization, so it must never be silently counted as a peer. A real
        cohort is populated deliberately via :func:`ingest_directory` (which
        tags a sector).
      - match ``size_bucket`` (derived from node_count) when provided,
      - match ``sector`` when provided (untagged/NULL rows are excluded from a
        sector-filtered query — never fabricated),
      - have a stored profile from which the requested metric can be read.

    Reuses already-precomputed profiles (tier 3, falling back to tier 2). It
    recomputes NOTHING.

    Returns a list of member dicts::

        {'network_id', 'name', 'node_count', 'sector', 'size_bucket',
         'metrics': {'relative_ascendency', 'robustness', 'oasis_overall',
                     'oasis_dimensions'}}
    """
    members: List[Dict[str, Any]] = []
    for net in db.list_networks():
        net_id = net.get('id')
        if exclude_network_id is not None and net_id == exclude_network_id:
            continue

        net_sector = net.get('sector')
        # Eligibility guard: untagged networks are not vetted peers.
        if require_sector_tag and not net_sector:
            continue

        node_count = net.get('node_count') or 0
        bucket = size_bucket_from_node_count(node_count)
        if size_bucket is not None and bucket != size_bucket:
            continue

        if sector is not None and net_sector != sector:
            continue

        blob = db.get_precomputed_metrics(net_id, tier=FULL_PROFILE_TIER)
        if not blob:
            blob = db.get_precomputed_metrics(net_id, tier=VECTORIZED_TIER)
        if not blob:
            continue

        metrics = _extract_member_metrics(blob)
        members.append({
            'network_id': net_id,
            'name': net.get('name'),
            'node_count': node_count,
            'sector': net_sector,
            'size_bucket': bucket,
            'metrics': metrics,
        })
    return members


def cohort_metric_values(members: Sequence[Dict[str, Any]],
                         metric_key: str) -> List[float]:
    """Collect non-null values of ``metric_key`` across cohort members."""
    values: List[float] = []
    for m in members:
        v = m.get('metrics', {}).get(metric_key)
        if v is not None:
            values.append(float(v))
    return values


# ---------------------------------------------------------------------------
# High-level convenience: benchmark alpha against a matched cohort
# ---------------------------------------------------------------------------

def peer_alpha_benchmark(db,
                         alpha: float,
                         node_count: int,
                         sector: Optional[str] = None,
                         exclude_network_id: Optional[int] = None,
                         metric_key: str = 'relative_ascendency',
                         require_sector_tag: bool = True) -> Dict[str, Any]:
    """Benchmark an organization's ``alpha`` against its size/sector cohort.

    Builds the cohort matched to ``node_count``'s size bucket (and ``sector`` if
    given), extracts peer ``metric_key`` values, and applies the size-gated
    :func:`peer_benchmark`. Returns the benchmark status dict augmented with
    ``size_bucket`` and ``sector`` for reporting context.

    Only sector-tagged networks are eligible peers by default
    (``require_sector_tag``) — untagged ecological/synthetic records are never
    counted as peers. On the real store today this returns
    ``insufficient_cohort`` because no >=10 matched, sector-tagged peers exist
    yet — the intended, honest default.
    """
    bucket = size_bucket_from_node_count(node_count)
    members = query_cohort(db, size_bucket=bucket, sector=sector,
                           exclude_network_id=exclude_network_id,
                           require_sector_tag=require_sector_tag)
    values = cohort_metric_values(members, metric_key)
    result = peer_benchmark(alpha, values)
    result['size_bucket'] = bucket
    result['sector'] = sector
    result['metric'] = metric_key
    return result


def format_peer_benchmark_note(result: Dict[str, Any], alpha: float) -> str:
    """Human-readable, HONEST one-liner for the benchmark section.

    - ``insufficient_cohort`` -> states we cannot yet peer-benchmark, gives the
      current cohort size N and the required minimum, and says the indicative
      reference is shown instead. NEVER states a percentile.
    - ``ok`` -> states the percentile within the peer cohort plus the cohort
      size and median (context, not a target).
    """
    status = result.get('status')
    if status == 'ok':
        pct = result['percentile']
        n = result['n']
        median = result.get('median')
        bucket = result.get('size_bucket')
        sector = result.get('sector')
        scope = f"{bucket}-size" + (f" / {sector}-sector" if sector else "")
        median_txt = (f", cohort median alpha = {median:.3f}"
                      if median is not None else "")
        return (
            f"Peer benchmark: this organization's alpha = {alpha:.3f} sits at the "
            f"{pct:.0f}th percentile of {n} matched peer organizations "
            f"({scope} cohort){median_txt}. Percentile reflects position within "
            f"peers, not an absolute target."
        )
    # insufficient_cohort (or any non-ok status): honest fallback
    n = result.get('n', 0)
    minimum = result.get('min', MIN_COHORT_SIZE)
    return (
        f"Peer benchmarking requires a larger comparison set (currently N={n}, "
        f"need >={minimum}); showing indicative reference instead. No peer "
        f"percentile is reported to avoid extrapolating from too few peers."
    )


# ---------------------------------------------------------------------------
# Ingestion path — how a real cohort gets populated
# ---------------------------------------------------------------------------

def _load_flow_matrix(data: Dict[str, Any]):
    """Extract (flow_matrix, node_names, org_name) from a network JSON dict.

    Supports the flow_matrix / flows / matrix naming conventions already used by
    the precompute pipeline.
    """
    if 'flow_matrix' in data:
        matrix = np.array(data['flow_matrix'], dtype=float)
        node_names = data.get('node_names', data.get('nodes'))
    elif 'flows' in data:
        matrix = np.array(data['flows'], dtype=float)
        node_names = data.get('nodes', data.get('node_names'))
    elif 'matrix' in data:
        matrix = np.array(data['matrix'], dtype=float)
        node_names = data.get('nodes', data.get('node_names'))
    else:
        return None, None, None
    org_name = data.get('organization', data.get('name'))
    return matrix, node_names, org_name


def ingest_directory(directory: str,
                     sector: Optional[str] = None,
                     pipeline=None,
                     db=None,
                     recursive: bool = False) -> Dict[str, Any]:
    """Bulk-ingest a directory of network JSONs into the store to grow a cohort.

    This is the documented way to BUILD a real peer cohort: point it at a folder
    of organization network JSONs (optionally all from one ``sector``); each is
    profiled via the standard full-profile precompute (compute-once, then
    persisted as tier-3) and tagged with the sector. Once >=``MIN_COHORT_SIZE``
    size/sector-matched peers exist, :func:`peer_alpha_benchmark` starts
    returning real percentiles automatically.

    Reuses ``pipeline.get_full_profile`` (which itself reuses
    ``precompute_full_profile``); it does not reimplement any metric.

    Args:
        directory: Folder containing ``*.json`` network files.
        sector: Optional sector tag applied to every ingested network.
        pipeline: A PrecomputePipeline (created against ``db`` if omitted).
        db: A DatabaseManager (singleton if omitted).
        recursive: Recurse into subdirectories when True.

    Returns:
        Summary dict: {'ingested', 'skipped', 'errors': [...], 'networks': [...]}.
    """
    # Lazy imports keep this module importable without the DB stack at import time.
    if db is None:
        from .db_manager import get_database_manager
        db = get_database_manager()
    if pipeline is None:
        from .precompute_pipeline import PrecomputePipeline
        pipeline = PrecomputePipeline(db_manager=db)

    base = Path(directory)
    summary: Dict[str, Any] = {
        'ingested': 0, 'skipped': 0, 'errors': [], 'networks': [],
        'sector': sector, 'directory': str(base),
    }
    if not base.exists():
        summary['errors'].append({'file': str(base), 'error': 'directory not found'})
        return summary

    pattern = '**/*.json' if recursive else '*.json'
    for filepath in sorted(base.glob(pattern)):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            matrix, node_names, org_name = _load_flow_matrix(data)
            if matrix is None or matrix.size == 0:
                summary['skipped'] += 1
                summary['errors'].append(
                    {'file': str(filepath), 'error': 'no/empty flow matrix'})
                continue

            org_name = org_name or filepath.stem
            result = pipeline.get_full_profile(matrix, node_names, org_name=org_name)
            network_id = result.get('network_id')

            # Tag the sector on the (now persisted) network record.
            if sector is not None and network_id is not None:
                net_hash = db.compute_network_hash(matrix, node_names)
                n_nodes = int(matrix.shape[0]) if matrix.ndim == 2 else 0
                n_edges = int(np.sum(matrix > 0))
                db.save_network(
                    name=org_name, source_file=str(filepath),
                    node_count=n_nodes, edge_count=n_edges,
                    network_hash=net_hash, sector=sector,
                )

            summary['ingested'] += 1
            summary['networks'].append({
                'name': org_name, 'network_id': network_id,
                'cache_hit': result.get('cache_hit', False), 'sector': sector,
            })
        except json.JSONDecodeError as e:
            summary['errors'].append({'file': str(filepath), 'error': f'JSON: {e}'})
        except Exception as e:  # pragma: no cover - defensive
            summary['errors'].append({'file': str(filepath), 'error': str(e)})

    logger.info(
        "Cohort ingest from %s: %d ingested, %d skipped, %d errors (sector=%s)",
        base, summary['ingested'], summary['skipped'],
        len(summary['errors']), sector,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI entry point — grow a cohort from the command line
# ---------------------------------------------------------------------------

def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest a directory of network JSONs into the OASIS store "
                    "to grow a peer-benchmarking cohort.")
    parser.add_argument('directory', help='Folder of *.json network files')
    parser.add_argument('--sector', default=None,
                        help='Optional sector tag applied to every network')
    parser.add_argument('--recursive', action='store_true',
                        help='Recurse into subdirectories')
    parser.add_argument('--db', default=None, help='Optional SQLite DB path')
    args = parser.parse_args(argv)

    db = None
    if args.db:
        from .db_manager import DatabaseManager
        db = DatabaseManager(db_path=args.db)

    summary = ingest_directory(args.directory, sector=args.sector,
                               db=db, recursive=args.recursive)
    print(json.dumps(summary, indent=2))
    return 0 if not summary['errors'] else 1


if __name__ == '__main__':  # pragma: no cover
    import sys
    sys.exit(_main(sys.argv[1:]))
