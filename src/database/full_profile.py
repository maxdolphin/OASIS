"""
Full-index precompute (Pass A: the core mechanism).

`precompute_full_profile` computes EVERY metric family the app/report displays,
ONCE, by reusing the existing calculators/analyzers (it does NOT reimplement any
metric formula). The result is a single nested dict, stamped with FORMULA_VERSION,
suitable for persisting as a tier=3 JSON blob keyed by network hash.

Families:
- core            : vectorized + extended Ulanowicz metrics (get_extended_metrics)
- oasis           : OASISCalculator(...).get_oasis_profile() (+ interpretation, recommendations)
- network_analysis: AdvancedNetworkAnalyzer(...).get_all_metrics()
- intelligence    : report_intelligence derived views (risk/benchmark/roadmap/esg)
- meta            : n_nodes / n_edges / organization

Each family is guarded independently: a failure on a tiny/degenerate graph
produces an `_error` marker for that family instead of aborting the whole profile.

FORMULA_VERSION
---------------
A short version stamp for the metric formulas. Bumping it INVALIDATES every
stored profile computed under an older version, forcing a recompute on next read
(`get_full_profile` treats a version mismatch as a cache MISS). Bump this whenever
any metric formula changes so stale precomputed values are never silently served.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Bump this whenever any scientific metric formula changes.
# Reflects this week's ENA/OASIS formula corrections
# (effective-connectivity sign fix, single-density definition, roll-up veto, etc.).
FORMULA_VERSION = "2026.07-fixes"


# ---------------------------------------------------------------------------
# Import shims (support both `from src.X` and `from X` execution contexts)
# ---------------------------------------------------------------------------

def _import_calculators():
    """Return (UlanowiczCalculator, OASISCalculator, AdvancedNetworkAnalyzer)."""
    try:
        from ulanowicz_calculator import UlanowiczCalculator
        from oasis_calculator import OASISCalculator
        from network_analyzer import AdvancedNetworkAnalyzer
    except ImportError:
        from src.ulanowicz_calculator import UlanowiczCalculator
        from src.oasis_calculator import OASISCalculator
        from src.network_analyzer import AdvancedNetworkAnalyzer
    return UlanowiczCalculator, OASISCalculator, AdvancedNetworkAnalyzer


def _import_vectorized():
    try:
        from vectorized_metrics import get_all_vectorized_metrics
    except ImportError:
        from src.vectorized_metrics import get_all_vectorized_metrics
    return get_all_vectorized_metrics


def _import_report_intelligence():
    try:
        import report_intelligence as ri
    except ImportError:
        from src import report_intelligence as ri
    return ri


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def precompute_full_profile(flow_matrix,
                            node_names: Optional[List[str]] = None,
                            org_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute the full index profile once and return it as a nested dict.

    Reuses the existing calculators/analyzers; does NOT reimplement any metric.
    A single UlanowiczCalculator is built and shared with the OASISCalculator so
    the Ulanowicz core is computed once.

    Args:
        flow_matrix: Square flow matrix (array-like).
        node_names:  Optional node labels.
        org_name:    Optional organization name (stored in meta).

    Returns:
        Nested dict:
        {
          'formula_version': FORMULA_VERSION,
          'core': {...}, 'oasis': {...}, 'network_analysis': {...},
          'intelligence': {...}, 'meta': {...}
        }
    """
    flow_matrix = np.asarray(flow_matrix, dtype=np.float64)
    n_nodes = int(flow_matrix.shape[0]) if flow_matrix.ndim == 2 else 0
    if node_names is None:
        node_names = [f"N{i}" for i in range(n_nodes)]
    n_edges = int(np.sum(flow_matrix > 0))

    UlanowiczCalculator, OASISCalculator, AdvancedNetworkAnalyzer = _import_calculators()

    profile: Dict[str, Any] = {
        'formula_version': FORMULA_VERSION,
        'core': {},
        'oasis': {},
        'network_analysis': {},
        'intelligence': {},
        'meta': {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'organization': org_name or 'Unknown',
        },
    }

    # --- Shared Ulanowicz calculator (built once, reused by core + oasis) ----
    calc = None
    try:
        calc = UlanowiczCalculator(flow_matrix, node_names)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"UlanowiczCalculator construction failed: {e}")

    # --- Family: core (vectorized + extended Ulanowicz) ---------------------
    profile['core'] = _family_core(flow_matrix, calc)

    # --- Family: network_analysis ------------------------------------------
    analyzer = None
    try:
        analyzer = AdvancedNetworkAnalyzer(flow_matrix, node_names)
    except Exception as e:
        logger.warning(f"AdvancedNetworkAnalyzer construction failed: {e}")
    profile['network_analysis'] = _family_network_analysis(analyzer)

    # --- Family: oasis (shares the Ulanowicz calculator) --------------------
    oasis_calc = None
    if calc is not None:
        try:
            oasis_calc = OASISCalculator(calc, network_analyzer=analyzer)
        except Exception as e:
            logger.warning(f"OASISCalculator construction failed: {e}")
    profile['oasis'] = _family_oasis(oasis_calc)

    # --- Family: intelligence (derived from oasis profile + core metrics) ---
    profile['intelligence'] = _family_intelligence(
        oasis_profile=profile['oasis'],
        core_metrics=profile['core'],
        oasis_calc=oasis_calc,
    )

    return profile


# ---------------------------------------------------------------------------
# Per-family builders (each guarded independently)
# ---------------------------------------------------------------------------

def _family_core(flow_matrix, calc) -> Dict[str, Any]:
    """Vectorized metrics + extended Ulanowicz metrics, merged."""
    core: Dict[str, Any] = {}
    try:
        get_all_vectorized_metrics = _import_vectorized()
        core.update(get_all_vectorized_metrics(flow_matrix))
    except Exception as e:
        logger.warning(f"vectorized metrics failed: {e}")
        core['_vectorized_error'] = str(e)

    if calc is not None:
        try:
            extended = calc.get_extended_metrics()
            # extended supersedes vectorized where keys overlap (fuller formulas)
            core.update(extended)
        except Exception as e:
            logger.warning(f"extended Ulanowicz metrics failed: {e}")
            core['_extended_error'] = str(e)

        # Finn Cycling Index (may be size-gated / None in the calculator)
        if 'finn_cycling_index' not in core:
            try:
                core['finn_cycling_index'] = calc.calculate_finn_cycling_index()
            except Exception:
                core['finn_cycling_index'] = None
    else:
        core['_error'] = 'UlanowiczCalculator unavailable'

    return core


def _family_network_analysis(analyzer) -> Dict[str, Any]:
    if analyzer is None:
        return {'_error': 'AdvancedNetworkAnalyzer unavailable'}
    try:
        return analyzer.get_all_metrics()
    except Exception as e:
        logger.warning(f"network analysis failed: {e}")
        return {'_error': str(e)}


def _family_oasis(oasis_calc) -> Dict[str, Any]:
    if oasis_calc is None:
        return {'_error': 'OASISCalculator unavailable'}
    oasis: Dict[str, Any] = {}
    try:
        oasis = oasis_calc.get_oasis_profile()
    except Exception as e:
        logger.warning(f"OASIS profile failed: {e}")
        return {'_error': str(e)}

    # Interpretation + recommendations are cheap, derived views; guard separately.
    try:
        oasis['interpretation'] = oasis_calc.get_oasis_interpretation()
    except Exception as e:
        oasis['_interpretation_error'] = str(e)
    try:
        oasis['recommendations'] = oasis_calc.get_recommendations()
    except Exception as e:
        oasis['_recommendations_error'] = str(e)

    return oasis


def _family_intelligence(oasis_profile: Dict[str, Any],
                         core_metrics: Dict[str, Any],
                         oasis_calc) -> Dict[str, Any]:
    """report_intelligence derived views. Pure lookups over the passed dicts."""
    if not isinstance(oasis_profile, dict) or '_error' in oasis_profile:
        return {'_error': 'OASIS profile unavailable; intelligence skipped'}

    ri = _import_report_intelligence()
    intel: Dict[str, Any] = {}

    metrics = core_metrics if isinstance(core_metrics, dict) else {}

    try:
        intel['risk'] = ri.build_risk_view(metrics, oasis_profile)
    except Exception as e:
        intel['_risk_error'] = str(e)

    try:
        intel['benchmark'] = ri.build_benchmark_view(metrics, oasis_profile)
    except Exception as e:
        intel['_benchmark_error'] = str(e)

    try:
        intel['esg_crosswalk'] = ri.build_esg_crosswalk(oasis_profile, metrics)
    except Exception as e:
        intel['_esg_error'] = str(e)

    try:
        recs = oasis_profile.get('recommendations')
        if recs is None and oasis_calc is not None:
            recs = oasis_calc.get_recommendations()
        intel['roadmap'] = ri.build_action_roadmap(recs or [], oasis_profile)
    except Exception as e:
        intel['_roadmap_error'] = str(e)

    return intel
