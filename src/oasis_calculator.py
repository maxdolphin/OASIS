"""
OASIS Model Calculator for Ecosystemic Organizational Sustainability

OASIS = Open, Autonomous, Symbiotic, Intelligent, Sustainable

This module implements the OASIS adaptive organization model integrated with
Ulanowicz's ecosystem theory and the 10 Principles of Regenerative Economics
from Fath et al. (2019) "Measuring regenerative economics: 10 principles and
measures undergirding systemic economic health" (Global Transitions 1, 15-27).

Scientific Foundation:
- Fath, Fiscus, Goerner, Berea, Ulanowicz (2019) - 10 Principles of Regenerative Economics
- Ulanowicz et al. (2009) - Quantifying Sustainability
- Zorach & Ulanowicz (2003) - Quantifying Complexity of Flow Networks

OASIS Dimension Mapping to Fath et al. (2019) Principles:
- OPEN: Principles 1, 3, 4 (Cross-scale circulation, inputs/outputs)
- AUTONOMOUS: Principles 2, 9 (Regenerative re-investment, autocatalytic cycles)
- SYMBIOTIC: Principles 5, 8 (Balance of sizes, mutualism)
- INTELLIGENT: Principles 7, 10 (Diversity of roles, adaptive learning)
- SUSTAINABLE: Principle 6 (Resilience-efficiency balance, Window of Vitality)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
import math


# ---------------------------------------------------------------------------
# Named context WEIGHTING PROFILES for the OASIS composite
# ---------------------------------------------------------------------------
# Per docs/business-revision/evidence/expert-org-management.md §3: keep the
# equal 20% weighting as the PUBLISHED, honest default (no false precision — no
# peer-reviewed weighting exists for these specific network constructs), but
# expose a small number of NAMED, context-tagged profiles a consultant can
# select as a diagnostic LENS. Only MODEST tilts are defensible (±0.05–0.08 from
# 0.20); NO extreme weightings. Dimension → org-design construct mapping (expert
# review §3.2):
#   open        ↔ external adaptability / boundary-spanning / environmental sensing
#   autonomous  ↔ distributed decision rights / empowerment / self-management
#   symbiotic   ↔ cross-functional collaboration / psychological safety
#   intelligent ↔ information-processing / learning / knowledge diversity
#   sustainable ↔ long-term resilience / structural balance / adaptive capacity
#
# IMPORTANT: re-weighting only recombines the FIVE ALREADY-COMPUTED dimension
# scores into a new OVERALL score + capped status. It does NOT change any
# dimension score or metric formula. Every profile's weights MUST sum to exactly
# 1.0 over exactly the five dimensions (validated at import time below).
WEIGHTING_PROFILES: Dict[str, Dict[str, Any]] = {
    'Balanced (default)': {
        'weights': {
            'open': 0.20,
            'autonomous': 0.20,
            'symbiotic': 0.20,
            'intelligent': 0.20,
            'sustainable': 0.20,
        },
        'description': (
            "Equal 20% across all five dimensions — the published, honest "
            "default. No lens applied; use when you have no reason to privilege "
            "one dimension over another (avoids false precision)."
        ),
    },
    'Scale-up / Growth': {
        'weights': {
            'open': 0.25,
            'intelligent': 0.25,
            'autonomous': 0.20,
            'symbiotic': 0.15,
            'sustainable': 0.15,
        },
        'description': (
            "Modest emphasis on Open + Intelligent (external adaptability and "
            "learning). Use for fast-growing organizations in changing markets "
            "where sensing and knowledge-processing dominate durable performance."
        ),
    },
    'Efficiency / Turnaround': {
        'weights': {
            'autonomous': 0.25,
            'sustainable': 0.25,
            'intelligent': 0.20,
            'open': 0.15,
            'symbiotic': 0.15,
        },
        'description': (
            "Modest emphasis on Autonomous + Sustainable (decision clarity and "
            "structural discipline / viability). Use for cost-out, restructuring "
            "or turnaround contexts prioritizing operational discipline."
        ),
    },
    'Regulated / Resilience-first': {
        'weights': {
            'sustainable': 0.28,
            'symbiotic': 0.25,
            'autonomous': 0.20,
            'open': 0.15,
            'intelligent': 0.12,
        },
        'description': (
            "Modest emphasis on Symbiotic + Sustainable (coordinated control and "
            "durability). Use for regulated, safety-critical or resilience-first "
            "organizations where long-term viability and coordination dominate."
        ),
    },
}


# Guard: every profile must cover exactly the five dimensions and sum to 1.0.
_OASIS_DIMENSIONS = frozenset(
    {'open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable'})
for _name, _profile in WEIGHTING_PROFILES.items():
    _w = _profile['weights']
    assert frozenset(_w.keys()) == _OASIS_DIMENSIONS, (
        f"Weighting profile '{_name}' must cover exactly the five OASIS "
        f"dimensions, got {sorted(_w.keys())}")
    assert abs(sum(_w.values()) - 1.0) < 1e-9, (
        f"Weighting profile '{_name}' weights must sum to 1.0, "
        f"got {sum(_w.values())}")
del _name, _profile, _w


class OASISCalculator:
    """
    Calculate OASIS organizational health scores from Ulanowicz and network metrics.

    The OASIS model provides a framework for assessing organizational health
    across five dimensions that map to ecological sustainability principles.

    Each dimension is scored 0-100, with higher scores indicating better health.
    """

    # Default weights for each dimension (equal by default)
    DEFAULT_WEIGHTS = {
        'open': 0.20,
        'autonomous': 0.20,
        'symbiotic': 0.20,
        'intelligent': 0.20,
        'sustainable': 0.20
    }

    # ------------------------------------------------------------------
    # Per-dimension normalization caps (0-100 mapping)
    # ------------------------------------------------------------------
    # Each dimension's raw score is a convex combination of sub-metrics that
    # are each in [0, 1], so the raw dimension score is itself in [0, 1]. These
    # caps are the `max_val` used by `_normalize_to_100(raw, 0, cap)`: a raw
    # sub-score >= cap maps to 100. A cap < 1.0 therefore compresses the top of
    # the scale (raw values above the cap all saturate at 100).
    #
    # IMPORTANT (research-integrity note, same conservative stance as the
    # viability-window caveat): these cap VALUES are CALIBRATION PARAMETERS that
    # are PENDING EMPIRICAL DERIVATION from a reference corpus of organizational
    # flow-networks. They are NOT theoretically-derived maxima. They are kept at
    # their historical values here and only CENTRALIZED + DOCUMENTED so that a
    # future empirical re-baseline is a single-line change. Do NOT substitute a
    # different arbitrary number without a corpus + re-baseline decision.
    #
    # Size dependence: OPEN, INTELLIGENT and SYMBIOTIC carry the network's SIZE
    # dependence (their sub-metrics — betweenness, clustering, roles, effective
    # nodes — scale with node count n and are size-normalized upstream). Those
    # dimensions are where a size-relative cap would eventually matter most.
    # SUSTAINABLE is SIZE-INVARIANT (built from alpha = A/C and the robustness
    # proxy R = -alpha*log(alpha), which are ratios independent of n), so its
    # cap is a pure scale choice, not a size gauge.
    #
    # No principled theoretical max is available for any of these five caps
    # (each raw score's true attainable maximum depends on the empirical
    # distribution of the constituent metrics, not on a closed-form bound), so
    # all values are left unchanged and marked calibration-pending.
    DIMENSION_NORMALIZATION_CAPS = {
        'open': 0.6,          # calibration pending (size-sensitive dimension)
        'autonomous': 0.5,    # calibration pending
        'symbiotic': 0.7,     # calibration pending (size-sensitive dimension)
        'intelligent': 0.6,   # calibration pending (size-sensitive dimension)
        'sustainable': 0.8,   # calibration pending (size-INVARIANT dimension)
    }

    # Health thresholds for interpretation
    HEALTH_THRESHOLDS = {
        'open': {'healthy': (50, 85), 'warning': (30, 50), 'critical': (0, 30)},
        'autonomous': {'healthy': (40, 80), 'warning': (25, 40), 'critical': (0, 25)},
        'symbiotic': {'healthy': (55, 90), 'warning': (35, 55), 'critical': (0, 35)},
        'intelligent': {'healthy': (45, 85), 'warning': (30, 45), 'critical': (0, 30)},
        'sustainable': {'healthy': (60, 95), 'warning': (40, 60), 'critical': (0, 40)}
    }

    def __init__(self, ulanowicz_calculator, network_analyzer=None,
                 dimension_weights: Optional[Dict[str, float]] = None):
        """
        Initialize OASIS calculator with Ulanowicz calculator and optional network analyzer.

        Args:
            ulanowicz_calculator: UlanowiczCalculator instance with computed metrics
            network_analyzer: Optional AdvancedNetworkAnalyzer instance for additional metrics
            dimension_weights: Optional custom weights for each dimension (must sum to 1.0)
        """
        self.ulanowicz = ulanowicz_calculator
        self.network_analyzer = network_analyzer

        # Set dimension weights
        if dimension_weights:
            total = sum(dimension_weights.values())
            if abs(total - 1.0) > 0.01:
                raise ValueError(f"Dimension weights must sum to 1.0, got {total}")
            self.weights = dimension_weights
        else:
            self.weights = self.DEFAULT_WEIGHTS.copy()

        # Cache for computed metrics
        self._metrics_cache = None
        self._network_metrics_cache = None

    def _get_ulanowicz_metrics(self) -> Dict[str, Any]:
        """Get cached or compute Ulanowicz metrics."""
        if self._metrics_cache is None:
            self._metrics_cache = self.ulanowicz.get_extended_metrics()
        return self._metrics_cache

    def _get_network_metrics(self) -> Dict[str, Any]:
        """Get cached or compute network analyzer metrics."""
        if self._network_metrics_cache is None:
            if self.network_analyzer:
                self._network_metrics_cache = self.network_analyzer.get_all_metrics()
            else:
                self._network_metrics_cache = {}
        return self._network_metrics_cache

    def _normalize_to_100(self, value: float, min_val: float = 0, max_val: float = 1) -> float:
        """Normalize a value to 0-100 scale."""
        if max_val <= min_val:
            return 50.0
        normalized = (value - min_val) / (max_val - min_val)
        return max(0, min(100, normalized * 100))

    def calculate_autocatalytic_index(self) -> Dict[str, Any]:
        """
        Detect and count autocatalytic (positive feedback) cycles in the network.

        An autocatalytic cycle is a closed loop where resources/information cycle
        back to reinforce earlier stages. This is key to Fath et al. (2019)
        Principle 9: Constructive vs Extractive.

        Returns:
            Dictionary with:
            - count: Number of cycles detected
            - max_length: Maximum cycle length
            - mean_length: Mean cycle length
            - cycle_flow_ratio: Proportion of flow involved in cycles
            - autocatalytic_index: Normalized index (0-1)
        """
        flow_matrix = self.ulanowicz.flow_matrix
        n_nodes = self.ulanowicz.n_nodes

        # Create directed graph
        G = nx.DiGraph()
        for i in range(n_nodes):
            G.add_node(i)
            for j in range(n_nodes):
                if flow_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=flow_matrix[i, j])

        # Find simple cycles (limit length for computational tractability)
        max_cycle_length = min(6, n_nodes)
        cycles = []

        try:
            # Use Johnson's algorithm for finding all simple cycles
            # Limit to reasonable number for large networks
            cycle_gen = nx.simple_cycles(G)
            cycle_count = 0
            max_cycles = 1000  # Limit for large networks

            for cycle in cycle_gen:
                if len(cycle) <= max_cycle_length:
                    cycles.append(cycle)
                    cycle_count += 1
                    if cycle_count >= max_cycles:
                        break
        except Exception:
            # Fall back to simpler approach for problematic graphs
            cycles = []

        # Calculate cycle statistics
        if not cycles:
            return {
                'count': 0,
                'max_length': 0,
                'mean_length': 0,
                'cycle_flow_ratio': 0,
                'autocatalytic_index': 0
            }

        cycle_lengths = [len(c) for c in cycles]

        # Calculate flow involved in cycles
        tst = self.ulanowicz.calculate_tst()
        cycle_flow = 0

        for cycle in cycles:
            # Get minimum flow in cycle (limiting factor)
            min_flow = float('inf')
            for i in range(len(cycle)):
                src = cycle[i]
                dst = cycle[(i + 1) % len(cycle)]
                flow = flow_matrix[src, dst]
                min_flow = min(min_flow, flow)
            if min_flow < float('inf'):
                cycle_flow += min_flow

        cycle_flow_ratio = cycle_flow / tst if tst > 0 else 0

        # Autocatalytic index: combination of cycle count and flow ratio
        # Normalized to account for network size
        expected_cycles = n_nodes * (n_nodes - 1) / 2  # Rough expectation
        count_factor = min(1, len(cycles) / max(1, expected_cycles))

        # Flow component: use the cycle_flow_ratio DIRECTLY (it is already a
        # proportion in [0, 1] — the fraction of total system throughput that
        # cycles). The former `* 10` amplifier had NO basis and saturated the
        # component to 1.0 for any network with >10% cycled flow, hiding real
        # variation in cyclic re-investment. Removing it de-saturates the term;
        # the clamp to 1 is retained only as a numerical guard.
        flow_component = min(1.0, cycle_flow_ratio)

        autocatalytic_index = 0.5 * count_factor + 0.5 * flow_component

        return {
            'count': len(cycles),
            'max_length': max(cycle_lengths) if cycle_lengths else 0,
            'mean_length': np.mean(cycle_lengths) if cycle_lengths else 0,
            'cycle_flow_ratio': cycle_flow_ratio,
            'autocatalytic_index': autocatalytic_index
        }

    # Condition-number cutoff for the integral-utility inversion. Real flow
    # networks yield cond(I - D) < ~10; a value this large signals a near-singular
    # (I - D) whose inverse would blow up (e.g. det ~ 1e-6 -> U ~ 1e6 -> b:c
    # explodes). We fall back to direct-only above this cutoff. (E-scale margin:
    # ~5-6 orders above any observed real-network condition number.)
    _INTEGRAL_UTILITY_COND_MAX = 1e6

    def _build_direct_utility_matrix(self) -> np.ndarray:
        """
        Patten direct utility matrix D.

            d_ij = (f_ij - f_ji) / T_i

        where T_i is the throughflow of node i.

        THROUGHFLOW CAVEAT (research-integrity note): this uses T_i = the internal
        row-sum of outgoing flows (internal outgoing throughflow) as a proxy for
        Patten throughflow. It OMITS boundary imports/exports (which a full Patten
        analysis includes in T_i = inflow + internal + outflow). This is an
        internal-flow-only-data proxy appropriate when the engine holds only the
        internal flow matrix; where boundary vectors are available a full
        throughflow should be substituted. A zero-throughflow node yields a zero
        row (no self-referential utility).

        References for the utility-analysis convention:
          - Patten's environ / utility analysis (Patten 1991, 1992).
          - Fath, B.D. & Patten, B.C. (1998) "Network mutualism: Positive
            community-level relations in ecosystems," Ecol. Modelling 107:127-143.
          - As generalized to organizations in Fath et al. (2019) Principle 8.
        NOTE: the primary Patten sources (Patten 1991/1992; Fath & Patten 1998) are
        NOT present in the local `_papers/` corpus; only Fath et al. (2019), which
        cites and applies them, is on hand. The construction here follows the
        convention as reported in Fath (2019) P8.
        """
        flow_matrix = self.ulanowicz.flow_matrix
        n = self.ulanowicz.n_nodes
        throughflow = np.sum(flow_matrix, axis=1)  # T_i (outgoing throughflow)
        D = np.zeros((n, n), dtype=float)
        for i in range(n):
            Ti = throughflow[i]
            if Ti <= 0:
                continue
            for j in range(n):
                if i == j:
                    continue
                D[i, j] = (flow_matrix[i, j] - flow_matrix[j, i]) / Ti
        return D

    def calculate_mutualism_index(self) -> Dict[str, Any]:
        """
        Classify relationships and compute mutualism via integral (direct + indirect)
        utility.

        Based on Fath et al. (2019) Principle 8: Mutualism. Fath (2019) is explicit that
        ecological/organizational mutualism is an *integral* property — the net benefit
        emerges "when considering the effects of all direct AND indirect relations."
        This is Patten's integral-utility construction:

            Direct utility     D:  d_ij = (f_ij - f_ji) / T_i
            Integral utility   U = (I - D)^(-1)      (guarded against ill-conditioning)
            Network mutualism b:c = sum(M > 0) / |sum(M < 0)|  over the OFF-DIAGONAL
                                    of M (i != j), for M in {D, U}. (>1 => net
                                    mutualistic.)

        DIAGONAL EXCLUSION: the benefit:cost sums run over off-diagonal relational
        pairings only (i != j). Network mutualism (Patten/Fath) is a property of
        the relations BETWEEN nodes; the diagonal of U is self-utility / return
        flow (always >= 0) and is not a "relation." Including it inflates the
        numerator for every network (e.g. the 4-ring integral b:c reads 6.0 with
        the diagonal but 3.0 without). Both D and U are aggregated the same way.

        The classic Patten result is that INDIRECT effects make relationships MORE
        mutualistic than direct effects alone; hence integral b:c >= direct b:c on
        a network with indirect paths. With NO indirect path (a 2-node network) the
        integral b:c EQUALS the direct b:c (no network-mutualism lift).

        The original direct-only reciprocity is retained as `direct_mutualism`
        (== the legacy `mutualism_ratio`) for back-compat and transparency.

        Returns:
            Dictionary with both direct and integral mutualism metrics.
        """
        flow_matrix = self.ulanowicz.flow_matrix
        n_nodes = self.ulanowicz.n_nodes

        # ---- Direct-only reciprocity (legacy, retained) --------------------
        mutual_pairs = 0
        one_way_pairs = 0

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                flow_ij = flow_matrix[i, j]
                flow_ji = flow_matrix[j, i]

                if flow_ij > 0 and flow_ji > 0:
                    mutual_pairs += 1
                elif flow_ij > 0 or flow_ji > 0:
                    one_way_pairs += 1

        total_connected = mutual_pairs + one_way_pairs
        mutualism_ratio = mutual_pairs / total_connected if total_connected > 0 else 0

        # Flow-weighted mutualism (considers strength of reciprocal flows)
        weighted_mutual = 0
        weighted_total = 0

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                flow_ij = flow_matrix[i, j]
                flow_ji = flow_matrix[j, i]

                if flow_ij > 0 or flow_ji > 0:
                    max_flow = max(flow_ij, flow_ji)
                    min_flow = min(flow_ij, flow_ji)

                    weighted_total += max_flow
                    weighted_mutual += min_flow  # Reciprocal component

        weighted_ratio = weighted_mutual / weighted_total if weighted_total > 0 else 0

        def _off_diagonal_bc(M: np.ndarray) -> float:
            """Benefit:cost ratio over OFF-DIAGONAL entries (i != j) of M.
            The diagonal (self-utility) is excluded — see method docstring."""
            M = np.array(M, dtype=float)
            np.fill_diagonal(M, 0.0)
            pos = float(np.sum(M[M > 0]))
            neg = float(np.abs(np.sum(M[M < 0])))
            if neg > 0:
                return pos / neg
            return float('inf') if pos > 0 else 0.0

        # ---- Direct benefit:cost ratio (off-diagonal of D) -----------------
        D = self._build_direct_utility_matrix()
        direct_bc = _off_diagonal_bc(D)

        # ---- Integral (direct + indirect) utility U = (I - D)^-1 -----------
        # Guard against a near-singular (I - D): a plain det<tiny test misses
        # ill-conditioned blow-ups (det ~ 1e-6 -> U ~ 1e6 -> b:c explodes). Use a
        # condition-number test and fall back to direct-only on ill-conditioning.
        fallback = False
        U = None
        IminusD = np.eye(n_nodes) - D
        try:
            cond = np.linalg.cond(IminusD)
        except np.linalg.LinAlgError:
            cond = np.inf
        if not np.isfinite(cond) or cond > self._INTEGRAL_UTILITY_COND_MAX:
            fallback = True
        else:
            try:
                U = np.linalg.inv(IminusD)
                if not np.all(np.isfinite(U)):
                    raise np.linalg.LinAlgError("non-finite U")
            except np.linalg.LinAlgError:
                fallback = True
                U = None

        if fallback:
            # Fall back to the direct component (no crash, flagged).
            integral_bc = direct_bc
        else:
            integral_bc = _off_diagonal_bc(U)

        # Normalize the integral b:c to [0,1] for use as a dimension input:
        # bc/(1+bc) maps [0, inf) -> [0, 1), with bc=1 (break-even) -> 0.5.
        if integral_bc == float('inf'):
            integral_mutualism = 1.0
        else:
            integral_mutualism = integral_bc / (1.0 + integral_bc)

        return {
            # --- back-compat keys (existing consumers read these) ---
            'mutual_pairs': mutual_pairs,
            'one_way_pairs': one_way_pairs,
            'mutualism_ratio': mutualism_ratio,
            'weighted_mutualism': weighted_ratio,
            'total_connected_pairs': total_connected,
            # --- new direct/integral utility decomposition ---
            'direct_mutualism': mutualism_ratio,
            'direct_benefit_cost_ratio': direct_bc,
            'integral_benefit_cost_ratio': integral_bc,
            'integral_mutualism': integral_mutualism,
            'direct_utility_matrix': D.tolist(),
            'integral_utility_matrix': (U.tolist() if U is not None else None),
            'fallback_direct_only': fallback,
        }

    def calculate_fitness_for_evolution(self, beta: float = 1.288) -> float:
        """
        Calculate the full fitness function with adjustable beta parameter.

        From Ulanowicz et al. (2009) Eq. 16:
        F = -(e/log(e)) * alpha^beta * log(alpha^beta)

        This represents evolutionary fitness where:
        - beta = 1.288 for ecosystems (empirically derived)
        - Maximum fitness at alpha = e^(-1/beta) ~ 0.4596 for beta=1.288

        Args:
            beta: Shape parameter (default 1.288 from ecological studies)

        Returns:
            Fitness value (0 to ~0.4)
        """
        alpha = self.ulanowicz.calculate_relative_ascendency()

        if alpha <= 0 or alpha >= 1:
            return 0.0

        # F = -(e/ln(e)) * alpha^beta * ln(alpha^beta)
        # Since ln(e) = 1, this simplifies to:
        # F = -e * alpha^beta * ln(alpha^beta)
        e = math.e
        alpha_beta = alpha ** beta

        if alpha_beta <= 0:
            return 0.0

        fitness = -e * alpha_beta * math.log(alpha_beta)

        return max(0, fitness)

    def calculate_open_score(self) -> Dict[str, Any]:
        """
        Calculate OPEN dimension score.

        OPEN measures the organization's ability to interconnect and exchange
        with its environment. Maps to Fath Principles 1, 3, 4 (Circulation).

        Key metrics:
        - connectance: Basic network connectivity
        - flow_diversity: Diversity of flow patterns (H)
        - clustering_coefficient: Local connectivity patterns
        - betweenness_centrality: Bridge/broker roles

        Formula: OPEN = 0.25*connectance + 0.30*flow_diversity +
                        0.25*avg_betweenness + 0.20*clustering

        Returns:
            Dictionary with score and contributing metrics
        """
        metrics = self._get_ulanowicz_metrics()
        net_metrics = self._get_network_metrics()

        # Get base metrics
        connectance = metrics.get('connectance', 0)
        flow_diversity = metrics.get('flow_diversity', 0)

        # Normalize flow diversity (typically 0-5 bits for organizational networks)
        max_flow_diversity = math.log(self.ulanowicz.n_nodes ** 2)
        norm_flow_diversity = flow_diversity / max_flow_diversity if max_flow_diversity > 0 else 0

        # Get clustering coefficient
        clustering = metrics.get('clustering_coefficient', 0)

        # Get average betweenness from network analyzer or topology metrics
        if net_metrics and 'centralities' in net_metrics:
            betweenness = net_metrics['centralities'].get('betweenness', {})
            avg_betweenness = np.mean(list(betweenness.values())) if betweenness else 0
        else:
            avg_betweenness = 0

        # Calculate weighted score
        raw_score = (
            0.25 * connectance +
            0.30 * norm_flow_diversity +
            0.25 * avg_betweenness +
            0.20 * clustering
        )

        # Convert to 0-100 scale (cap centralized in DIMENSION_NORMALIZATION_CAPS)
        score = self._normalize_to_100(raw_score, 0, self.DIMENSION_NORMALIZATION_CAPS['open'])

        return {
            'score': score,
            'metrics': {
                'connectance': connectance,
                'flow_diversity': flow_diversity,
                'norm_flow_diversity': norm_flow_diversity,
                'clustering_coefficient': clustering,
                'avg_betweenness': avg_betweenness
            },
            'weights': {
                'connectance': 0.25,
                'flow_diversity': 0.30,
                'betweenness': 0.25,
                'clustering': 0.20
            }
        }

    def calculate_autonomous_score(self) -> Dict[str, Any]:
        """
        Calculate AUTONOMOUS dimension score.

        AUTONOMOUS measures the organization's ability to learn and encode routines
        through cycling and feedback. Maps to Fath Principles 2, 9.

        Key metrics:
        - finn_cycling_index: Cycling/re-investment of resources
        - flow_reciprocity: Bidirectional flow patterns
        - AMI/H_max: Information organization ratio
        - autocatalytic_index: Self-reinforcing feedback cycles

        Formula: AUTONOMOUS = 0.35*FCI + 0.25*reciprocity +
                              0.25*(AMI/H_max) + 0.15*autocatalytic

        Returns:
            Dictionary with score and contributing metrics
        """
        metrics = self._get_ulanowicz_metrics()
        net_metrics = self._get_network_metrics()

        # Finn Cycling Index
        fci = metrics.get('finn_cycling_index')
        if fci is None:
            # Compute simplified FCI if not available
            fci = 0.1  # Default low value

        # Flow reciprocity
        if net_metrics and 'flow' in net_metrics:
            reciprocity = net_metrics['flow'].get('flow_reciprocity', 0)
        else:
            # Calculate from mutualism
            mutualism = self.calculate_mutualism_index()
            reciprocity = mutualism.get('mutualism_ratio', 0)

        # AMI normalized by max entropy
        ami = metrics.get('average_mutual_information', 0)
        max_entropy = math.log(self.ulanowicz.n_nodes ** 2)
        norm_ami = ami / max_entropy if max_entropy > 0 else 0

        # Autocatalytic index
        autocatalytic = self.calculate_autocatalytic_index()
        autocatalytic_idx = autocatalytic.get('autocatalytic_index', 0)

        # Calculate weighted score
        raw_score = (
            0.35 * min(fci, 1.0) +
            0.25 * reciprocity +
            0.25 * norm_ami +
            0.15 * autocatalytic_idx
        )

        # Convert to 0-100 scale (cap centralized in DIMENSION_NORMALIZATION_CAPS)
        score = self._normalize_to_100(raw_score, 0, self.DIMENSION_NORMALIZATION_CAPS['autonomous'])

        return {
            'score': score,
            'metrics': {
                'finn_cycling_index': fci,
                'flow_reciprocity': reciprocity,
                'ami': ami,
                'norm_ami': norm_ami,
                'autocatalytic_index': autocatalytic_idx,
                'autocatalytic_details': autocatalytic
            },
            'weights': {
                'finn_cycling_index': 0.35,
                'reciprocity': 0.25,
                'ami': 0.25,
                'autocatalytic': 0.15
            }
        }

    def calculate_symbiotic_score(self) -> Dict[str, Any]:
        """
        Calculate SYMBIOTIC dimension score.

        SYMBIOTIC measures human-machine integration and balanced cooperation.
        Maps to Fath Principles 5, 8 (Balance of sizes, Mutualism).

        Key metrics:
        - gini_coefficient: Flow inequality (inverted - lower is better)
        - modularity: Community structure strength
        - effective_nodes/actual: Node utilization efficiency
        - integral_mutualism: Integral (direct + indirect) utility, Patten / Fath 2019 P8

        Formula: SYMBIOTIC = 0.30*(1-gini) + 0.25*modularity +
                             0.25*(eff_nodes/actual) + 0.20*integral_mutualism

        Returns:
            Dictionary with score and contributing metrics
        """
        metrics = self._get_ulanowicz_metrics()
        net_metrics = self._get_network_metrics()

        # Gini coefficient (lower is more equal)
        flow_matrix = self.ulanowicz.flow_matrix
        flows = flow_matrix[flow_matrix > 0].flatten()
        if len(flows) > 1:
            sorted_flows = np.sort(flows)
            n = len(sorted_flows)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_flows)) / (n * np.sum(sorted_flows)) - (n + 1) / n
        else:
            gini = 0

        # Modularity from community detection
        if net_metrics and 'communities' in net_metrics:
            louvain = net_metrics['communities'].get('louvain', {})
            modularity = louvain.get('modularity', 0)
        else:
            modularity = 0.3  # Default moderate value

        # Effective nodes ratio
        effective_nodes = metrics.get('effective_nodes', self.ulanowicz.n_nodes)
        actual_nodes = self.ulanowicz.n_nodes
        node_ratio = effective_nodes / actual_nodes if actual_nodes > 0 else 1

        # Mutualism: use INTEGRAL (direct + indirect) utility per Fath (2019)
        # Principle 8 / Patten. The integral b:c ratio is normalized to [0,1]
        # (integral_mutualism). The legacy direct-only mutualism_ratio is retained
        # in the metrics block below for back-compat and transparency.
        mutualism = self.calculate_mutualism_index()
        mutualism_ratio = mutualism.get('mutualism_ratio', 0)  # direct (legacy)
        integral_mutualism = mutualism.get('integral_mutualism', mutualism_ratio)

        # Calculate weighted score (mutualism input = integral utility)
        raw_score = (
            0.30 * (1 - gini) +
            0.25 * min(modularity, 1) +
            0.25 * min(node_ratio, 1) +
            0.20 * integral_mutualism
        )

        # Convert to 0-100 scale (cap centralized in DIMENSION_NORMALIZATION_CAPS)
        score = self._normalize_to_100(raw_score, 0, self.DIMENSION_NORMALIZATION_CAPS['symbiotic'])

        return {
            'score': score,
            'metrics': {
                'gini_coefficient': gini,
                'equality': 1 - gini,
                'modularity': modularity,
                'effective_nodes': effective_nodes,
                'actual_nodes': actual_nodes,
                'node_utilization': node_ratio,
                'mutualism_ratio': mutualism_ratio,
                'integral_mutualism': integral_mutualism,
                'mutualism_details': mutualism
            },
            'weights': {
                'equality': 0.30,
                'modularity': 0.25,
                'node_utilization': 0.25,
                'mutualism': 0.20
            }
        }

    def calculate_intelligent_score(self) -> Dict[str, Any]:
        """
        Calculate INTELLIGENT dimension score.

        INTELLIGENT measures functional diversity and ability to leverage
        different types of intelligence. Maps to Fath Principles 7, 10.

        Key metrics:
        - number_of_roles: Functional differentiation (Zorach-Ulanowicz)
        - functional_diversity: Log of roles
        - roles_per_node: Distribution of roles
        - conditional_entropy: Remaining uncertainty/flexibility

        Formula: INTELLIGENT = 0.35*roles + 0.25*diversity +
                               0.20*roles_per_node + 0.20*cond_entropy

        Returns:
            Dictionary with score and contributing metrics
        """
        metrics = self._get_ulanowicz_metrics()

        # Number of roles, SIZE-RELATIVE normalization (principled).
        #
        # R = number_of_roles = exp(AMI). The Zorach & Ulanowicz (2003) identity
        # block gives R = N / C, where N = effective_nodes and C = effective
        # connectivity. The connectivity floor C >= 1 for a connected network
        # (Ulanowicz 2004, p.334 — the lower edge of the window of vitality is
        # C = 1) implies R <= N. Hence R / N in [0, 1] is a PRINCIPLED,
        # size-relative normalizer: it gauges how close the network is to its
        # own maximum functional differentiation (one distinct role per
        # effective node), independent of node count. This replaces the former
        # fixed `roles / 10` ceiling, which implicitly assumed a ~10-role
        # organization and systematically penalized small nets / inflated large
        # ones purely as a size artifact.
        num_roles = metrics.get('number_of_roles', 1)
        effective_nodes = metrics.get('effective_nodes', self.ulanowicz.n_nodes)
        if effective_nodes and effective_nodes > 0:
            norm_roles = min(num_roles / effective_nodes, 1)
        else:
            norm_roles = 0.0

        # Functional diversity (log of roles = AMI)
        functional_diversity = metrics.get('functional_diversity', 0)
        max_diversity = math.log(self.ulanowicz.n_nodes)
        norm_diversity = functional_diversity / max_diversity if max_diversity > 0 else 0

        # Roles per node = R / N. By the same R <= N bound above, roles_per_node
        # is already in [0, 1], so the principled max is 1.0. We normalize by
        # min(rpn, 1.0) rather than the former arbitrary `/ 2` (which implied a
        # "2 roles per effective node" ceiling with no theoretical basis).
        roles_per_node = metrics.get('roles_per_node', 1)
        norm_roles_per_node = min(roles_per_node / 1.0, 1)

        # Conditional entropy (flexibility in the system)
        cond_entropy = metrics.get('conditional_entropy', 0)
        flow_diversity = metrics.get('flow_diversity', 1)
        norm_cond_entropy = cond_entropy / flow_diversity if flow_diversity > 0 else 0

        # Calculate weighted score
        raw_score = (
            0.35 * norm_roles +
            0.25 * norm_diversity +
            0.20 * norm_roles_per_node +
            0.20 * norm_cond_entropy
        )

        # Convert to 0-100 scale (cap centralized in DIMENSION_NORMALIZATION_CAPS)
        score = self._normalize_to_100(raw_score, 0, self.DIMENSION_NORMALIZATION_CAPS['intelligent'])

        return {
            'score': score,
            'metrics': {
                'number_of_roles': num_roles,
                'norm_roles': norm_roles,
                'functional_diversity': functional_diversity,
                'norm_functional_diversity': norm_diversity,
                'roles_per_node': roles_per_node,
                'norm_roles_per_node': norm_roles_per_node,
                'conditional_entropy': cond_entropy,
                'norm_conditional_entropy': norm_cond_entropy
            },
            'weights': {
                'roles': 0.35,
                'diversity': 0.25,
                'roles_per_node': 0.20,
                'conditional_entropy': 0.20
            }
        }

    def calculate_sustainable_score(self) -> Dict[str, Any]:
        """
        Calculate SUSTAINABLE dimension score.

        SUSTAINABLE measures the balance between order and freedom,
        centered on the Window of Vitality. Maps to Fath Principle 6.

        Key metrics:
        - robustness: R = -alpha * log(alpha)
        - is_in_window: Boolean for Window of Viability
        - regenerative_capacity: Self-renewal ability
        - alpha_optimality: Distance from optimal alpha (0.37)

        Formula: SUSTAINABLE = 0.30*robustness + 0.20*is_in_window +
                               0.20*regen_capacity + 0.30*alpha_optimality

        Returns:
            Dictionary with score and contributing metrics
        """
        metrics = self._get_ulanowicz_metrics()

        # Robustness (normalized to 0-1, max theoretical is ~0.368)
        robustness = metrics.get('robustness', 0)
        max_robustness = 1 / math.e  # Theoretical max at alpha = 1/e
        norm_robustness = robustness / max_robustness if max_robustness > 0 else 0

        # Is in Window of Viability
        is_viable = metrics.get('is_viable', False)
        in_window = 1.0 if is_viable else 0.0

        # Regenerative capacity
        regen_capacity = metrics.get('regenerative_capacity', 0)
        # Normalize: expect 0-0.3 range
        norm_regen = min(regen_capacity / 0.3, 1)

        # Alpha optimality: How close to optimal 0.37
        alpha = metrics.get('relative_ascendency', 0.5)
        optimal_alpha = 0.37
        distance_from_optimal = abs(alpha - optimal_alpha)
        # Convert to score: 1 at optimal, 0 at extremes
        alpha_optimality = max(0, 1 - (distance_from_optimal / optimal_alpha))

        # Fitness for evolution (bonus metric)
        fitness = self.calculate_fitness_for_evolution()
        norm_fitness = min(fitness / 0.4, 1)  # Max fitness ~0.4

        # Calculate weighted score
        raw_score = (
            0.30 * norm_robustness +
            0.20 * in_window +
            0.20 * norm_regen +
            0.30 * alpha_optimality
        )

        # Convert to 0-100 scale (cap centralized in DIMENSION_NORMALIZATION_CAPS)
        score = self._normalize_to_100(raw_score, 0, self.DIMENSION_NORMALIZATION_CAPS['sustainable'])

        return {
            'score': score,
            'metrics': {
                'robustness': robustness,
                'norm_robustness': norm_robustness,
                'is_viable': is_viable,
                'in_window_score': in_window,
                'regenerative_capacity': regen_capacity,
                'norm_regenerative': norm_regen,
                'relative_ascendency': alpha,
                'optimal_alpha': optimal_alpha,
                'distance_from_optimal': distance_from_optimal,
                'alpha_optimality': alpha_optimality,
                'fitness_for_evolution': fitness,
                'norm_fitness': norm_fitness
            },
            'weights': {
                'robustness': 0.30,
                'in_window': 0.20,
                'regenerative': 0.20,
                'alpha_optimality': 0.30
            }
        }

    # Ordered status bands for the roll-up band cap: CRITICAL < WARNING < HEALTHY
    _STATUS_LEVELS = {'CRITICAL': 0, 'WARNING': 1, 'HEALTHY': 2}
    _LEVEL_TO_STATUS = {0: 'CRITICAL', 1: 'WARNING', 2: 'HEALTHY'}

    @classmethod
    def _dimension_status(cls, dim: str, score: float) -> str:
        """Per-dimension status band using HEALTH_THRESHOLDS (O9 logic)."""
        thresholds = cls.HEALTH_THRESHOLDS[dim]
        if score >= thresholds['healthy'][0]:
            return 'HEALTHY'
        elif score >= thresholds['warning'][0]:
            return 'WARNING'
        return 'CRITICAL'

    @classmethod
    def compute_overall_status(cls, scores: Dict[str, float],
                               weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Compute the overall OASIS status with the dimension-agnostic worst-dimension
        band cap veto.

        Rule (expert-guided, see docs/business-revision/evidence/expert-org-management.md
        section 2 and expert-ecosystem-dynamics.md):
            Order bands CRITICAL=0 < WARNING=1 < HEALTHY=2.
            - raw_overall_level from the weighted mean (>=60 HEALTHY / >=40 WARNING / else CRITICAL)
            - each dimension's level from HEALTH_THRESHOLDS
            - worst_dim_level = min(level over the 5 dimensions)
            - final_overall_level = min(raw_overall_level, worst_dim_level + 1)
              ("overall can never be more than one band above the worst dimension")
            - the numeric overall score is UNCHANGED; only the STATUS LABEL is capped.

        Args:
            scores: dict of dimension -> 0..100 score (open/autonomous/symbiotic/
                    intelligent/sustainable).
            weights: optional dimension weights; defaults to DEFAULT_WEIGHTS.

        Returns:
            dict with:
                overall_score: weighted mean (unchanged by the cap)
                raw_overall_status: label before the cap
                overall_status: final label after the cap
                dimension_status: per-dimension status labels
                capped: whether the cap lowered the label
                capped_by: dimension(s) at the worst band that drove the cap
                           (empty list if no cap applied)
        """
        if weights is None:
            weights = cls.DEFAULT_WEIGHTS

        # Weighted-mean numeric score (unchanged by the cap).
        overall = sum(scores[dim] * weights[dim] for dim in scores)

        # Raw overall band from the score, exactly as before.
        if overall >= 60:
            raw_level = cls._STATUS_LEVELS['HEALTHY']
        elif overall >= 40:
            raw_level = cls._STATUS_LEVELS['WARNING']
        else:
            raw_level = cls._STATUS_LEVELS['CRITICAL']

        # Per-dimension bands.
        dim_status = {dim: cls._dimension_status(dim, score)
                      for dim, score in scores.items()}
        dim_levels = {dim: cls._STATUS_LEVELS[s] for dim, s in dim_status.items()}

        worst_dim_level = min(dim_levels.values())

        # Band cap: overall can never be more than one band above the worst dimension.
        final_level = min(raw_level, worst_dim_level + 1)

        capped = final_level < raw_level
        # Dimensions sitting at the worst band are the ones that drove the cap.
        capped_by = (
            sorted(dim for dim, lvl in dim_levels.items() if lvl == worst_dim_level)
            if capped else []
        )

        return {
            'overall_score': overall,
            'raw_overall_status': cls._LEVEL_TO_STATUS[raw_level],
            'overall_status': cls._LEVEL_TO_STATUS[final_level],
            'dimension_status': dim_status,
            'capped': capped,
            'capped_by': capped_by,
        }

    @classmethod
    def resolve_weights(cls, profile: Union[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Resolve a weighting-profile NAME or an explicit weight dict to a validated
        weight dict over the five dimensions.

        Args:
            profile: either a key of WEIGHTING_PROFILES (e.g. "Scale-up / Growth")
                     or an explicit {dimension: weight} dict (manual "Custom").

        Returns:
            A copy of the weight dict (covering the five dimensions, summing to 1.0).

        Raises:
            ValueError: if the name is unknown, the dimensions are wrong, or the
                        weights do not sum to 1.0 (within 1e-2, matching __init__).
        """
        if isinstance(profile, str):
            if profile not in WEIGHTING_PROFILES:
                raise ValueError(
                    f"Unknown weighting profile '{profile}'. "
                    f"Available: {sorted(WEIGHTING_PROFILES)}")
            return dict(WEIGHTING_PROFILES[profile]['weights'])

        # Explicit weight dict (manual "Custom" override).
        weights = dict(profile)
        if frozenset(weights.keys()) != _OASIS_DIMENSIONS:
            raise ValueError(
                f"Weights must cover exactly the five OASIS dimensions, "
                f"got {sorted(weights.keys())}")
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        return weights

    @classmethod
    def apply_weighting_profile(
            cls,
            dimension_scores: Dict[str, float],
            profile: Union[str, Dict[str, float]] = 'Balanced (default)',
    ) -> Dict[str, Any]:
        """
        Cheaply RE-WEIGHT the OASIS overall from ALREADY-COMPUTED dimension scores.

        This is a pure recombination: it takes the five STORED (0..100) dimension
        scores and a named profile (or explicit weight dict) and returns the new
        weighted-mean overall + the worst-dimension band-capped status, reusing
        `compute_overall_status`. It does NOT recompute any dimension metric —
        weights never touch the dimension scores — so the app can switch profiles
        instantly on a precomputed profile.

        Args:
            dimension_scores: {dimension: 0..100 score} for the five dimensions.
            profile: a WEIGHTING_PROFILES name (default "Balanced (default)") or an
                     explicit weight dict (manual "Custom").

        Returns:
            The `compute_overall_status` result dict (overall_score, overall_status,
            raw_overall_status, dimension_status, capped, capped_by), plus:
                'weights': the resolved weight dict used
                'profile_name': the profile name if a name was passed, else 'Custom'
        """
        weights = cls.resolve_weights(profile)
        rollup = cls.compute_overall_status(dimension_scores, weights)
        rollup['weights'] = weights
        rollup['profile_name'] = profile if isinstance(profile, str) else 'Custom'
        return rollup

    def get_oasis_profile(self) -> Dict[str, Any]:
        """
        Calculate complete OASIS profile with all dimension scores.

        Returns:
            Dictionary containing:
            - dimension_scores: Individual scores for each dimension
            - overall_score: Weighted average of all dimensions
            - weights: Current dimension weights
            - status: Health status for each dimension and overall
        """
        # Calculate each dimension
        open_result = self.calculate_open_score()
        autonomous_result = self.calculate_autonomous_score()
        symbiotic_result = self.calculate_symbiotic_score()
        intelligent_result = self.calculate_intelligent_score()
        sustainable_result = self.calculate_sustainable_score()

        # Extract scores
        scores = {
            'open': open_result['score'],
            'autonomous': autonomous_result['score'],
            'symbiotic': symbiotic_result['score'],
            'intelligent': intelligent_result['score'],
            'sustainable': sustainable_result['score']
        }

        # Compute the weighted overall score, per-dimension status, and the
        # worst-dimension band cap on the overall status label. The numeric
        # overall score is the weighted mean and is UNCHANGED by the cap.
        rollup = self.compute_overall_status(scores, self.weights)

        overall = rollup['overall_score']
        status = rollup['dimension_status']
        overall_status = rollup['overall_status']

        return {
            'dimension_scores': scores,
            'dimension_details': {
                'open': open_result,
                'autonomous': autonomous_result,
                'symbiotic': symbiotic_result,
                'intelligent': intelligent_result,
                'sustainable': sustainable_result
            },
            'overall_score': overall,
            'weights': self.weights.copy(),
            'dimension_status': status,
            'overall_status': overall_status,
            # Roll-up band cap veto metadata (worst-dimension cap):
            'raw_overall_status': rollup['raw_overall_status'],
            'overall_status_capped': rollup['capped'],
            'capped_by': rollup['capped_by']
        }

    def get_oasis_interpretation(self) -> Dict[str, str]:
        """
        Get human-readable interpretations for each OASIS dimension.

        Returns:
            Dictionary with interpretation text for each dimension
        """
        profile = self.get_oasis_profile()
        scores = profile['dimension_scores']
        details = profile['dimension_details']

        interpretations = {}

        # OPEN interpretation
        open_score = scores['open']
        if open_score >= 70:
            interpretations['open'] = (
                f"Strong interconnectivity (score: {open_score:.0f}/100). "
                "The organization demonstrates excellent flow diversity and connectivity. "
                "Cross-departmental communication is healthy."
            )
        elif open_score >= 50:
            interpretations['open'] = (
                f"Moderate interconnectivity (score: {open_score:.0f}/100). "
                "The organization has reasonable connectivity but could improve "
                "cross-functional collaboration and information sharing."
            )
        else:
            interpretations['open'] = (
                f"Limited interconnectivity (score: {open_score:.0f}/100). "
                "The organization shows siloed behavior. Consider improving "
                "communication channels and encouraging cross-team collaboration."
            )

        # AUTONOMOUS interpretation
        auto_score = scores['autonomous']
        if auto_score >= 60:
            interpretations['autonomous'] = (
                f"Strong learning capacity (score: {auto_score:.0f}/100). "
                "The organization effectively encodes routines and shows healthy "
                "feedback loops for continuous improvement."
            )
        elif auto_score >= 40:
            interpretations['autonomous'] = (
                f"Moderate learning capacity (score: {auto_score:.0f}/100). "
                "Some feedback mechanisms exist but knowledge encoding could be "
                "strengthened through better documentation and process cycles."
            )
        else:
            interpretations['autonomous'] = (
                f"Limited learning capacity (score: {auto_score:.0f}/100). "
                "The organization struggles to encode and retain institutional knowledge. "
                "Implement stronger feedback loops and knowledge management systems."
            )

        # SYMBIOTIC interpretation
        symb_score = scores['symbiotic']
        if symb_score >= 70:
            interpretations['symbiotic'] = (
                f"Excellent role integration (score: {symb_score:.0f}/100). "
                "Strong mutualistic relationships exist across the organization. "
                "Resources are distributed equitably."
            )
        elif symb_score >= 50:
            interpretations['symbiotic'] = (
                f"Moderate role integration (score: {symb_score:.0f}/100). "
                "Some imbalance in resource distribution or cooperation patterns. "
                "Consider addressing flow inequalities and fostering reciprocal relationships."
            )
        else:
            interpretations['symbiotic'] = (
                f"Limited role integration (score: {symb_score:.0f}/100). "
                "Significant imbalances in how resources flow and roles interact. "
                "Address inequalities and build more cooperative structures."
            )

        # INTELLIGENT interpretation
        intel_score = scores['intelligent']
        if intel_score >= 65:
            interpretations['intelligent'] = (
                f"High functional diversity (score: {intel_score:.0f}/100). "
                "The organization effectively leverages diverse specialized roles. "
                "Good balance of expertise across functions."
            )
        elif intel_score >= 45:
            interpretations['intelligent'] = (
                f"Moderate functional diversity (score: {intel_score:.0f}/100). "
                "Some role differentiation exists but could be enhanced. "
                "Consider developing more specialized capabilities."
            )
        else:
            interpretations['intelligent'] = (
                f"Limited functional diversity (score: {intel_score:.0f}/100). "
                "Insufficient role differentiation may limit organizational intelligence. "
                "Develop more specialized functions and cross-functional capabilities."
            )

        # SUSTAINABLE interpretation
        sust_score = scores['sustainable']
        sust_metrics = details['sustainable']['metrics']
        alpha = sust_metrics.get('relative_ascendency', 0)
        is_viable = sust_metrics.get('is_viable', False)

        # Reframed: position-on-a-gradient + direction-of-travel against the
        # INDICATIVE ecological reference band (single source of truth). The
        # numeric score is unchanged; is_viable is still computed upstream and
        # available, but is presented as a gradient position, not a PASS/FAIL.
        try:
            from src.report_intelligence import sustainable_verdict_narrative
        except Exception:
            from report_intelligence import sustainable_verdict_narrative
        interpretations['sustainable'] = sustainable_verdict_narrative(
            sust_score, alpha
        )

        return interpretations

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on OASIS assessment.

        Returns:
            List of recommendation dictionaries with priority, dimension, and action
        """
        profile = self.get_oasis_profile()
        scores = profile['dimension_scores']
        details = profile['dimension_details']

        recommendations = []

        # OPEN recommendations
        if scores['open'] < 50:
            recommendations.append({
                'priority': 'HIGH' if scores['open'] < 30 else 'MEDIUM',
                'dimension': 'OPEN',
                'issue': 'Low interconnectivity',
                'action': 'Establish regular cross-functional meetings and communication channels',
                'metrics_to_improve': ['flow_diversity', 'connectance', 'clustering_coefficient']
            })

        # AUTONOMOUS recommendations
        if scores['autonomous'] < 40:
            recommendations.append({
                'priority': 'HIGH' if scores['autonomous'] < 25 else 'MEDIUM',
                'dimension': 'AUTONOMOUS',
                'issue': 'Weak feedback and learning loops',
                'action': 'Implement knowledge management systems and regular retrospectives',
                'metrics_to_improve': ['finn_cycling_index', 'flow_reciprocity']
            })

        # SYMBIOTIC recommendations
        symb_metrics = details['symbiotic']['metrics']
        if symb_metrics['gini_coefficient'] > 0.5:
            recommendations.append({
                'priority': 'MEDIUM',
                'dimension': 'SYMBIOTIC',
                'issue': 'High resource inequality',
                'action': 'Redistribute resources and responsibilities more equitably',
                'metrics_to_improve': ['gini_coefficient', 'mutualism_ratio']
            })

        # INTELLIGENT recommendations
        intel_metrics = details['intelligent']['metrics']
        if intel_metrics['number_of_roles'] < 3:
            recommendations.append({
                'priority': 'MEDIUM',
                'dimension': 'INTELLIGENT',
                'issue': 'Insufficient functional differentiation',
                'action': 'Develop specialized capabilities and clearer role definitions',
                'metrics_to_improve': ['number_of_roles', 'functional_diversity']
            })

        # SUSTAINABLE recommendations (highest priority for viability issues)
        sust_metrics = details['sustainable']['metrics']
        if not sust_metrics['is_viable']:
            alpha = sust_metrics['relative_ascendency']
            if alpha < 0.2:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'dimension': 'SUSTAINABLE',
                    'issue': 'Under-organized relative to the indicative reference '
                             'band (alpha < 0.2)',
                    'action': 'Direction of travel: increase structure / coordination '
                              '(standardize processes, strengthen coordination)',
                    'metrics_to_improve': ['relative_ascendency', 'robustness']
                })
            elif alpha > 0.6:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'dimension': 'SUSTAINABLE',
                    'issue': 'Over-organized relative to the indicative reference '
                             'band (alpha > 0.6)',
                    'action': 'Direction of travel: increase redundancy / flexibility '
                              '(reduce constraints, diversify pathways)',
                    'metrics_to_improve': ['relative_ascendency', 'redundancy', 'overhead_ratio']
                })

        # Sort by priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))

        return recommendations

    def set_dimension_weights(self, weights: Dict[str, float]) -> None:
        """
        Update dimension weights for overall score calculation.

        Args:
            weights: Dictionary with weights for each dimension (must sum to 1.0)
        """
        required_keys = {'open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable'}
        if set(weights.keys()) != required_keys:
            raise ValueError(f"Weights must include exactly these keys: {required_keys}")

        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        self.weights = weights.copy()

    def get_dimension_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of each dimension with key metrics and status.

        Returns:
            Dictionary with dimension summaries for UI display
        """
        profile = self.get_oasis_profile()
        interpretations = self.get_oasis_interpretation()

        summary = {}

        dimension_info = {
            'open': {
                'full_name': 'OPEN',
                'description': 'Ability to interconnect and exchange',
                'fath_principles': [1, 3, 4],
                'icon': 'globe'
            },
            'autonomous': {
                'full_name': 'AUTONOMOUS',
                'description': 'Ability to learn and encode routines',
                'fath_principles': [2, 9],
                'icon': 'brain'
            },
            'symbiotic': {
                'full_name': 'SYMBIOTIC',
                'description': 'Human-machine integration balance',
                'fath_principles': [5, 8],
                'icon': 'handshake'
            },
            'intelligent': {
                'full_name': 'INTELLIGENT',
                'description': 'Leverage diverse intelligence types',
                'fath_principles': [7, 10],
                'icon': 'lightbulb'
            },
            'sustainable': {
                'full_name': 'SUSTAINABLE',
                'description': 'Balance order and freedom',
                'fath_principles': [6],
                'icon': 'leaf'
            }
        }

        for dim, info in dimension_info.items():
            summary[dim] = {
                **info,
                'score': profile['dimension_scores'][dim],
                'status': profile['dimension_status'][dim],
                'interpretation': interpretations[dim],
                'key_metrics': profile['dimension_details'][dim]['metrics'],
                'weights': profile['dimension_details'][dim]['weights']
            }

        return summary
