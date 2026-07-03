"""
Published Metrics Database

Centralized database of all published values from research papers for validation.
All metrics are extracted directly from peer-reviewed publications with proper
citations to enable reproducible scientific validation.

References:
- Ulanowicz & Norden (1990): Int. J. Systems Sci. 21(2), 429-437
- Ulanowicz (1986): Growth and Development: Ecosystems Phenomenology
- Ulanowicz et al. (2009): Ecological Complexity 6, 27-36
- Heymans et al. (2002): Various ecosystem analyses
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum


class LogBase(Enum):
    """Logarithm base used in paper calculations."""
    NATURAL = "natural"  # ln (base e)
    LOG2 = "log2"  # log base 2
    LOG10 = "log10"  # log base 10


# =============================================================================
# BASE-DEPENDENCE OF METRICS
# -----------------------------------------------------------------------------
# Information-theoretic MAGNITUDES scale linearly with the logarithm base: a
# value computed in nats (natural log, as the UlanowiczCalculator does) is
# ``1/ln(2)`` times SMALLER than the same quantity in bits (log2). When a stored
# published value is quoted in a different base than the engine computes, these
# metrics MUST be base-converted before comparison (see
# ``scientific_validation_agent.nats_to_bits``).
#
# Ratios / dimensionless indices are base-INVARIANT: the log base cancels in a
# quotient (e.g. relative ascendency alpha = A/C) or the quantity never involves
# a logarithm at all (e.g. total system throughput = a sum of flows). These must
# NEVER be converted, or a correct value would be corrupted.
# =============================================================================

# Metrics whose magnitude changes with the log base (nats vs bits vs digits).
BASE_DEPENDENT_METRICS = frozenset({
    "ascendency",
    "development_capacity",
    "reserve",
    "overhead",
    "average_mutual_information",
    "statistical_entropy",
    "flow_diversity",
    "conditional_entropy",
    "structural_information",
})

# Metrics that are invariant to the log base (ratios, indices, raw flow sums).
BASE_INVARIANT_METRICS = frozenset({
    "relative_ascendency",
    "ascendency_ratio",
    "robustness",
    "total_system_throughput",
    "network_efficiency",
    "finn_cycling_index",
    "is_viable",
    "regenerative_capacity",
    "redundancy",
})


def is_base_dependent(metric_name: str) -> bool:
    """Return True if a metric's magnitude scales with the logarithm base.

    Base-dependent metrics require a nats<->bits conversion before a
    cross-base published-value comparison; base-invariant ones must not be
    converted. Unknown metric names default to base-invariant (no conversion)
    so a comparison is never silently corrupted by an unexpected key.
    """
    return metric_name in BASE_DEPENDENT_METRICS


@dataclass
class PublishedMetric:
    """A single published metric with its value and metadata."""
    value: float
    unit: str
    reported: bool = True
    note: Optional[str] = None


@dataclass
class NetworkPublishedData:
    """Published data for a single network from research papers."""
    source: str
    doi: Optional[str] = None
    figure: Optional[str] = None
    page: Optional[int] = None
    log_base: LogBase = LogBase.NATURAL
    tolerance: float = 0.05  # 5% default tolerance
    metrics: Dict[str, PublishedMetric] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    # When True, the entry is a published-literature reference value (e.g. a
    # benchmark anchor quoted directly from a paper's prose) that is NOT tied to
    # a recomputable flow matrix in NETWORK_DATA_FILES. The computational
    # validation agent skips these instead of erroring on a missing data file;
    # they are still exposed as scientific reference anchors in the report.
    reference_only: bool = False


# =============================================================================
# PUBLISHED METRICS DATABASE
# =============================================================================

PUBLISHED_METRICS: Dict[str, NetworkPublishedData] = {

    # =========================================================================
    # CONE SPRING ECOSYSTEM - Original
    # =========================================================================
    "cone_spring_original": NetworkPublishedData(
        source="Ulanowicz & Norden 1990",
        doi="10.1080/00207729008910393",
        figure="Table 1",
        log_base=LogBase.LOG2,  # Paper uses log base 2 (bits)
        tolerance=0.05,
        metrics={
            "total_system_throughput": PublishedMetric(
                value=42016,
                unit="mgC/m2/day",
                note="Sum of all flows in network"
            ),
            "development_capacity": PublishedMetric(
                value=135000,
                unit="mgC-bits/m2/day",
                note="C = -Sum(T_ij * log2(T_ij/TST))"
            ),
            "ascendency": PublishedMetric(
                value=68191,
                unit="mgC-bits/m2/day",
                note="A = Sum(T_ij * log2((T_ij*TST)/(T_i.*T_.j)))"
            ),
            "reserve": PublishedMetric(
                value=66809,
                unit="mgC-bits/m2/day",
                note="Phi = C - A"
            ),
            "relative_ascendency": PublishedMetric(
                value=0.505,
                unit="dimensionless",
                note="alpha = A/C"
            ),
            "average_mutual_information": PublishedMetric(
                value=1.623,
                unit="bits",
                note="AMI = A/TST"
            ),
            "statistical_entropy": PublishedMetric(
                value=3.213,
                unit="bits",
                note="H = C/TST"
            ),
        },
        notes=[
            "This is the original Cone Spring ecosystem before eutrophication",
            "Flow matrix is 5x5 internal flows only",
            "Paper uses log base 2 (bits) for information calculations"
        ]
    ),

    # =========================================================================
    # CONE SPRING ECOSYSTEM - Eutrophicated (from Ulanowicz 2009)
    # =========================================================================
    "cone_spring_eutrophicated": NetworkPublishedData(
        source="Ulanowicz et al. 2009",
        doi="10.1016/j.ecocom.2008.10.005",
        figure="Figure 6",
        page=31,
        log_base=LogBase.NATURAL,
        tolerance=0.05,
        metrics={
            "relative_ascendency": PublishedMetric(
                value=0.529,
                unit="dimensionless",
                note="Above optimal (0.460), indicating excess organization"
            ),
        },
        notes=[
            "Modified Cone Spring with 8000 kcal/m2/y added to pathway 1->2->3",
            "System has excess ascendency, reduced reserve capacity",
            "Marginal contributions of main pathway now <1"
        ]
    ),

    # =========================================================================
    # CRYSTAL RIVER CREEK
    # =========================================================================
    "crystal_river_creek": NetworkPublishedData(
        source="Ulanowicz 1986",
        doi=None,  # Book, not journal article
        figure="Various",
        log_base=LogBase.LOG2,  # Assumes log base 2 based on era
        tolerance=0.10,  # Higher tolerance due to system boundary differences
        metrics={
            "total_system_throughput": PublishedMetric(
                value=97916,
                unit="kcal/m2/year",
                note="Full system with external flows"
            ),
            "development_capacity": PublishedMetric(
                value=204355,
                unit="kcal-bits/m2/year",
                reported=True
            ),
            "ascendency": PublishedMetric(
                value=112891,
                unit="kcal-bits/m2/year",
                reported=True
            ),
            "reserve": PublishedMetric(
                value=91464,
                unit="kcal-bits/m2/year",
                reported=True
            ),
            "relative_ascendency": PublishedMetric(
                value=0.552,
                unit="dimensionless",
                note="Moderately organized system"
            ),
        },
        notes=[
            "Crystal River control ecosystem from Florida",
            "Published values include full system with external flows",
            "Internal-only calculations will differ significantly",
            "Different system boundary definition may cause discrepancies"
        ]
    ),

    # =========================================================================
    # SOUTH FLORIDA EVERGLADES - Heymans et al. (2002) reference anchors
    # -------------------------------------------------------------------------
    # DATA-PROVENANCE FIX (replaces the former mislabeled "florida_bay" entry):
    # The prior entry stored relative ascendency alpha = 0.367 citing
    # "Heymans et al. 2002" with a "subtropical seagrass / shallow marine"
    # description. That is unsourceable: Heymans, Ulanowicz & Bondavalli (2002),
    # "Network analysis of the South Florida Everglades graminoid marshes and
    # comparison with nearby cypress ecosystems", Ecological Modelling 149:5-23,
    # is about a FRESHWATER graminoid marsh and a cypress swamp, NOT a marine
    # seagrass bay, and it never reports 0.367 (which happens to equal 1/e used
    # elsewhere in the code as the robustness optimum).
    #
    # The paper reports relative ascendency directly, as whole-percent prose on
    # p.20 (Section 3.3, "System-level analysis"):
    #   "... the relative ascendency of 52% for the graminoids is higher than
    #    any such index they had encountered ... The relative ascendency of 34%
    #    reported for the cypress is lower than most of the relative
    #    ascendencies calculated by NETWRK ..."
    # (%AC, ascendency as a percentage of development capacity, IS the relative
    # ascendency alpha = A/C.) These two values are stored below.
    #
    # They are reference_only anchors: the paper gives alpha as a percentage in
    # prose without a published A/C/Phi breakdown, and no flow matrix shipped in
    # this repo reproduces the paper's alpha, so they are quoted literature
    # values (benchmark anchors), not recomputable-from-JSON entries.
    # =========================================================================
    "everglades_graminoid": NetworkPublishedData(
        source="Heymans et al. 2002",
        doi="10.1016/S0304-3800(01)00511-7",
        page=20,  # Section 3.3 "System-level analysis": "relative ascendency of 52%"
        log_base=LogBase.NATURAL,
        tolerance=0.10,
        reference_only=True,
        metrics={
            "relative_ascendency": PublishedMetric(
                value=0.52,  # 52% per Heymans et al. 2002, p.20 (prose)
                unit="dimensionless",
                note="alpha = A/C = 52% (Heymans et al. 2002, p.20)"
            ),
        },
        notes=[
            "South Florida Everglades freshwater graminoid marsh (sawgrass)",
            "Two-dimensional wetland dominated by periphyton primary production",
            "Exceptionally high relative ascendency (52%) -> tightly organized, "
            "efficient but relatively fragile system",
        ]
    ),
    "everglades_cypress": NetworkPublishedData(
        source="Heymans et al. 2002",
        doi="10.1016/S0304-3800(01)00511-7",
        page=20,  # Section 3.3 "System-level analysis": "relative ascendency of 34%"
        log_base=LogBase.NATURAL,
        tolerance=0.10,
        reference_only=True,
        metrics={
            "relative_ascendency": PublishedMetric(
                value=0.34,  # 34% per Heymans et al. 2002, p.20 (prose)
                unit="dimensionless",
                note="alpha = A/C = 34% (Heymans et al. 2002, p.20)"
            ),
        },
        notes=[
            "Big Cypress Preserve / Fakahatchee Strand cypress swamp",
            "Three-dimensional forested wetland with higher primary-producer diversity",
            "Lower relative ascendency (34%) -> more overhead/redundancy, "
            "greater long-term resilience",
        ]
    ),

    # =========================================================================
    # PRAWNS-ALLIGATOR - Original (3 pathways)
    # =========================================================================
    "prawns_alligator_original": NetworkPublishedData(
        source="Ulanowicz et al. 2009",
        doi="10.1016/j.ecocom.2008.10.005",
        figure="Figure 1",
        page=28,
        log_base=LogBase.NATURAL,
        tolerance=0.05,
        metrics={
            "total_system_throughput": PublishedMetric(
                value=102.6,
                unit="mg C/m2/year",
                note="Sum of all flows across 3 pathways"
            ),
            "ascendency": PublishedMetric(
                value=53.9,
                unit="mg C-bits/m2/year",
                reported=True
            ),
            "reserve": PublishedMetric(
                value=121.3,
                unit="mg C-bits/m2/year",
                reported=True
            ),
        },
        notes=[
            "Three parallel pathways from prawns to alligators",
            "Demonstrates importance of pathway diversity for resilience",
            "If fish pathway fails, system can adapt using alternatives"
        ]
    ),

    # =========================================================================
    # PRAWNS-ALLIGATOR - Efficient Only (single pathway)
    # =========================================================================
    "prawns_alligator_efficient": NetworkPublishedData(
        source="Ulanowicz et al. 2009",
        doi="10.1016/j.ecocom.2008.10.005",
        figure="Figure 2",
        page=29,
        log_base=LogBase.NATURAL,
        tolerance=0.05,
        metrics={
            "total_system_throughput": PublishedMetric(
                value=121.8,
                unit="mg C/m2/year",
                note="Higher throughput but zero resilience"
            ),
            "ascendency": PublishedMetric(
                value=100.3,
                unit="mg C-bits/m2/year",
                reported=True
            ),
            "reserve": PublishedMetric(
                value=0.0,
                unit="mg C-bits/m2/year",
                note="ZERO reserve - system has no resilience"
            ),
        },
        notes=[
            "Only the most efficient fish pathway",
            "Maximum throughput but zero reserve capacity",
            "Complete collapse if fish population fails",
            "Demonstrates danger of over-optimization"
        ]
    ),

    # =========================================================================
    # PRAWNS-ALLIGATOR - Adapted (after fish loss)
    # =========================================================================
    "prawns_alligator_adapted": NetworkPublishedData(
        source="Ulanowicz et al. 2009",
        doi="10.1016/j.ecocom.2008.10.005",
        figure="Figure 3",
        page=30,
        log_base=LogBase.NATURAL,
        tolerance=0.05,
        metrics={
            "total_system_throughput": PublishedMetric(
                value=99.7,
                unit="mg C/m2/year",
                note="Slightly reduced from original but system persists"
            ),
            "ascendency": PublishedMetric(
                value=44.5,
                unit="mg C-bits/m2/year",
                reported=True
            ),
            "reserve": PublishedMetric(
                value=68.2,
                unit="mg C-bits/m2/year",
                reported=True
            ),
        },
        notes=[
            "System after fish pathway loss",
            "Demonstrates adaptation through alternative pathways",
            "Lower throughput but maintained function",
            "Reserve capacity enabled survival"
        ]
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_published_metric(network_id: str, metric_name: str) -> Optional[float]:
    """
    Get a published metric value for a network.

    Args:
        network_id: Network identifier (e.g., 'cone_spring_original')
        metric_name: Metric name (e.g., 'relative_ascendency')

    Returns:
        Published value or None if not found
    """
    if network_id not in PUBLISHED_METRICS:
        return None

    network_data = PUBLISHED_METRICS[network_id]
    if metric_name not in network_data.metrics:
        return None

    return network_data.metrics[metric_name].value


def get_tolerance(network_id: str) -> float:
    """Get the validation tolerance for a network."""
    if network_id in PUBLISHED_METRICS:
        return PUBLISHED_METRICS[network_id].tolerance
    return 0.05  # Default 5%


def get_log_base(network_id: str) -> LogBase:
    """Get the logarithm base used in the paper."""
    if network_id in PUBLISHED_METRICS:
        return PUBLISHED_METRICS[network_id].log_base
    return LogBase.NATURAL


def list_networks() -> List[str]:
    """List all networks in the database."""
    return list(PUBLISHED_METRICS.keys())


def list_metrics(network_id: str) -> List[str]:
    """List all metrics available for a network."""
    if network_id not in PUBLISHED_METRICS:
        return []
    return list(PUBLISHED_METRICS[network_id].metrics.keys())


def get_network_info(network_id: str) -> Optional[Dict[str, Any]]:
    """Get full information about a network."""
    if network_id not in PUBLISHED_METRICS:
        return None

    data = PUBLISHED_METRICS[network_id]
    return {
        "source": data.source,
        "doi": data.doi,
        "figure": data.figure,
        "page": data.page,
        "log_base": data.log_base.value,
        "tolerance": data.tolerance,
        "metrics": {
            name: {
                "value": m.value,
                "unit": m.unit,
                "reported": m.reported,
                "note": m.note
            }
            for name, m in data.metrics.items()
        },
        "notes": data.notes
    }


# =============================================================================
# VALIDATION CHECK DEFINITIONS
# =============================================================================

VALIDATION_CHECKS = {
    "fundamental_relationship": {
        "description": "C = A + Phi must hold",
        "tolerance": 0.001,  # 0.1% error tolerance
        "formula": "development_capacity == ascendency + reserve"
    },
    "range_alpha": {
        "description": "Relative ascendency must be in [0, 1]",
        "min": 0.0,
        "max": 1.0,
        "formula": "0 <= relative_ascendency <= 1"
    },
    "range_fci": {
        "description": "Finn Cycling Index must be in [0, 1]",
        "min": 0.0,
        "max": 1.0,
        "formula": "0 <= finn_cycling_index <= 1"
    },
    "thermodynamic_constraint": {
        "description": "Ascendency cannot exceed Development Capacity",
        "formula": "ascendency <= development_capacity"
    },
    "positive_tst": {
        "description": "Total System Throughput must be positive",
        "min": 0.0,
        "formula": "total_system_throughput > 0"
    },
    "non_negative_reserve": {
        "description": "Reserve must be non-negative",
        "min": 0.0,
        "formula": "reserve >= 0"
    }
}


# =============================================================================
# REFERENCE NETWORKS MAPPING
# =============================================================================

# Map network IDs to their data file paths
NETWORK_DATA_FILES = {
    "cone_spring_original": "data/ecosystem_samples/cone_spring_original.json",
    "cone_spring_eutrophicated": "data/ecosystem_samples/cone_spring_eutrophicated.json",
    "crystal_river_creek": "data/ecosystem_samples/crystal_river_creek.json",
    # florida_bay.json is a genuine Ulanowicz et al. (1998) Florida Bay marine
    # food web (see the file's own metadata) and is still used by
    # validation/test_florida_bay.py. It is intentionally NOT tied to a
    # PUBLISHED_METRICS entry: the former "florida_bay" metrics entry that cited
    # Heymans 2002 was mislabeled and has been replaced by the everglades_*
    # reference anchors above.
    "florida_bay": "data/ecosystem_samples/florida_bay.json",
    "prawns_alligator_original": "data/ecosystem_samples/prawns_alligator_original.json",
    "prawns_alligator_efficient": "data/ecosystem_samples/prawns_alligator_efficient.json",
    "prawns_alligator_adapted": "data/ecosystem_samples/prawns_alligator_adapted.json",
}
