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
    # FLORIDA BAY
    # =========================================================================
    "florida_bay": NetworkPublishedData(
        source="Heymans et al. 2002",
        doi=None,
        log_base=LogBase.NATURAL,
        tolerance=0.10,
        metrics={
            "relative_ascendency": PublishedMetric(
                value=0.367,
                unit="dimensionless",
                note="Lower organization, higher resilience"
            ),
        },
        notes=[
            "Subtropical seagrass-dominated ecosystem",
            "Shallow marine environment",
            "Value indicates good balance between efficiency and resilience"
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
    "florida_bay": "data/ecosystem_samples/florida_bay.json",
    "prawns_alligator_original": "data/ecosystem_samples/prawns_alligator_original.json",
    "prawns_alligator_efficient": "data/ecosystem_samples/prawns_alligator_efficient.json",
    "prawns_alligator_adapted": "data/ecosystem_samples/prawns_alligator_adapted.json",
}
