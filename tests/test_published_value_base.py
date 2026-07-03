"""
Base-awareness tests for the OASIS published-value VALIDATION comparison layer.

Background
----------
The Ulanowicz engine (`UlanowiczCalculator`) computes information-theoretic
MAGNITUDES (Ascendency A, Development Capacity C, Overhead/Reserve Phi, Average
Mutual Information AMI, flow diversity / statistical entropy H) with the natural
logarithm -> the results are in **nats**. Several stored published reference
values (e.g. Ulanowicz & Norden 1990 "cone_spring_original", Ulanowicz 1986
"crystal_river_creek") are quoted in **bits** (log base 2), per the papers.

Comparing a nats value against a bits value is a base mismatch: it fails every
base-DEPENDENT magnitude even when the flow computation is otherwise sound.
Base-INVARIANT metrics -- relative ascendency alpha = A/C, robustness, and any
pure ratio -- cancel the log base and must NOT be converted.

These tests lock in a single, explicit, per-metric, base-aware conversion in the
comparison layer:

  * ``nats_to_bits(x) = x / ln(2) = x * log2(e)``  (magnitude INCREASES; nats->bits)
  * conversion is applied ONLY to base-dependent metrics of LOG2 networks
  * base-invariant metrics and NATURAL/unknown-base networks are left untouched.

The tests deliberately exercise the conversion LOGIC with controlled inputs, so
they are independent of any data-provenance issues in the stored flow matrices.
"""

import math

import pytest

from src.services import published_metrics_db as pdb
from src.services.published_metrics_db import LogBase
from src.services.scientific_validation_agent import (
    ScientificValidationAgent,
    nats_to_bits,
    ValidationStatus,
)


LN2 = math.log(2)


# ---------------------------------------------------------------------------
# 1. Conversion direction (hand-computed): nats -> bits
# ---------------------------------------------------------------------------

def test_nats_to_bits_hand_computed():
    """ln(2) nats is exactly 1 bit; the helper reproduces the hand value."""
    assert nats_to_bits(math.log(2)) == pytest.approx(1.0, rel=1e-12)


def test_nats_to_bits_matches_log2_of_e_scaling():
    """bits = nats / ln2 = nats * log2(e). Both forms must agree."""
    for x in (0.5, 1.0, 1.623, 68191.0):
        assert nats_to_bits(x) == pytest.approx(x / LN2, rel=1e-12)
        assert nats_to_bits(x) == pytest.approx(x * math.log2(math.e), rel=1e-12)


def test_nats_to_bits_increases_magnitude():
    """Because ln2 < 1, converting nats->bits must INCREASE a positive value."""
    for x in (1e-6, 1.0, 1000.0):
        assert nats_to_bits(x) > x


def test_wrong_direction_is_detectably_different():
    """A guard: bits->nats (multiply by ln2) is NOT the same as nats->bits."""
    x = 1.623
    assert (x * LN2) != pytest.approx(nats_to_bits(x), rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Base-dependence classification
# ---------------------------------------------------------------------------

def test_magnitudes_are_base_dependent():
    for name in (
        "ascendency",
        "development_capacity",
        "reserve",
        "overhead",
        "average_mutual_information",
        "statistical_entropy",
        "flow_diversity",
    ):
        assert pdb.is_base_dependent(name), f"{name} should be base-DEPENDENT"


def test_ratios_and_indices_are_base_invariant():
    for name in (
        "relative_ascendency",
        "ascendency_ratio",
        "robustness",
        "total_system_throughput",
        "network_efficiency",
        "finn_cycling_index",
        "is_viable",
    ):
        assert not pdb.is_base_dependent(name), f"{name} should be base-INVARIANT"


# ---------------------------------------------------------------------------
# 3. Per-metric conversion applied only for LOG2 base-dependent metrics
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return ScientificValidationAgent()


def test_log2_base_dependent_metric_is_converted(agent):
    """A LOG2 network converts a base-dependent engine nats value to bits."""
    value_nats = 100.0
    converted = agent._convert_engine_value("ascendency", value_nats, LogBase.LOG2)
    assert converted == pytest.approx(value_nats / LN2, rel=1e-12)


def test_log2_base_invariant_metric_is_not_converted(agent):
    """alpha and robustness must be compared raw even for a LOG2 network."""
    for name in ("relative_ascendency", "robustness", "ascendency_ratio"):
        raw = 0.505
        assert agent._convert_engine_value(name, raw, LogBase.LOG2) == raw


def test_natural_base_dependent_metric_is_not_force_converted(agent):
    """A NATURAL-base network's magnitude stays in nats (no /ln2 applied)."""
    value_nats = 53.9
    assert agent._convert_engine_value(
        "ascendency", value_nats, LogBase.NATURAL
    ) == value_nats


def test_unknown_base_is_not_force_converted(agent):
    """Guard: an unrecognized/LOG10 base must not be silently divided by ln2."""
    value_nats = 42.0
    # LOG10 is a real enum member but no network uses it and the engine is nats;
    # it must NOT be treated as if it were LOG2.
    assert agent._convert_engine_value(
        "ascendency", value_nats, LogBase.LOG10
    ) == value_nats


# ---------------------------------------------------------------------------
# 4. End-to-end comparison: base reconciliation flips a nats/bits mismatch
#    from FAIL to PASS when the ONLY difference is the log base.
# ---------------------------------------------------------------------------

def test_comparison_passes_when_only_difference_is_base(agent):
    """
    Published value in bits, engine value the SAME quantity in nats.
    After the base-aware conversion the comparison must PASS (0% error);
    without conversion (raw nats vs bits) it must FAIL. This isolates the
    base fix from any flow-data provenance issue.
    """
    published_bits = 68191.0
    engine_nats = published_bits * LN2  # same physical quantity, in nats

    # With base-aware conversion (LOG2 network, base-dependent metric):
    converted = agent._convert_engine_value("ascendency", engine_nats, LogBase.LOG2)
    passed = agent._compare_metric(
        metric_name="ascendency",
        published_value=published_bits,
        computed_value=converted,
        tolerance=0.05,
    )
    assert passed.status == ValidationStatus.PASS
    assert passed.percent_error == pytest.approx(0.0, abs=1e-6)

    # Without conversion (the historical nats-vs-bits bug): FAIL.
    unconverted = agent._compare_metric(
        metric_name="ascendency",
        published_value=published_bits,
        computed_value=engine_nats,
        tolerance=0.05,
    )
    assert unconverted.status == ValidationStatus.FAIL


def test_ami_bits_reconciliation_hand_value(agent):
    """Cone-spring published AMI = 1.623 bits == 1.1249 nats; converting the
    nats value back must reproduce 1.623 bits (right direction, hand-checked)."""
    ami_nats = 1.623 * LN2  # 1.1249... nats
    assert agent._convert_engine_value(
        "average_mutual_information", ami_nats, LogBase.LOG2
    ) == pytest.approx(1.623, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. Real-network wiring: for a LOG2 network the comparison value is in bits;
#    for base-invariant alpha it is unchanged.
# ---------------------------------------------------------------------------

def test_cone_spring_base_dependent_comparison_is_in_bits(agent):
    """
    For the LOG2 cone_spring_original network, the value the comparison layer
    puts up against the published (bits) ascendency must be the engine nats
    value converted to bits -- NOT the raw nats value. (We assert the wiring,
    not a published-value match: the stored internal-only flow matrix does not
    reproduce the paper's full-system magnitude -- a separate data-provenance
    issue documented in the provenance tests.)
    """
    result = agent.validate_network("cone_spring_original")
    comps = {c.metric_name: c for c in result.metric_comparisons}

    engine_nats_A = result.computed_metrics["ascendency"]
    asc = comps["ascendency"]
    assert asc.computed_value == pytest.approx(nats_to_bits(engine_nats_A), rel=1e-9)
    # And it is strictly larger than the raw nats value (conversion happened).
    assert asc.computed_value > engine_nats_A


def test_cone_spring_alpha_comparison_is_raw(agent):
    """Base-invariant alpha for the same LOG2 network is compared WITHOUT
    conversion: the compared value equals the engine's relative_ascendency."""
    result = agent.validate_network("cone_spring_original")
    comps = {c.metric_name: c for c in result.metric_comparisons}
    engine_alpha = result.computed_metrics["relative_ascendency"]
    assert comps["relative_ascendency"].computed_value == pytest.approx(
        engine_alpha, rel=1e-12
    )


def test_natural_network_ascendency_comparison_is_raw(agent):
    """A NATURAL-base network (prawns) must compare ascendency in raw nats."""
    result = agent.validate_network("prawns_alligator_original")
    comps = {c.metric_name: c for c in result.metric_comparisons}
    engine_nats_A = result.computed_metrics["ascendency"]
    assert comps["ascendency"].computed_value == pytest.approx(
        engine_nats_A, rel=1e-12
    )


def test_reference_only_networks_still_skip(agent):
    """Everglades reference anchors have no flow matrix -> SKIP, unchanged."""
    for net in ("everglades_graminoid", "everglades_cypress"):
        result = agent.validate_network(net)
        assert result.overall_status == ValidationStatus.SKIP
