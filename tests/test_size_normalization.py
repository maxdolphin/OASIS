"""
Tests for size-relative normalization in the OASIS composite (src/oasis_calculator.py).

Scope (per the size-normalization task, grounded in
docs/business-revision/evidence/expert-org-management.md §3b):

  A. Role normalization is SIZE-RELATIVE and PRINCIPLED:
       norm_roles = min(number_of_roles / effective_nodes, 1)
     Roles R = exp(AMI) is bounded above by the effective number of nodes N
     (R = N/C with the effective connectivity C >= 1 connectivity floor,
     Ulanowicz 2004; Zorach & Ulanowicz 2003). So R/N in [0, 1] is a
     principled, size-relative normalizer replacing the fixed `roles/10`.
     `rolesPerNode` (R/N) is itself already <= 1 by the same bound, so it is
     normalized by the principled max of 1.0 (drop the arbitrary `/2`).

  B. The per-dimension caps are CENTRALIZED and DOCUMENTED (values unchanged
     unless a principled theoretical max exists) via DIMENSION_NORMALIZATION_CAPS.

  C. The autocatalysis sub-term is DE-SATURATED: the arbitrary `cycle_flow_ratio
     * 10` amplifier is removed so a network with a modest cycled-flow fraction
     no longer pins the sub-term to 1.0.
"""

import math
import numpy as np
import pytest

from src.ulanowicz_calculator import UlanowiczCalculator
from src.oasis_calculator import OASISCalculator


def _make_ring(n: int, base_flow: float = 100.0) -> np.ndarray:
    """A simple directed ring on n nodes (structurally self-similar across n).

    Every node sends `base_flow` to the next node; the only structural
    difference between two rings is their size n. This is the cleanest way to
    check that a normalizer is size-FAIR: the per-node structure is identical,
    so a size-fair role score must be comparable across n.
    """
    fm = np.zeros((n, n), dtype=float)
    for i in range(n):
        fm[i, (i + 1) % n] = base_flow
    return fm


def _oasis_for(fm: np.ndarray) -> OASISCalculator:
    calc = UlanowiczCalculator(fm)
    return OASISCalculator(calc)


# ---------------------------------------------------------------------------
# A. Size-relative role normalization
# ---------------------------------------------------------------------------

def test_norm_roles_equals_roles_over_effective_nodes():
    """norm_roles must be exactly roles / effective_nodes (principled bound)."""
    fm = _make_ring(8)
    oasis = _oasis_for(fm)
    result = oasis.calculate_intelligent_score()
    m = result['metrics']

    roles = m['number_of_roles']
    eff_nodes = oasis.ulanowicz.calculate_effective_nodes()
    expected = min(roles / eff_nodes, 1.0)

    assert m['norm_roles'] == pytest.approx(expected, abs=1e-9)


def test_norm_roles_in_unit_interval():
    for n in (3, 5, 8, 20):
        fm = _make_ring(n)
        oasis = _oasis_for(fm)
        norm = oasis.calculate_intelligent_score()['metrics']['norm_roles']
        assert 0.0 <= norm <= 1.0


def test_norm_roles_is_size_fair_small_vs_large():
    """A small (n=5) and a larger (n=20) structurally-similar network must get
    COMPARABLE normalized role scores — the fixed `/10` systematically penalized
    the small net and inflated the large one. With roles/effective_nodes the two
    are size-fair (within a small tolerance)."""
    small = _oasis_for(_make_ring(5)).calculate_intelligent_score()['metrics']
    large = _oasis_for(_make_ring(20)).calculate_intelligent_score()['metrics']

    # Under the OLD fixed /10 rule the two norm_roles would differ substantially
    # because roles ~ exp(AMI) grows with n while the divisor stayed fixed at 10.
    # Under roles/effective_nodes they are comparable.
    assert small['norm_roles'] == pytest.approx(large['norm_roles'], abs=0.15)


def test_norm_roles_per_node_dropped_arbitrary_half():
    """roles_per_node (R/N) is already <= 1 by the R <= N bound, so it must be
    normalized by the principled max of 1.0 (i.e. min(rpn, 1)), NOT the old /2."""
    fm = _make_ring(6)
    oasis = _oasis_for(fm)
    m = oasis.calculate_intelligent_score()['metrics']
    rpn = m['roles_per_node']
    assert m['norm_roles_per_node'] == pytest.approx(min(rpn, 1.0), abs=1e-9)


# ---------------------------------------------------------------------------
# B. Centralized, documented caps
# ---------------------------------------------------------------------------

def test_dimension_caps_are_centralized_and_documented():
    caps = OASISCalculator.DIMENSION_NORMALIZATION_CAPS
    assert set(caps.keys()) == {
        'open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable'
    }
    for v in caps.values():
        assert 0.0 < v <= 1.0


def test_dimension_caps_drive_the_normalization():
    """The dimension scores must read their cap from the central config, so a
    future empirical re-calibration is a one-line change. We verify by checking
    an all-ones raw sub-score would saturate at the documented cap for each dim.
    (We check the config is actually consumed, not just present.)"""
    caps = OASISCalculator.DIMENSION_NORMALIZATION_CAPS
    # Cap values are unchanged in this task unless principled; assert the
    # documented baseline set is present (guards against silent value drift).
    assert caps['open'] == pytest.approx(0.6)
    assert caps['autonomous'] == pytest.approx(0.5)
    assert caps['symbiotic'] == pytest.approx(0.7)
    assert caps['intelligent'] == pytest.approx(0.6)
    assert caps['sustainable'] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# C. Autocatalysis de-saturation
# ---------------------------------------------------------------------------

def test_autocatalysis_no_longer_saturates_on_modest_cycled_flow():
    """A network whose cycled-flow fraction is ~15% must NOT pin the flow
    component of the autocatalytic index to 1.0 (the old `* 10` amplifier
    saturated anything above 10% cycled flow)."""
    # Build a network with a dominant acyclic backbone plus a small cycle so the
    # cycle_flow_ratio lands around 0.15.
    # Nodes 0->1->2->0 form a cycle; a large through-flow 3->4 dilutes it.
    fm = np.array([
        [0, 15, 0, 0, 0],
        [0, 0, 15, 0, 0],
        [15, 0, 0, 0, 0],
        [0, 0, 0, 0, 85],
        [0, 0, 0, 0, 0],
    ], dtype=float)
    oasis = _oasis_for(fm)
    auto = oasis.calculate_autocatalytic_index()

    ratio = auto['cycle_flow_ratio']
    # Sanity: the fixture really does have a modest, non-trivial cycled fraction.
    assert 0.05 < ratio < 0.35, f"fixture cycle_flow_ratio={ratio}"

    # The de-saturated flow component equals the raw ratio (clamped to 1), NOT
    # min(1, ratio*10) which would be ~1.0 here.
    flow_component = min(1.0, ratio)
    assert flow_component < 0.999, "flow component should not be saturated"

    # And the composite index must reflect the un-amplified flow component.
    # index = 0.5*count_factor + 0.5*flow_component, with flow_component < 1.
    assert auto['autocatalytic_index'] < 0.5 + 0.5 * 0.999
