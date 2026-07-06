"""
Track-1 mutualism correction — test-driven.

Fath et al. (2019) Principle 8 defines ecological mutualism as an *integral*
(direct + indirect) utility property, not the direct-only reciprocity the engine
originally computed. The correct construction is Patten's integral utility matrix:

    Direct utility:    d_ij = (f_ij - f_ji) / T_i     (T_i = throughflow of i)
    Integral utility:  U    = (I - D)^(-1)
    Network mutualism (benefit:cost) = sum(U>0) / |sum(U<0)|   OVER OFF-DIAGONAL (i != j)

The benefit:cost sums exclude the diagonal: network mutualism is a property of the
OFF-DIAGONAL relational pairings (i != j). The diagonal of U is self-utility /
return-flow (always >= 0) and is not a "relation"; including it inflates the
numerator for every network. See §A5 of the ENA review.

Patten's classic result: INDIRECT effects make relationships more mutualistic, so
the integral b:c >= direct b:c on a network with a closed loop of indirect paths.
Corollary: with NO indirect path (a 2-node network) integral b:c EQUALS direct b:c
(no network-mutualism lift).

See docs/business-revision/evidence/expert-ena-methods.md §A5 (A5-mutualism CONFIRM).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ulanowicz_calculator import UlanowiczCalculator
from oasis_calculator import OASISCalculator


def _oasis(flow_matrix):
    uc = UlanowiczCalculator(np.asarray(flow_matrix, dtype=float))
    return OASISCalculator(uc)


# ---------------------------------------------------------------------------
# Hand-checkable U = (I - D)^-1 on a small directed network
# ---------------------------------------------------------------------------

def test_direct_utility_matrix_hand_checked():
    """
    3-node ring A->B->C->A with equal flow f=1.
    Throughflow T_i = out_i (for a balanced ring, in=out=1) so T_i = 1 for each.
    d_ij = (f_ij - f_ji)/T_i.
      Row A (T=1): to B f_AB=1, f_BA=0 -> +1 ; to C f_AC=0, f_CA=1 -> -1
      Row B (T=1): to A -1 ; to C +1
      Row C (T=1): to A +1 ; to B -1
    """
    F = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=float)
    oc = _oasis(F)
    res = oc.calculate_mutualism_index()

    D = np.asarray(res['direct_utility_matrix'])
    expected_D = np.array([
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, 1.0],
        [1.0, -1.0, 0.0],
    ])
    assert np.allclose(D, expected_D), f"D mismatch:\n{D}"

    # Integral utility U = (I - D)^-1, verified against numpy on the same D.
    U = np.asarray(res['integral_utility_matrix'])
    expected_U = np.linalg.inv(np.eye(3) - expected_D)
    assert np.allclose(U, expected_U), f"U mismatch:\n{U}\nexpected:\n{expected_U}"


def test_integral_utility_2node_hand_checked():
    """
    2-node exploitative pair A->B (f=4), B->A (f=1).
    T_A = 4, T_B = 1.
      d_AB = (f_AB - f_BA)/T_A = (4-1)/4 = 0.75
      d_BA = (f_BA - f_AB)/T_B = (1-4)/1 = -3.0
    D = [[0, 0.75],[-3, 0]].
    U = (I-D)^-1 computed analytically:  det(I-D) = 1 - (0.75*3) ... via numpy.

    A 2-node network has NO indirect path, so the OFF-DIAGONAL integral b:c must
    EQUAL the direct b:c (no network-mutualism lift). Here both = 0.25 (the sole
    off-diagonal positive is 0.75, the sole off-diagonal negative is -3.0 in BOTH
    D and U, since the 2-node U merely rescales those off-diagonal signs). This
    proves "no indirect path -> no lift" and that the diagonal is excluded (the
    prior diagonal-inclusive value of ~0.917 was a self-utility artifact).
    """
    F = np.array([
        [0, 4],
        [1, 0],
    ], dtype=float)
    oc = _oasis(F)
    res = oc.calculate_mutualism_index()

    D = np.asarray(res['direct_utility_matrix'])
    expected_D = np.array([[0.0, 0.75], [-3.0, 0.0]])
    assert np.allclose(D, expected_D), f"D mismatch:\n{D}"

    U = np.asarray(res['integral_utility_matrix'])
    expected_U = np.linalg.inv(np.eye(2) - expected_D)
    assert np.allclose(U, expected_U)

    # No indirect path => integral b:c == direct b:c (off-diagonal aggregation).
    assert res['direct_benefit_cost_ratio'] == pytest.approx(0.25)
    assert res['integral_benefit_cost_ratio'] == pytest.approx(0.25)
    assert res['integral_benefit_cost_ratio'] == pytest.approx(
        res['direct_benefit_cost_ratio'])


# ---------------------------------------------------------------------------
# Patten's result: indirect effects increase mutualism (integral b:c >= direct b:c)
# ---------------------------------------------------------------------------

def test_indirect_effects_increase_mutualism():
    """
    A directed cycle of exploitative (one-way) links has ZERO direct pairwise
    mutualism (no reciprocal pairs), but the closed loop of indirect effects makes
    the network net-mutualistic under integral utility (Patten). With the OFF-
    DIAGONAL aggregation the 4-ring gives a GENUINE indirect lift:
        direct b:c   = 1.0   (off-diagonal +/- of D balance on a symmetric ring)
        integral b:c = 3.0   (indirect loop lifts the positive utility)
    (The prior diagonal-inclusive integral value was 6.0, a self-utility artifact.)
    """
    # 4-node ring: pure one-way exploitation around the loop.
    F = np.array([
        [0, 5, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 5],
        [5, 0, 0, 0],
    ], dtype=float)
    oc = _oasis(F)
    res = oc.calculate_mutualism_index()

    direct_bc = res['direct_benefit_cost_ratio']
    integral_bc = res['integral_benefit_cost_ratio']

    # Direct-only pairwise: no reciprocal (bidirectional) pairs -> mutualism is 0.
    assert res['direct_mutualism'] == pytest.approx(0.0)
    # Off-diagonal corrected b:c values.
    assert direct_bc == pytest.approx(1.0)
    assert integral_bc == pytest.approx(3.0)
    # Patten: indirect effects make it strictly MORE mutualistic here.
    assert integral_bc > direct_bc
    assert res['fallback_direct_only'] is False


def test_benefit_cost_excludes_diagonal():
    """
    Guard: the b:c aggregation must exclude U's diagonal (self-utility). The
    diagonal of U is always >= 0, so including it would inflate the numerator.
    We verify the reported integral b:c matches an off-diagonal recomputation of
    the returned U, and does NOT match the diagonal-inclusive value.
    """
    F = np.array([
        [0, 5, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 5],
        [5, 0, 0, 0],
    ], dtype=float)
    oc = _oasis(F)
    res = oc.calculate_mutualism_index()
    U = np.asarray(res['integral_utility_matrix'])

    off = U.copy()
    np.fill_diagonal(off, 0.0)
    off_bc = off[off > 0].sum() / abs(off[off < 0].sum())

    incl_bc = U[U > 0].sum() / abs(U[U < 0].sum())

    assert res['integral_benefit_cost_ratio'] == pytest.approx(off_bc)
    assert res['integral_benefit_cost_ratio'] != pytest.approx(incl_bc)


def test_singular_matrix_fallback():
    """
    If (I - D) is singular / non-invertible, the calculator must fall back to
    direct-only reporting with a flag, and must not raise.
    """
    # An empty / disconnected network (all zero flows). Every throughflow is 0,
    # D is all-zero, (I-D)=I is invertible, so construct a genuinely singular case
    # by monkeypatching is not ideal; instead use a network that yields singular I-D.
    # A perfectly balanced 2-cycle A<->B with equal flows gives d_AB=d_BA=0 (D=0),
    # which is invertible; to force singularity we build I-D singular directly.
    # Use a 2-node network whose D makes (I-D) singular:
    #   want det(I-D)=0. With D=[[0,a],[b,0]], det(I-D)=1-ab=0 -> ab=1.
    #   d_AB = (f_AB-f_BA)/T_A, d_BA=(f_BA-f_AB)/T_B. Pick flows so a*b=1.
    # f_AB=2,f_BA=0 -> T_A=2 -> a=1 ; f: for b we need (f_BA-f_AB)/T_B=1 -> impossible sign.
    # Simpler: assert the API exposes the fallback flag and handles a forced singular D.
    F = np.array([
        [0, 2],
        [0, 0],
    ], dtype=float)
    oc = _oasis(F)
    # Force a singular (I - D) by patching the direct-utility builder's output.
    orig = oc._build_direct_utility_matrix
    oc._build_direct_utility_matrix = lambda: np.array([[0.0, 1.0], [1.0, 0.0]])  # det(I-D)=1-1=0
    try:
        res = oc.calculate_mutualism_index()
    finally:
        oc._build_direct_utility_matrix = orig

    assert res['fallback_direct_only'] is True
    # On fallback, integral values fall back to the direct component (no crash).
    assert 'direct_mutualism' in res
    assert res['integral_utility_matrix'] is None


def test_near_singular_matrix_fallback():
    """
    A merely-singular check (det < tiny) misses near-singular blow-ups: det ~ 1e-6
    -> U entries ~ 1e6 -> b:c explodes -> integral_mutualism pins to 1.0. The guard
    must use a CONDITION-NUMBER test and fall back to direct-only on ill-conditioning.

    Build D so (I - D) is near-singular: D = [[0, 1],[1-1e-6, 0]] ->
    det(I - D) = 1 - (1)(1-1e-6) = 1e-6, cond(I - D) ~ 4e6 (well above any real
    network's cond, which is < ~10). The calculator must set fallback_direct_only.
    """
    F = np.array([
        [0, 2],
        [0, 0],
    ], dtype=float)
    oc = _oasis(F)
    orig = oc._build_direct_utility_matrix
    # near-singular (not exactly singular): det(I-D) = 1e-6, cond ~ 4e6
    oc._build_direct_utility_matrix = lambda: np.array([[0.0, 1.0],
                                                        [1.0 - 1e-6, 0.0]])
    try:
        res = oc.calculate_mutualism_index()
    finally:
        oc._build_direct_utility_matrix = orig

    assert res['fallback_direct_only'] is True
    assert res['integral_utility_matrix'] is None
    # integral b:c must NOT have exploded / pinned to 1.0 — it falls back to direct.
    assert res['integral_benefit_cost_ratio'] == pytest.approx(
        res['direct_benefit_cost_ratio'])


def test_backcompat_keys_present():
    """The original public keys must survive for existing consumers."""
    F = np.array([
        [0, 3, 1],
        [2, 0, 4],
        [1, 5, 0],
    ], dtype=float)
    oc = _oasis(F)
    res = oc.calculate_mutualism_index()
    for k in ('mutual_pairs', 'one_way_pairs', 'mutualism_ratio',
              'weighted_mutualism', 'total_connected_pairs'):
        assert k in res, f"back-compat key missing: {k}"
