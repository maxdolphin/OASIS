"""
Tests for named OASIS context WEIGHTING PROFILES.

Rationale: docs/business-revision/evidence/expert-org-management.md §3 —
keep equal 20% as the published (honest) default, but expose a small number of
named, context-tagged weighting profiles a consultant selects as a lens. Only
MODEST tilts (no false precision, no extreme weightings).

Re-weighting is a CHEAP recombination: it only changes the OVERALL score + the
capped status label, computed as a weighted mean of the FIVE ALREADY-COMPUTED
dimension scores. It must NOT recompute any dimension metric.

Contract under test (in src/oasis_calculator.py):
    - WEIGHTING_PROFILES: dict name -> {'weights': {5 dims -> w}, 'description': str}
    - OASISCalculator.apply_weighting_profile(dimension_scores, profile) ->
      dict with overall_score + capped status metadata (reuses compute_overall_status).
"""

import math

import numpy as np
import pytest

from src.oasis_calculator import OASISCalculator, WEIGHTING_PROFILES

DIMENSIONS = {'open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable'}


# --------------------------------------------------------------------------
# Profile-definition invariants
# --------------------------------------------------------------------------
def test_profiles_exist_and_include_balanced_default():
    assert 'Balanced (default)' in WEIGHTING_PROFILES
    # 2-4 named profiles per the expert guidance (at least the default + a few).
    assert 2 <= len(WEIGHTING_PROFILES) <= 6


@pytest.mark.parametrize("name", list(WEIGHTING_PROFILES.keys()))
def test_every_profile_weights_sum_to_one_over_five_dims(name):
    profile = WEIGHTING_PROFILES[name]
    weights = profile['weights']
    # Covers exactly the 5 dimensions.
    assert set(weights.keys()) == DIMENSIONS, name
    # Sums to 1.0 within 1e-9.
    assert abs(sum(weights.values()) - 1.0) < 1e-9, (name, sum(weights.values()))


@pytest.mark.parametrize("name", list(WEIGHTING_PROFILES.keys()))
def test_every_profile_has_human_description(name):
    desc = WEIGHTING_PROFILES[name].get('description', '')
    assert isinstance(desc, str) and len(desc.strip()) >= 10, name


@pytest.mark.parametrize("name", list(WEIGHTING_PROFILES.keys()))
def test_tilts_are_modest(name):
    """No extreme weightings: every weight stays within a modest band of 0.20."""
    weights = WEIGHTING_PROFILES[name]['weights']
    for dim, w in weights.items():
        assert 0.10 <= w <= 0.30, (name, dim, w)


def test_balanced_default_is_equal_weights():
    weights = WEIGHTING_PROFILES['Balanced (default)']['weights']
    for dim in DIMENSIONS:
        assert abs(weights[dim] - 0.20) < 1e-9, dim


# --------------------------------------------------------------------------
# Cheap recombination behavior
# --------------------------------------------------------------------------
# A non-uniform org (deliberately uneven across dimensions) so a tilt bites.
NON_UNIFORM = {
    'open': 80.0,
    'autonomous': 40.0,
    'symbiotic': 70.0,
    'intelligent': 30.0,
    'sustainable': 90.0,
}


def test_balanced_reproduces_equal_weight_overall_exactly():
    """'Balanced (default)' recombination == the equal-weight compute_overall_status."""
    baseline = OASISCalculator.compute_overall_status(
        NON_UNIFORM, OASISCalculator.DEFAULT_WEIGHTS)
    got = OASISCalculator.apply_weighting_profile(NON_UNIFORM, 'Balanced (default)')
    assert abs(got['overall_score'] - baseline['overall_score']) < 1e-12
    assert got['overall_status'] == baseline['overall_status']


@pytest.mark.parametrize("name", list(WEIGHTING_PROFILES.keys()))
def test_recombination_equals_full_compute_for_overall(name):
    """
    PARITY: cheap recombination overall == a full OASISCalculator(weights=profile)
    overall computed from scratch. We drive the calculator through a real flow
    matrix, read the STORED dimension scores, then check that applying the profile
    to those stored scores equals building a fresh calculator with those weights.
    """
    calc = _make_calculator()
    profile_scores = calc.get_oasis_profile()['dimension_scores']

    weights = WEIGHTING_PROFILES[name]['weights']

    # Full compute-from-scratch with the profile weights.
    full = OASISCalculator(_make_ulanowicz(), dimension_weights=weights)
    full_profile = full.get_oasis_profile()

    # Cheap recombination on the STORED (equal-weight) dimension scores.
    cheap = OASISCalculator.apply_weighting_profile(profile_scores, name)

    # Dimension scores are identical (weights never touch them), so overalls match.
    assert abs(cheap['overall_score'] - full_profile['overall_score']) < 1e-9, name
    assert cheap['overall_status'] == full_profile['overall_status'], name


def test_tilted_profile_changes_overall_vs_balanced_on_non_uniform_org():
    balanced = OASISCalculator.apply_weighting_profile(NON_UNIFORM, 'Balanced (default)')
    changed = False
    for name in WEIGHTING_PROFILES:
        if name == 'Balanced (default)':
            continue
        tilted = OASISCalculator.apply_weighting_profile(NON_UNIFORM, name)
        if abs(tilted['overall_score'] - balanced['overall_score']) > 1e-6:
            changed = True
    assert changed, "at least one tilted profile must move the overall on a non-uniform org"


def test_capped_status_still_applies_after_reweighting():
    """A CRITICAL dimension still caps the overall label after re-weighting."""
    # Four carriers high, sustainable CRITICAL. Even a profile that de-emphasizes
    # sustainable keeps a high numeric overall but the band cap must hold.
    scores = {
        'open': 100.0, 'autonomous': 100.0, 'symbiotic': 100.0,
        'intelligent': 100.0, 'sustainable': 0.0,  # critical (<40)
    }
    for name in WEIGHTING_PROFILES:
        res = OASISCalculator.apply_weighting_profile(scores, name)
        assert res['overall_status'] != 'HEALTHY', name
        assert res['capped'] is True, name
        assert 'sustainable' in res['capped_by'], name


def test_apply_accepts_explicit_weight_dict():
    """The method accepts a raw weight dict (manual 'Custom') as well as a name."""
    custom = {'open': 0.20, 'autonomous': 0.20, 'symbiotic': 0.20,
              'intelligent': 0.20, 'sustainable': 0.20}
    got = OASISCalculator.apply_weighting_profile(NON_UNIFORM, custom)
    baseline = OASISCalculator.compute_overall_status(NON_UNIFORM)
    assert abs(got['overall_score'] - baseline['overall_score']) < 1e-12


def test_unknown_profile_name_raises():
    with pytest.raises(ValueError):
        OASISCalculator.apply_weighting_profile(NON_UNIFORM, 'No Such Profile')


# --------------------------------------------------------------------------
# Helpers: build a real calculator from a small flow matrix.
# --------------------------------------------------------------------------
def _make_ulanowicz():
    from src.ulanowicz_calculator import UlanowiczCalculator
    # Small asymmetric flow network (non-trivial, non-uniform).
    flow = np.array([
        [0.0, 5.0, 2.0, 0.0],
        [1.0, 0.0, 4.0, 3.0],
        [0.0, 2.0, 0.0, 6.0],
        [4.0, 0.0, 1.0, 0.0],
    ])
    names = ['A', 'B', 'C', 'D']
    return UlanowiczCalculator(flow, names)


def _make_calculator():
    return OASISCalculator(_make_ulanowicz())
