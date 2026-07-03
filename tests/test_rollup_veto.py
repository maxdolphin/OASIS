"""
Tests for the OASIS composite roll-up "band cap" veto.

Rule (dimension-agnostic worst-dimension band cap):
    Order bands CRITICAL=0 < WARNING=1 < HEALTHY=2.
    - raw_overall_level from weighted mean (>=60 HEALTHY / >=40 WARNING / else CRITICAL)
    - each dimension's level from HEALTH_THRESHOLDS / get_status
    - worst_dim_level = min(level over 5 dimensions)
    - final_overall_level = min(raw_overall_level, worst_dim_level + 1)
    - numeric overall score is UNCHANGED; only the label is capped.
    - `capped_by` lists the dimension(s) that drove the cap.

The "Non-Viable org labeled HEALTHY" contradiction:
    (OPEN 100, AUT 100, SYM 100, INT 100, SUSTAINABLE 0) -> weighted mean 80
    Raw -> HEALTHY, but SUSTAINABLE is CRITICAL so overall must be capped to WARNING.
"""

from src.oasis_calculator import OASISCalculator


# Scores chosen so each dimension lands in the intended band via HEALTH_THRESHOLDS.
# open healthy>=50, autonomous>=40, symbiotic>=55, intelligent>=45, sustainable>=60.
ALL_HEALTHY = {
    'open': 100.0,
    'autonomous': 100.0,
    'symbiotic': 100.0,
    'intelligent': 100.0,
    'sustainable': 100.0,
}


def _apply(scores, weights=None):
    """Invoke the pure roll-up logic under test."""
    return OASISCalculator.compute_overall_status(scores, weights)


def test_non_viable_org_not_labeled_healthy():
    """(100,100,100,100, SUSTAINABLE critical) -> mean ~80 but overall WARNING not HEALTHY."""
    scores = dict(ALL_HEALTHY, sustainable=0.0)  # sustainable critical (<40)
    result = _apply(scores)

    assert abs(result['overall_score'] - 80.0) < 1e-6, result['overall_score']
    assert result['raw_overall_status'] == 'HEALTHY'
    assert result['overall_status'] == 'WARNING'
    assert result['capped'] is True
    assert 'sustainable' in result['capped_by']


def test_all_dimensions_healthy_overall_healthy():
    """All dims HEALTHY -> overall HEALTHY, no cap applied."""
    result = _apply(ALL_HEALTHY)
    assert result['overall_status'] == 'HEALTHY'
    assert result['capped'] is False
    assert result['capped_by'] == []


def test_one_warning_rest_healthy_stays_healthy():
    """One dim WARNING (worst=1), rest HEALTHY, high mean -> overall may stay HEALTHY."""
    # autonomous in warning band [25,40); rest very high -> mean still >= 60.
    scores = dict(ALL_HEALTHY, autonomous=30.0)
    result = _apply(scores)
    assert result['raw_overall_status'] == 'HEALTHY'
    # worst_dim_level = 1 (WARNING); +1 = 2 (HEALTHY) allowed -> not capped down.
    assert result['overall_status'] == 'HEALTHY'
    assert result['capped'] is False


def test_two_dimensions_critical_capped_at_warning():
    """Two dims CRITICAL -> still capped at WARNING (worst=0, +1=1)."""
    scores = dict(ALL_HEALTHY, sustainable=0.0, symbiotic=0.0)
    result = _apply(scores)
    assert result['overall_status'] == 'WARNING'
    assert result['capped'] is True
    assert set(result['capped_by']) == {'sustainable', 'symbiotic'}


def test_numeric_score_unchanged_by_cap():
    """The cap changes only the label, never the numeric weighted-mean score."""
    scores = dict(ALL_HEALTHY, sustainable=0.0)
    result = _apply(scores)
    expected_mean = sum(scores[d] * 0.20 for d in scores)
    assert abs(result['overall_score'] - expected_mean) < 1e-6


def test_low_mean_stays_critical_even_if_dims_ok():
    """If the weighted mean itself is CRITICAL, cap never raises it above raw."""
    # All dims warning-ish but mean < 40 -> raw CRITICAL. worst_dim_level+1 can't raise it.
    scores = {
        'open': 35.0,        # warning
        'autonomous': 30.0,  # warning
        'symbiotic': 40.0,   # warning
        'intelligent': 32.0, # warning
        'sustainable': 45.0, # warning
    }
    result = _apply(scores)
    # mean = 36.4 -> CRITICAL raw; min(CRITICAL, WARNING+1=HEALTHY) = CRITICAL
    assert result['raw_overall_status'] == 'CRITICAL'
    assert result['overall_status'] == 'CRITICAL'


def test_profile_integration_exposes_capped_fields():
    """The full get_oasis_profile() path exposes the new fields."""
    import numpy as np
    from src.ulanowicz_calculator import UlanowiczCalculator

    flow = np.array([
        [0, 10, 0, 0],
        [0, 0, 8, 0],
        [0, 0, 0, 6],
        [4, 0, 0, 0],
    ], dtype=float)
    uc = UlanowiczCalculator(flow, node_names=['A', 'B', 'C', 'D'])
    profile = OASISCalculator(uc).get_oasis_profile()

    assert 'overall_status' in profile
    assert 'raw_overall_status' in profile
    assert 'overall_status_capped' in profile
    assert 'capped_by' in profile
    assert isinstance(profile['capped_by'], list)
