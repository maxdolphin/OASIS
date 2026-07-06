"""
Tests for the gradient reframe of the OASIS viability verdict.

The old binary "Viable / Non-Viable (PASS/FAIL)" language is replaced by a
position-on-a-gradient + direction-of-travel, framed against the *indicative*
ecological reference band [0.2, 0.6]. These tests pin the classifier API and
assert that reframed verdict text carries the gradient position, the
direction-of-travel, and the indicative caveat — and never a bare absolute-fail
string.

No threshold constants or score formulas are changed by the reframe.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.report_intelligence import (  # noqa: E402
    assess_alpha_position,
    sustainable_verdict_narrative,
    VIABILITY_LOWER,
    VIABILITY_UPPER,
    INDICATIVE_REFERENCE_CAVEAT,
)


# ---------------------------------------------------------------------------
# Classifier: position + direction-of-travel
# ---------------------------------------------------------------------------
def test_under_organized():
    r = assess_alpha_position(0.09)
    assert r['position'] == 'under-organized'
    assert r['direction_of_travel'] == 'increase structure / coordination'
    # gradient value, signed below the lower edge
    assert r['relative_distance'] < 0


def test_balanced_mid():
    r = assess_alpha_position(0.45)
    assert r['position'] == 'balanced'
    assert r['direction_of_travel'] == 'maintain balance'


def test_over_organized():
    r = assess_alpha_position(0.72)
    assert r['position'] == 'over-organized'
    assert r['direction_of_travel'] == 'increase redundancy / flexibility'
    assert r['relative_distance'] > 0


def test_boundary_lower_is_balanced():
    r = assess_alpha_position(VIABILITY_LOWER)  # exactly 0.2
    assert r['position'] == 'balanced'
    assert r['direction_of_travel'] == 'maintain balance'


def test_boundary_upper_is_balanced():
    r = assess_alpha_position(VIABILITY_UPPER)  # exactly 0.6
    assert r['position'] == 'balanced'
    assert r['direction_of_travel'] == 'maintain balance'


def test_descriptor_and_caveat_present():
    r = assess_alpha_position(0.09)
    assert isinstance(r['descriptor'], str) and r['descriptor']
    assert 'indicative' in r['descriptor'].lower()
    assert r['caveat'] == INDICATIVE_REFERENCE_CAVEAT
    assert 'directional indicator' in r['caveat']
    assert 'not a compliance threshold' in r['caveat']


def test_constants_unchanged():
    # Guard against accidental threshold drift.
    assert VIABILITY_LOWER == 0.2
    assert VIABILITY_UPPER == 0.6


# ---------------------------------------------------------------------------
# Reframed sustainability verdict text (oasis_calculator interpretations)
# ---------------------------------------------------------------------------
def test_reframed_low_alpha_verdict_uses_gradient_and_caveat():
    # Low SUSTAINABLE score + low alpha: the reframed verdict must read as a
    # gradient position + direction-of-travel, not a bare absolute fail.
    text = sustainable_verdict_narrative(30, 0.09)
    lowered = text.lower()
    # gradient position + direction-of-travel present
    assert 'under-organized' in lowered
    assert 'increase structure' in lowered
    # indicative-reference framing present
    assert 'indicative' in lowered
    # NO bare absolute pass/fail language
    assert 'non-viable' not in lowered
    assert 'unsustainable' not in lowered
