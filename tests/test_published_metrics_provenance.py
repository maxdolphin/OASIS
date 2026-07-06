"""
Data-provenance tests for the published-metrics reference database.

These lock in the fix for the mislabeled "florida_bay" benchmark anchor, which
previously stored relative ascendency alpha = 0.367 citing "Heymans et al. 2002"
with a "subtropical seagrass / shallow marine" description. That value was
unsourceable: Heymans, Ulanowicz & Bondavalli (2002), "Network analysis of the
South Florida Everglades graminoid marshes and comparison with nearby cypress
ecosystems", Ecological Modelling 149:5-23, is about a FRESHWATER graminoid marsh
and a cypress swamp (not a marine seagrass bay) and never reports 0.367.

The paper reports relative ascendency (alpha = A/C = ascendency as a percentage
of development capacity) directly in prose on p.20, Section 3.3 "System-level
analysis":
    "... the relative ascendency of 52% for the graminoids is higher than any
     such index they had encountered ... The relative ascendency of 34% reported
     for the cypress is lower than most of the relative ascendencies calculated
     by NETWRK ..."
=> graminoid alpha = 0.52, cypress alpha = 0.34.
"""

import pytest

from src.services import published_metrics_db as pdb


# --- The mislabeled entry/value must be gone ---------------------------------

def test_bogus_florida_bay_metrics_entry_removed():
    """The Heymans-cited florida_bay published-metric anchor (alpha=0.367) is gone."""
    assert "florida_bay" not in pdb.PUBLISHED_METRICS
    assert pdb.get_published_metric("florida_bay", "relative_ascendency") is None


def test_no_published_anchor_stores_the_1_over_e_value():
    """0.367 (== 1/e used elsewhere) must not survive as a published alpha anchor."""
    for net_id in pdb.list_networks():
        ra = pdb.get_published_metric(net_id, "relative_ascendency")
        if ra is not None:
            assert abs(ra - 0.367) > 1e-3, (
                f"{net_id} stores alpha≈0.367 (== 1/e); this was the unsourceable value"
            )


# --- The corrected, genuinely-sourced Heymans anchors ------------------------

def test_everglades_graminoid_matches_heymans_p20():
    # Heymans et al. 2002, Ecological Modelling 149:5-23, p.20 (Section 3.3):
    # "relative ascendency of 52% for the graminoids"
    alpha = pdb.get_published_metric("everglades_graminoid", "relative_ascendency")
    assert alpha == pytest.approx(0.52, abs=1e-9)

    info = pdb.get_network_info("everglades_graminoid")
    assert info is not None
    assert "Heymans" in info["source"]
    assert info["page"] == 20
    # Must be labeled as the freshwater graminoid marsh, NOT a marine seagrass bay.
    notes = " ".join(info.get("notes", [])).lower()
    assert "graminoid" in notes or "marsh" in notes
    assert "seagrass" not in notes and "marine" not in notes


def test_everglades_cypress_matches_heymans_p20():
    # Heymans et al. 2002, p.20 (Section 3.3):
    # "relative ascendency of 34% reported for the cypress"
    alpha = pdb.get_published_metric("everglades_cypress", "relative_ascendency")
    assert alpha == pytest.approx(0.34, abs=1e-9)

    info = pdb.get_network_info("everglades_cypress")
    assert info is not None
    assert "Heymans" in info["source"]
    assert info["page"] == 20
    notes = " ".join(info.get("notes", [])).lower()
    assert "cypress" in notes
    assert "seagrass" not in notes and "marine" not in notes


def test_relative_ascendency_is_a_c_ratio_in_unit_interval():
    """alpha = A/C is a dimensionless ratio in [0, 1] for the corrected anchors."""
    for net_id in ("everglades_graminoid", "everglades_cypress"):
        metric = pdb.PUBLISHED_METRICS[net_id].metrics["relative_ascendency"]
        assert metric.unit == "dimensionless"
        assert 0.0 <= metric.value <= 1.0


def test_reference_only_anchors_are_flagged():
    """Prose-quoted anchors with no recomputable flow matrix are reference_only."""
    for net_id in ("everglades_graminoid", "everglades_cypress"):
        assert pdb.PUBLISHED_METRICS[net_id].reference_only is True
        # ... and therefore intentionally have no NETWORK_DATA_FILES mapping.
        assert net_id not in pdb.NETWORK_DATA_FILES


def test_reference_only_anchors_are_skipped_by_validation_agent():
    """The computational validator skips reference_only anchors (no ERROR)."""
    from src.services.scientific_validation_agent import (
        ScientificValidationAgent,
        ValidationStatus,
    )

    agent = ScientificValidationAgent()
    for net_id in ("everglades_graminoid", "everglades_cypress"):
        result = agent.validate_network(net_id)
        assert result.overall_status == ValidationStatus.SKIP
