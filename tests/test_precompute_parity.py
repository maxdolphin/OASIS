"""
Parity test for the compute-once / read-thereafter wiring.

The precompute wiring changes the SOURCE of displayed/reported values from a
live recompute to a READ of the stored full profile. It must NOT change the
math. This test asserts, for three sample organizations, that the values read
from the precomputed profile are IDENTICAL to the values produced by a fresh
computation with the same calculators/analyzers.

Compared values:
- OASIS overall score + 5 dimension scores + overall status
- robustness, relative ascendency (alpha)  [core / Ulanowicz]
- a network-analysis metric (small-world sigma + density)

Uses a throwaway SQLite DB so the real DB is never touched.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.database.db_manager import DatabaseManager
from src.database.precompute_pipeline import PrecomputePipeline


SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "ecosystem_samples"
SAMPLE_ORGS = [
    "cone_spring_original",
    "crystal_river_creek",
    "chesapeake_bay_simplified",
]


def _load_sample(name):
    data = json.load(open(SAMPLE_DIR / f"{name}.json"))
    flow_matrix = np.array(data.get("flow_matrix", data.get("flows")), dtype=float)
    node_names = data.get("node_names", data.get("nodes"))
    org_name = data.get("organization", name)
    return flow_matrix, node_names, org_name


def _fresh_values(flow_matrix, node_names):
    """Compute the compared values live, the way the pre-wiring app did."""
    try:
        from ulanowicz_calculator import UlanowiczCalculator
        from oasis_calculator import OASISCalculator
        from network_analyzer import AdvancedNetworkAnalyzer
        from vectorized_metrics import get_all_vectorized_metrics
    except ImportError:
        from src.ulanowicz_calculator import UlanowiczCalculator
        from src.oasis_calculator import OASISCalculator
        from src.network_analyzer import AdvancedNetworkAnalyzer
        from src.vectorized_metrics import get_all_vectorized_metrics

    calc = UlanowiczCalculator(flow_matrix, node_names)
    analyzer = AdvancedNetworkAnalyzer(flow_matrix, node_names)
    oasis = OASISCalculator(calc, network_analyzer=analyzer)
    profile = oasis.get_oasis_profile()
    # Mirror _family_core: vectorized metrics, then extended overlay.
    core = dict(get_all_vectorized_metrics(flow_matrix))
    core.update(calc.get_extended_metrics())
    na = analyzer.get_all_metrics()

    return {
        "oasis_overall": profile["overall_score"],
        "oasis_status": profile["overall_status"],
        "oasis_dims": dict(profile["dimension_scores"]),
        "robustness": core.get("robustness"),
        "alpha": core.get("relative_ascendency"),
        "structural_information": core.get("structural_information"),
        "na_sigma": na["small_world"]["small_world_sigma"],
        "na_density": na["basic"]["density"],
    }


def _profile_values(profile):
    """Extract the same compared values by READING the stored profile."""
    oasis = profile["oasis"]
    core = profile["core"]
    na = profile["network_analysis"]
    return {
        "oasis_overall": oasis["overall_score"],
        "oasis_status": oasis["overall_status"],
        "oasis_dims": dict(oasis["dimension_scores"]),
        "robustness": core.get("robustness"),
        "alpha": core.get("relative_ascendency"),
        "structural_information": core.get("structural_information"),
        "na_sigma": na["small_world"]["small_world_sigma"],
        "na_density": na["basic"]["density"],
    }


@pytest.fixture
def pipeline(tmp_path):
    db = DatabaseManager(db_path=str(tmp_path / "parity.db"))
    return PrecomputePipeline(db_manager=db)


@pytest.mark.parametrize("org", SAMPLE_ORGS)
def test_read_from_profile_matches_fresh_compute(pipeline, org):
    flow_matrix, node_names, org_name = _load_sample(org)

    # Provision: compute + store the full profile once.
    result = pipeline.get_full_profile(flow_matrix, node_names, org_name=org_name)
    assert result["cache_hit"] is False
    read = _profile_values(result["profile"])

    # Fresh, independent computation.
    fresh = _fresh_values(flow_matrix, node_names)

    # Scalars must be numerically identical (same math, different source).
    for key in ("oasis_overall", "robustness", "alpha",
                "structural_information", "na_sigma", "na_density"):
        r, f = read[key], fresh[key]
        if r is None or f is None:
            assert r == f, f"{org}: {key} None mismatch (read={r}, fresh={f})"
        else:
            assert r == pytest.approx(f, rel=0, abs=0), \
                f"{org}: {key} mismatch (read={r}, fresh={f})"

    # OASIS status + per-dimension scores.
    assert read["oasis_status"] == fresh["oasis_status"], f"{org}: OASIS status differs"
    for dim in ("open", "autonomous", "symbiotic", "intelligent", "sustainable"):
        assert read["oasis_dims"][dim] == pytest.approx(fresh["oasis_dims"][dim], rel=0, abs=0), \
            f"{org}: OASIS dim {dim} differs (read={read['oasis_dims'][dim]}, fresh={fresh['oasis_dims'][dim]})"


@pytest.mark.parametrize("org", SAMPLE_ORGS)
def test_second_read_is_cache_hit(pipeline, org):
    """After provision, a second read HITs the store (does not recompute)."""
    flow_matrix, node_names, org_name = _load_sample(org)
    pipeline.get_full_profile(flow_matrix, node_names, org_name=org_name)
    second = pipeline.get_full_profile(flow_matrix, node_names, org_name=org_name)
    assert second["cache_hit"] is True
