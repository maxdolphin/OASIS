"""
Tests for the full-index precompute mechanism (Pass A).

Covers:
- precompute_full_profile returns all four families with representative keys.
- get_full_profile: compute-once, read-thereafter (cache HIT does not recompute).
- Version mismatch forces recompute (MISS -> restore with current version).
- Degenerate graph returns a profile with per-family error markers, no crash.

Uses a throwaway SQLite DB so the real DB is never touched.
"""

import numpy as np
import pytest

from src.database.db_manager import DatabaseManager
from src.database.precompute_pipeline import PrecomputePipeline
from src.database import full_profile as fp_mod
from src.database.full_profile import (
    precompute_full_profile,
    FORMULA_VERSION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flow_matrix():
    """A small but non-degenerate directed flow network (5 nodes, cyclic)."""
    return np.array([
        [0, 10, 0, 0, 5],
        [0, 0, 8, 2, 0],
        [0, 0, 0, 7, 1],
        [3, 0, 0, 0, 6],
        [0, 4, 0, 0, 0],
    ], dtype=float)


@pytest.fixture
def node_names():
    return ['A', 'B', 'C', 'D', 'E']


@pytest.fixture
def pipeline(tmp_path):
    """PrecomputePipeline wired to a throwaway SQLite DB."""
    db_path = tmp_path / "test_networks.db"
    db = DatabaseManager(db_path=str(db_path))
    return PrecomputePipeline(db_manager=db)


# ---------------------------------------------------------------------------
# precompute_full_profile: family coverage
# ---------------------------------------------------------------------------

def test_full_profile_has_all_families(flow_matrix, node_names):
    profile = precompute_full_profile(flow_matrix, node_names, org_name='Test Org')

    assert profile['formula_version'] == FORMULA_VERSION
    for family in ('core', 'oasis', 'network_analysis', 'intelligence', 'meta'):
        assert family in profile, f"missing family: {family}"


def test_core_family_keys(flow_matrix, node_names):
    core = precompute_full_profile(flow_matrix, node_names)['core']
    for key in ('ascendency', 'relative_ascendency', 'robustness',
                'number_of_roles', 'finn_cycling_index'):
        assert key in core, f"core missing {key}"


def test_oasis_family_keys(flow_matrix, node_names):
    oasis = precompute_full_profile(flow_matrix, node_names)['oasis']
    scores = oasis['dimension_scores']
    for dim in ('open', 'autonomous', 'symbiotic', 'intelligent', 'sustainable'):
        assert dim in scores, f"oasis dimension missing {dim}"
    assert 'overall_status' in oasis
    assert 'capped_by' in oasis


def test_network_analysis_family_keys(flow_matrix, node_names):
    na = precompute_full_profile(flow_matrix, node_names)['network_analysis']
    assert 'centralities' in na
    assert 'communities' in na


def test_intelligence_family_keys(flow_matrix, node_names):
    intel = precompute_full_profile(flow_matrix, node_names)['intelligence']
    assert 'risk' in intel
    assert 'benchmark' in intel


def test_meta_family(flow_matrix, node_names):
    meta = precompute_full_profile(flow_matrix, node_names, org_name='Acme')['meta']
    assert meta['n_nodes'] == 5
    assert meta['n_edges'] == int(np.sum(flow_matrix > 0))
    assert meta['organization'] == 'Acme'


# ---------------------------------------------------------------------------
# get_full_profile: compute-once, read-thereafter
# ---------------------------------------------------------------------------

def test_get_full_profile_first_call_is_miss(pipeline, flow_matrix, node_names):
    result = pipeline.get_full_profile(flow_matrix, node_names, org_name='Test Org')
    assert result['cache_hit'] is False
    assert result['profile']['formula_version'] == FORMULA_VERSION


def test_cache_hit_does_not_recompute(pipeline, flow_matrix, node_names, monkeypatch):
    """Second call on the same matrix is a HIT and must NOT recompute."""
    # First call populates the cache.
    first = pipeline.get_full_profile(flow_matrix, node_names, org_name='Test Org')
    assert first['cache_hit'] is False

    # Spy: fail loudly if precompute_full_profile is called again.
    call_counter = {'n': 0}
    real = fp_mod.precompute_full_profile

    def spy(*args, **kwargs):
        call_counter['n'] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(fp_mod, 'precompute_full_profile', spy)
    # Also patch the reference imported into the pipeline module, if any.
    import src.database.precompute_pipeline as pp_mod
    if hasattr(pp_mod, 'precompute_full_profile'):
        monkeypatch.setattr(pp_mod, 'precompute_full_profile', spy)

    second = pipeline.get_full_profile(flow_matrix, node_names, org_name='Test Org')

    assert second['cache_hit'] is True
    assert call_counter['n'] == 0, "cache HIT must not recompute the profile"


def test_version_mismatch_forces_recompute(pipeline, flow_matrix, node_names):
    """A stored profile with an old formula_version is treated as a MISS."""
    # Populate the cache.
    first = pipeline.get_full_profile(flow_matrix, node_names, org_name='Test Org')
    assert first['cache_hit'] is False

    # Corrupt the stored version to simulate a pre-fix formula version.
    network_hash = pipeline.db.compute_network_hash(flow_matrix, node_names)
    network = pipeline.db.get_network_by_hash(network_hash)
    pipeline.db.save_precomputed_metrics(
        network_id=network['id'],
        tier=3,
        metrics={'formula_version': 'OLD-0000', 'core': {}, 'stale': True},
        formula_version='OLD-0000',
    )

    # Now a read with the current version must MISS and recompute.
    result = pipeline.get_full_profile(flow_matrix, node_names, org_name='Test Org')
    assert result['cache_hit'] is False
    assert result['profile']['formula_version'] == FORMULA_VERSION
    assert 'stale' not in result['profile']

    # And the store is refreshed to the current version.
    stored = pipeline.db.get_precomputed_metrics(
        network['id'], tier=3, required_version=FORMULA_VERSION
    )
    assert stored is not None
    assert stored['formula_version'] == FORMULA_VERSION


# ---------------------------------------------------------------------------
# Degenerate graph: per-family error markers, no crash
# ---------------------------------------------------------------------------

def test_degenerate_graph_returns_profile_with_error_markers():
    tiny = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    names = ['x', 'y']

    # Must not raise.
    profile = precompute_full_profile(tiny, names, org_name='Tiny')

    assert profile['formula_version'] == FORMULA_VERSION
    for family in ('core', 'oasis', 'network_analysis', 'intelligence', 'meta'):
        assert family in profile

    # If any family failed on the degenerate graph it must carry an error marker
    # rather than having taken down the whole profile.
    for family in ('core', 'oasis', 'network_analysis', 'intelligence'):
        fam = profile[family]
        if isinstance(fam, dict) and '_error' in fam:
            assert isinstance(fam['_error'], str)


def test_degenerate_graph_via_pipeline(pipeline):
    tiny = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    result = pipeline.get_full_profile(tiny, ['x', 'y'], org_name='Tiny')
    assert result['cache_hit'] is False
    assert result['profile']['formula_version'] == FORMULA_VERSION
