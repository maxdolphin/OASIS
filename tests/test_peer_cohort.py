"""
Tests for the PEER-COHORT benchmarking scaffold (percentile-vs-peers).

HONESTY CONTRACT (mirrors the product requirement):
- No peer number is ever fabricated.
- When the size/sector-matched cohort has fewer than MIN_COHORT_SIZE members
  the mechanism MUST return an ``insufficient_cohort`` status and NO percentile.
- ``insufficient_cohort`` is the expected DEFAULT state today (the real store
  does not yet hold >=10 sector-matched peers).

All DB work happens in a throwaway temp SQLite file; the real DB is untouched.
"""

import numpy as np
import pytest

from src.database.db_manager import DatabaseManager
from src.database.precompute_pipeline import PrecomputePipeline, FULL_PROFILE_TIER
from src.database import peer_cohort as pc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """A throwaway DatabaseManager backed by a temp SQLite file."""
    return DatabaseManager(db_path=str(tmp_path / "cohort_test.db"))


def _seed_member(db, name, node_count, alpha, sector='tech',
                 robustness=0.3, overall=60.0):
    """Persist one synthetic cohort member (network + tier-3 profile blob).

    Uses a *fabricated* profile blob purely to exercise the query/percentile
    plumbing quickly; it does NOT run the real calculators. The alpha value is
    the number under test. Members default to a sector tag because only tagged
    (deliberately registered) networks are eligible peers.
    """
    net_hash = f"hash_{name}"
    net_id = db.save_network(
        name=name, source_file="", node_count=node_count,
        edge_count=node_count, network_hash=net_hash, sector=sector,
    )
    blob = {
        'core': {'relative_ascendency': alpha, 'robustness': robustness},
        'oasis': {'overall_score': overall,
                  'dimension_scores': {'open': overall}},
    }
    db.save_precomputed_metrics(net_id, FULL_PROFILE_TIER, blob)
    return net_id


# ---------------------------------------------------------------------------
# 1. Percentile computation
# ---------------------------------------------------------------------------

def test_percentile_middle_of_five_is_about_50():
    # value equal to the 3rd of 5 sorted peers -> median -> ~50th percentile
    cohort = [0.10, 0.20, 0.30, 0.40, 0.50]
    res = pc.compute_peer_percentile(0.30, cohort)
    assert res['n'] == 5
    assert res['percentile'] == pytest.approx(50.0, abs=1e-6)
    assert res['median'] == pytest.approx(0.30)


def test_percentile_top_and_bottom():
    cohort = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    top = pc.compute_peer_percentile(100, cohort)
    bottom = pc.compute_peer_percentile(-100, cohort)
    assert top['percentile'] == pytest.approx(100.0)
    assert bottom['percentile'] == pytest.approx(0.0)


def test_percentile_reports_quartiles():
    cohort = [10, 20, 30, 40, 50]
    res = pc.compute_peer_percentile(35, cohort)
    assert res['q1'] == pytest.approx(20.0)
    assert res['median'] == pytest.approx(30.0)
    assert res['q3'] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# 2. Size-bucket derivation (boundary correctness)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (1, 'micro'), (9, 'micro'),
    (10, 'small'), (49, 'small'),
    (50, 'mid'), (249, 'mid'),
    (250, 'large'), (10000, 'large'),
])
def test_size_bucket_boundaries(n, expected):
    assert pc.size_bucket_from_node_count(n) == expected


# ---------------------------------------------------------------------------
# 3. Insufficient cohort (CRITICAL honesty guarantee)
# ---------------------------------------------------------------------------

def test_insufficient_cohort_returns_status_not_percentile():
    cohort = [0.3, 0.4, 0.5]  # only 3 peers, below MIN_COHORT_SIZE
    res = pc.peer_benchmark(0.35, cohort)
    assert res['status'] == 'insufficient_cohort'
    assert res['n'] == 3
    assert res['min'] == pc.MIN_COHORT_SIZE
    assert 'percentile' not in res  # NO fabricated number


def test_empty_cohort_is_insufficient():
    res = pc.peer_benchmark(0.42, [])
    assert res['status'] == 'insufficient_cohort'
    assert res['n'] == 0
    assert 'percentile' not in res


def test_default_state_of_fresh_store_is_insufficient(db):
    """The honest DEFAULT: a store with no peers cannot benchmark alpha."""
    res = pc.peer_alpha_benchmark(db, alpha=0.4, node_count=30)
    assert res['status'] == 'insufficient_cohort'
    assert 'percentile' not in res


def test_untagged_networks_are_not_counted_as_peers(db):
    """HONESTY GUARD: untagged records (ecological samples / synthetic fixtures)
    must NEVER be counted as peer organizations, even if >=10 exist and match
    the size bucket. This is the safeguard that keeps the real store's default
    state at insufficient_cohort."""
    for i in range(12):
        _seed_member(db, f"untagged_{i}", node_count=20, alpha=0.3, sector=None)
    res = pc.peer_alpha_benchmark(db, alpha=0.4, node_count=30)
    assert res['status'] == 'insufficient_cohort'
    assert res['n'] == 0
    assert 'percentile' not in res


# ---------------------------------------------------------------------------
# 4. Sufficient cohort -> real percentile + stats
# ---------------------------------------------------------------------------

def test_sufficient_cohort_returns_real_percentile(db):
    # 10 peers all in the 'small' bucket (node_count 10-49), alphas 0.10..0.55
    alphas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    for i, a in enumerate(alphas):
        _seed_member(db, f"peer_{i}", node_count=20, alpha=a)

    # Org alpha = 0.40 sits above 7 peers -> upper part of the distribution.
    res = pc.peer_alpha_benchmark(db, alpha=0.40, node_count=30)
    assert res['status'] == 'ok'
    assert res['n'] == 10
    assert 0.0 <= res['percentile'] <= 100.0
    assert res['percentile'] > 50.0
    assert 'median' in res and 'q1' in res and 'q3' in res


def test_cohort_excludes_self(db):
    """The org's own stored record must not count as one of its peers."""
    alphas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    ids = [_seed_member(db, f"peer_{i}", node_count=20, alpha=a)
           for i, a in enumerate(alphas)]
    # Excluding one member drops the cohort to 9 -> insufficient.
    res = pc.peer_alpha_benchmark(db, alpha=0.4, node_count=30,
                                  exclude_network_id=ids[0])
    assert res['status'] == 'insufficient_cohort'
    assert res['n'] == 9


# ---------------------------------------------------------------------------
# 5. Filters narrow the cohort
# ---------------------------------------------------------------------------

def test_sector_filter_narrows_cohort(db):
    # 10 'tech' peers + 6 'finance' peers, all in the same size bucket.
    for i in range(10):
        _seed_member(db, f"tech_{i}", node_count=20, alpha=0.3, sector='tech')
    for i in range(6):
        _seed_member(db, f"fin_{i}", node_count=20, alpha=0.5, sector='finance')

    tech = pc.peer_alpha_benchmark(db, alpha=0.35, node_count=30, sector='tech')
    assert tech['status'] == 'ok'
    assert tech['n'] == 10

    fin = pc.peer_alpha_benchmark(db, alpha=0.55, node_count=30, sector='finance')
    assert fin['status'] == 'insufficient_cohort'  # only 6 finance peers
    assert fin['n'] == 6


def test_size_bucket_filter_narrows_cohort(db):
    # 10 'small'-bucket peers + 4 'large'-bucket peers.
    for i in range(10):
        _seed_member(db, f"small_{i}", node_count=20, alpha=0.3)
    for i in range(4):
        _seed_member(db, f"large_{i}", node_count=500, alpha=0.5)

    # Org with 30 nodes -> 'small' bucket -> only the 10 small peers match.
    res = pc.peer_alpha_benchmark(db, alpha=0.35, node_count=30)
    assert res['status'] == 'ok'
    assert res['n'] == 10

    # Org with 400 nodes -> 'large' bucket -> only 4 peers -> insufficient.
    res_large = pc.peer_alpha_benchmark(db, alpha=0.4, node_count=400)
    assert res_large['status'] == 'insufficient_cohort'
    assert res_large['n'] == 4


# ---------------------------------------------------------------------------
# 6. query_cohort returns key metrics for members
# ---------------------------------------------------------------------------

def test_query_cohort_returns_member_metrics(db):
    _seed_member(db, "m1", node_count=20, alpha=0.3, robustness=0.28, overall=55.0)
    _seed_member(db, "m2", node_count=30, alpha=0.4, robustness=0.31, overall=65.0)
    members = pc.query_cohort(db, size_bucket='small')
    assert len(members) == 2
    m = {x['name']: x for x in members}['m1']
    assert m['metrics']['relative_ascendency'] == pytest.approx(0.3)
    assert m['metrics']['robustness'] == pytest.approx(0.28)
    assert m['metrics']['oasis_overall'] == pytest.approx(55.0)
    assert m['size_bucket'] == 'small'


# ---------------------------------------------------------------------------
# 7. Honest fallback note (never fabricates numbers)
# ---------------------------------------------------------------------------

def test_note_for_insufficient_is_honest():
    res = {'status': 'insufficient_cohort', 'n': 3, 'min': pc.MIN_COHORT_SIZE}
    note = pc.format_peer_benchmark_note(res, alpha=0.4)
    assert 'indicative' in note.lower()
    assert 'N=3' in note
    assert str(pc.MIN_COHORT_SIZE) in note
    # must NOT claim a percentile
    assert 'percentile' not in note.lower() or 'requires' in note.lower()


def test_note_for_sufficient_reports_percentile():
    res = {'status': 'ok', 'n': 12, 'percentile': 62.5,
           'median': 0.33, 'q1': 0.25, 'q3': 0.41}
    note = pc.format_peer_benchmark_note(res, alpha=0.4)
    assert '62' in note
    assert '12' in note
    assert 'percentile' in note.lower()


# ---------------------------------------------------------------------------
# 8. Ingestion path grows the cohort
# ---------------------------------------------------------------------------

def test_ingest_directory_grows_cohort(db, tmp_path):
    import json
    src = tmp_path / "nets"
    src.mkdir()
    # Two small valid networks written as JSON flow matrices.
    for i in range(2):
        m = np.array([[0, 5, 0], [0, 0, 3], [2, 0, 0]], dtype=float) + i
        (src / f"net_{i}.json").write_text(json.dumps({
            'organization': f"IngestOrg_{i}",
            'flow_matrix': m.tolist(),
            'node_names': ['A', 'B', 'C'],
        }))

    pipeline = PrecomputePipeline(db_manager=db)
    summary = pc.ingest_directory(str(src), sector='logistics',
                                  pipeline=pipeline, db=db)
    assert summary['ingested'] == 2
    # Both networks are now persisted and tagged with the sector.
    members = pc.query_cohort(db, sector='logistics')
    assert len(members) == 2
    assert all(x['sector'] == 'logistics' for x in members)
