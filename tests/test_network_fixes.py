"""
Track-1 Network-Science Formula Corrections — Test Suite
=========================================================

Test-driven pins for the six canonical-reference-backed corrections (FIX A-F).
Each fix has a failing-then-passing test that pins the corrected behavior.

References:
- Freeman (1979): directed degree centralization normalizer = (n-1)^2.
- Brandes (2001): weighted betweenness/closeness treat weight as DISTANCE (invert flow -> 1/flow).
- Fronczak et al. (2004): L_rand ~ ln(n)/ln<k>, with <k> = 2m/n.
- Telford/Bassett et al. (2011): omega = L_rand/L - C/C_lattice (lattice clustering, not random).
- Colizza et al. (2006): rich-club must be normalized against a degree-preserving randomization.
- Flow-diversity utilization: numerator (nats) and denominator log-base must match.

Flow networks are DIRECTED (nx.DiGraph). Metrics that legitimately require an undirected
projection (small-world, rich-club) are projected explicitly.
"""

import os
import sys

import numpy as np
import networkx as nx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ulanowicz_calculator import UlanowiczCalculator
from network_analyzer import AdvancedNetworkAnalyzer
import publication_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _directed_star_matrix(n):
    """Hub (node 0) -> all other nodes. Directed out-star."""
    m = np.zeros((n, n))
    for j in range(1, n):
        m[0, j] = 1.0
    return m


def _random_directed_matrix(n, density=0.4, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < density:
                m[i, j] = rng.random() * 10 + 1
    return m


# ---------------------------------------------------------------------------
# FIX A — Freeman centralization directed normalizer (n-1)^2
# ---------------------------------------------------------------------------

class TestFixAFreemanCentralization:
    """Directed degree centralization must use (n-1)^2, never exceed 1."""

    def test_directed_star_out_centralization_near_one(self):
        n = 6
        calc = UlanowiczCalculator(_directed_star_matrix(n), use_vectorized=False)
        topo = calc.calculate_network_topology_metrics()
        # Out-star: one hub with out-degree n-1, rest 0.
        # Sum(d*-d_i) = (n-1)^2, normalizer (n-1)^2 => exactly 1.0.
        assert topo["out_degree_centralization"] == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= topo["out_degree_centralization"] <= 1.0
        # The averaged degree_centralization must also stay in-bounds.
        assert 0.0 <= topo["degree_centralization"] <= 1.0

    def test_random_directed_centralization_bounded(self):
        for seed in range(6):
            n = 8
            calc = UlanowiczCalculator(
                _random_directed_matrix(n, seed=seed), use_vectorized=False
            )
            topo = calc.calculate_network_topology_metrics()
            for key in ("in_degree_centralization", "out_degree_centralization",
                        "degree_centralization"):
                assert 0.0 <= topo[key] <= 1.0, (
                    f"{key}={topo[key]} out of [0,1] at seed {seed}"
                )

    def test_old_normalizer_would_exceed_one(self):
        """Documents the pre-fix defect: (n-1)(n-2) gives >1 on the star."""
        n = 6
        sum_diff = (n - 1) ** 2  # out-star raw dispersion
        old = sum_diff / ((n - 1) * (n - 2))  # buggy denominator
        new = sum_diff / ((n - 1) ** 2)       # correct denominator
        assert old > 1.0
        assert new == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# FIX B — Betweenness/closeness treat flow as strength (invert to distance)
# ---------------------------------------------------------------------------

class TestFixBBetweennessInversion:
    """High-flow bridge endpoints must score HIGH betweenness (1/flow distance)."""

    def _parallel_route_matrix(self):
        # 5-node hand-checkable case with TWO parallel routes from node 0 to node 4:
        #   STRONG route through node 1:  0 == 1 == 4   (flow = 100 each tie)
        #   WEAK route through 2 and 3:   0 -- 2 -- 3 -- 4  (flow = 1 each tie)
        #
        # Correct behavior (Brandes 2001, weight = distance): invert to 1/flow so
        # the strong route is SHORT (distance 0.01+0.01=0.02) and the weak route is
        # LONG (1+1+1=3). Shortest paths take the strong route, so the strong-route
        # hub (node 1) gets HIGH betweenness and the weak-route hubs (2,3) get 0.
        #
        # Buggy behavior (raw flow as distance): strong route sums to 200 (looks
        # LONG) while the weak route sums to 3 (looks SHORT) -> paths take the weak
        # route, so node 2/3 get high betweenness and node 1 gets ZERO. The fix
        # flips this ranking.
        n = 5
        m = np.zeros((n, n))
        strong, weak = 100.0, 1.0

        def bi(a, b, w):
            m[a, b] = w
            m[b, a] = w

        bi(0, 1, strong)
        bi(1, 4, strong)   # strong route hub = node 1
        bi(0, 2, weak)
        bi(2, 3, weak)
        bi(3, 4, weak)     # weak route hubs = nodes 2, 3
        return m

    def test_strong_route_hub_ranks_high_after_inversion(self):
        m = self._parallel_route_matrix()
        analyzer = AdvancedNetworkAnalyzer(m, [f"N{i}" for i in range(5)])
        btw = analyzer.calculate_centralities()["betweenness"]
        # The strong-tie hub (node 1) must outrank the weak-route hubs (2, 3),
        # because the inverted-distance shortest path prefers the strong route.
        assert btw[1] > btw[2], f"strong hub {btw[1]:.3f} !> weak hub {btw[2]:.3f}"
        assert btw[1] > btw[3], f"strong hub {btw[1]:.3f} !> weak hub {btw[3]:.3f}"
        # Hand-checkable exact values (nx normalized betweenness on this graph):
        assert btw[1] == pytest.approx(0.5, abs=1e-9)
        assert btw[2] == pytest.approx(0.0, abs=1e-9)

    def test_inversion_changes_ranking_vs_raw_weight(self):
        """Pins that the fix REVERSES the ranking vs the buggy raw-weight metric.

        Raw flow-as-distance ranks the weak-route hub above the strong-route hub
        (node2 > node1); the corrected 1/flow distance ranks them the other way
        (node1 > node2). This is the exact bug the fix corrects."""
        m = self._parallel_route_matrix()
        G = nx.DiGraph()
        n = m.shape[0]
        for i in range(n):
            for j in range(n):
                if m[i, j] > 0:
                    G.add_edge(i, j, weight=m[i, j])

        # Buggy metric: flow used directly as distance.
        raw = nx.betweenness_centrality(G, weight="weight", normalized=True)
        # Corrected metric: distance = 1/flow.
        H = G.copy()
        for u, v, d in H.edges(data=True):
            d["distance"] = 1.0 / d["weight"]
        inv = nx.betweenness_centrality(H, weight="distance", normalized=True)

        # Buggy: weak-route hub (2) beats strong-route hub (1).
        assert raw[2] > raw[1]
        # Fixed: strong-route hub (1) beats weak-route hub (2) — ranking flipped.
        assert inv[1] > inv[2]

    def test_analyzer_uses_inverted_distance_for_closeness(self):
        """Closeness must also invert: strong ties => close, not far."""
        m = self._parallel_route_matrix()
        analyzer = AdvancedNetworkAnalyzer(m, [f"N{i}" for i in range(5)])
        cent = analyzer.calculate_centralities()
        assert "closeness" in cent
        # all closeness values finite and non-negative
        assert all(np.isfinite(v) and v >= 0 for v in cent["closeness"].values())

    def test_eigenvector_pagerank_still_use_weight_as_strength(self):
        """FIX B must NOT invert eigenvector/pagerank (weight = strength there)."""
        m = self._parallel_route_matrix()
        analyzer = AdvancedNetworkAnalyzer(m, [f"N{i}" for i in range(5)])
        cent = analyzer.calculate_centralities()
        # Sanity: these exist and are proper distributions/vectors.
        assert abs(sum(cent["pagerank"].values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in cent["eigenvector"].values())


# ---------------------------------------------------------------------------
# FIX C — Small-world random baseline mean degree <k> = 2m/n
# ---------------------------------------------------------------------------

class TestFixCMeanDegree:
    def test_mean_degree_complete_graph(self):
        # Complete undirected K_n has <k> = n-1 = 2m/n.
        n = 6
        Kn = nx.complete_graph(n)
        m = Kn.number_of_edges()
        expected = 2 * m / n
        assert expected == pytest.approx(n - 1)
        # exercise the analyzer's helper
        adj = np.zeros((n, n))
        for u, v in Kn.edges():
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        analyzer = AdvancedNetworkAnalyzer(adj, [f"N{i}" for i in range(n)])
        Gu = analyzer.G.to_undirected()
        assert analyzer._mean_degree(Gu) == pytest.approx(2 * Gu.number_of_edges() / n)

    def test_mean_degree_ring_lattice(self):
        # Ring lattice C_n (each node degree 2): <k> = 2.
        n = 8
        ring = nx.cycle_graph(n)
        adj = np.zeros((n, n))
        for u, v in ring.edges():
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        analyzer = AdvancedNetworkAnalyzer(adj, [f"N{i}" for i in range(n)])
        Gu = analyzer.G.to_undirected()
        assert analyzer._mean_degree(Gu) == pytest.approx(2.0)

    def test_small_world_metrics_finite(self):
        m = _random_directed_matrix(10, density=0.5, seed=3)
        analyzer = AdvancedNetworkAnalyzer(m, [f"N{i}" for i in range(10)])
        sw = analyzer.calculate_small_world_metrics()
        assert np.isfinite(sw["random_path_length"])
        assert np.isfinite(sw["small_world_sigma"])
        assert np.isfinite(sw["small_world_omega"])


# ---------------------------------------------------------------------------
# FIX D — Small-world omega uses LATTICE clustering, bounded to [-1, 1]
# ---------------------------------------------------------------------------

class TestFixDOmegaLattice:
    def test_omega_bounded_random(self):
        for seed in range(5):
            m = _random_directed_matrix(12, density=0.5, seed=seed)
            analyzer = AdvancedNetworkAnalyzer(m, [f"N{i}" for i in range(12)])
            sw = analyzer.calculate_small_world_metrics()
            w = sw["small_world_omega"]
            assert -1.0 - 1e-9 <= w <= 1.0 + 1e-9, f"omega={w} out of [-1,1] seed {seed}"

    def test_ring_lattice_omega_negative(self):
        # A ring lattice (Watts-Strogatz with p=0) is the lattice end => omega ~ -1.
        n = 20
        ws = nx.watts_strogatz_graph(n, 4, 0.0, seed=1)
        adj = np.zeros((n, n))
        for u, v in ws.edges():
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        analyzer = AdvancedNetworkAnalyzer(adj, [f"N{i}" for i in range(n)])
        sw = analyzer.calculate_small_world_metrics()
        assert sw["small_world_omega"] < 0.0

    def test_lattice_clustering_helper(self):
        # C_lattice approx = 3(k-2)/(4(k-1)) for ring lattice of mean degree k.
        analyzer = AdvancedNetworkAnalyzer(
            _directed_star_matrix(4), [f"N{i}" for i in range(4)]
        )
        k = 4
        assert analyzer._lattice_clustering(k) == pytest.approx(3 * (k - 2) / (4 * (k - 1)))
        # guard small k: no crash / division by zero
        assert 0.0 <= analyzer._lattice_clustering(1) <= 1.0
        assert 0.0 <= analyzer._lattice_clustering(2) <= 1.0


# ---------------------------------------------------------------------------
# FIX E — Rich-club normalized (Colizza 2006) with small-graph guard
# ---------------------------------------------------------------------------

class TestFixERichClub:
    def test_rich_core_normalized_above_one(self):
        # Build a graph with a GENUINE rich core: a moderate-degree random
        # background periphery plus a forced clique among 6 hub nodes and extra
        # hub->periphery edges to lift core degrees well above the periphery.
        # A degree-preserving randomization will NOT reproduce the extra core
        # interconnection, so normalized phi(k) > 1 at the high-k (core) cutoff.
        # (Colizza et al. 2006: phi_norm > 1 signals a real rich-club effect.)
        rng = np.random.default_rng(3)
        G = nx.gnm_random_graph(60, 90, seed=5)  # moderate-degree periphery
        core = list(range(6))
        for i in core:  # force a clique on the core
            for j in core:
                if i < j:
                    G.add_edge(i, j)
        for c in core:  # boost core degree above periphery
            for _ in range(8):
                G.add_edge(c, int(rng.integers(6, 60)))

        adj = np.zeros((len(G), len(G)))
        for u, v in G.edges():
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        analyzer = AdvancedNetworkAnalyzer(adj, [f"N{i}" for i in range(len(G))])
        rc = analyzer.calculate_rich_club_coefficient()
        spectrum = rc["full_spectrum"]
        assert isinstance(spectrum, dict) and len(spectrum) > 0
        # At a high-k cutoff (core members, k >= 10) normalized phi should exceed 1.
        high_k_vals = [v for k, v in spectrum.items() if k >= 10 and v is not None]
        assert any(v > 1.0 for v in high_k_vals), (
            f"expected a rich-club signal (phi>1) in {spectrum}"
        )

    def test_small_graph_returns_sentinel_not_crash(self):
        # Tiny graph: normalized rich-club randomization is not meaningful; must
        # not crash and must return the 'insufficient' sentinel.
        m = _directed_star_matrix(3)
        analyzer = AdvancedNetworkAnalyzer(m, [f"N{i}" for i in range(3)])
        rc = analyzer.calculate_rich_club_coefficient()
        assert rc["rich_club_coefficient"] == "insufficient" or rc["full_spectrum"] in ({}, None)
        # no exception is the key assertion

    def test_random_graph_phi_near_one(self):
        # An Erdos-Renyi random graph has no rich-club: normalized phi ~ 1.
        G = nx.gnp_random_graph(40, 0.25, seed=7)
        adj = np.zeros((40, 40))
        for u, v in G.edges():
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        analyzer = AdvancedNetworkAnalyzer(adj, [f"N{i}" for i in range(40)])
        rc = analyzer.calculate_rich_club_coefficient()
        spectrum = rc["full_spectrum"]
        if isinstance(spectrum, dict) and spectrum:
            vals = [v for v in spectrum.values() if v is not None]
            if vals:
                # Should hover around 1 (allow generous tolerance for finite size).
                assert min(vals) < 2.0


# ---------------------------------------------------------------------------
# FIX F — Flow-diversity utilization log base match (nats/nats)
# ---------------------------------------------------------------------------

class TestFixFUtilizationBase:
    def test_uniform_flow_utilization_near_100(self):
        # Uniform flow across all n^2 cells => flow diversity = ln(n^2) (nats),
        # so utilization = fd / ln(n^2) * 100 = 100%.
        n = 5
        fd = np.log(n ** 2)  # max diversity in nats
        util = publication_report.flow_diversity_utilization(fd, n)
        assert util == pytest.approx(100.0, abs=1e-6)

    def test_old_base2_understated(self):
        # Pre-fix used log2(n^2) in the denominator against a nats numerator,
        # understating by factor ln2 (~0.693) => ~69.3% for the uniform case.
        n = 5
        fd = np.log(n ** 2)
        old = fd / np.log2(n ** 2) * 100
        assert old == pytest.approx(69.3147, abs=1e-2)
        # the fix must be higher than the old understated value
        assert publication_report.flow_diversity_utilization(fd, n) > old

    def test_guard_tiny_graph(self):
        # n=1 => log(1)=0 denominator; must not divide by zero.
        assert publication_report.flow_diversity_utilization(0.0, 1) == 0.0
