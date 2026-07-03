"""
Track-1 ENA-method formula corrections — test-driven.

These tests encode the paper-expected behavior for the four standard-backed
corrections confirmed by the expert panel (see
docs/business-revision/evidence/expert-ena-methods.md and
validation-CD-roles-cycling.md):

  FIX 1 — Effective connectivity must be F/N (>= 1), not the inverted N/F.
          Zorach & Ulanowicz (2003) p.72: C = F/N; identity R = F/C².
  FIX 2 — Finn Cycling Index via column-normalized Leontief inverse
          (Finn 1976; Ulanowicz 2004 §5). Pure ring -> ~1, chain -> 0.
          The old short-cycle proxy returns ~0 on a pure ring.
  FIX 3 — Trophic level must be flow-weighted (Levine 1980; Ulanowicz 2004 §4),
          producing fractional effective levels, not unweighted shortest-path hops.
  FIX 4 — "Lindeman efficiency" relabeled to respiratory_retention_ratio.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ulanowicz_calculator import UlanowiczCalculator
from ecosystem_flow_calculator import EcosystemFlowCalculator
from vectorized_metrics import vectorized_effective_connectivity


def _random_flow_matrix(n, seed):
    rng = np.random.default_rng(seed)
    # positive off-diagonal flows, zero diagonal (typical directed flow network)
    m = rng.uniform(1.0, 10.0, size=(n, n))
    np.fill_diagonal(m, 0.0)
    return m


# ---------------------------------------------------------------------------
# FIX 1 — Effective connectivity = F/N
# ---------------------------------------------------------------------------

class TestFix1EffectiveConnectivity:

    SEEDS = [(4, 1), (5, 3), (6, 9), (5, 42), (3, 7)]

    @pytest.mark.parametrize("n,seed", SEEDS)
    def test_connectivity_floor_geq_one(self, n, seed):
        """Connectivity is flows-per-node and must be >= 1.0 for a connected
        network (Ulanowicz 2004 p.334: lower window bound = 1.0)."""
        m = _random_flow_matrix(n, seed)
        calc = UlanowiczCalculator(m, use_vectorized=False)
        c = calc.calculate_effective_connectivity()
        assert c >= 1.0 - 1e-9, f"connectivity {c} < 1.0 (n={n}, seed={seed})"

    @pytest.mark.parametrize("n,seed", SEEDS)
    def test_connectivity_equals_F_over_N(self, n, seed):
        """C == F/N to 1e-9 (Zorach & Ulanowicz 2003 p.72)."""
        m = _random_flow_matrix(n, seed)
        calc = UlanowiczCalculator(m, use_vectorized=False)
        c = calc.calculate_effective_connectivity()
        f = calc.calculate_effective_flows()
        nn = calc.calculate_effective_nodes()
        assert c == pytest.approx(f / nn, abs=1e-9)

    @pytest.mark.parametrize("n,seed", SEEDS)
    def test_identity_R_equals_F_over_C_squared(self, n, seed):
        """R = exp(AMI) must equal F/C² to 1e-9 (Z-U 2003 identity block)."""
        m = _random_flow_matrix(n, seed)
        calc = UlanowiczCalculator(m, use_vectorized=False)
        c = calc.calculate_effective_connectivity()
        f = calc.calculate_effective_flows()
        r = np.exp(calc.calculate_ami())
        assert r == pytest.approx(f / c**2, abs=1e-9)

    @pytest.mark.parametrize("n,seed", SEEDS)
    def test_loop_matches_vectorized(self, n, seed):
        """Loop and vectorized effective connectivity agree to 1e-9."""
        m = _random_flow_matrix(n, seed)
        loop = UlanowiczCalculator(m, use_vectorized=False).calculate_effective_connectivity()
        calc_v = UlanowiczCalculator(m, use_vectorized=True)
        vec = vectorized_effective_connectivity(
            calc_v.flow_matrix,
            calc_v.output_throughput,
            calc_v.input_throughput,
            calc_v._tst,
        )
        assert loop == pytest.approx(vec, abs=1e-9)


# ---------------------------------------------------------------------------
# FIX 2 — Finn Cycling Index (canonical Leontief) + short-cycle proxy relabel
# ---------------------------------------------------------------------------

class TestFix2FinnCyclingIndex:

    @staticmethod
    def _ring4():
        # 1->2->3->4->1, unit flows
        m = np.zeros((4, 4))
        m[0, 1] = m[1, 2] = m[2, 3] = m[3, 0] = 1.0
        return m

    @staticmethod
    def _chain4():
        # 1->2->3->4, acyclic
        m = np.zeros((4, 4))
        m[0, 1] = m[1, 2] = m[2, 3] = 1.0
        return m

    def test_full_finn_ring_is_fully_cycled(self):
        """A pure directed ring recycles ~100% of its medium: FCI ~= 1."""
        calc = EcosystemFlowCalculator(self._ring4())
        fci = calc.calculate_finn_cycling_index()
        assert fci == pytest.approx(1.0, abs=0.05), f"ring FCI={fci}"

    def test_full_finn_chain_is_acyclic(self):
        """An acyclic chain has no cycling: FCI ~= 0."""
        calc = EcosystemFlowCalculator(self._chain4())
        fci = calc.calculate_finn_cycling_index()
        assert fci == pytest.approx(0.0, abs=0.05), f"chain FCI={fci}"

    def test_short_cycle_proxy_returns_zero_on_ring(self):
        """The short-cycle proxy (self-loops + 2-cycles only) misses the
        length-4 cycle and returns ~0 — documenting why it's only a proxy."""
        calc = UlanowiczCalculator(self._ring4(), use_vectorized=False)
        proxy = calc.calculate_short_cycle_proxy()
        assert proxy == pytest.approx(0.0, abs=1e-9), f"proxy={proxy}"

    def test_short_cycle_proxy_backcompat_alias(self):
        """The old method name must still exist and equal the proxy."""
        calc = UlanowiczCalculator(self._ring4(), use_vectorized=False)
        assert calc.calculate_finn_cycling_index() == pytest.approx(
            calc.calculate_short_cycle_proxy(), abs=1e-12
        )

    def test_full_finn_available_on_ulanowicz_calculator(self):
        """The corrected full Finn (internal-only) is available and returns
        ~1 on the ring, unlike the short-cycle proxy."""
        calc = UlanowiczCalculator(self._ring4(), use_vectorized=False)
        full = calc.calculate_finn_cycling_index_full()
        assert full == pytest.approx(1.0, abs=0.05), f"full FCI={full}"


# ---------------------------------------------------------------------------
# FIX 3 — Flow-weighted effective trophic level (Levine)
# ---------------------------------------------------------------------------

class TestFix3TrophicDepth:

    @staticmethod
    def _chain4():
        m = np.zeros((4, 4))
        m[0, 1] = m[1, 2] = m[2, 3] = 1.0
        return m

    def test_effective_levels_increase_along_chain(self):
        """On a linear chain, effective trophic levels increase 1,2,3,4."""
        calc = UlanowiczCalculator(self._chain4(), use_vectorized=False)
        levels = calc.calculate_effective_trophic_levels()
        assert levels[0] < levels[1] < levels[2] < levels[3]

    def test_effective_levels_can_be_fractional(self):
        """Levine effective levels are flow-weighted and can be fractional —
        the Ulanowicz 2004 worked example yields 2.5 for a mixed feeder."""
        # Compartment 4 fed 60% from L1, 30% from L2(=via 2), 10% from L3(=via 3)
        # Build a network reproducing the 2.5 example (Ulanowicz 2004 Fig.4).
        m = np.zeros((4, 4))
        m[0, 1] = 1.0          # 1 -> 2   (level 2)
        m[1, 2] = 1.0          # 2 -> 3   (level 3)
        # node 4 (index 3) is fed 0.6 from 1, 0.3 from 2, 0.1 from 3
        m[0, 3] = 0.6
        m[1, 3] = 0.3
        m[2, 3] = 0.1
        calc = UlanowiczCalculator(m, use_vectorized=False)
        levels = calc.calculate_effective_trophic_levels()
        assert levels[3] == pytest.approx(2.5, abs=1e-6), f"levels={levels}"
        # fractional (not an integer hop count)
        assert abs(levels[3] - round(levels[3])) > 1e-6

    def test_trophic_depth_differs_from_shortest_path(self):
        """The flow-weighted depth must differ from the unweighted shortest
        path on a flow-weighted example."""
        m = np.zeros((4, 4))
        m[0, 1] = 1.0
        m[1, 2] = 1.0
        m[0, 3] = 0.6
        m[1, 3] = 0.3
        m[2, 3] = 0.1
        calc = UlanowiczCalculator(m, use_vectorized=False)
        depth = calc.calculate_trophic_depth()
        # flow-weighted depth (max effective level ~3) != unweighted mean hops
        assert depth == pytest.approx(3.0, abs=1e-6), f"depth={depth}"


# ---------------------------------------------------------------------------
# FIX 4 — "Lindeman efficiency" relabel -> respiratory_retention_ratio
# ---------------------------------------------------------------------------

class TestFix4RespiratoryRetention:

    @staticmethod
    def _calc():
        m = np.zeros((3, 3))
        m[0, 1] = 5.0
        m[1, 2] = 3.0
        imports = np.array([10.0, 0.0, 0.0])
        exports = np.array([0.0, 0.0, 2.0])
        respiration = np.array([1.0, 1.0, 1.0])
        return EcosystemFlowCalculator(m, imports=imports, exports=exports,
                                       respiration=respiration)

    def test_respiratory_retention_ratio_value(self):
        """Renamed metric equals the documented formula
        1 - respiration/(TST + imports)."""
        calc = self._calc()
        tst = calc.calculate_tst()
        expected = 1 - (np.sum(calc.respiration) / (tst + np.sum(calc.imports)))
        expected = max(0.0, min(1.0, expected))
        got = calc.calculate_respiratory_retention_ratio()
        assert got == pytest.approx(expected, abs=1e-12)

    def test_metric_key_present_in_output(self):
        """The renamed key is present in ecosystem metrics."""
        calc = self._calc()
        metrics = calc.get_ecosystem_metrics()
        assert 'respiratory_retention_ratio' in metrics

    def test_lindeman_backcompat_alias(self):
        """Old key/method preserved as a back-compat alias so consumers
        don't break."""
        calc = self._calc()
        assert calc.calculate_lindeman_efficiency() == pytest.approx(
            calc.calculate_respiratory_retention_ratio(), abs=1e-12
        )
        metrics = calc.get_ecosystem_metrics()
        assert 'lindeman_efficiency' in metrics
