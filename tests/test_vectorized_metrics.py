"""
Tests for Vectorized Metrics Module

This test suite verifies that vectorized implementations produce
numerically equivalent results to the original loop-based implementations.

The tests also include performance benchmarks to measure speedup.
"""

import sys
import os
import time
import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ulanowicz_calculator import UlanowiczCalculator
from vectorized_metrics import (
    precompute_sums,
    vectorized_flow_diversity,
    vectorized_ami,
    vectorized_ascendency,
    vectorized_development_capacity,
    vectorized_reserve,
    vectorized_relative_ascendency,
    vectorized_robustness,
    vectorized_effective_flows,
    vectorized_effective_nodes,
    vectorized_effective_connectivity,
    vectorized_number_of_roles,
    get_all_vectorized_metrics,
    VectorizedMetricsCalculator,
)


# Test tolerance for floating point comparison
# Using 1e-9 to account for accumulated floating point differences
# in large matrices while still ensuring numerical equivalence
TOLERANCE = 1e-9


def generate_random_flow_matrix(n_nodes: int, density: float = 0.3, seed: int = 42) -> np.ndarray:
    """Generate a random flow matrix for testing."""
    np.random.seed(seed)
    matrix = np.random.random((n_nodes, n_nodes)) * 100

    # Apply density mask
    mask = np.random.random((n_nodes, n_nodes)) < density
    matrix = matrix * mask

    # Zero out diagonal (no self-loops typically)
    np.fill_diagonal(matrix, 0)

    return matrix


def generate_small_test_matrix() -> np.ndarray:
    """Generate a small deterministic test matrix for verification."""
    return np.array([
        [0.0, 10.0, 5.0, 2.0],
        [8.0, 0.0, 3.0, 1.0],
        [4.0, 6.0, 0.0, 7.0],
        [1.0, 2.0, 4.0, 0.0]
    ])


class TestPrecomputeSums:
    """Test precompute_sums function."""

    def test_basic_sums(self):
        """Test that sums are computed correctly."""
        matrix = generate_small_test_matrix()
        row_sums, col_sums, tst = precompute_sums(matrix)

        # Verify row sums (output throughput)
        expected_row_sums = np.sum(matrix, axis=1)
        np.testing.assert_array_almost_equal(row_sums, expected_row_sums)

        # Verify column sums (input throughput)
        expected_col_sums = np.sum(matrix, axis=0)
        np.testing.assert_array_almost_equal(col_sums, expected_col_sums)

        # Verify TST
        expected_tst = np.sum(matrix)
        assert abs(tst - expected_tst) < TOLERANCE

    def test_empty_matrix(self):
        """Test behavior with zero matrix."""
        matrix = np.zeros((5, 5))
        row_sums, col_sums, tst = precompute_sums(matrix)

        assert tst == 0
        assert np.all(row_sums == 0)
        assert np.all(col_sums == 0)


class TestVectorizedFlowDiversity:
    """Test vectorized flow diversity calculation."""

    def test_matches_original(self):
        """Verify vectorized matches original implementation."""
        matrix = generate_small_test_matrix()

        # Original implementation
        calc = UlanowiczCalculator(matrix, use_vectorized=False)
        original_result = calc.calculate_flow_diversity()

        # Vectorized implementation
        vectorized_result = vectorized_flow_diversity(matrix)

        assert abs(original_result - vectorized_result) < TOLERANCE, \
            f"Mismatch: original={original_result}, vectorized={vectorized_result}"

    def test_with_precomputed_tst(self):
        """Test using precomputed TST value."""
        matrix = generate_small_test_matrix()
        _, _, tst = precompute_sums(matrix)

        result1 = vectorized_flow_diversity(matrix)
        result2 = vectorized_flow_diversity(matrix, tst=tst)

        assert abs(result1 - result2) < TOLERANCE

    def test_empty_matrix_returns_zero(self):
        """Test that empty matrix returns 0."""
        matrix = np.zeros((5, 5))
        result = vectorized_flow_diversity(matrix)
        assert result == 0.0


class TestVectorizedAMI:
    """Test vectorized AMI calculation."""

    def test_matches_original(self):
        """Verify vectorized matches original implementation."""
        matrix = generate_small_test_matrix()

        calc = UlanowiczCalculator(matrix, use_vectorized=False)
        original_result = calc.calculate_ami()

        row_sums, col_sums, tst = precompute_sums(matrix)
        vectorized_result = vectorized_ami(matrix, row_sums, col_sums, tst)

        assert abs(original_result - vectorized_result) < TOLERANCE, \
            f"Mismatch: original={original_result}, vectorized={vectorized_result}"

    def test_larger_network(self):
        """Test on a larger random network."""
        matrix = generate_random_flow_matrix(20, density=0.4)

        calc = UlanowiczCalculator(matrix, use_vectorized=False)
        original_result = calc.calculate_ami()

        row_sums, col_sums, tst = precompute_sums(matrix)
        vectorized_result = vectorized_ami(matrix, row_sums, col_sums, tst)

        assert abs(original_result - vectorized_result) < TOLERANCE


class TestVectorizedAscendency:
    """Test vectorized ascendency calculation."""

    def test_matches_original(self):
        """Verify vectorized matches original implementation."""
        matrix = generate_small_test_matrix()

        calc = UlanowiczCalculator(matrix, use_vectorized=False)
        original_result = calc.calculate_ascendency()

        row_sums, col_sums, tst = precompute_sums(matrix)
        vectorized_result = vectorized_ascendency(matrix, row_sums, col_sums, tst)

        assert abs(original_result - vectorized_result) < TOLERANCE, \
            f"Mismatch: original={original_result}, vectorized={vectorized_result}"

    def test_multiple_sizes(self):
        """Test on various network sizes."""
        for n_nodes in [5, 10, 25, 50]:
            matrix = generate_random_flow_matrix(n_nodes, seed=n_nodes)

            calc = UlanowiczCalculator(matrix, use_vectorized=False)
            original = calc.calculate_ascendency()

            row_sums, col_sums, tst = precompute_sums(matrix)
            vectorized = vectorized_ascendency(matrix, row_sums, col_sums, tst)

            assert abs(original - vectorized) < TOLERANCE, \
                f"Mismatch for n={n_nodes}: original={original}, vectorized={vectorized}"


class TestVectorizedDevelopmentCapacity:
    """Test vectorized development capacity calculation."""

    def test_matches_original(self):
        """Verify vectorized matches original implementation."""
        matrix = generate_small_test_matrix()

        calc = UlanowiczCalculator(matrix, use_vectorized=False)
        original_result = calc.calculate_development_capacity()

        _, _, tst = precompute_sums(matrix)
        vectorized_result = vectorized_development_capacity(matrix, tst)

        assert abs(original_result - vectorized_result) < TOLERANCE, \
            f"Mismatch: original={original_result}, vectorized={vectorized_result}"


class TestVectorizedReserve:
    """Test vectorized reserve calculation."""

    def test_matches_original(self):
        """Verify vectorized matches original implementation."""
        matrix = generate_small_test_matrix()

        calc = UlanowiczCalculator(matrix, use_vectorized=False)
        original_result = calc.calculate_reserve()

        row_sums, col_sums, tst = precompute_sums(matrix)
        vectorized_result = vectorized_reserve(matrix, row_sums, col_sums, tst)

        assert abs(original_result - vectorized_result) < TOLERANCE, \
            f"Mismatch: original={original_result}, vectorized={vectorized_result}"

    def test_fundamental_relationship(self):
        """Test that C = A + Φ holds for vectorized metrics."""
        matrix = generate_random_flow_matrix(15)

        row_sums, col_sums, tst = precompute_sums(matrix)

        A = vectorized_ascendency(matrix, row_sums, col_sums, tst)
        C = vectorized_development_capacity(matrix, tst)
        Phi = vectorized_reserve(matrix, row_sums, col_sums, tst)

        # C should equal A + Φ
        assert abs(C - (A + Phi)) < TOLERANCE, \
            f"Fundamental relationship violated: C={C}, A+Φ={A + Phi}"


class TestVectorizedEffectiveMetrics:
    """Test vectorized effective flows, nodes, and connectivity."""

    def test_effective_flows_matches_original(self):
        """Verify effective flows matches original."""
        matrix = generate_small_test_matrix()

        calc = UlanowiczCalculator(matrix, use_vectorized=False)
        original = calc.calculate_effective_flows()

        _, _, tst = precompute_sums(matrix)
        vectorized = vectorized_effective_flows(matrix, tst)

        assert abs(original - vectorized) < TOLERANCE, \
            f"Effective flows mismatch: original={original}, vectorized={vectorized}"

    def test_effective_nodes_matches_original(self):
        """Verify effective nodes matches original."""
        matrix = generate_small_test_matrix()

        calc = UlanowiczCalculator(matrix, use_vectorized=False)
        original = calc.calculate_effective_nodes()

        row_sums, col_sums, tst = precompute_sums(matrix)
        vectorized = vectorized_effective_nodes(matrix, row_sums, col_sums, tst)

        assert abs(original - vectorized) < TOLERANCE, \
            f"Effective nodes mismatch: original={original}, vectorized={vectorized}"

    def test_effective_connectivity_matches_original(self):
        """Verify effective connectivity matches original."""
        matrix = generate_small_test_matrix()

        calc = UlanowiczCalculator(matrix, use_vectorized=False)
        original = calc.calculate_effective_connectivity()

        row_sums, col_sums, tst = precompute_sums(matrix)
        vectorized = vectorized_effective_connectivity(matrix, row_sums, col_sums, tst)

        assert abs(original - vectorized) < TOLERANCE, \
            f"Effective connectivity mismatch: original={original}, vectorized={vectorized}"


class TestVectorizedRobustness:
    """Test vectorized robustness calculation."""

    def test_matches_original(self):
        """Verify vectorized matches original implementation."""
        matrix = generate_small_test_matrix()

        calc = UlanowiczCalculator(matrix, use_vectorized=False)
        original_result = calc.calculate_robustness()

        row_sums, col_sums, tst = precompute_sums(matrix)
        vectorized_result = vectorized_robustness(matrix, row_sums, col_sums, tst)

        assert abs(original_result - vectorized_result) < TOLERANCE, \
            f"Mismatch: original={original_result}, vectorized={vectorized_result}"


class TestGetAllVectorizedMetrics:
    """Test the batch computation function."""

    def test_all_metrics_match(self):
        """Verify all metrics match original implementations."""
        matrix = generate_random_flow_matrix(15, seed=123)

        # Get all vectorized metrics
        vectorized = get_all_vectorized_metrics(matrix)

        # Compare with original
        calc = UlanowiczCalculator(matrix, use_vectorized=False)

        assert abs(vectorized['total_system_throughput'] - calc.calculate_tst()) < TOLERANCE
        assert abs(vectorized['flow_diversity'] - calc.calculate_flow_diversity()) < TOLERANCE
        assert abs(vectorized['average_mutual_information'] - calc.calculate_ami()) < TOLERANCE
        assert abs(vectorized['ascendency'] - calc.calculate_ascendency()) < TOLERANCE
        assert abs(vectorized['development_capacity'] - calc.calculate_development_capacity()) < TOLERANCE
        assert abs(vectorized['reserve'] - calc.calculate_reserve()) < TOLERANCE
        assert abs(vectorized['relative_ascendency'] - calc.calculate_relative_ascendency()) < TOLERANCE
        assert abs(vectorized['robustness'] - calc.calculate_robustness()) < TOLERANCE


class TestVectorizedMetricsCalculator:
    """Test the VectorizedMetricsCalculator class."""

    def test_caching_works(self):
        """Verify that caching returns same values."""
        matrix = generate_small_test_matrix()
        calc = VectorizedMetricsCalculator(matrix)

        # First call
        ami1 = calc.calculate_ami()

        # Second call should use cache
        ami2 = calc.calculate_ami()

        assert ami1 == ami2

    def test_clear_cache(self):
        """Test cache clearing."""
        matrix = generate_small_test_matrix()
        calc = VectorizedMetricsCalculator(matrix)

        # Populate cache
        calc.calculate_ami()
        assert len(calc._cache) > 0

        # Clear cache
        calc.clear_cache()
        assert len(calc._cache) == 0

    def test_matches_original(self):
        """Verify calculator matches original implementations."""
        matrix = generate_small_test_matrix()

        vec_calc = VectorizedMetricsCalculator(matrix)
        orig_calc = UlanowiczCalculator(matrix, use_vectorized=False)

        assert abs(vec_calc.calculate_ami() - orig_calc.calculate_ami()) < TOLERANCE
        assert abs(vec_calc.calculate_ascendency() - orig_calc.calculate_ascendency()) < TOLERANCE
        assert abs(vec_calc.calculate_robustness() - orig_calc.calculate_robustness()) < TOLERANCE


class TestUlanowiczCalculatorIntegration:
    """Test UlanowiczCalculator with vectorized mode."""

    def test_vectorized_mode_toggle(self):
        """Test switching between vectorized and non-vectorized modes."""
        matrix = generate_small_test_matrix()

        # Non-vectorized
        calc_orig = UlanowiczCalculator(matrix, use_vectorized=False)
        ami_orig = calc_orig.calculate_ami()

        # Vectorized
        calc_vec = UlanowiczCalculator(matrix, use_vectorized=True)
        ami_vec = calc_vec.calculate_ami()

        assert abs(ami_orig - ami_vec) < TOLERANCE

    def test_auto_vectorization_threshold(self):
        """Test automatic vectorization based on network size."""
        # Small network - should not auto-vectorize
        small_matrix = generate_random_flow_matrix(30)
        calc_small = UlanowiczCalculator(small_matrix)
        assert not calc_small.use_vectorized

        # Large network - should auto-vectorize
        large_matrix = generate_random_flow_matrix(60)
        calc_large = UlanowiczCalculator(large_matrix)
        assert calc_large.use_vectorized

    def test_get_precomputed_sums(self):
        """Test the precomputed sums accessor."""
        matrix = generate_small_test_matrix()
        calc = UlanowiczCalculator(matrix)

        row_sums, col_sums, tst = calc.get_precomputed_sums()

        assert len(row_sums) == 4
        assert len(col_sums) == 4
        assert tst == np.sum(matrix)

    def test_cache_info(self):
        """Test cache info method."""
        matrix = generate_random_flow_matrix(60)
        calc = UlanowiczCalculator(matrix, use_vectorized=True)

        # Populate some cache
        calc.calculate_ami()
        calc.calculate_ascendency()

        info = calc.get_cache_info()
        assert info['use_vectorized'] is True
        assert info['n_nodes'] == 60
        assert 'ami' in info['cached_metrics']
        assert 'ascendency' in info['cached_metrics']


class TestPerformanceBenchmarks:
    """Performance benchmarks comparing original vs vectorized."""

    @pytest.mark.parametrize("n_nodes", [50, 100, 200])
    def test_speedup(self, n_nodes):
        """Measure speedup of vectorized implementation."""
        matrix = generate_random_flow_matrix(n_nodes, density=0.3, seed=n_nodes)
        node_names = [f"Node_{i}" for i in range(n_nodes)]

        # Time original implementation
        calc_orig = UlanowiczCalculator(matrix, node_names, use_vectorized=False)

        start = time.time()
        for _ in range(3):
            calc_orig._vectorized_cache.clear()  # Clear cache between runs
            _ = calc_orig.calculate_ami()
            _ = calc_orig.calculate_ascendency()
            _ = calc_orig.calculate_development_capacity()
            _ = calc_orig.calculate_robustness()
        original_time = time.time() - start

        # Time vectorized implementation
        calc_vec = UlanowiczCalculator(matrix, node_names, use_vectorized=True)

        start = time.time()
        for _ in range(3):
            calc_vec._vectorized_cache.clear()  # Clear cache between runs
            _ = calc_vec.calculate_ami()
            _ = calc_vec.calculate_ascendency()
            _ = calc_vec.calculate_development_capacity()
            _ = calc_vec.calculate_robustness()
        vectorized_time = time.time() - start

        speedup = original_time / vectorized_time if vectorized_time > 0 else float('inf')

        print(f"\n[n={n_nodes}] Original: {original_time:.3f}s, Vectorized: {vectorized_time:.3f}s, Speedup: {speedup:.1f}x")

        # Vectorized should generally be faster for larger networks
        if n_nodes >= 100:
            assert speedup > 1.0, f"Vectorized should be faster for n={n_nodes}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_node(self):
        """Test with single node matrix."""
        matrix = np.array([[0.0]])
        calc = UlanowiczCalculator(matrix, use_vectorized=True)

        assert calc.calculate_tst() == 0
        assert calc.calculate_flow_diversity() == 0
        assert calc.calculate_ascendency() == 0

    def test_sparse_matrix(self):
        """Test with very sparse matrix."""
        matrix = np.zeros((10, 10))
        matrix[0, 1] = 100.0  # Single flow

        calc_orig = UlanowiczCalculator(matrix, use_vectorized=False)
        calc_vec = UlanowiczCalculator(matrix, use_vectorized=True)

        assert abs(calc_orig.calculate_ami() - calc_vec.calculate_ami()) < TOLERANCE
        assert abs(calc_orig.calculate_ascendency() - calc_vec.calculate_ascendency()) < TOLERANCE

    def test_dense_matrix(self):
        """Test with fully connected matrix."""
        np.random.seed(42)
        matrix = np.random.random((8, 8)) * 50
        np.fill_diagonal(matrix, 0)

        calc_orig = UlanowiczCalculator(matrix, use_vectorized=False)
        calc_vec = UlanowiczCalculator(matrix, use_vectorized=True)

        assert abs(calc_orig.calculate_ami() - calc_vec.calculate_ami()) < TOLERANCE
        assert abs(calc_orig.calculate_robustness() - calc_vec.calculate_robustness()) < TOLERANCE

    def test_uniform_flows(self):
        """Test with uniform flow values."""
        matrix = np.ones((5, 5)) * 10
        np.fill_diagonal(matrix, 0)

        calc_orig = UlanowiczCalculator(matrix, use_vectorized=False)
        calc_vec = UlanowiczCalculator(matrix, use_vectorized=True)

        assert abs(calc_orig.calculate_flow_diversity() - calc_vec.calculate_flow_diversity()) < TOLERANCE

    def test_large_values(self):
        """Test with large flow values."""
        matrix = generate_random_flow_matrix(10) * 1e6

        calc_orig = UlanowiczCalculator(matrix, use_vectorized=False)
        calc_vec = UlanowiczCalculator(matrix, use_vectorized=True)

        # Results should be relatively close (allow slightly larger tolerance for large values)
        rel_tol = 1e-8
        assert abs(calc_orig.calculate_ami() - calc_vec.calculate_ami()) / (abs(calc_orig.calculate_ami()) + 1e-10) < rel_tol


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
