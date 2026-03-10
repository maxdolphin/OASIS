#!/usr/bin/env python
"""
Performance Benchmark Script for Precomputation Strategy

This script benchmarks the vectorized vs original metric calculations
to verify the expected performance improvements for large networks.

Expected Performance Improvements:
| Network Size | Original Time | After Optimization |
|--------------|--------------|-------------------|
| 50 nodes     | ~2s          | ~0.2s (10x)       |
| 200 nodes    | ~30s         | ~3s (10x)         |
| 500 nodes    | ~5min        | ~30s (10x)        |
| 1000 nodes   | Timeout      | ~2min             |
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ulanowicz_calculator import UlanowiczCalculator
from vectorized_metrics import get_all_vectorized_metrics, VectorizedMetricsCalculator


def generate_test_matrix(n_nodes: int, density: float = 0.3, seed: int = 42) -> np.ndarray:
    """Generate a random flow matrix for benchmarking."""
    np.random.seed(seed)
    matrix = np.random.random((n_nodes, n_nodes)) * 100

    # Apply density mask
    mask = np.random.random((n_nodes, n_nodes)) < density
    matrix = matrix * mask

    # Zero diagonal
    np.fill_diagonal(matrix, 0)

    return matrix


def benchmark_individual_metrics(matrix: np.ndarray,
                                  node_names: list,
                                  iterations: int = 3) -> dict:
    """Benchmark individual metric calculations."""

    results = {}

    # Original implementation
    calc_orig = UlanowiczCalculator(matrix, node_names, use_vectorized=False)
    start = time.time()
    for _ in range(iterations):
        calc_orig._vectorized_cache.clear()
        _ = calc_orig.calculate_ami()
        _ = calc_orig.calculate_ascendency()
        _ = calc_orig.calculate_development_capacity()
        _ = calc_orig.calculate_robustness()
        _ = calc_orig.calculate_flow_diversity()
        _ = calc_orig.calculate_effective_flows()
        _ = calc_orig.calculate_effective_nodes()
        _ = calc_orig.calculate_effective_connectivity()
    results['original_time'] = (time.time() - start) / iterations

    # Vectorized implementation
    calc_vec = UlanowiczCalculator(matrix, node_names, use_vectorized=True)
    start = time.time()
    for _ in range(iterations):
        calc_vec._vectorized_cache.clear()
        _ = calc_vec.calculate_ami()
        _ = calc_vec.calculate_ascendency()
        _ = calc_vec.calculate_development_capacity()
        _ = calc_vec.calculate_robustness()
        _ = calc_vec.calculate_flow_diversity()
        _ = calc_vec.calculate_effective_flows()
        _ = calc_vec.calculate_effective_nodes()
        _ = calc_vec.calculate_effective_connectivity()
    results['vectorized_time'] = (time.time() - start) / iterations

    # Batch vectorized computation
    start = time.time()
    for _ in range(iterations):
        _ = get_all_vectorized_metrics(matrix)
    results['batch_vectorized_time'] = (time.time() - start) / iterations

    # Calculate speedups
    results['speedup_vectorized'] = results['original_time'] / results['vectorized_time']
    results['speedup_batch'] = results['original_time'] / results['batch_vectorized_time']

    return results


def run_benchmark_suite():
    """Run the complete benchmark suite."""

    print("=" * 70)
    print("PRECOMPUTATION STRATEGY PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()

    test_sizes = [50, 100, 200, 500]

    # Check if we should include larger sizes
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--large', action='store_true', help='Include larger network sizes')
    args, _ = parser.parse_known_args()

    if args.large:
        test_sizes.extend([1000, 2000])

    print("Network Size | Original (s) | Vectorized (s) | Batch (s) | Speedup")
    print("-" * 70)

    for n_nodes in test_sizes:
        matrix = generate_test_matrix(n_nodes, density=0.3)
        node_names = [f"Node_{i}" for i in range(n_nodes)]

        # Fewer iterations for large networks
        iterations = 1 if n_nodes >= 500 else 3

        results = benchmark_individual_metrics(matrix, node_names, iterations)

        print(f"{n_nodes:12d} | {results['original_time']:12.3f} | "
              f"{results['vectorized_time']:14.3f} | "
              f"{results['batch_vectorized_time']:9.3f} | "
              f"{results['speedup_vectorized']:.1f}x / {results['speedup_batch']:.1f}x")

    print("-" * 70)
    print()
    print("Speedup columns show: Vectorized speedup / Batch speedup")
    print()

    # Cache performance test
    print("=" * 70)
    print("CACHE PERFORMANCE TEST (200 nodes)")
    print("=" * 70)

    matrix = generate_test_matrix(200)
    node_names = [f"Node_{i}" for i in range(200)]

    # First computation (cold cache)
    calc = UlanowiczCalculator(matrix, node_names, use_vectorized=True)
    start = time.time()
    _ = calc.calculate_ami()
    _ = calc.calculate_ascendency()
    _ = calc.calculate_robustness()
    cold_time = time.time() - start
    print(f"Cold cache: {cold_time:.4f}s")

    # Second computation (warm cache)
    start = time.time()
    _ = calc.calculate_ami()
    _ = calc.calculate_ascendency()
    _ = calc.calculate_robustness()
    warm_time = time.time() - start
    print(f"Warm cache: {warm_time:.6f}s")
    print(f"Cache speedup: {cold_time / warm_time:.0f}x")

    print()
    print("Benchmark complete!")


if __name__ == '__main__':
    run_benchmark_suite()
