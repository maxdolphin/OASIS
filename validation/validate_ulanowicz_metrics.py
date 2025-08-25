#!/usr/bin/env python3
"""
Validation Script for Ulanowicz et al. (2009) Metrics
Validates our implementation against published values from:
"Quantifying sustainability: Resilience, efficiency and the return of information theory"
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ulanowicz_calculator import UlanowiczCalculator


def load_ecosystem_data(filename):
    """Load ecosystem network data from JSON file."""
    filepath = Path(__file__).parent.parent / 'data' / 'ecosystem_samples' / filename
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def validate_cone_spring_original():
    """Validate metrics for original Cone Spring ecosystem."""
    print("\n" + "="*60)
    print("VALIDATING: Cone Spring Ecosystem (Original)")
    print("="*60)
    
    # Load data
    data = load_ecosystem_data('cone_spring_original.json')
    flows = np.array(data['flows'])
    
    # Expected values from paper
    expected = {
        'relative_ascendency': 0.418,
        'optimal_alpha': 0.460,
        'status': 'below optimal'
    }
    
    print(f"\nExpected metrics from Ulanowicz et al. (2009):")
    print(f"  Relative Ascendency (α): {expected['relative_ascendency']:.3f}")
    print(f"  Optimal α: {expected['optimal_alpha']:.3f}")
    print(f"  Status: {expected['status']}")
    
    # Calculate using our implementation
    calculator = UlanowiczCalculator(flows, node_names=data['nodes'])
    metrics = calculator.get_sustainability_metrics()
    
    print(f"\nCalculated metrics from our implementation:")
    print(f"  Total System Throughput (T): {metrics['total_system_throughput']:.1f}")
    print(f"  Development Capacity (C): {metrics['development_capacity']:.1f}")
    print(f"  Ascendency (A): {metrics['ascendency']:.1f}")
    print(f"  Reserve/Overhead (Φ): {metrics['reserve']:.1f}")
    print(f"  Relative Ascendency (α): {metrics['relative_ascendency']:.3f}")
    
    # Get extended metrics for robustness
    extended = calculator.get_extended_metrics()
    print(f"  Robustness (R): {extended['robustness']:.3f}")
    
    # Validate fundamental relationship: C = A + Φ
    c_check = metrics['ascendency'] + metrics['reserve']
    c_error = abs(c_check - metrics['development_capacity']) / metrics['development_capacity'] * 100
    print(f"\nFundamental relationship validation:")
    print(f"  C = A + Φ")
    print(f"  {metrics['development_capacity']:.1f} = {metrics['ascendency']:.1f} + {metrics['reserve']:.1f}")
    print(f"  Check: {c_check:.1f}")
    print(f"  Error: {c_error:.3f}%")
    
    # Compare α values
    alpha_diff = abs(metrics['relative_ascendency'] - expected['relative_ascendency'])
    print(f"\nRelative Ascendency (α) comparison:")
    print(f"  Expected: {expected['relative_ascendency']:.3f}")
    print(f"  Calculated: {metrics['relative_ascendency']:.3f}")
    print(f"  Difference: {alpha_diff:.4f}")
    
    if alpha_diff < 0.01:
        print(f"  ✓ PASS: Within tolerance (< 0.01)")
    else:
        print(f"  ✗ FAIL: Outside tolerance")
    
    return metrics['relative_ascendency']


def validate_cone_spring_eutrophicated():
    """Validate metrics for eutrophicated Cone Spring ecosystem."""
    print("\n" + "="*60)
    print("VALIDATING: Cone Spring Ecosystem (Eutrophicated)")
    print("="*60)
    
    # Load data
    data = load_ecosystem_data('cone_spring_eutrophicated.json')
    flows = np.array(data['flows'])
    
    # Expected values from paper
    expected = {
        'relative_ascendency': 0.529,
        'optimal_alpha': 0.460,
        'status': 'above optimal',
        'modification': 'Added 8000 kcal m⁻² y⁻¹ to pathway 1→2→3'
    }
    
    print(f"\nExpected metrics from Ulanowicz et al. (2009):")
    print(f"  Relative Ascendency (α): {expected['relative_ascendency']:.3f}")
    print(f"  Optimal α: {expected['optimal_alpha']:.3f}")
    print(f"  Status: {expected['status']}")
    print(f"  Modification: {expected['modification']}")
    
    # Calculate using our implementation
    calculator = UlanowiczCalculator(flows, node_names=data['nodes'])
    metrics = calculator.get_sustainability_metrics()
    
    print(f"\nCalculated metrics from our implementation:")
    print(f"  Total System Throughput (T): {metrics['total_system_throughput']:.1f}")
    print(f"  Development Capacity (C): {metrics['development_capacity']:.1f}")
    print(f"  Ascendency (A): {metrics['ascendency']:.1f}")
    print(f"  Reserve/Overhead (Φ): {metrics['reserve']:.1f}")
    print(f"  Relative Ascendency (α): {metrics['relative_ascendency']:.3f}")
    
    # Get extended metrics for robustness
    extended = calculator.get_extended_metrics()
    print(f"  Robustness (R): {extended['robustness']:.3f}")
    
    # Validate fundamental relationship: C = A + Φ
    c_check = metrics['ascendency'] + metrics['reserve']
    c_error = abs(c_check - metrics['development_capacity']) / metrics['development_capacity'] * 100
    print(f"\nFundamental relationship validation:")
    print(f"  C = A + Φ")
    print(f"  {metrics['development_capacity']:.1f} = {metrics['ascendency']:.1f} + {metrics['reserve']:.1f}")
    print(f"  Check: {c_check:.1f}")
    print(f"  Error: {c_error:.3f}%")
    
    # Compare α values
    alpha_diff = abs(metrics['relative_ascendency'] - expected['relative_ascendency'])
    print(f"\nRelative Ascendency (α) comparison:")
    print(f"  Expected: {expected['relative_ascendency']:.3f}")
    print(f"  Calculated: {metrics['relative_ascendency']:.3f}")
    print(f"  Difference: {alpha_diff:.4f}")
    
    if alpha_diff < 0.01:
        print(f"  ✓ PASS: Within tolerance (< 0.01)")
    else:
        print(f"  ✗ FAIL: Outside tolerance")
    
    return metrics['relative_ascendency']


def compare_networks(alpha_original, alpha_eutrophicated):
    """Compare the two networks and validate the effect of eutrophication."""
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    
    optimal_alpha = 0.460
    
    print(f"\nOptimal α (from Window of Vitality): {optimal_alpha:.3f}")
    print(f"\nOriginal ecosystem:")
    print(f"  α = {alpha_original:.3f}")
    print(f"  Distance from optimal: {abs(alpha_original - optimal_alpha):.3f}")
    print(f"  Status: {'Below' if alpha_original < optimal_alpha else 'Above'} optimal")
    print(f"  Interpretation: System can still grow and develop")
    
    print(f"\nEutrophicated ecosystem:")
    print(f"  α = {alpha_eutrophicated:.3f}")
    print(f"  Distance from optimal: {abs(alpha_eutrophicated - optimal_alpha):.3f}")
    print(f"  Status: {'Below' if alpha_eutrophicated < optimal_alpha else 'Above'} optimal")
    print(f"  Interpretation: System has excess ascendency, reduced reserve")
    
    print(f"\nChange due to eutrophication:")
    print(f"  Δα = {alpha_eutrophicated - alpha_original:.3f}")
    print(f"  Relative change: {(alpha_eutrophicated - alpha_original)/alpha_original * 100:.1f}%")
    
    # Check if both are within Window of Vitality
    print(f"\nWindow of Vitality (0.20 < α < 0.60):")
    print(f"  Original: {'✓ Within' if 0.20 < alpha_original < 0.60 else '✗ Outside'} window")
    print(f"  Eutrophicated: {'✓ Within' if 0.20 < alpha_eutrophicated < 0.60 else '✗ Outside'} window")
    
    # Marginal contribution interpretation
    print(f"\nMarginal Contribution Effects:")
    print(f"  Original (α < α_opt):")
    print(f"    - Main pathway (1→2→3) contributions > 1")
    print(f"    - Parallel flow contributions < 1")
    print(f"    - System benefits from strengthening main pathway")
    
    print(f"\n  Eutrophicated (α > α_opt):")
    print(f"    - Main pathway (1→2→3) contributions < 1")
    print(f"    - Parallel flow contributions > 1")
    print(f"    - System needs more diversity/redundancy")


def main():
    """Run all validation tests."""
    print("\n" + "#"*60)
    print("# ULANOWICZ ET AL. (2009) METRICS VALIDATION")
    print("# Testing our implementation against published values")
    print("#"*60)
    
    try:
        # Validate both networks
        alpha_original = validate_cone_spring_original()
        alpha_eutrophicated = validate_cone_spring_eutrophicated()
        
        # Compare results
        compare_networks(alpha_original, alpha_eutrophicated)
        
        print("\n" + "#"*60)
        print("# VALIDATION COMPLETE")
        print("#"*60)
        
    except Exception as e:
        print(f"\n✗ ERROR during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())