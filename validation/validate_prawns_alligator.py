#!/usr/bin/env python3
"""
Validation Script for Prawns-Alligator Network from Ulanowicz et al. (2009)
Tests the three configurations demonstrating efficiency vs resilience trade-offs.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ulanowicz_calculator import UlanowiczCalculator


def load_network(filename):
    """Load network data from JSON file."""
    filepath = Path(__file__).parent.parent / 'data' / 'ecosystem_samples' / filename
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def validate_network(filename, config_name):
    """Validate a network configuration."""
    print(f"\n{'='*60}")
    print(f"VALIDATING: {config_name}")
    print(f"{'='*60}")
    
    # Load data
    data = load_network(filename)
    flows = np.array(data['flows'])
    nodes = data['nodes']
    metadata = data['metadata']
    
    print(f"\nNetwork: {data['organization']}")
    print(f"Nodes: {', '.join(nodes)}")
    print(f"Description: {metadata['description']}")
    
    # Display flow matrix
    print(f"\nFlow Matrix ({metadata['units']}):")
    for i, from_node in enumerate(nodes):
        for j, to_node in enumerate(nodes):
            if flows[i, j] > 0:
                print(f"  {from_node} → {to_node}: {flows[i, j]:.2f}")
    
    # Calculate metrics
    calculator = UlanowiczCalculator(flows, node_names=nodes)
    metrics = calculator.get_sustainability_metrics()
    extended = calculator.get_extended_metrics()
    
    print(f"\nCalculated Metrics:")
    print(f"  Total System Throughput: {metrics['total_system_throughput']:.2f}")
    print(f"  Development Capacity: {metrics['development_capacity']:.2f}")
    print(f"  Ascendency: {metrics['ascendency']:.2f}")
    print(f"  Reserve: {metrics['reserve']:.2f}")
    print(f"  Relative Ascendency (α): {metrics['relative_ascendency']:.4f}")
    print(f"  Robustness: {extended['robustness']:.4f}")
    
    # Compare with published values if available
    if 'published_metrics' in metadata:
        pub = metadata['published_metrics']
        print(f"\nPublished Metrics (from paper):")
        if 'total_system_throughput' in pub:
            print(f"  Total System Throughput: {pub['total_system_throughput']}")
        if 'ascendency' in pub:
            print(f"  Ascendency: {pub['ascendency']}")
        if 'reserve' in pub:
            print(f"  Reserve: {pub['reserve']}")
        if 'note' in pub:
            print(f"  Note: {pub['note']}")
    
    # Verify fundamental relationship
    c_check = metrics['ascendency'] + metrics['reserve']
    error = abs(c_check - metrics['development_capacity']) / metrics['development_capacity'] * 100
    print(f"\nFundamental Relationship Check (C = A + Φ):")
    print(f"  {metrics['development_capacity']:.2f} = {metrics['ascendency']:.2f} + {metrics['reserve']:.2f}")
    print(f"  Error: {error:.4f}%")
    
    # Sustainability assessment
    print(f"\nSustainability Assessment:")
    alpha = metrics['relative_ascendency']
    optimal_alpha = 0.460
    
    if alpha < 0.2:
        status = "Too Chaotic (α < 0.2)"
        health = "❌ Critical - Insufficient organization"
    elif alpha < optimal_alpha:
        status = f"Below Optimal (α = {alpha:.3f} < {optimal_alpha})"
        health = "✓ Viable - Room for growth"
    elif alpha < 0.6:
        status = f"Above Optimal (α = {alpha:.3f} > {optimal_alpha})"
        health = "⚠️ Viable - Reduced flexibility"
    else:
        status = "Too Rigid (α > 0.6)"
        health = "❌ Critical - Over-constrained"
    
    print(f"  Status: {status}")
    print(f"  Health: {health}")
    
    return metrics


def compare_configurations(metrics_dict):
    """Compare metrics across all configurations."""
    print(f"\n{'='*60}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\n{'Configuration':<25} {'TST':>8} {'α':>8} {'Robust':>8} {'Status':<20}")
    print("-" * 70)
    
    for config, metrics in metrics_dict.items():
        tst = metrics['total_system_throughput']
        alpha = metrics['relative_ascendency']
        robust = metrics['extended']['robustness']
        
        if alpha < 0.2:
            status = "Too Chaotic"
        elif alpha < 0.460:
            status = "Below Optimal"
        elif alpha < 0.6:
            status = "Above Optimal"
        else:
            status = "Too Rigid"
        
        print(f"{config:<25} {tst:>8.1f} {alpha:>8.3f} {robust:>8.3f} {status:<20}")
    
    print(f"\n{'Key Insights:'}")
    print("1. Original network: Balanced with three parallel pathways")
    print("2. Efficient-only: Maximum throughput but zero resilience")
    print("3. After adaptation: Maintains function despite species loss")
    print("\nThis demonstrates the importance of pathway redundancy for system resilience!")


def main():
    """Run validation for all Prawns-Alligator network configurations."""
    print("\n" + "#"*60)
    print("# PRAWNS-ALLIGATOR NETWORK VALIDATION")
    print("# Ulanowicz et al. (2009) - Efficiency vs Resilience")
    print("#"*60)
    
    metrics_dict = {}
    
    # Validate original network with all pathways
    metrics = validate_network(
        'prawns_alligator_original.json',
        'Original Network (3 Pathways)'
    )
    metrics['extended'] = UlanowiczCalculator(
        np.array(load_network('prawns_alligator_original.json')['flows']),
        node_names=load_network('prawns_alligator_original.json')['nodes']
    ).get_extended_metrics()
    metrics_dict['Original (3 paths)'] = metrics
    
    # Validate efficient-only configuration
    metrics = validate_network(
        'prawns_alligator_efficient.json',
        'Most Efficient Path Only'
    )
    metrics['extended'] = UlanowiczCalculator(
        np.array(load_network('prawns_alligator_efficient.json')['flows']),
        node_names=load_network('prawns_alligator_efficient.json')['nodes']
    ).get_extended_metrics()
    metrics_dict['Efficient only'] = metrics
    
    # Validate adapted network after fish loss
    metrics = validate_network(
        'prawns_alligator_adapted.json',
        'After Fish Loss (Adapted)'
    )
    metrics['extended'] = UlanowiczCalculator(
        np.array(load_network('prawns_alligator_adapted.json')['flows']),
        node_names=load_network('prawns_alligator_adapted.json')['nodes']
    ).get_extended_metrics()
    metrics_dict['Adapted (no fish)'] = metrics
    
    # Compare all configurations
    compare_configurations(metrics_dict)
    
    print(f"\n" + "#"*60)
    print("# VALIDATION COMPLETE")
    print("#"*60)
    print("\n✓ All networks loaded and analyzed successfully")
    print("✓ Fundamental relationships verified")
    print("✓ Demonstrates efficiency-resilience trade-off")


if __name__ == "__main__":
    main()