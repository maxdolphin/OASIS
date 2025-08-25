#!/usr/bin/env python3
"""
Validation with exogenous flows included.
The Ulanowicz paper includes external inputs and outputs in the total system.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ulanowicz_calculator import UlanowiczCalculator


def create_extended_flow_matrix(data):
    """
    Create extended flow matrix including exogenous inputs and outputs.
    Following Ulanowicz convention where the full system includes:
    - Internal flows (node to node)
    - Exogenous inputs (from environment to nodes)
    - Exogenous outputs (from nodes to environment)
    - Dissipations (energy lost as respiration/heat)
    """
    internal_flows = np.array(data['flows'])
    n = len(internal_flows)
    
    # Create extended matrix (n+2 x n+2) with environment as nodes 0 and n+1
    # Row 0: Environment (source)
    # Rows 1 to n: Internal nodes
    # Row n+1: Environment (sink)
    extended = np.zeros((n+2, n+2))
    
    # Copy internal flows (offset by 1)
    extended[1:n+1, 1:n+1] = internal_flows
    
    # Add exogenous inputs (from environment node 0 to internal nodes)
    if 'exogenous_inputs' in data['metadata']:
        inputs = data['metadata']['exogenous_inputs']
        extended[0, 1] = inputs.get('to_plants', 0)
        extended[0, 2] = inputs.get('to_detritus', 0)
        extended[0, 3] = inputs.get('to_bacteria', 0)
        extended[0, 4] = inputs.get('to_detritivores', 0)
        extended[0, 5] = inputs.get('to_carnivores', 0)
    
    # Add exogenous outputs (from internal nodes to environment node n+1)
    if 'exogenous_outputs' in data['metadata']:
        outputs = data['metadata']['exogenous_outputs']
        extended[1, n+1] = outputs.get('from_plants', 0)
        extended[2, n+1] = outputs.get('from_detritus', 0)
        extended[3, n+1] = outputs.get('from_bacteria', 0)
        extended[4, n+1] = outputs.get('from_detritivores', 0)
        extended[5, n+1] = outputs.get('from_carnivores', 0)
    
    # Add dissipations (respiration) - also goes to environment
    if 'dissipations' in data['metadata']:
        dissipations = data['metadata']['dissipations']
        extended[1, n+1] += dissipations.get('plants', 0)
        extended[2, n+1] += dissipations.get('detritus', 0)
        extended[3, n+1] += dissipations.get('bacteria', 0)
        extended[4, n+1] += dissipations.get('detritivores', 0)
        extended[5, n+1] += dissipations.get('carnivores', 0)
    
    # Extended node names
    extended_names = ['Environment_In'] + data['nodes'] + ['Environment_Out']
    
    return extended, extended_names


def validate_with_full_system(filename, expected_alpha):
    """Validate using the full system including exogenous flows."""
    print(f"\nValidating: {filename}")
    print("="*50)
    
    # Load data
    filepath = Path(__file__).parent.parent / 'data' / 'ecosystem_samples' / filename
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Create extended flow matrix
    extended_flows, extended_names = create_extended_flow_matrix(data)
    
    print(f"Internal flow matrix shape: {len(data['flows'])}x{len(data['flows'])}")
    print(f"Extended flow matrix shape: {extended_flows.shape}")
    print(f"Total flows in extended matrix: {np.sum(extended_flows):.1f}")
    
    # Calculate metrics
    calculator = UlanowiczCalculator(extended_flows, node_names=extended_names)
    metrics = calculator.get_sustainability_metrics()
    
    print(f"\nMetrics:")
    print(f"  Total System Throughput: {metrics['total_system_throughput']:.1f}")
    print(f"  Development Capacity: {metrics['development_capacity']:.1f}")
    print(f"  Ascendency: {metrics['ascendency']:.1f}")
    print(f"  Reserve: {metrics['reserve']:.1f}")
    print(f"  Relative Ascendency (α): {metrics['relative_ascendency']:.3f}")
    
    print(f"\nComparison:")
    print(f"  Expected α: {expected_alpha:.3f}")
    print(f"  Calculated α: {metrics['relative_ascendency']:.3f}")
    print(f"  Difference: {abs(metrics['relative_ascendency'] - expected_alpha):.4f}")
    
    if abs(metrics['relative_ascendency'] - expected_alpha) < 0.01:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")
    
    return metrics


def main():
    print("\n" + "#"*60)
    print("# VALIDATION WITH EXOGENOUS FLOWS")
    print("#"*60)
    
    # Test original network
    metrics_orig = validate_with_full_system('cone_spring_original.json', 0.418)
    
    # Test eutrophicated network  
    metrics_eutr = validate_with_full_system('cone_spring_eutrophicated.json', 0.529)
    
    print("\n" + "#"*60)
    print("# SUMMARY")
    print("#"*60)
    print(f"\nOriginal α: {metrics_orig['relative_ascendency']:.3f} (expected: 0.418)")
    print(f"Eutrophicated α: {metrics_eutr['relative_ascendency']:.3f} (expected: 0.529)")


if __name__ == "__main__":
    main()