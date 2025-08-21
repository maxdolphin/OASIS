#!/usr/bin/env python3
"""
Validation test for Cone Spring ecosystem data.
Compares our calculations with published values from Ulanowicz & Norden (1990).
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ulanowicz_calculator import UlanowiczCalculator

def load_cone_spring_data():
    """Load the Cone Spring flow matrix and published metrics."""
    
    # Load flow matrix
    with open('extracted_data/cone_spring_flow_matrix.json', 'r') as f:
        flow_data = json.load(f)
    
    # Load published metrics
    with open('extracted_data/cone_spring_published_metrics.json', 'r') as f:
        published_data = json.load(f)
    
    return flow_data, published_data

def run_validation():
    """Run validation test comparing our calculations with published values."""
    
    print("=" * 80)
    print("CONE SPRING ECOSYSTEM VALIDATION")
    print("Comparing with Ulanowicz & Norden (1990)")
    print("=" * 80)
    
    # Load data
    flow_data, published_data = load_cone_spring_data()
    
    flow_matrix = np.array(flow_data['flow_matrix'])
    node_names = flow_data['nodes']
    published_metrics = published_data['metrics']
    
    print(f"\nSource: {flow_data['source']}")
    print(f"Units: {flow_data['units']}")
    print(f"Nodes: {', '.join(node_names)}")
    
    # Create calculator
    calculator = UlanowiczCalculator(flow_matrix, node_names)
    
    # Calculate metrics
    metrics = calculator.get_sustainability_metrics()
    
    # Compare with published values
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS (using log₂)")
    print("=" * 80)
    
    comparisons = [
        ('Total System Throughput', 'total_system_throughput', 'total_system_throughput', 0),
        ('Development Capacity', 'development_capacity', 'development_capacity', 1),
        ('Ascendency', 'ascendency', 'ascendency', 1),
        ('Overhead/Reserve', 'overhead', 'reserve', 1),
        ('Ascendency Ratio (A/C)', 'ascendency_ratio', 'ascendency_ratio', 3),
        ('AMI', 'average_mutual_information', None, 3)
    ]
    
    print(f"\n{'Metric':<30} {'Published':<15} {'Calculated':<15} {'Error %':<10}")
    print("-" * 70)
    
    total_error = 0
    count = 0
    
    for name, pub_key, calc_key, decimals in comparisons:
        if pub_key in published_metrics:
            published_val = published_metrics[pub_key]['value']
            
            if calc_key:
                if calc_key == 'average_mutual_information':
                    # AMI needs to be calculated manually as A/TST
                    calculated_val = metrics['ascendency'] / metrics['total_system_throughput']
                else:
                    calculated_val = metrics.get(calc_key, 0)
            else:
                # Skip if no calculation available
                continue
            
            # Calculate error
            if published_val != 0:
                error = abs((calculated_val - published_val) / published_val) * 100
            else:
                error = 0 if calculated_val == 0 else 100
            
            total_error += error
            count += 1
            
            # Format output
            if decimals == 0:
                print(f"{name:<30} {published_val:<15.0f} {calculated_val:<15.0f} {error:<10.1f}%")
            elif decimals == 3:
                print(f"{name:<30} {published_val:<15.3f} {calculated_val:<15.3f} {error:<10.1f}%")
            else:
                print(f"{name:<30} {published_val:<15.1f} {calculated_val:<15.1f} {error:<10.1f}%")
    
    avg_error = total_error / count if count > 0 else 0
    
    print("-" * 70)
    print(f"{'AVERAGE ERROR':<30} {'':<15} {'':<15} {avg_error:<10.1f}%")
    
    # Additional calculations
    print("\n" + "=" * 80)
    print("ADDITIONAL CALCULATIONS")
    print("=" * 80)
    
    # Calculate robustness using corrected formula
    robustness = calculator.calculate_robustness()
    print(f"\nRobustness (using -α·log₂(α)): {robustness:.4f}")
    
    # Check window of viability
    lower, upper = calculator.calculate_window_of_viability()
    print(f"\nWindow of Viability:")
    print(f"  Lower bound (20% of C): {lower:.1f}")
    print(f"  Upper bound (60% of C): {upper:.1f}")
    print(f"  Current Ascendency: {metrics['ascendency']:.1f}")
    print(f"  Is Viable: {'YES' if metrics['is_viable'] else 'NO'}")
    
    # Verify fundamental relationship
    verification = calculator.verify_fundamental_relationship()
    print(f"\nFundamental Relationship (C = A + Φ):")
    print(f"  C (calculated): {verification['development_capacity']:.1f}")
    print(f"  A + Φ: {verification['calculated_capacity']:.1f}")
    print(f"  Error: {verification['relative_error']*100:.3f}%")
    print(f"  Valid: {'YES' if verification['is_valid'] else 'NO'}")
    
    print("\n" + "=" * 80)
    
    if avg_error < 5:
        print("✅ VALIDATION SUCCESSFUL - Average error < 5%")
    elif avg_error < 10:
        print("⚠️  VALIDATION WARNING - Average error 5-10%")
    else:
        print("❌ VALIDATION FAILED - Average error > 10%")
    
    print("=" * 80)
    
    return avg_error

if __name__ == "__main__":
    # Change to validation directory
    import os
    os.chdir(Path(__file__).parent)
    
    try:
        error = run_validation()
        sys.exit(0 if error < 10 else 1)
    except Exception as e:
        print(f"\n❌ Error running validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)