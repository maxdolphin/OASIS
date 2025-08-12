#!/usr/bin/env python3
"""
Example usage of the Adaptive Organization Analysis system.

This script demonstrates how to use the system to analyze organizational
networks using Ulanowicz's ecosystem sustainability theory.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ulanowicz_calculator import UlanowiczCalculator
from visualizer import SustainabilityVisualizer


def example_simple_organization():
    """Example with a simple 4-node organizational network."""
    print("="*60)
    print("EXAMPLE 1: Simple Organization (4 departments)")
    print("="*60)
    
    # Define departments
    departments = ["Management", "Operations", "Sales", "Support"]
    
    # Define flow matrix (information/resource flows between departments)
    flow_matrix = np.array([
        #  Mgmt  Ops  Sales Support
        [  0.0, 3.0,  2.0,  1.0 ],  # Management
        [  1.5, 0.0,  1.5,  2.0 ],  # Operations  
        [  2.0, 1.0,  0.0,  1.5 ],  # Sales
        [  1.0, 2.5,  2.0,  0.0 ]   # Support
    ])
    
    # Calculate metrics
    calc = UlanowiczCalculator(flow_matrix, departments)
    metrics = calc.get_sustainability_metrics()
    
    # Print results
    print(f"Sustainability Status: {calc.assess_sustainability()}")
    print(f"Ascendency: {metrics['ascendency']:.3f}")
    print(f"Development Capacity: {metrics['development_capacity']:.3f}")
    print(f"Ascendency Ratio: {metrics['ascendency_ratio']:.3f}")
    print(f"Is Viable: {'YES' if metrics['is_viable'] else 'NO'}")
    
    return calc


def example_complex_organization():
    """Example with a more complex 7-node organizational network."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Complex Organization (7 departments)")
    print("="*60)
    
    # Define departments
    departments = [
        "Executive", "HR", "Finance", "Operations", 
        "Marketing", "R&D", "Customer_Service"
    ]
    
    # More complex flow matrix
    flow_matrix = np.array([
        # Exec  HR   Fin  Ops  Mkt  R&D  CS
        [0.0,  2.0, 3.0, 4.0, 2.5, 1.5, 1.0],  # Executive
        [1.5,  0.0, 1.0, 2.0, 1.0, 0.5, 2.5],  # HR
        [2.0,  1.5, 0.0, 3.0, 2.0, 1.0, 0.5],  # Finance
        [1.0,  1.0, 2.5, 0.0, 1.5, 2.0, 3.0],  # Operations
        [1.0,  0.5, 1.5, 2.0, 0.0, 1.0, 2.5],  # Marketing
        [0.5,  0.3, 1.0, 1.5, 1.5, 0.0, 0.5],  # R&D
        [0.5,  1.5, 0.3, 2.5, 2.0, 0.2, 0.0],  # Customer Service
    ])
    
    # Calculate metrics
    calc = UlanowiczCalculator(flow_matrix, departments)
    metrics = calc.get_sustainability_metrics()
    
    # Print results
    print(f"Sustainability Status: {calc.assess_sustainability()}")
    print(f"Ascendency: {metrics['ascendency']:.3f}")
    print(f"Development Capacity: {metrics['development_capacity']:.3f}")
    print(f"Ascendency Ratio: {metrics['ascendency_ratio']:.3f}")
    print(f"Is Viable: {'YES' if metrics['is_viable'] else 'NO'}")
    
    return calc


def example_highly_organized():
    """Example of an over-organized (rigid) system."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Highly Organized (Rigid) System")
    print("="*60)
    
    departments = ["CEO", "VP1", "VP2", "Manager1", "Manager2"]
    
    # Highly hierarchical flow pattern (mostly top-down)
    flow_matrix = np.array([
        [0.0, 5.0, 5.0, 0.0, 0.0],  # CEO
        [1.0, 0.0, 0.5, 4.0, 0.0],  # VP1
        [1.0, 0.5, 0.0, 0.0, 4.0],  # VP2
        [0.1, 2.0, 0.0, 0.0, 0.1],  # Manager1
        [0.1, 0.0, 2.0, 0.1, 0.0],  # Manager2
    ])
    
    calc = UlanowiczCalculator(flow_matrix, departments)
    metrics = calc.get_sustainability_metrics()
    
    print(f"Sustainability Status: {calc.assess_sustainability()}")
    print(f"Ascendency: {metrics['ascendency']:.3f}")
    print(f"Development Capacity: {metrics['development_capacity']:.3f}")
    print(f"Ascendency Ratio: {metrics['ascendency_ratio']:.3f}")
    print(f"Is Viable: {'YES' if metrics['is_viable'] else 'NO'}")
    
    return calc


def example_chaotic_system():
    """Example of an under-organized (chaotic) system."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Chaotic (Under-organized) System")  
    print("="*60)
    
    departments = ["Team1", "Team2", "Team3", "Team4", "Team5"]
    
    # Random, unstructured flow pattern
    np.random.seed(42)  # For reproducibility
    flow_matrix = np.random.uniform(0, 2, (5, 5))
    np.fill_diagonal(flow_matrix, 0)  # No self-loops
    
    calc = UlanowiczCalculator(flow_matrix, departments)
    metrics = calc.get_sustainability_metrics()
    
    print(f"Sustainability Status: {calc.assess_sustainability()}")
    print(f"Ascendency: {metrics['ascendency']:.3f}")
    print(f"Development Capacity: {metrics['development_capacity']:.3f}")
    print(f"Ascendency Ratio: {metrics['ascendency_ratio']:.3f}")
    print(f"Is Viable: {'YES' if metrics['is_viable'] else 'NO'}")
    
    return calc


def generate_visualizations():
    """Generate visualization examples."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Use the complex organization example
    departments = [
        "Executive", "HR", "Finance", "Operations", 
        "Marketing", "R&D", "Customer_Service"
    ]
    
    flow_matrix = np.array([
        [0.0,  2.0, 3.0, 4.0, 2.5, 1.5, 1.0],
        [1.5,  0.0, 1.0, 2.0, 1.0, 0.5, 2.5],
        [2.0,  1.5, 0.0, 3.0, 2.0, 1.0, 0.5],
        [1.0,  1.0, 2.5, 0.0, 1.5, 2.0, 3.0],
        [1.0,  0.5, 1.5, 2.0, 0.0, 1.0, 2.5],
        [0.5,  0.3, 1.0, 1.5, 1.5, 0.0, 0.5],
        [0.5,  1.5, 0.3, 2.5, 2.0, 0.2, 0.0],
    ])
    
    calc = UlanowiczCalculator(flow_matrix, departments)
    viz = SustainabilityVisualizer(calc)
    
    # Generate interactive HTML visualization
    try:
        output_path = Path(__file__).parent / "sustainability_analysis.html"
        viz.save_visualization(str(output_path), 'html')
        print(f"Interactive visualization saved to: {output_path}")
    except Exception as e:
        print(f"Could not save HTML visualization: {e}")
    
    # Generate static PNG visualization  
    try:
        output_path = Path(__file__).parent / "sustainability_analysis.png"
        viz.save_visualization(str(output_path), 'png')
        print(f"Static visualization saved to: {output_path}")
    except Exception as e:
        print(f"Could not save PNG visualization: {e}")


def main():
    """Run all examples."""
    print("ADAPTIVE ORGANIZATION ANALYSIS - EXAMPLES")
    print("Based on Ulanowicz's Ecosystem Sustainability Theory")
    
    # Run examples
    calc1 = example_simple_organization()
    calc2 = example_complex_organization() 
    calc3 = example_highly_organized()
    calc4 = example_chaotic_system()
    
    # Generate visualizations
    generate_visualizations()
    
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    examples = [
        ("Simple Org", calc1),
        ("Complex Org", calc2), 
        ("Rigid Org", calc3),
        ("Chaotic Org", calc4)
    ]
    
    print(f"{'Organization':<15} {'Status':<25} {'A/C Ratio':<10} {'Viable':<8}")
    print("-" * 60)
    
    for name, calc in examples:
        metrics = calc.get_sustainability_metrics()
        status = calc.assess_sustainability().split(' - ')[0]
        print(f"{name:<15} {status:<25} {metrics['ascendency_ratio']:.3f}      {'YES' if metrics['is_viable'] else 'NO'}")


if __name__ == "__main__":
    main()