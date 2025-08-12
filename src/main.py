"""
Main application interface for Adaptive Organization Analysis.

This module provides the main entry point and user interface for analyzing
organizational networks using Ulanowicz's ecosystem sustainability theory.
"""

import numpy as np
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Union

try:
    from .ulanowicz_calculator import UlanowiczCalculator
    from .visualizer import SustainabilityVisualizer
except ImportError:
    # If running as script, add current directory to path
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ulanowicz_calculator import UlanowiczCalculator
    from visualizer import SustainabilityVisualizer


def load_network_from_json(filepath: str) -> tuple:
    """
    Load network data from JSON file.
    
    Expected format:
    {
        "nodes": ["Node1", "Node2", ...],
        "flows": [[0, 1.5, 0], [2.0, 0, 1.0], ...] // flow_matrix[i][j] = flow from i to j
    }
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Tuple of (flow_matrix, node_names)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'flows' not in data:
        raise ValueError("JSON file must contain 'flows' key with flow matrix")
    
    flow_matrix = np.array(data['flows'])
    node_names = data.get('nodes', [f"Node_{i}" for i in range(len(flow_matrix))])
    
    return flow_matrix, node_names


def load_network_from_csv(filepath: str, has_header: bool = True) -> tuple:
    """
    Load network data from CSV file.
    
    CSV should contain the flow matrix with optional header row for node names.
    
    Args:
        filepath: Path to CSV file
        has_header: Whether CSV has header row with node names
        
    Returns:
        Tuple of (flow_matrix, node_names)
    """
    import pandas as pd
    
    df = pd.read_csv(filepath, header=0 if has_header else None)
    
    if has_header:
        node_names = df.columns.tolist()
        flow_matrix = df.values
    else:
        flow_matrix = df.values
        node_names = [f"Node_{i}" for i in range(len(flow_matrix))]
    
    return flow_matrix, node_names


def create_example_network() -> tuple:
    """
    Create an example organizational network for demonstration.
    
    This represents a hypothetical organization with different departments
    and their information/resource flows.
    
    Returns:
        Tuple of (flow_matrix, node_names)
    """
    node_names = [
        "Executive", "HR", "Finance", "Operations", 
        "Marketing", "R&D", "Customer_Service"
    ]
    
    # Flow matrix representing information/resource flows between departments
    # Higher values indicate stronger connections/dependencies
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
    
    return flow_matrix, node_names


def print_analysis_report(calculator: UlanowiczCalculator, detailed: bool = True):
    """
    Print a comprehensive analysis report to console.
    
    Args:
        calculator: UlanowiczCalculator instance
        detailed: Whether to include detailed explanations
    """
    metrics = calculator.get_sustainability_metrics()
    extended_metrics = calculator.get_extended_metrics()
    assessments = calculator.assess_regenerative_health()
    sustainability_status = calculator.assess_sustainability()
    
    print("\n" + "="*70)
    print("REGENERATIVE ORGANIZATIONAL SUSTAINABILITY ANALYSIS")
    print("Based on Extended Ulanowicz-Fath Ecosystem Theory")
    print("="*70)
    
    print(f"\nSUSTAINABILITY STATUS: {sustainability_status}")
    
    print(f"\nCORE ULANOWICZ METRICS:")
    print(f"  Total System Throughput (TST): {extended_metrics['total_system_throughput']:.3f}")
    print(f"  Average Mutual Information (AMI): {extended_metrics['average_mutual_information']:.3f}")
    print(f"  Ascendency (A): {extended_metrics['ascendency']:.3f}")
    print(f"  Development Capacity (C): {extended_metrics['development_capacity']:.3f}")
    print(f"  Overhead (Φ): {extended_metrics['overhead']:.3f}")
    
    print(f"\nEXTENDED REGENERATIVE METRICS:")
    print(f"  Flow Diversity (H): {extended_metrics['flow_diversity']:.3f}")
    print(f"  Structural Information (SI): {extended_metrics['structural_information']:.3f}")
    print(f"  Robustness (R): {extended_metrics['robustness']:.3f}")
    print(f"  Network Efficiency: {extended_metrics['network_efficiency']:.3f}")
    print(f"  Regenerative Capacity: {extended_metrics['regenerative_capacity']:.3f}")
    
    print(f"\nSYSTEM RATIOS:")
    print(f"  Ascendency Ratio (A/C): {extended_metrics['ascendency_ratio']:.3f}")
    print(f"  Overhead Ratio (Φ/C): {extended_metrics['overhead_ratio']:.3f}")
    print(f"  Redundancy: {extended_metrics['redundancy']:.3f}")
    
    print(f"\nWINDOW OF VIABILITY:")
    print(f"  Lower Bound: {extended_metrics['viability_lower_bound']:.3f}")
    print(f"  Upper Bound: {extended_metrics['viability_upper_bound']:.3f}")
    print(f"  Current Ascendency: {extended_metrics['ascendency']:.3f}")
    print(f"  Is Viable: {'YES' if extended_metrics['is_viable'] else 'NO'}")
    
    print(f"\nREGENERATIVE HEALTH ASSESSMENT:")
    for category, assessment in assessments.items():
        print(f"  {category.title()}: {assessment}")
    
    if detailed:
        print(f"\nDETAILED INTERPRETATION:")
        efficiency = extended_metrics['network_efficiency']
        robustness = extended_metrics['robustness']
        
        if efficiency < 0.2:
            print("  - System is underutilized with low efficiency")
            print("  - Consider streamlining processes and improving coordination")
        elif efficiency > 0.6:
            print("  - System may be over-optimized and brittle")
            print("  - Consider adding redundancy and flexibility")
        else:
            print("  - System efficiency is within sustainable range")
        
        if robustness < 0.1:
            print("  - System lacks robustness to handle disturbances")
            print("  - Focus on building resilience and adaptive capacity")
        elif robustness > 0.25:
            print("  - System demonstrates strong robustness")
            print("  - Well-balanced between efficiency and resilience")
        
        regen_capacity = extended_metrics['regenerative_capacity']
        if regen_capacity < 0.1:
            print("  - Limited regenerative capacity for self-renewal")
        elif regen_capacity > 0.2:
            print("  - Strong regenerative potential for adaptation")
        
        print(f"\nNETWORK PROPERTIES:")
        print(f"  Nodes: {calculator.n_nodes}")
        print(f"  Total Connections: {np.count_nonzero(calculator.flow_matrix)}")
        print(f"  Network Density: {np.count_nonzero(calculator.flow_matrix) / (calculator.n_nodes ** 2):.3f}")
        print(f"  Effective Link Density: {extended_metrics['effective_link_density']:.3f}")
        print(f"  Trophic Depth: {extended_metrics['trophic_depth']:.3f}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze organizational sustainability using Ulanowicz's ecosystem theory"
    )
    parser.add_argument(
        '--input', '-i',
        help='Input file path (JSON or CSV format)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['json', 'csv'],
        help='Input file format'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output visualization file path'
    )
    parser.add_argument(
        '--viz-format',
        choices=['html', 'png', 'pdf', 'robustness'],
        default='html',
        help='Visualization output format (default: html, robustness for robustness curve)'
    )
    parser.add_argument(
        '--example',
        action='store_true',
        help='Use example organizational network'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed analysis report with extended metrics'
    )
    parser.add_argument(
        '--regenerative',
        action='store_true',
        help='Focus on regenerative economics analysis'
    )
    
    args = parser.parse_args()
    
    try:
        # Load network data
        if args.example:
            print("Using example organizational network...")
            flow_matrix, node_names = create_example_network()
        elif args.input:
            print(f"Loading network from {args.input}...")
            if args.format == 'json' or args.input.endswith('.json'):
                flow_matrix, node_names = load_network_from_json(args.input)
            elif args.format == 'csv' or args.input.endswith('.csv'):
                flow_matrix, node_names = load_network_from_csv(args.input)
            else:
                raise ValueError("Could not determine file format. Use --format or file extension.")
        else:
            print("No input specified. Use --example for demo or --input for your data.")
            print("Run with --help for usage information.")
            return
        
        # Calculate sustainability metrics
        print("Calculating sustainability metrics...")
        calculator = UlanowiczCalculator(flow_matrix, node_names)
        
        # Print analysis report
        print_analysis_report(calculator, detailed=args.detailed)
        
        # Create visualizations
        if args.output:
            print(f"\nGenerating visualization...")
            visualizer = SustainabilityVisualizer(calculator)
            visualizer.save_visualization(args.output, args.viz_format)
            
            if args.viz_format == 'robustness':
                robustness_file = args.output.replace('.html', '_robustness.html')
                print(f"Robustness curve saved to: {robustness_file}")
            
            print(f"Visualization saved to: {args.output}")
        else:
            print(f"\nTo generate visualization, use: --output filename.html")
            print(f"For robustness analysis, use: --output filename.html --viz-format robustness")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()