#!/usr/bin/env python3
"""
Display the Cone Spring ecosystem matrices and metrics.
Run this to see the key data from Ulanowicz et al. (2009).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
# from tabulate import tabulate  # Optional, will use fallback if not available


def load_network_data():
    """Load both network configurations."""
    data_path = Path(__file__).parent.parent / 'data' / 'ecosystem_samples'
    
    with open(data_path / 'cone_spring_original.json', 'r') as f:
        original = json.load(f)
    
    with open(data_path / 'cone_spring_eutrophicated.json', 'r') as f:
        eutrophicated = json.load(f)
    
    return original, eutrophicated


def display_flow_matrix(data, title):
    """Display a flow matrix in a nice format."""
    flows = np.array(data['flows'])
    nodes = data['nodes']
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Create DataFrame for display
    df = pd.DataFrame(flows, index=nodes, columns=nodes)
    
    print("\nFlow Matrix (kcal m⁻² y⁻¹):")
    print(df.to_string(float_format=lambda x: f'{x:.0f}'))
    
    print(f"\nTotal Internal Flows: {flows.sum():.0f}")
    print(f"Row sums (outputs): {flows.sum(axis=1)}")
    print(f"Column sums (inputs): {flows.sum(axis=0)}")
    
    return flows


def display_published_metrics():
    """Display the published metrics from the paper."""
    print("\n" + "="*60)
    print("PUBLISHED METRICS FROM ULANOWICZ ET AL. (2009)")
    print("="*60)
    
    metrics = [
        ["Metric", "Original Network", "Eutrophicated Network"],
        ["─"*20, "─"*25, "─"*25],
        ["Relative Ascendency (α)", "0.418", "0.529"],
        ["Optimal α", "0.460", "0.460"],
        ["Distance from Optimal", "0.042 (below)", "0.069 (above)"],
        ["System Status", "Below optimal", "Above optimal"],
        ["Marginal Contributions", "Main path > 1, Parallel < 1", "Main path < 1, Parallel > 1"],
        ["Sustainability", "Can grow and develop", "Excess ascendency, reduced reserve"]
    ]
    
    for row in metrics:
        print(f"{row[0]:<25} {row[1]:<30} {row[2]:<30}")


def display_formulas():
    """Display the key Information Theory formulas."""
    print("\n" + "="*60)
    print("KEY INFORMATION THEORY FORMULAS")
    print("="*60)
    
    formulas = [
        ["Metric", "Formula", "Equation", "Units"],
        ["─"*20, "─"*40, "─"*10, "─"*15],
        ["Development Capacity", "C = -Σ(Tᵢⱼ × ln(Tᵢⱼ/T..))", "Eq. 11", "flow-nats"],
        ["Ascendency", "A = Σ(Tᵢⱼ × ln(Tᵢⱼ×T.. / (Tᵢ.×T.ⱼ)))", "Eq. 12", "flow-nats"],
        ["Reserve/Overhead", "Φ = C - A", "Eq. 13", "flow-nats"],
        ["Relative Ascendency", "α = A/C", "Derived", "dimensionless"],
        ["Fundamental Relation", "C = A + Φ", "Eq. 14", "-"]
    ]
    
    for row in formulas:
        print(f"{row[0]:<22} {row[1]:<42} {row[2]:<10} {row[3]:<15}")


def display_exogenous_flows(data, title):
    """Display exogenous flows for a network."""
    metadata = data['metadata']
    
    print(f"\n{title}")
    print("─"*40)
    
    if 'exogenous_inputs' in metadata:
        print("\nExogenous Inputs (kcal m⁻² y⁻¹):")
        for key, value in metadata['exogenous_inputs'].items():
            node = key.replace('to_', '').capitalize()
            print(f"  {node:<15}: {value:>6.0f}")
        total_in = sum(metadata['exogenous_inputs'].values())
        print(f"  {'Total':<15}: {total_in:>6.0f}")
    
    if 'exogenous_outputs' in metadata:
        print("\nExogenous Outputs (kcal m⁻² y⁻¹):")
        for key, value in metadata['exogenous_outputs'].items():
            node = key.replace('from_', '').capitalize()
            print(f"  {node:<15}: {value:>6.0f}")
        total_out = sum(metadata['exogenous_outputs'].values())
        print(f"  {'Total':<15}: {total_out:>6.0f}")
    
    if 'dissipations' in metadata:
        print("\nDissipations/Respiration (kcal m⁻² y⁻¹):")
        for key, value in metadata['dissipations'].items():
            print(f"  {key.capitalize():<15}: {value:>6.0f}")
        total_diss = sum(metadata['dissipations'].values())
        print(f"  {'Total':<15}: {total_diss:>6.0f}")


def display_difference(original_flows, eutrophicated_flows, nodes):
    """Display the difference between the two networks."""
    print("\n" + "="*60)
    print("EUTROPHICATION EFFECT (Difference Matrix)")
    print("="*60)
    
    diff = eutrophicated_flows - original_flows
    df_diff = pd.DataFrame(diff, index=nodes, columns=nodes)
    
    print("\nFlow Changes (kcal m⁻² y⁻¹):")
    print(df_diff.to_string(float_format=lambda x: f'{x:.0f}'))
    
    print(f"\nTotal flow increase: {diff.sum():.0f} kcal m⁻² y⁻¹")
    print("Note: Added 8000 kcal to Plants→Detritus→Bacteria pathway")


def main():
    """Main function to display all information."""
    print("\n" + "#"*60)
    print("# ULANOWICZ ET AL. (2009) - CONE SPRING ECOSYSTEM")
    print("# Network Matrices and Published Metrics")
    print("#"*60)
    
    # Load data
    original, eutrophicated = load_network_data()
    nodes = original['nodes']
    
    # Display flow matrices
    original_flows = display_flow_matrix(original, "ORIGINAL CONE SPRING ECOSYSTEM")
    eutrophicated_flows = display_flow_matrix(eutrophicated, "EUTROPHICATED CONE SPRING ECOSYSTEM")
    
    # Display the difference
    display_difference(original_flows, eutrophicated_flows, nodes)
    
    # Display published metrics
    display_published_metrics()
    
    # Display formulas
    display_formulas()
    
    # Display exogenous flows
    print("\n" + "="*60)
    print("EXOGENOUS FLOWS AND DISSIPATIONS")
    print("="*60)
    display_exogenous_flows(original, "Original Network")
    display_exogenous_flows(eutrophicated, "Eutrophicated Network")
    
    # Summary
    print("\n" + "="*60)
    print("ECOSYSTEM TRANSFORMATION SUMMARY")
    print("="*60)
    print("\nEutrophication Effect:")
    print("  • Added 8000 kcal m⁻² y⁻¹ to Plants→Detritus→Bacteria")
    print("  • Relative Ascendency: 0.418 → 0.529 (+26.6%)")
    print("  • Moved from below optimal to above optimal")
    print("  • Changed from growth-favorable to diversity-needed state")
    print("\nSustainability Implications:")
    print("  • Original: Healthy with room for development")
    print("  • Eutrophicated: Over-constrained, vulnerable to shocks")
    print("  • Recommendation: Increase pathway diversity")
    
    print("\n" + "#"*60)
    print("# END OF REPORT")
    print("#"*60)


if __name__ == "__main__":
    main()