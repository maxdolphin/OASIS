#!/usr/bin/env python3
"""
Extended example showcasing all regenerative economics indicators.

This script demonstrates the full capability of the extended Ulanowicz-Fath
regenerative economics framework with comprehensive analysis.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ulanowicz_calculator import UlanowiczCalculator
from visualizer import SustainabilityVisualizer


def demo_regenerative_indicators():
    """Comprehensive demo of all regenerative economics indicators."""
    
    print("="*80)
    print("COMPREHENSIVE REGENERATIVE ECONOMICS ANALYSIS DEMO")
    print("Extended Ulanowicz-Fath Framework Implementation")
    print("="*80)
    
    # Create a realistic organizational network
    departments = [
        "Executive", "Strategy", "HR", "Finance", "Operations", 
        "Marketing", "R&D", "Customer_Service", "IT", "Legal"
    ]
    
    # Complex organizational flow matrix (10x10)
    flow_matrix = np.array([
        # Exec Strat  HR   Fin  Ops  Mkt  R&D  CS   IT   Legal
        [0.0, 4.0,  2.0, 3.5, 3.0, 2.5, 2.0, 1.5, 2.0, 1.0],  # Executive
        [2.0, 0.0,  1.0, 2.0, 2.5, 3.0, 3.5, 1.0, 1.5, 0.5],  # Strategy
        [1.5, 0.8,  0.0, 1.5, 2.0, 1.0, 0.5, 2.5, 1.0, 2.0],  # HR
        [3.0, 2.0,  1.5, 0.0, 4.0, 2.5, 1.5, 1.0, 2.0, 1.5],  # Finance
        [1.0, 1.5,  1.8, 3.5, 0.0, 2.0, 2.5, 4.0, 3.0, 0.5],  # Operations
        [1.5, 2.5,  0.8, 2.0, 1.5, 0.0, 1.0, 3.5, 2.0, 0.3],  # Marketing
        [1.0, 3.0,  0.3, 1.0, 2.0, 1.5, 0.0, 0.5, 2.5, 1.0],  # R&D
        [0.8, 0.5,  2.0, 0.8, 3.5, 3.0, 0.3, 0.0, 1.5, 0.8],  # Customer Service
        [1.5, 1.0,  1.2, 2.5, 3.5, 2.0, 2.0, 2.0, 0.0, 1.0],  # IT
        [0.5, 0.3,  1.8, 2.0, 0.8, 0.5, 0.8, 1.0, 0.8, 0.0],  # Legal
    ])
    
    # Initialize calculator
    calc = UlanowiczCalculator(flow_matrix, departments)
    
    # Get all metrics
    extended_metrics = calc.get_extended_metrics()
    assessments = calc.assess_regenerative_health()
    
    # Print comprehensive analysis
    print_comprehensive_analysis(calc, extended_metrics, assessments)
    
    # Generate visualizations
    generate_comprehensive_visualizations(calc)
    
    # Demonstrate scenario analysis
    scenario_analysis(flow_matrix, departments)


def print_comprehensive_analysis(calc, extended_metrics, assessments):
    """Print detailed analysis of all indicators."""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE METRICS ANALYSIS")
    print("="*60)
    
    print("\n1. CORE ULANOWICZ INDICATORS:")
    print(f"   Total System Throughput (TST): {extended_metrics['total_system_throughput']:.3f}")
    print(f"   Average Mutual Information (AMI): {extended_metrics['average_mutual_information']:.3f}")
    print(f"   Ascendency (A): {extended_metrics['ascendency']:.3f}")
    print(f"   Development Capacity (C): {extended_metrics['development_capacity']:.3f}")
    print(f"   Overhead (Φ): {extended_metrics['overhead']:.3f}")
    
    print("\n2. REGENERATIVE ECONOMICS INDICATORS:")
    print(f"   Flow Diversity (H): {extended_metrics['flow_diversity']:.3f}")
    print(f"   Structural Information (SI): {extended_metrics['structural_information']:.3f}")
    print(f"   Robustness (R): {extended_metrics['robustness']:.3f}")
    print(f"   Network Efficiency: {extended_metrics['network_efficiency']:.3f}")
    print(f"   Regenerative Capacity: {extended_metrics['regenerative_capacity']:.3f}")
    
    print("\n3. NETWORK STRUCTURE INDICATORS:")
    print(f"   Effective Link Density: {extended_metrics['effective_link_density']:.3f}")
    print(f"   Trophic Depth: {extended_metrics['trophic_depth']:.3f}")
    print(f"   Redundancy: {extended_metrics['redundancy']:.3f}")
    
    print("\n4. SUSTAINABILITY RATIOS:")
    print(f"   Ascendency Ratio (A/C): {extended_metrics['ascendency_ratio']:.3f}")
    print(f"   Overhead Ratio (Φ/C): {extended_metrics['overhead_ratio']:.3f}")
    
    print("\n5. WINDOW OF VIABILITY:")
    print(f"   Lower Bound: {extended_metrics['viability_lower_bound']:.3f}")
    print(f"   Upper Bound: {extended_metrics['viability_upper_bound']:.3f}")
    print(f"   Current Position: {extended_metrics['ascendency']:.3f}")
    print(f"   Is Viable: {'YES' if extended_metrics['is_viable'] else 'NO'}")
    
    print("\n6. REGENERATIVE HEALTH ASSESSMENT:")
    for category, assessment in assessments.items():
        status = assessment.split(' - ')[0]
        print(f"   {category.title()}: {status}")
    
    print("\n" + "="*60)
    print("INTERPRETATION SUMMARY")
    print("="*60)
    
    # Robustness interpretation
    robustness = extended_metrics['robustness']
    efficiency = extended_metrics['network_efficiency']
    
    print(f"\nROBUSTNESS ANALYSIS:")
    if robustness > 0.25:
        print("  🟢 EXCELLENT: System demonstrates high robustness")
        print("     - Well-balanced between efficiency and resilience")
        print("     - Can handle significant disturbances")
    elif robustness > 0.15:
        print("  🟡 GOOD: System has moderate robustness")
        print("     - Generally stable with room for improvement")
    else:
        print("  🔴 CONCERNING: System lacks sufficient robustness")
        print("     - Vulnerable to disturbances and shocks")
    
    # Efficiency analysis
    print(f"\nEFFICIENCY ANALYSIS:")
    if 0.2 <= efficiency <= 0.6:
        print("  🟢 OPTIMAL: Efficiency within sustainable range")
        print(f"     - Current: {efficiency:.3f}, Optimal range: 0.2-0.6")
    elif efficiency < 0.2:
        print("  🟡 UNDERUTILIZED: System efficiency is low")
        print("     - Consider streamlining and better coordination")
    else:
        print("  🔴 OVER-OPTIMIZED: System may be too efficient")
        print("     - Risk of brittleness, consider adding redundancy")
    
    # Regenerative capacity analysis
    regen_capacity = extended_metrics['regenerative_capacity']
    print(f"\nREGENERATIVE CAPACITY ANALYSIS:")
    if regen_capacity > 0.2:
        print("  🟢 HIGH: Strong capacity for self-renewal and adaptation")
    elif regen_capacity > 0.1:
        print("  🟡 MODERATE: Some regenerative potential exists")
    else:
        print("  🔴 LOW: Limited capacity for adaptation and renewal")
    
    print(f"\nOVERALL SYSTEM HEALTH:")
    viable = extended_metrics['is_viable']
    if viable and robustness > 0.15 and 0.2 <= efficiency <= 0.6:
        print("  🟢 HEALTHY: System is operating within sustainable parameters")
    elif viable:
        print("  🟡 STABLE: System is viable but could be improved")
    else:
        print("  🔴 AT RISK: System requires significant intervention")


def generate_comprehensive_visualizations(calc):
    """Generate all available visualizations."""
    
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    visualizer = SustainabilityVisualizer(calc)
    
    # Generate main dashboard
    try:
        dashboard_path = Path(__file__).parent / "comprehensive_analysis.html"
        visualizer.save_visualization(str(dashboard_path), 'html')
        print(f"✅ Comprehensive dashboard: {dashboard_path}")
    except Exception as e:
        print(f"❌ Failed to generate dashboard: {e}")
    
    # Generate robustness curve
    try:
        robustness_path = Path(__file__).parent / "robustness_analysis.html" 
        robustness_fig = visualizer.create_robustness_curve()
        robustness_fig.write_html(str(robustness_path))
        print(f"✅ Robustness analysis: {robustness_path}")
    except Exception as e:
        print(f"❌ Failed to generate robustness curve: {e}")
    
    # Generate static charts
    try:
        static_path = Path(__file__).parent / "static_analysis.png"
        visualizer.save_visualization(str(static_path), 'png')
        print(f"✅ Static charts: {static_path}")
    except Exception as e:
        print(f"❌ Failed to generate static charts: {e}")


def scenario_analysis(base_flow_matrix, departments):
    """Demonstrate scenario analysis with different network configurations."""
    
    print("\n" + "="*60)
    print("SCENARIO ANALYSIS")
    print("="*60)
    
    scenarios = {
        "Current": base_flow_matrix,
        "Centralized": create_centralized_scenario(base_flow_matrix),
        "Distributed": create_distributed_scenario(base_flow_matrix),
        "Hybrid": create_hybrid_scenario(base_flow_matrix)
    }
    
    results = []
    
    for scenario_name, flow_matrix in scenarios.items():
        calc = UlanowiczCalculator(flow_matrix, departments)
        metrics = calc.get_extended_metrics()
        
        results.append({
            'scenario': scenario_name,
            'robustness': metrics['robustness'],
            'efficiency': metrics['network_efficiency'],
            'viable': metrics['is_viable'],
            'regenerative_capacity': metrics['regenerative_capacity']
        })
    
    print(f"\n{'Scenario':<12} {'Robust.':<8} {'Effic.':<8} {'Viable':<7} {'Regen.':<8} {'Assessment'}")
    print("-" * 70)
    
    for result in results:
        viable_str = "YES" if result['viable'] else "NO"
        
        if result['robustness'] > 0.2 and result['viable']:
            assessment = "🟢 STRONG"
        elif result['robustness'] > 0.1 and result['viable']:
            assessment = "🟡 GOOD"
        else:
            assessment = "🔴 WEAK"
        
        print(f"{result['scenario']:<12} {result['robustness']:<8.3f} {result['efficiency']:<8.3f} "
              f"{viable_str:<7} {result['regenerative_capacity']:<8.3f} {assessment}")
    
    # Find optimal scenario
    best_scenario = max(results, key=lambda x: x['robustness'] if x['viable'] else 0)
    print(f"\n🎯 RECOMMENDED SCENARIO: {best_scenario['scenario']}")
    print(f"   Robustness: {best_scenario['robustness']:.3f}")
    print(f"   Efficiency: {best_scenario['efficiency']:.3f}")
    print(f"   Regenerative Capacity: {best_scenario['regenerative_capacity']:.3f}")


def create_centralized_scenario(base_matrix):
    """Create a more centralized network scenario."""
    centralized = base_matrix.copy()
    
    # Increase flows to/from executive (node 0)
    centralized[0, :] *= 1.5  # Executive outputs more
    centralized[:, 0] *= 1.3  # Executive receives more
    
    # Reduce direct inter-departmental flows
    for i in range(1, centralized.shape[0]):
        for j in range(1, centralized.shape[1]):
            if i != j:
                centralized[i, j] *= 0.7
    
    return centralized


def create_distributed_scenario(base_matrix):
    """Create a more distributed network scenario."""
    distributed = base_matrix.copy()
    
    # Reduce flows to/from executive
    distributed[0, :] *= 0.7
    distributed[:, 0] *= 0.8
    
    # Increase direct inter-departmental flows
    for i in range(1, distributed.shape[0]):
        for j in range(1, distributed.shape[1]):
            if i != j:
                distributed[i, j] *= 1.3
    
    return distributed


def create_hybrid_scenario(base_matrix):
    """Create a hybrid network scenario."""
    hybrid = base_matrix.copy()
    
    # Strategic departments get more connections
    strategic_nodes = [1, 4, 5, 6]  # Strategy, Operations, Marketing, R&D
    
    for i in strategic_nodes:
        for j in strategic_nodes:
            if i != j:
                hybrid[i, j] *= 1.2
    
    return hybrid


if __name__ == "__main__":
    demo_regenerative_indicators()