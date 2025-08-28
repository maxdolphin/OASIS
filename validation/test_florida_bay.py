#!/usr/bin/env python3
"""
Test Florida Bay ecosystem network metrics
Compare with any published values we can find
"""

import sys
import json
import numpy as np
sys.path.append('..')
from src.ulanowicz_calculator import UlanowiczCalculator

# Load Florida Bay data
with open('../data/ecosystem_samples/florida_bay.json', 'r') as f:
    data = json.load(f)

# Extract data
flow_matrix = np.array(data['flows'])
node_names = data['nodes']

print("="*60)
print("FLORIDA BAY ECOSYSTEM NETWORK ANALYSIS")
print("="*60)
print(f"Source: {data['metadata']['source']}")
print(f"Units: {data['metadata']['units']}")
print(f"Nodes: {len(node_names)}")
print(f"Location: {data['metadata']['location']}")
print()

# Calculate metrics
calc = UlanowiczCalculator(flow_matrix, node_names)
basic_metrics = calc.get_sustainability_metrics()
extended_metrics = calc.get_extended_metrics()

print("CALCULATED METRICS:")
print("-"*40)
print(f"Total System Throughput (TST): {basic_metrics['total_system_throughput']:.1f}")
print(f"Development Capacity (C): {basic_metrics['development_capacity']:.2f}")
print(f"Ascendency (A): {basic_metrics['ascendency']:.2f}")
print(f"Reserve/Overhead (Φ): {basic_metrics['reserve']:.2f}")
print(f"Relative Ascendency (α): {basic_metrics['relative_ascendency']:.4f}")
print(f"Robustness: {extended_metrics['robustness']:.4f}")
print()

# Verify fundamental relationship
c_check = basic_metrics['ascendency'] + basic_metrics['reserve']
error = abs(c_check - basic_metrics['development_capacity']) / basic_metrics['development_capacity'] * 100
print("VALIDATION:")
print("-"*40)
print(f"C = A + Φ verification:")
print(f"  {basic_metrics['development_capacity']:.2f} = {basic_metrics['ascendency']:.2f} + {basic_metrics['reserve']:.2f}")
print(f"  Error: {error:.6f}%")
print()

# Network properties
total_flows = np.sum(flow_matrix)
n_connections = np.count_nonzero(flow_matrix)
connectance = n_connections / (len(node_names) ** 2)

print("NETWORK PROPERTIES:")
print("-"*40)
print(f"Total flows: {total_flows:.1f}")
print(f"Active connections: {n_connections}")
print(f"Connectance: {connectance:.3f}")
print()

# Check cycling - important for Florida Bay
print("CYCLING ANALYSIS:")
print("-"*40)
# Detritus is node 9
detritus_idx = 9
detritus_inflow = np.sum(flow_matrix[:, detritus_idx])
detritus_outflow = np.sum(flow_matrix[detritus_idx, :])
print(f"Detritus inflow: {detritus_inflow:.1f}")
print(f"Detritus outflow: {detritus_outflow:.1f}")
print(f"Detritus recycling ratio: {detritus_outflow/detritus_inflow:.3f}")

# Estimate cycling percentage
# This is simplified - proper cycling analysis would use environ analysis
recycled_flow = min(detritus_inflow, detritus_outflow)
cycling_percentage = (recycled_flow / total_flows) * 100
print(f"Estimated cycling: {cycling_percentage:.1f}% of total system activity")
print()

print("PUBLISHED VALUES (from search):")
print("-"*40)
print("• >26% of total system activity involves recycling")
print("• Proportion exceeded only by coral reef ecosystems")
print("• Most recycling by pelagic and benthic flagellates")
print()

print("WINDOW OF VIABILITY:")
print("-"*40)
alpha = basic_metrics['relative_ascendency']
if 0.2 <= alpha <= 0.6:
    print(f"✅ System is VIABLE (α = {alpha:.3f})")
else:
    print(f"⚠️ System is OUTSIDE viability window (α = {alpha:.3f})")
    
if alpha < 0.2:
    print("   Status: Too chaotic/disorganized")
elif alpha > 0.6:
    print("   Status: Too rigid/brittle")
else:
    if alpha < 0.35:
        print("   Status: Lower end of viability (resilient but inefficient)")
    elif alpha > 0.5:
        print("   Status: Upper end of viability (efficient but less resilient)")
    else:
        print("   Status: Near optimal balance")