#!/usr/bin/env python3
"""Test script for Roles Analysis implementation"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ulanowicz_calculator import UlanowiczCalculator

print("=" * 60)
print("TESTING ROLES ANALYSIS IMPLEMENTATION")
print("=" * 60)

# Test Case 1: Simple 3-node network
print("\nTest 1: Simple 3-node network")
print("-" * 40)

flow_matrix = np.array([
    [0, 10, 5],
    [5, 0, 10],
    [10, 5, 0]
])
nodes = ["Node A", "Node B", "Node C"]

calc = UlanowiczCalculator(flow_matrix, nodes)
metrics = calc.get_extended_metrics()

print(f"AMI: {metrics.get('average_mutual_information', 0):.4f}")
print(f"Number of Roles (R): {metrics.get('number_of_roles', 0):.4f}")
print(f"Effective Nodes (N): {metrics.get('effective_nodes', 0):.4f}")
print(f"Effective Flows (F): {metrics.get('effective_flows', 0):.4f}")
print(f"Effective Connectivity (C): {metrics.get('effective_connectivity', 0):.4f}")
print(f"Roles per Node: {metrics.get('roles_per_node', 0):.4f}")
print(f"Specialization Index: {metrics.get('specialization_index', 0):.4f}")

# Verify relationships
R = metrics.get('number_of_roles', 0)
N = metrics.get('effective_nodes', 0)
F = metrics.get('effective_flows', 0)
C = metrics.get('effective_connectivity', 0)

if F > 0:
    check1 = N**2 / F
    print(f"\nVerification R = N²/F: R={R:.4f}, N²/F={check1:.4f}")

if C > 0:
    check2 = F / C**2
    check3 = N / C
    print(f"Verification R = F/C²: R={R:.4f}, F/C²={check2:.4f}")
    print(f"Verification R = N/C: R={R:.4f}, N/C={check3:.4f}")

# Verify log(R) = AMI
ami = metrics.get('average_mutual_information', 0)
log_R = np.log(R) if R > 0 else 0
print(f"\nlog(R) = AMI check: log(R)={log_R:.4f}, AMI={ami:.4f}")

print("\n" + "=" * 60)

# Test Case 2: Load an ecosystem sample
print("\nTest 2: Crystal River (Control) Ecosystem")
print("-" * 40)

try:
    import json
    with open('data/ecosystem_samples/crystal_river_control.json', 'r') as f:
        data = json.load(f)
    
    flow_matrix = np.array(data['flows'])
    nodes = data['nodes']
    
    calc = UlanowiczCalculator(flow_matrix, nodes)
    metrics = calc.get_extended_metrics()
    
    print(f"Network: {data['organization']}")
    print(f"Nodes: {len(nodes)}")
    print(f"Non-zero flows: {np.count_nonzero(flow_matrix)}")
    print()
    print(f"Number of Roles (R): {metrics.get('number_of_roles', 0):.4f}")
    print(f"Effective Nodes (N): {metrics.get('effective_nodes', 0):.4f}")
    print(f"Effective Flows (F): {metrics.get('effective_flows', 0):.4f}")
    print(f"Effective Connectivity (C): {metrics.get('effective_connectivity', 0):.4f}")
    print(f"Verification Error: {metrics.get('roles_verification_error', 0):.6f}")
    
    # Check if in natural range
    R = metrics.get('number_of_roles', 0)
    if 2 <= R <= 5:
        print(f"✅ Roles in natural range [2, 5]: {R:.2f}")
    else:
        print(f"⚠️ Roles outside natural range [2, 5]: {R:.2f}")
    
except FileNotFoundError:
    print("Crystal River data not found - skipping ecosystem test")
except Exception as e:
    print(f"Error loading ecosystem data: {e}")

print("\n" + "=" * 60)
print("TESTS COMPLETED")
print("=" * 60)