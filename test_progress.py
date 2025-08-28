#!/usr/bin/env python3
"""Test script to verify enhanced progress tracking for Graminoids dataset."""

import json
import numpy as np
import time
from src.ulanowicz_calculator import UlanowiczCalculator

# Load Graminoids dataset
with open('data/ecosystem_samples/graminoid_everglades.json', 'r') as f:
    data = json.load(f)

nodes = data['nodes']
flows = np.array(data['flows'])

print(f"Testing Graminoids dataset: {len(nodes)} nodes, {np.sum(flows > 0)} flows")
print("=" * 60)

# Initialize calculator
print("\n1. Initializing calculator...")
calc = UlanowiczCalculator(flows, nodes)

# Test the slow functions with timing
print("\n2. Testing Finn Cycling Index (known bottleneck)...")
start = time.time()
try:
    fci = calc.calculate_finn_cycling_index()
    elapsed = time.time() - start
    print(f"   ✅ FCI = {fci:.4f} (computed in {elapsed:.2f}s)")
except Exception as e:
    elapsed = time.time() - start
    print(f"   ❌ Failed after {elapsed:.2f}s: {e}")

print("\n3. Testing Network Topology Metrics...")
start = time.time()
try:
    topo = calc.calculate_network_topology_metrics()
    elapsed = time.time() - start
    print(f"   ✅ Topology metrics computed in {elapsed:.2f}s")
    print(f"      - Average path length: {topo.get('average_path_length', 'N/A')}")
    print(f"      - Clustering coefficient: {topo.get('clustering_coefficient', 'N/A')}")
except Exception as e:
    elapsed = time.time() - start
    print(f"   ❌ Failed after {elapsed:.2f}s: {e}")

print("\n4. Testing full extended metrics...")
start = time.time()
try:
    extended = calc.get_extended_metrics()
    elapsed = time.time() - start
    print(f"   ✅ All extended metrics computed in {elapsed:.2f}s")
    print(f"      - Total metrics computed: {len(extended)}")
except Exception as e:
    elapsed = time.time() - start
    print(f"   ❌ Failed after {elapsed:.2f}s: {e}")

print("\n" + "=" * 60)
print("Test completed! Check app.py for enhanced progress indicators.")