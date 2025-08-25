#!/usr/bin/env python3
"""
Debug the calculation step by step for Cone Spring ecosystem.
"""

import numpy as np
import math
import json
from pathlib import Path


def calculate_metrics_manually(flow_matrix):
    """Calculate metrics manually following Ulanowicz formulas exactly."""
    n = len(flow_matrix)
    
    # Calculate Total System Throughput (TST)
    tst = np.sum(flow_matrix)
    print(f"TST (T..): {tst:.1f}")
    
    # Calculate row sums (outputs) Ti.
    row_sums = np.sum(flow_matrix, axis=1)
    print(f"Row sums (Ti.): {row_sums}")
    
    # Calculate column sums (inputs) T.j
    col_sums = np.sum(flow_matrix, axis=0)
    print(f"Column sums (T.j): {col_sums}")
    
    # Calculate Development Capacity: C = -Σ(Tij * ln(Tij/T..))
    C = 0
    for i in range(n):
        for j in range(n):
            if flow_matrix[i,j] > 0:
                Tij = flow_matrix[i,j]
                C += Tij * math.log(Tij / tst)
    C = -C
    print(f"\nDevelopment Capacity (C): {C:.1f}")
    
    # Calculate Ascendency: A = Σ(Tij * ln(Tij*T.. / (Ti.*T.j)))
    A = 0
    for i in range(n):
        for j in range(n):
            if flow_matrix[i,j] > 0:
                Tij = flow_matrix[i,j]
                Ti_dot = row_sums[i]
                T_dot_j = col_sums[j]
                if Ti_dot > 0 and T_dot_j > 0:
                    A += Tij * math.log((Tij * tst) / (Ti_dot * T_dot_j))
    print(f"Ascendency (A): {A:.1f}")
    
    # Calculate Reserve
    reserve = C - A
    print(f"Reserve (Φ): {reserve:.1f}")
    
    # Calculate Relative Ascendency
    alpha = A / C if C > 0 else 0
    print(f"Relative Ascendency (α): {alpha:.3f}")
    
    return {
        'tst': tst,
        'C': C,
        'A': A,
        'reserve': reserve,
        'alpha': alpha
    }


def main():
    # Load Cone Spring original data
    filepath = Path(__file__).parent.parent / 'data' / 'ecosystem_samples' / 'cone_spring_original.json'
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print("="*60)
    print("CONE SPRING ORIGINAL - Manual Calculation")
    print("="*60)
    
    flows = np.array(data['flows'], dtype=float)
    print(f"\nFlow matrix shape: {flows.shape}")
    print(f"Flow matrix:\n{flows}")
    
    print("\n--- Using Internal Flows Only ---")
    metrics = calculate_metrics_manually(flows)
    
    print(f"\nExpected α from paper: 0.418")
    print(f"Calculated α: {metrics['alpha']:.3f}")
    print(f"Difference: {abs(metrics['alpha'] - 0.418):.3f}")
    
    # Now try with a complete system matrix including inputs/outputs as separate nodes
    print("\n" + "="*60)
    print("Testing different matrix configurations...")
    print("="*60)
    
    # Configuration 1: Add exogenous as self-loops
    print("\n--- Configuration 1: Adding exogenous as diagonal ---")
    flows_with_diag = flows.copy()
    # The dissipations might need to be on diagonal
    if 'dissipations' in data['metadata']:
        dissipations = data['metadata']['dissipations']
        flows_with_diag[0,0] = dissipations.get('plants', 0)
        flows_with_diag[1,1] = dissipations.get('detritus', 0)
        flows_with_diag[2,2] = dissipations.get('bacteria', 0)
        flows_with_diag[3,3] = dissipations.get('detritivores', 0)
        flows_with_diag[4,4] = dissipations.get('carnivores', 0)
    metrics2 = calculate_metrics_manually(flows_with_diag)
    print(f"α with dissipations on diagonal: {metrics2['alpha']:.3f}")


if __name__ == "__main__":
    main()