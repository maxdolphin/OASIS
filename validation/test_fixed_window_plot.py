#!/usr/bin/env python3
"""
Test the fixed Window of Viability visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ulanowicz_calculator import UlanowiczCalculator

def load_network(filename):
    filepath = Path(__file__).parent.parent / 'data' / 'ecosystem_samples' / filename
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_metrics(data):
    flows = np.array(data['flows'])
    calc = UlanowiczCalculator(flows)
    basic = calc.get_sustainability_metrics()
    extended = calc.get_extended_metrics()
    return {
        'alpha': basic['relative_ascendency'],
        'robustness': extended['robustness']
    }

# Load and calculate metrics
original_metrics = calculate_metrics(load_network('prawns_alligator_original.json'))
efficient_metrics = calculate_metrics(load_network('prawns_alligator_efficient.json'))
adapted_metrics = calculate_metrics(load_network('prawns_alligator_adapted.json'))

# Plot all three networks on the Window of Viability
fig, ax = plt.subplots(figsize=(14, 9))

# Define the window
alpha_range = np.linspace(0, 1, 100)
optimal_alpha = 0.460

# Robustness curve
b = 1.288
robustness_curve = []
for a in alpha_range:
    if a > 0 and a < 1:
        r = -(np.e/np.log(np.e)) * a**b * np.log(a**b)
    else:
        r = 0
    robustness_curve.append(max(0, r))

# Plot robustness curve
ax.plot(alpha_range, robustness_curve, 'k-', linewidth=2, label='Theoretical Robustness', alpha=0.5)

# Mark the window of viability
ax.axvspan(0.2, 0.6, alpha=0.2, color='green', label='Window of Viability')
ax.axvline(optimal_alpha, color='orange', linestyle='--', linewidth=2, label=f'Optimal α = {optimal_alpha}')

# Plot our three networks
network_data = [
    ('Original\n(3 paths)', original_metrics['alpha'], original_metrics['robustness'], 'green', 'o'),
    ('Efficient\nOnly', efficient_metrics['alpha'], efficient_metrics['robustness'], 'red', 's'),
    ('Adapted\n(no fish)', adapted_metrics['alpha'], adapted_metrics['robustness'], 'blue', '^')
]

for name, alpha, robustness, color, marker in network_data:
    ax.scatter(alpha, robustness, s=250, c=color, marker=marker, 
              edgecolors='black', linewidth=2, label=name, zorder=5)

# Add annotations with improved positioning
# Original network
ax.annotate('Original\n(3 paths)', 
            (original_metrics['alpha'], original_metrics['robustness']),
            xytext=(0, 20), textcoords='offset points',
            fontsize=10, fontweight='bold', ha='center', va='bottom')

# Adapted network  
ax.annotate('Adapted\n(no fish)', 
            (adapted_metrics['alpha'], adapted_metrics['robustness']),
            xytext=(20, -10), textcoords='offset points',
            fontsize=10, fontweight='bold', ha='left', va='center')

# Efficient network - special handling for edge position
ax.annotate('Efficient\nOnly', 
            (efficient_metrics['alpha'], efficient_metrics['robustness']),
            xytext=(-80, 30), textcoords='offset points',
            fontsize=10, fontweight='bold', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='red', alpha=0.9),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                          color='red', lw=1.5))

# Add regions labels
ax.text(0.1, 0.25, 'TOO\nCHAOTIC', ha='center', va='center', 
        fontsize=12, color='red', fontweight='bold', alpha=0.7)
ax.text(0.4, 0.35, 'VIABLE\nSYSTEMS', ha='center', va='center', 
        fontsize=12, color='green', fontweight='bold', alpha=0.7)
ax.text(0.8, 0.15, 'TOO\nRIGID', ha='center', va='center', 
        fontsize=12, color='red', fontweight='bold', alpha=0.7)

# Add warning text about efficient network
ax.text(0.85, 0.05, '⚠️ Zero resilience!\nSystem fails completely\nif any connection breaks', 
        ha='center', va='center', fontsize=9, color='darkred',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

ax.set_xlabel('Relative Ascendency (α)', fontsize=13)
ax.set_ylabel('Robustness', fontsize=13)
ax.set_title('Prawns-Alligator Networks on the Window of Viability', 
            fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=10)

# Extend axes to accommodate labels
ax.set_xlim(-0.05, 1.15)
ax.set_ylim(-0.05, 0.52)

plt.tight_layout()
plt.savefig('window_viability_fixed.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nSYSTEM POSITIONS:")
print("=" * 50)
print(f"Original:  α = {original_metrics['alpha']:.3f} - VIABLE (below optimal)")
print(f"Efficient: α = {efficient_metrics['alpha']:.3f} - CRITICAL (too rigid!)")
print(f"Adapted:   α = {adapted_metrics['alpha']:.3f} - VIABLE (near optimal)")
print("\n✓ Fixed plot saved as 'window_viability_fixed.png'")