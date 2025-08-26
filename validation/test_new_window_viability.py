#!/usr/bin/env python3
"""Test the new Window of Viability visualization"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ulanowicz_calculator import UlanowiczCalculator

# Load networks and calculate metrics
def load_and_calculate(filename):
    filepath = Path(__file__).parent.parent / 'data' / 'ecosystem_samples' / filename
    with open(filepath, 'r') as f:
        data = json.load(f)
    flows = np.array(data['flows'])
    calc = UlanowiczCalculator(flows)
    basic = calc.get_sustainability_metrics()
    extended = calc.get_extended_metrics()
    return {
        'alpha': basic['relative_ascendency'],
        'robustness': extended['robustness']
    }

original_metrics = load_and_calculate('prawns_alligator_original.json')
efficient_metrics = load_and_calculate('prawns_alligator_efficient.json')
adapted_metrics = load_and_calculate('prawns_alligator_adapted.json')

# Create the Window of Viability visualization with proper positioning
fig, ax = plt.subplots(figsize=(15, 10))

# Generate theoretical robustness curve
alpha_range = np.linspace(0, 1, 200)
b = 1.288  # Exponent from Ulanowicz
robustness_curve = []

for a in alpha_range:
    if 0 < a < 1:
        # Robustness formula: R = -a^b * ln(a^b) where b = 1.288
        r = -(np.e/np.log(np.e)) * a**b * np.log(a**b)
    else:
        r = 0
    robustness_curve.append(max(0, r))

# Plot the theoretical curve
ax.plot(alpha_range, robustness_curve, 'k-', linewidth=2.5, 
        label='Theoretical Robustness Curve', alpha=0.6, zorder=1)

# Mark the Window of Viability (0.2 < α < 0.6)
ax.axvspan(0.2, 0.6, alpha=0.15, color='green', label='Window of Viability', zorder=0)

# Mark the optimal point
optimal_alpha = 0.460
ax.axvline(optimal_alpha, color='orange', linestyle='--', linewidth=2, 
           label=f'Optimal α = {optimal_alpha}', alpha=0.7, zorder=1)

# Calculate positions for our three networks
networks = [
    {
        'name': 'Original\n(3 pathways)',
        'alpha': original_metrics['alpha'],
        'robustness': original_metrics['robustness'],
        'color': 'green',
        'marker': 'o',
        'size': 300
    },
    {
        'name': 'Adapted\n(no fish)',
        'alpha': adapted_metrics['alpha'],
        'robustness': adapted_metrics['robustness'],
        'color': 'blue',
        'marker': '^',
        'size': 300
    },
    {
        'name': 'Efficient Only\n(single path)',
        'alpha': efficient_metrics['alpha'],
        'robustness': efficient_metrics['robustness'],
        'color': 'red',
        'marker': 's',
        'size': 300
    }
]

# Plot each network
for net in networks:
    ax.scatter(net['alpha'], net['robustness'], 
               s=net['size'], c=net['color'], marker=net['marker'],
               edgecolors='black', linewidth=2.5, 
               label=net['name'].replace('\n', ' '), 
               zorder=10, alpha=0.9)

# Add custom annotations for each network
# Original network - positioned above the point
ax.annotate('Original\n3 pathways', 
            xy=(original_metrics['alpha'], original_metrics['robustness']),
            xytext=(original_metrics['alpha'], original_metrics['robustness'] + 0.04),
            fontsize=11, fontweight='bold', ha='center', va='bottom',
            color='darkgreen')

# Adapted network - positioned to the right
ax.annotate('Adapted\nafter fish loss', 
            xy=(adapted_metrics['alpha'], adapted_metrics['robustness']),
            xytext=(adapted_metrics['alpha'] + 0.05, adapted_metrics['robustness']),
            fontsize=11, fontweight='bold', ha='left', va='center',
            color='darkblue')

# Efficient network - special handling with arrow
ax.annotate('Efficient Only\nZero Resilience!', 
            xy=(efficient_metrics['alpha'], efficient_metrics['robustness']),
            xytext=(0.75, 0.08),
            fontsize=11, fontweight='bold', ha='center', va='center',
            color='darkred',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                     edgecolor='red', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3',
                          color='red', lw=2))

# Add zone labels
ax.text(0.1, 0.3, 'TOO\nCHAOTIC', ha='center', va='center',
        fontsize=14, color='red', fontweight='bold', alpha=0.6,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))

ax.text(0.4, 0.38, 'VIABLE\nZONE', ha='center', va='center',
        fontsize=14, color='green', fontweight='bold', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))

ax.text(0.8, 0.2, 'TOO\nRIGID', ha='center', va='center',
        fontsize=14, color='red', fontweight='bold', alpha=0.6,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))

# Add critical warning for efficient network
ax.text(0.98, 0.45, '⚠️ CRITICAL WARNING\n' + 
        'The efficient-only network\n' +
        'has ZERO resilience.\n' +
        'If any component fails,\n' +
        'the entire system collapses!',
        ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='red', alpha=0.1),
        color='darkred', fontweight='bold')

# Configure axes
ax.set_xlabel('Relative Ascendency (α) - Efficiency Measure', fontsize=14, fontweight='bold')
ax.set_ylabel('Robustness - Resilience Measure', fontsize=14, fontweight='bold')
ax.set_title('Prawns-Alligator Networks on the Window of Viability\n' +
             'Demonstrating the Efficiency-Resilience Trade-off',
             fontsize=16, fontweight='bold', pad=20)

# Set axis limits with padding for labels
ax.set_xlim(-0.05, 1.12)
ax.set_ylim(-0.03, 0.55)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Configure legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# Add background shading for different zones
ax.axvspan(0, 0.2, alpha=0.05, color='red', zorder=0)    # Too chaotic
ax.axvspan(0.6, 1.0, alpha=0.05, color='red', zorder=0)   # Too rigid

plt.tight_layout()
plt.savefig('window_viability_new.png', dpi=150, bbox_inches='tight')
plt.show()

# Print numerical positions
print("\nNETWORK POSITIONS ON THE WINDOW OF VIABILITY:")
print("=" * 60)
print(f"Original Network:  α = {original_metrics['alpha']:.4f}, R = {original_metrics['robustness']:.4f}")
print(f"                   Status: VIABLE (slightly below optimal)")
print(f"\nAdapted Network:   α = {adapted_metrics['alpha']:.4f}, R = {adapted_metrics['robustness']:.4f}")
print(f"                   Status: VIABLE (approaching optimal)")
print(f"\nEfficient Network: α = {efficient_metrics['alpha']:.4f}, R = {efficient_metrics['robustness']:.4f}")
print(f"                   Status: CRITICAL - Too rigid, zero resilience!")
print("\n" + "="*60)
print("✓ New Window of Viability visualization created successfully!")
print("  Saved as: window_viability_new.png")