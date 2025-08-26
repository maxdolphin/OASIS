#!/usr/bin/env python3
"""
Test Window of Viability plot with larger text for better readability
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ulanowicz_calculator import UlanowiczCalculator

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

# Get metrics
original_metrics = load_and_calculate('prawns_alligator_original.json')
adapted_metrics = load_and_calculate('prawns_alligator_adapted.json')
efficient_metrics = load_and_calculate('prawns_alligator_efficient.json')

# Create figure with larger size for better readability
fig, ax = plt.subplots(figsize=(20, 12), constrained_layout=True)

# Generate theoretical robustness curve
alpha_range = np.linspace(0, 1, 200)
b = 1.288
robustness_curve = []
for a in alpha_range:
    if 0 < a < 1:
        r = - (a**b) * np.log(a**b)
    else:
        r = 0.0
    robustness_curve.append(max(0.0, r))

# Plot with thicker line
ax.plot(alpha_range, robustness_curve, 'k-', linewidth=3,
        label='Theoretical Robustness Curve', alpha=0.6, zorder=1)

# Window of Viability
ax.axvspan(0.2, 0.6, alpha=0.15, color='green', label='Window of Viability', zorder=0)

# Optimal point with thicker line
optimal_alpha = 0.460
ax.axvline(optimal_alpha, color='orange', linestyle='--', linewidth=3,
           label=f'Optimal α = {optimal_alpha}', alpha=0.7, zorder=1)

# Networks with larger markers
networks = [
    {'name': 'Original\n(3 pathways)', 'alpha': original_metrics['alpha'],
     'robustness': original_metrics['robustness'], 'color': 'green', 'marker': 'o', 'size': 400},
    {'name': 'Adapted\n(no fish)', 'alpha': adapted_metrics['alpha'],
     'robustness': adapted_metrics['robustness'], 'color': 'blue', 'marker': '^', 'size': 400},
    {'name': 'Efficient Only\n(single path)', 'alpha': efficient_metrics['alpha'],
     'robustness': efficient_metrics['robustness'], 'color': 'red', 'marker': 's', 'size': 400}
]

for net in networks:
    ax.scatter(net['alpha'], net['robustness'],
               s=net['size'], c=net['color'], marker=net['marker'],
               edgecolors='black', linewidth=3,
               label=net['name'].replace('\n', ' '),
               zorder=10, alpha=0.9)

# LARGER annotations
ax.annotate('Original\n3 pathways',
            xy=(original_metrics['alpha'], original_metrics['robustness']),
            xytext=(original_metrics['alpha'], original_metrics['robustness'] + 0.04),
            fontsize=14, fontweight='bold', ha='center', va='bottom',
            color='darkgreen', clip_on=False)

ax.annotate('Adapted\nafter fish loss',
            xy=(adapted_metrics['alpha'], adapted_metrics['robustness']),
            xytext=(adapted_metrics['alpha'] + 0.05, adapted_metrics['robustness']),
            fontsize=14, fontweight='bold', ha='left', va='center',
            color='darkblue', clip_on=False)

ax.annotate('Efficient Only\nZero Resilience!',
            xy=(efficient_metrics['alpha'], efficient_metrics['robustness']),
            xytext=(0.75, 0.08),
            fontsize=14, fontweight='bold', ha='center', va='center',
            color='darkred',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow',
                      edgecolor='red', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3',
                            color='red', lw=2.5),
            clip_on=False)

# LARGER zone labels
ax.text(0.1, 0.3, 'TOO\nCHAOTIC', ha='center', va='center',
        fontsize=18, color='red', fontweight='bold', alpha=0.6,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5),
        clip_on=False)

ax.text(0.4, 0.38, 'VIABLE\nZONE', ha='center', va='center',
        fontsize=18, color='green', fontweight='bold', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3),
        clip_on=False)

ax.text(0.8, 0.2, 'TOO\nRIGID', ha='center', va='center',
        fontsize=18, color='red', fontweight='bold', alpha=0.6,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5),
        clip_on=False)

# LARGER warning text
ax.text(0.98, 0.45,
        '⚠️ CRITICAL WARNING\n'
        'The efficient-only network\n'
        'has ZERO resilience.\n'
        'If any component fails,\n'
        'the entire system collapses!',
        ha='right', va='top', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='red', alpha=0.1),
        color='darkred', fontweight='bold', clip_on=False)

# LARGER axis labels and title
ax.set_xlabel('Relative Ascendency (α) - Efficiency Measure', fontsize=16, fontweight='bold')
ax.set_ylabel('Robustness - Resilience Measure', fontsize=16, fontweight='bold')
ax.set_title('Prawns-Alligator Networks on the Window of Viability\n'
             'Demonstrating the Efficiency-Resilience Trade-off',
             fontsize=19, fontweight='bold', pad=20)

ax.set_xlim(-0.05, 1.12)
ax.set_ylim(-0.03, 0.55)
ax.grid(True, alpha=0.3, linestyle='--')

# LARGER legend
ax.legend(loc='upper left', fontsize=13, framealpha=0.9)

# Background shading
ax.axvspan(0, 0.2, alpha=0.05, color='red', zorder=0)
ax.axvspan(0.6, 1.0, alpha=0.05, color='red', zorder=0)

# Draw and save
fig.canvas.draw()
plt.savefig('window_viability_large_text.png', dpi=300, pad_inches=0.2)
plt.savefig('window_viability_large_text.pdf', pad_inches=0.2)

print("✓ Plot saved with LARGER TEXT for better readability:")
print("  - window_viability_large_text.png (300 DPI)")
print("  - window_viability_large_text.pdf")
print("\nText size improvements:")
print("  • Annotations: 14pt (was 11pt)")
print("  • Zone labels: 18pt (was 14pt)")
print("  • Warning text: 13pt (was 10pt)")
print("  • Axis labels: 16pt (was 14pt)")
print("  • Title: 19pt (was 16pt)")
print("  • Legend: 13pt (was 11pt)")
print("  • Markers: 400 (was 300)")
print("  • Figure: 20x12 inches (was 18x11)")

plt.show()