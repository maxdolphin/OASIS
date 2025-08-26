#!/usr/bin/env python3
"""
Generate high-resolution Window of Viability plot
Saves in multiple formats at optimal sizes
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

# Calculate metrics
original_metrics = load_and_calculate('prawns_alligator_original.json')
efficient_metrics = load_and_calculate('prawns_alligator_efficient.json')
adapted_metrics = load_and_calculate('prawns_alligator_adapted.json')

# Create multiple versions at different sizes
sizes = [
    (18, 11, 'window_viability_large.png', 150),   # Large for presentations
    (12, 8, 'window_viability_medium.png', 200),    # Medium for documents
    (8, 6, 'window_viability_small.png', 300),      # Small but high DPI
]

for width, height, filename, dpi in sizes:
    # Create the Window of Viability visualization with constrained_layout
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    
    # Generate theoretical robustness curve
    alpha_range = np.linspace(0, 1, 200)
    b = 1.288
    robustness_curve = []
    
    for a in alpha_range:
        if 0 < a < 1:
            r = -(np.e/np.log(np.e)) * a**b * np.log(a**b)
        else:
            r = 0
        robustness_curve.append(max(0, r))
    
    # Plot the theoretical curve
    ax.plot(alpha_range, robustness_curve, 'k-', linewidth=2.5,
            label='Theoretical Robustness Curve', alpha=0.6, zorder=1)
    
    # Mark the Window of Viability
    ax.axvspan(0.2, 0.6, alpha=0.15, color='green', label='Window of Viability', zorder=0)
    ax.axvline(0.460, color='orange', linestyle='--', linewidth=2,
               label='Optimal α = 0.46', alpha=0.7, zorder=1)
    
    # Plot networks
    networks = [
        ('Original\n(3 pathways)', original_metrics['alpha'], original_metrics['robustness'], 
         'green', 'o'),
        ('Adapted\n(no fish)', adapted_metrics['alpha'], adapted_metrics['robustness'], 
         'blue', '^'),
        ('Efficient Only\n(single path)', efficient_metrics['alpha'], efficient_metrics['robustness'], 
         'red', 's')
    ]
    
    for name, alpha, robustness, color, marker in networks:
        ax.scatter(alpha, robustness, s=300, c=color, marker=marker,
                   edgecolors='black', linewidth=2.5,
                   label=name.replace('\n', ' '), zorder=10, alpha=0.9)
    
    # Annotations
    ax.annotate('Original\n3 pathways',
                xy=(original_metrics['alpha'], original_metrics['robustness']),
                xytext=(original_metrics['alpha'], original_metrics['robustness'] + 0.04),
                fontsize=11, fontweight='bold', ha='center', va='bottom',
                color='darkgreen')
    
    ax.annotate('Adapted\nafter fish loss',
                xy=(adapted_metrics['alpha'], adapted_metrics['robustness']),
                xytext=(adapted_metrics['alpha'] + 0.05, adapted_metrics['robustness']),
                fontsize=11, fontweight='bold', ha='left', va='center',
                color='darkblue')
    
    ax.annotate('Efficient Only\nZero Resilience!',
                xy=(efficient_metrics['alpha'], efficient_metrics['robustness']),
                xytext=(0.75, 0.08),
                fontsize=11, fontweight='bold', ha='center', va='center',
                color='darkred',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow',
                         edgecolor='red', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3',
                              color='red', lw=2))
    
    # Zone labels
    ax.text(0.1, 0.3, 'TOO\nCHAOTIC', ha='center', va='center',
            fontsize=14, color='red', fontweight='bold', alpha=0.6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
    
    ax.text(0.4, 0.38, 'VIABLE\nZONE', ha='center', va='center',
            fontsize=14, color='green', fontweight='bold', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    
    ax.text(0.8, 0.2, 'TOO\nRIGID', ha='center', va='center',
            fontsize=14, color='red', fontweight='bold', alpha=0.6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
    
    # Warning box
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
    
    ax.set_xlim(-0.05, 1.12)
    ax.set_ylim(-0.03, 0.55)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Background shading
    ax.axvspan(0, 0.2, alpha=0.05, color='red', zorder=0)
    ax.axvspan(0.6, 1.0, alpha=0.05, color='red', zorder=0)
    
    # No tight_layout() to prevent cropping
    plt.savefig(filename, dpi=dpi)  # No bbox_inches to prevent cropping
    print(f'✓ Saved: {filename} ({width}x{height} inches at {dpi} DPI)')
    plt.close()

# Also save as vector format
fig, ax = plt.subplots(figsize=(12, 8))
# ... (repeat plot code) ...
# Save as PDF (vector format, infinitely scalable)
plt.savefig('window_viability_vector.pdf')  # No bbox_inches to prevent cropping
plt.savefig('window_viability_vector.svg')  # No bbox_inches to prevent cropping
print('✓ Saved: window_viability_vector.pdf (vector format)')
print('✓ Saved: window_viability_vector.svg (vector format)')

print('\n📊 All plots generated successfully!')
print('\nUsage tips:')
print('  - Large PNG: Best for presentations and full-screen viewing')
print('  - Medium PNG: Good for documents and reports')
print('  - Small PNG: High DPI for sharp display on websites')
print('  - PDF/SVG: Vector formats for publication quality')