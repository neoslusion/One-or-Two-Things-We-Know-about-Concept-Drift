"""
Create SNR-Adaptive Architecture Diagram

This script generates a professional architecture diagram with ENGLISH labels
for the thesis.

Output: report/latex/image/snr_adaptive_architecture.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../../../report/latex/image/')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up figure with high quality
plt.figure(figsize=(12, 10), dpi=300)
ax = plt.gca()
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
color_input = '#E3F2FD'      # Light blue
color_process = '#BBDEFB'    # Medium blue
color_decision = '#FFE082'   # Yellow
color_aggressive = '#FFB74D' # Orange
color_conservative = '#81C784' # Green
color_output = '#B39DDB'     # Purple

# Font settings
title_font = {'family': 'serif', 'size': 14, 'weight': 'bold'}
box_font = {'family': 'serif', 'size': 11, 'weight': 'normal'}
small_font = {'family': 'serif', 'size': 9, 'weight': 'normal'}

def draw_box(ax, x, y, width, height, text, color, fontsize=11):
    """Draw a rounded box with text"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=color,
        linewidth=2
    )
    ax.add_patch(box)

    # Add text
    lines = text.split('\n')
    for i, line in enumerate(lines):
        y_offset = (len(lines) - 1) * 0.15 / 2 - i * 0.15
        ax.text(x, y + y_offset, line,
                ha='center', va='center',
                fontsize=fontsize, weight='bold' if i == 0 else 'normal',
                family='serif')

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """Draw an arrow with optional label"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=30,
        linewidth=2,
        color='black'
    )
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label,
                ha='left', va='center',
                fontsize=9, style='italic',
                family='serif',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))

def draw_diamond(ax, x, y, width, height, text, color):
    """Draw a diamond shape (decision)"""
    points = np.array([
        [x, y + height/2],      # top
        [x + width/2, y],       # right
        [x, y - height/2],      # bottom
        [x - width/2, y],       # left
    ])
    diamond = mpatches.Polygon(points, closed=True,
                              edgecolor='black', facecolor=color,
                              linewidth=2)
    ax.add_patch(diamond)

    # Add text
    lines = text.split('\n')
    for i, line in enumerate(lines):
        y_offset = (len(lines) - 1) * 0.12 / 2 - i * 0.12
        ax.text(x, y + y_offset, line,
                ha='center', va='center',
                fontsize=10, weight='bold',
                family='serif')

# Title (ENGLISH)
ax.text(5, 11.5, 'ShapeDD SNR-Adaptive Architecture',
        ha='center', va='top', **title_font)

# 1. Input: Data Stream (ENGLISH)
draw_box(ax, 5, 10.5, 3.5, 0.6,
         'Input Data Stream\n(X_t)',
         color_input)

# Arrow to SNR estimation
draw_arrow(ax, 5, 10.2, 5, 9.5)

# 2. SNR Estimation Module (ENGLISH)
draw_box(ax, 5, 9, 4, 1,
         'SNR Estimation Module\nSNR = var(recent_changes) / var(noise)\nBuffer size: 750 samples',
         color_process, fontsize=10)

# Arrow to decision
draw_arrow(ax, 5, 8.5, 5, 7.8)

# 3. Decision Diamond (ENGLISH)
draw_diamond(ax, 5, 7, 2.5, 1.2,
             'SNR > τ?\n(τ = 0.010)',
             color_decision)

# Left arrow (YES - High SNR) (ENGLISH)
draw_arrow(ax, 3.75, 7, 2.5, 5.5, label='YES\nHigh SNR', style='->')

# Right arrow (NO - Low SNR) (ENGLISH)
draw_arrow(ax, 6.25, 7, 7.5, 5.5, label='NO\nLow SNR', style='->')

# 4a. Aggressive Strategy (Left) (ENGLISH)
draw_box(ax, 2.5, 4, 3.5, 2.2,
         'AGGRESSIVE Strategy\n(High SNR)\n\nShapeDD_Adaptive_v2\n• sensitivity: "medium"\n• threshold: 0.01\n• n_perm: 2500',
         color_aggressive, fontsize=9)

# 4b. Conservative Strategy (Right) (ENGLISH)
draw_box(ax, 7.5, 4, 3.5, 2.2,
         'CONSERVATIVE Strategy\n(Low SNR)\n\nShapeDD (original)\n• threshold: 0.01\n• n_perm: 2500\n• No FDR filtering',
         color_conservative, fontsize=9)

# Arrows converging to output
draw_arrow(ax, 2.5, 2.9, 4, 2)
draw_arrow(ax, 7.5, 2.9, 6, 2)

# 5. Output: Drift Decision (ENGLISH)
draw_box(ax, 5, 1.2, 3, 0.8,
         'Drift Detection Decision\n(Detected / Not detected)',
         color_output)

# Add statistics boxes (ENGLISH)
# Updated with current benchmark values (8 methods, 7 datasets)
stats_box_text = 'Strategy Distribution\n(empirical):\n58.7% Aggressive\n41.3% Conservative'
ax.text(0.8, 7, stats_box_text,
        ha='left', va='center',
        fontsize=8, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', edgecolor='black', linewidth=1.5))

# UPDATED PERFORMANCE VALUES from current benchmark
# F1 = 0.236, Recall = 15.1%, Rank = 6/8
performance_text = 'Performance:\nF1 = 0.236\nRecall = 15.1%\nRank: 6/8'
ax.text(9.2, 7, performance_text,
        ha='right', va='center',
        fontsize=8, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#C8E6C9', edgecolor='black', linewidth=1.5))

# Add legend at bottom (ENGLISH)
legend_elements = [
    mpatches.Rectangle((0, 0), 1, 1, fc=color_input, ec='black', label='Input'),
    mpatches.Rectangle((0, 0), 1, 1, fc=color_process, ec='black', label='Processing'),
    mpatches.Rectangle((0, 0), 1, 1, fc=color_decision, ec='black', label='Decision'),
    mpatches.Rectangle((0, 0), 1, 1, fc=color_aggressive, ec='black', label='Aggressive'),
    mpatches.Rectangle((0, 0), 1, 1, fc=color_conservative, ec='black', label='Conservative'),
    mpatches.Rectangle((0, 0), 1, 1, fc=color_output, ec='black', label='Output'),
]
ax.legend(handles=legend_elements, loc='lower center',
          ncol=6, fontsize=8, frameon=True,
          bbox_to_anchor=(0.5, -0.05))

# Add note at bottom (ENGLISH)
note_text = ('Note: Threshold τ=0.010 is calibrated for buffer-based detection (750 samples). '
             'Observed SNR (0.005-0.020) is ~100× lower than theoretical SNR (0.4-4.0) '
             'due to buffer dilution effect.')
ax.text(5, 0.2, note_text,
        ha='center', va='top',
        fontsize=7, family='serif', style='italic',
        wrap=True,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', linewidth=1))

plt.tight_layout()

# Save with high quality
output_path = os.path.join(OUTPUT_DIR, 'snr_adaptive_architecture.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Architecture diagram saved to: {output_path}")
print(f"  Image size: {plt.gcf().get_size_inches()[0]:.1f} x {plt.gcf().get_size_inches()[1]:.1f} inches")
print(f"  Resolution: 300 DPI")
print()
print("Performance values updated to current benchmark results:")
print("  F1 = 0.236, Recall = 15.1%, Rank = 6/8")

plt.close()
