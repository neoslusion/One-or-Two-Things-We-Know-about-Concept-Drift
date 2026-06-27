"""
Create SNR-Adaptive Strategy Selection Visualization

This script generates the strategy selection distribution figure.

Output: report/latex/image/strategy_selection.png
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../../../report/latex/image/')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set professional style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

print("Creating Strategy Selection Distribution...")

# Strategy counts from empirical results
aggressive_count = 291
conservative_count = 205
total = aggressive_count + conservative_count

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Pie chart (ENGLISH labels)
labels = [f'Aggressive\n(SNR > 0.010)\n58.7%', f'Conservative\n(SNR ≤ 0.010)\n41.3%']
sizes = [aggressive_count, conservative_count]
colors = ['#ff7f0e', '#1f77b4']
explode = (0.05, 0)

ax1.pie(sizes, explode=explode, labels=labels, autopct='%d selections',
        colors=colors, startangle=90, textprops={'fontsize': 12})
ax1.set_title('Strategy Selection Distribution\n(SNR-Adaptive v2)',
              fontsize=14, fontweight='bold')

# Right: Comparison bar chart (ENGLISH labels)
versions = ['v1\n(τ=0.008,\ns=high)', 'v2\n(τ=0.010,\ns=medium)', 'Optimal\n(theoretical)']
aggressive_pct = [64.7, 58.7, 50.0]
conservative_pct = [35.3, 41.3, 50.0]

x = np.arange(len(versions))
width = 0.35

bars1 = ax2.bar(x - width/2, aggressive_pct, width, label='Aggressive',
                color='#ff7f0e', alpha=0.8)
bars2 = ax2.bar(x + width/2, conservative_pct, width, label='Conservative',
                color='#1f77b4', alpha=0.8)

ax2.axhline(50, color='green', linestyle='--', linewidth=2, alpha=0.7,
            label='Target (50%)')
ax2.set_ylabel('Percentage (%)', fontsize=12)
ax2.set_title('Strategy Balance by Configuration', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(versions, fontsize=11)
ax2.legend(fontsize=11)
ax2.set_ylim([0, 100])
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, 'strategy_selection.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()
