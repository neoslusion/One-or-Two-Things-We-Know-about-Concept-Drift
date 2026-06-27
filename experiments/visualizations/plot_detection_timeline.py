"""
Generate Detection Timeline Visualization.

Shows detected drift points vs ground truth over time for each method.
This is a key visualization for drift detection papers.

Output: results/plots/detection_timeline_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import BENCHMARK_PROPER_OUTPUTS, PLOTS_DIR


def load_benchmark_results():
    """Load benchmark results from pickle file."""
    results_path = BENCHMARK_PROPER_OUTPUTS["results_pkl"]
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    df = pd.read_pickle(results_path)
    return df


def plot_detection_timeline_single(
    events,
    cdt_detections,
    se_detections,
    stream_length,
    scenario_name,
    output_path
):
    """
    Create detection timeline for a single scenario.
    
    Args:
        events: List of ground truth events
        cdt_detections: List of CDT_MSW detections
        se_detections: List of SE-CDT detections
        stream_length: Total stream length
        scenario_name: Name of scenario
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    
    # Color scheme
    colors = {
        'Sudden': '#e74c3c',
        'Gradual': '#f39c12',
        'Incremental': '#9b59b6',
        'Recurrent': '#1abc9c',
        'Blip': '#3498db',
        'Unknown': '#95a5a6'
    }
    
    # === Panel 1: Ground Truth ===
    ax1 = axes[0]
    ax1.set_title(f'Ground Truth Drift Events ({scenario_name})', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Drift Type')
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([0])
    ax1.set_yticklabels(['GT'])
    
    for i, evt in enumerate(events):
        pos = evt['pos']
        dtype = evt['type']
        width = evt.get('width', 0)
        color = colors.get(dtype, colors['Unknown'])
        
        # Draw marker at drift position
        ax1.scatter(pos, 0, marker='v', s=200, color=color, zorder=5, edgecolor='black', linewidth=1)
        
        # Draw transition region if applicable
        if width > 0:
            ax1.axvspan(pos, pos + width, alpha=0.2, color=color)
        
        # Label
        ax1.annotate(
            dtype[:3],
            xy=(pos, 0),
            xytext=(pos, 0.3),
            ha='center',
            fontsize=8,
            fontweight='bold',
            color=color
        )
    
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax1.grid(True, axis='x', alpha=0.3)
    
    # === Panel 2: CDT_MSW Detections ===
    ax2 = axes[1]
    ax2.set_title('CDT_MSW Detections (Supervised)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Detection')
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([0])
    ax2.set_yticklabels(['CDT'])
    
    # Draw ground truth lines (faint)
    for evt in events:
        ax2.axvline(x=evt['pos'], color='green', linestyle='--', alpha=0.3, linewidth=1)
    
    # Draw CDT detections
    for det in cdt_detections:
        pos = det.get('pos', det) if isinstance(det, dict) else det
        dtype = det.get('type', 'Unknown') if isinstance(det, dict) else 'Unknown'
        color = colors.get(dtype, colors['Unknown'])
        
        ax2.scatter(pos, 0, marker='s', s=150, color=color, zorder=5, edgecolor='black', linewidth=1)
        
        # Check if TP or FP
        is_tp = False
        for evt in events:
            if abs(pos - evt['pos']) < 250:  # Within tolerance
                is_tp = True
                break
        
        marker_edge = 'green' if is_tp else 'red'
        ax2.scatter(pos, 0, marker='s', s=200, facecolor='none', edgecolor=marker_edge, linewidth=2, zorder=6)
    
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Add detection count
    ax2.text(
        0.02, 0.95,
        f'Detections: {len(cdt_detections)}',
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment='top'
    )
    
    # === Panel 3: SE-CDT Detections ===
    ax3 = axes[2]
    ax3.set_title('SE-CDT Detections (Unsupervised)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Detection')
    ax3.set_xlabel('Sample Index', fontsize=11)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_yticks([0])
    ax3.set_yticklabels(['SE'])
    
    # Draw ground truth lines (faint)
    for evt in events:
        ax3.axvline(x=evt['pos'], color='green', linestyle='--', alpha=0.3, linewidth=1)
    
    # Draw SE-CDT detections
    for det in se_detections:
        pos = det.get('pos', det) if isinstance(det, dict) else det
        color = '#2ecc71'  # Green for SE-CDT
        
        ax3.scatter(pos, 0, marker='o', s=150, color=color, zorder=5, edgecolor='black', linewidth=1)
        
        # Check if TP or FP
        is_tp = False
        for evt in events:
            if abs(pos - evt['pos']) < 250:
                is_tp = True
                break
        
        marker_edge = 'green' if is_tp else 'red'
        ax3.scatter(pos, 0, marker='o', s=200, facecolor='none', edgecolor=marker_edge, linewidth=2, zorder=6)
    
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax3.grid(True, axis='x', alpha=0.3)
    
    ax3.text(
        0.02, 0.95,
        f'Detections: {len(se_detections)}',
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment='top'
    )
    
    # Set x-axis limits
    for ax in axes:
        ax.set_xlim(0, stream_length)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='green', linewidth=2, label='True Positive'),
        mpatches.Patch(facecolor='none', edgecolor='red', linewidth=2, label='False Positive'),
        plt.Line2D([0], [0], color='green', linestyle='--', label='Ground Truth'),
    ]
    
    # Add drift type legend
    for dtype, color in colors.items():
        if dtype != 'Unknown':
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                          markersize=10, label=dtype)
            )
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02)
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Timeline saved: {output_path}")


def plot_comparison_summary(results_df, output_path):
    """
    Create summary comparison of CDT_MSW vs SE-CDT across all scenarios.
    
    Shows TP/FP/FN counts per scenario.
    """
    # Aggregate metrics by scenario
    scenarios = results_df['Scenario'].unique()
    
    cdt_metrics = {'TP': [], 'FP': [], 'FN': []}
    se_metrics = {'TP': [], 'FP': [], 'FN': []}
    
    for scenario in scenarios:
        scenario_df = results_df[results_df['Scenario'] == scenario]
        
        # Aggregate CDT metrics
        cdt_tp = scenario_df['CDT_Metrics'].apply(lambda x: x['TP']).sum()
        cdt_fp = scenario_df['CDT_Metrics'].apply(lambda x: x['FP']).sum()
        cdt_fn = scenario_df['CDT_Metrics'].apply(lambda x: x['FN']).sum()
        
        cdt_metrics['TP'].append(cdt_tp)
        cdt_metrics['FP'].append(cdt_fp)
        cdt_metrics['FN'].append(cdt_fn)
        
        # Aggregate SE-CDT metrics (using Standard MMD variant)
        se_tp = scenario_df['SE_STD_Metrics'].apply(lambda x: x['TP']).sum()
        se_fp = scenario_df['SE_STD_Metrics'].apply(lambda x: x['FP']).sum()
        se_fn = scenario_df['SE_STD_Metrics'].apply(lambda x: x['FN']).sum()
        
        se_metrics['TP'].append(se_tp)
        se_metrics['FP'].append(se_fp)
        se_metrics['FN'].append(se_fn)
    
    # Create comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # TP comparison
    ax1 = axes[0]
    ax1.bar(x - width/2, cdt_metrics['TP'], width, label='CDT_MSW', color='#3498db')
    ax1.bar(x + width/2, se_metrics['TP'], width, label='SE-CDT', color='#2ecc71')
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('True Positives (↑ better)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # FP comparison
    ax2 = axes[1]
    ax2.bar(x - width/2, cdt_metrics['FP'], width, label='CDT_MSW', color='#3498db')
    ax2.bar(x + width/2, se_metrics['FP'], width, label='SE-CDT', color='#2ecc71')
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('False Positives (↓ better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # FN comparison
    ax3 = axes[2]
    ax3.bar(x - width/2, cdt_metrics['FN'], width, label='CDT_MSW', color='#3498db')
    ax3.bar(x + width/2, se_metrics['FN'], width, label='SE-CDT', color='#2ecc71')
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('False Negatives (↓ better)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    plt.suptitle('CDT_MSW vs SE-CDT: Detection Performance by Scenario', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Comparison summary saved: {output_path}")


def main():
    """Generate all detection timeline visualizations."""
    print("="*60)
    print("Generating Detection Timeline Visualizations")
    print("="*60)
    
    # Load results
    try:
        df = load_benchmark_results()
        print(f"Loaded {len(df)} benchmark results")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python main.py compare' first to generate results.")
        return
    
    # Create output directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Generate timeline for each scenario (seed=0 as example)
    scenarios = df['Scenario'].unique()
    
    for scenario in scenarios:
        scenario_df = df[(df['Scenario'] == scenario) & (df['Seed'] == 0)]
        
        if len(scenario_df) == 0:
            print(f"  No results for {scenario} seed=0, skipping...")
            continue
        
        row = scenario_df.iloc[0]
        
        # Extract data
        events = row['Events']
        cdt_dets = row['CDT_Detections']
        se_dets = row.get('SE_Det_STD', [])  # Use Standard MMD detections
        stream_length = row['Stream_Length']
        
        output_path = PLOTS_DIR / f"timeline_{scenario.lower()}.png"
        
        plot_detection_timeline_single(
            events=events,
            cdt_detections=cdt_dets,
            se_detections=se_dets,
            stream_length=stream_length,
            scenario_name=scenario,
            output_path=output_path
        )
    
    # Generate comparison summary
    summary_path = PLOTS_DIR / "detection_comparison_summary.png"
    plot_comparison_summary(df, summary_path)
    
    print("\n✓ All detection timeline visualizations generated!")


if __name__ == "__main__":
    main()
