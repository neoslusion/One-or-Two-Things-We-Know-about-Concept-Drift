"""
Generate Prequential Evaluation Results Visualization.

Creates comprehensive visualizations for the prequential accuracy evaluation,
showing adaptation effectiveness with SE-CDT's type-specific strategies.

Output: results/plots/prequential_*.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import PREQUENTIAL_OUTPUTS, PLOTS_DIR


def plot_adaptation_strategy_effectiveness(
    results: Dict,
    drift_points: List[int],
    drift_types: List[str],
    output_path: Path
):
    """
    Create visualization showing effectiveness of different adaptation strategies.
    
    Args:
        results: Dictionary containing accuracy history for each mode
        drift_points: List of ground truth drift positions
        drift_types: List of drift types at each position
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Strategy colors
    strategy_colors = {
        'sudden': '#e74c3c',
        'gradual': '#f39c12',
        'incremental': '#9b59b6',
        'recurrent': '#1abc9c',
        'blip_ignored': '#95a5a6',
        'simple_retrain': '#3498db',
    }
    
    # Mode colors
    mode_colors = {
        'type_specific': '#2ecc71',
        'simple_retrain': '#3498db',
        'no_adaptation': '#e74c3c'
    }
    
    # === Panel 1: Accuracy Over Time (All Modes) ===
    ax1 = axes[0, 0]
    
    for mode, data in results.items():
        if 'accuracy' in data:
            idx_list = [r['idx'] for r in data['accuracy']]
            acc_list = [r['accuracy'] for r in data['accuracy']]
            
            color = mode_colors.get(mode, 'gray')
            label = mode.replace('_', ' ').title()
            
            ax1.plot(idx_list, acc_list, color=color, linewidth=1.5, label=label, alpha=0.8)
    
    # Mark drift points
    for i, dp in enumerate(drift_points):
        ax1.axvline(x=dp, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        if i < len(drift_types):
            ax1.annotate(
                drift_types[i][:3].upper(),
                xy=(dp, 1.02),
                xycoords=('data', 'axes fraction'),
                ha='center', fontsize=8
            )
    
    ax1.set_xlabel('Sample Index', fontsize=11)
    ax1.set_ylabel('Prequential Accuracy', fontsize=11)
    ax1.set_title('Accuracy Over Time: All Adaptation Modes', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.3, 1.05])
    
    # === Panel 2: Post-Drift Recovery ===
    ax2 = axes[0, 1]
    
    # For each drift, compute recovery curve (next 500 samples after drift)
    recovery_window = 500
    
    for mode, data in results.items():
        if 'accuracy' not in data:
            continue
            
        acc_map = {r['idx']: r['accuracy'] for r in data['accuracy']}
        
        # Average recovery curve across all drifts
        recovery_curves = []
        
        for dp in drift_points:
            curve = []
            for i in range(recovery_window):
                if dp + i in acc_map:
                    curve.append(acc_map[dp + i])
            
            if len(curve) >= recovery_window * 0.8:  # At least 80% coverage
                # Pad to same length
                while len(curve) < recovery_window:
                    curve.append(curve[-1])
                recovery_curves.append(curve[:recovery_window])
        
        if recovery_curves:
            avg_recovery = np.mean(recovery_curves, axis=0)
            std_recovery = np.std(recovery_curves, axis=0)
            
            x = np.arange(recovery_window)
            color = mode_colors.get(mode, 'gray')
            label = mode.replace('_', ' ').title()
            
            ax2.plot(x, avg_recovery, color=color, linewidth=2, label=label)
            ax2.fill_between(x, avg_recovery - std_recovery, avg_recovery + std_recovery,
                           color=color, alpha=0.2)
    
    ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% Baseline')
    ax2.set_xlabel('Samples After Drift', fontsize=11)
    ax2.set_ylabel('Prequential Accuracy', fontsize=11)
    ax2.set_title('Average Recovery Curve (Post-Drift)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.3, 1.0])
    
    # === Panel 3: Strategy Usage Distribution ===
    ax3 = axes[1, 0]
    
    # Count strategy usage from type_specific mode
    if 'type_specific' in results and 'adaptations' in results['type_specific']:
        strategy_counts = {}
        for adapt in results['type_specific']['adaptations']:
            strategy = adapt.get('strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        if strategy_counts:
            strategies = list(strategy_counts.keys())
            counts = list(strategy_counts.values())
            colors = [strategy_colors.get(s, '#95a5a6') for s in strategies]
            
            bars = ax3.bar(strategies, counts, color=colors, edgecolor='black', linewidth=1)
            
            ax3.set_xlabel('Strategy', fontsize=11)
            ax3.set_ylabel('Count', fontsize=11)
            ax3.set_title('Adaptation Strategy Usage', fontsize=12, fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, counts):
                ax3.annotate(str(count),
                           xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords='offset points',
                           ha='center', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No adaptation data available',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Adaptation Strategy Usage', fontsize=12, fontweight='bold')
    
    # === Panel 4: Performance Summary ===
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    summary_lines = [
        "PERFORMANCE SUMMARY",
        "=" * 40,
        ""
    ]
    
    for mode, data in results.items():
        if 'accuracy' in data:
            acc_list = [r['accuracy'] for r in data['accuracy']]
            mean_acc = np.mean(acc_list)
            n_adaptations = len(data.get('adaptations', []))
            n_detections = len(data.get('detections', []))
            
            mode_name = mode.replace('_', ' ').title()
            summary_lines.append(f"{mode_name}:")
            summary_lines.append(f"  Mean Accuracy: {mean_acc:.2%}")
            summary_lines.append(f"  Detections: {n_detections}")
            summary_lines.append(f"  Adaptations: {n_adaptations}")
            summary_lines.append("")
    
    summary_text = '\n'.join(summary_lines)
    
    ax4.text(0.1, 0.9, summary_text,
            transform=ax4.transAxes,
            fontfamily='monospace',
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('SE-CDT Prequential Evaluation: Adaptation Effectiveness',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Adaptation effectiveness plot saved: {output_path}")


def plot_detection_metrics_comparison(
    metrics: Dict,
    output_path: Path
):
    """
    Create bar chart comparing detection metrics across adaptation modes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    modes = list(metrics.keys())
    if not modes:
        print("No metrics data available")
        return
    
    # Filter out non-mode entries
    modes = [m for m in modes if isinstance(metrics[m], dict) and 'EDR' in metrics[m]]
    
    if not modes:
        print("No valid metrics found")
        return
    
    mode_colors = {
        'type_specific': '#2ecc71',
        'simple_retrain': '#3498db',
        'no_adaptation': '#e74c3c'
    }
    
    x = np.arange(len(modes))
    
    # === EDR (Detection Rate) ===
    ax1 = axes[0]
    values = [metrics[m].get('EDR', 0) for m in modes]
    colors = [mode_colors.get(m, 'gray') for m in modes]
    bars = ax1.bar(x, values, color=colors, edgecolor='black', linewidth=1)
    ax1.set_ylabel('EDR (Detection Rate)', fontsize=11)
    ax1.set_title('EDR (↑ better)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', '\n') for m in modes], fontsize=9)
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=10)
    
    # === Precision ===
    ax2 = axes[1]
    values = [metrics[m].get('Precision', 0) for m in modes]
    bars = ax2.bar(x, values, color=colors, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Precision (↑ better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', '\n') for m in modes], fontsize=9)
    ax2.set_ylim([0, 1.1])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax2.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=10)
    
    # === Overall Accuracy ===
    ax3 = axes[2]
    values = [metrics[m].get('overall_accuracy', 0) for m in modes]
    bars = ax3.bar(x, values, color=colors, edgecolor='black', linewidth=1)
    ax3.set_ylabel('Overall Accuracy', fontsize=11)
    ax3.set_title('Overall Accuracy (↑ better)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace('_', '\n') for m in modes], fontsize=9)
    ax3.set_ylim([0, 1.1])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax3.annotate(f'{val:.2%}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=10)
    
    plt.suptitle('Detection & Adaptation Metrics Comparison',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Detection metrics comparison saved: {output_path}")


def main():
    """Generate prequential evaluation visualizations."""
    print("="*60)
    print("Generating Prequential Evaluation Visualizations")
    print("="*60)
    
    # Create output directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Note: This script expects data from evaluate_prequential.py
    # For standalone usage, we'll generate sample data
    
    print("\nNote: For full visualizations, run 'python main.py monitoring' first")
    print("      This script will then use the generated results.")
    
    # Check if prequential results exist
    results_path = PREQUENTIAL_OUTPUTS.get("results_pkl", PLOTS_DIR.parent / "raw" / "prequential_results.pkl")
    
    if os.path.exists(results_path):
        import pickle
        with open(results_path, 'rb') as f:
            saved_results = pickle.load(f)
        
        # Extract data
        results = saved_results.get('results', {})
        drift_points = saved_results.get('drift_points', [])
        drift_types = saved_results.get('drift_types', [])
        metrics = saved_results.get('metrics', {})
        
        # Generate visualizations
        output_path = PLOTS_DIR / "prequential_adaptation_effectiveness.png"
        plot_adaptation_strategy_effectiveness(results, drift_points, drift_types, output_path)
        
        metrics_path = PLOTS_DIR / "prequential_metrics_comparison.png"
        plot_detection_metrics_comparison(metrics, metrics_path)
        
        print("\n✓ All prequential visualizations generated!")
    else:
        print(f"\nResults file not found: {results_path}")
        print("Run 'python main.py monitoring' first to generate prequential results.")
        
        # Create placeholder plot showing what would be generated
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 
               "Run 'python main.py monitoring' first\nto generate prequential evaluation data",
               ha='center', va='center', fontsize=14,
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat'))
        ax.set_title('Prequential Evaluation (Placeholder)', fontsize=14)
        ax.axis('off')
        
        placeholder_path = PLOTS_DIR / "prequential_placeholder.png"
        plt.savefig(placeholder_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Placeholder saved: {placeholder_path}")


if __name__ == "__main__":
    main()
