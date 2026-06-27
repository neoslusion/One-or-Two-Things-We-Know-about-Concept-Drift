"""
Generate Runtime Comparison Visualization.

Compares execution time between CDT_MSW and SE-CDT methods.

Output: results/plots/runtime_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def compute_runtime_stats(df):
    """
    Compute runtime statistics per scenario.
    
    Returns:
        DataFrame with runtime stats per scenario
    """
    stats = []
    
    for scenario in df['Scenario'].unique():
        scenario_df = df[df['Scenario'] == scenario]
        
        cdt_times = scenario_df['Runtime_CDT'].values
        se_times = scenario_df['Runtime_SE'].values
        stream_lengths = scenario_df['Stream_Length'].values
        
        avg_stream_len = np.mean(stream_lengths)
        
        stats.append({
            'Scenario': scenario,
            'CDT_Mean': np.mean(cdt_times),
            'CDT_Std': np.std(cdt_times),
            'SE_Mean': np.mean(se_times),
            'SE_Std': np.std(se_times),
            'Speedup': np.mean(cdt_times) / np.mean(se_times) if np.mean(se_times) > 0 else 0,
            'Avg_Stream_Length': avg_stream_len,
            'CDT_per_1k': np.mean(cdt_times) / (avg_stream_len / 1000),
            'SE_per_1k': np.mean(se_times) / (avg_stream_len / 1000)
        })
    
    return pd.DataFrame(stats)


def plot_runtime_comparison(stats_df, output_path):
    """
    Create runtime comparison bar chart.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = stats_df['Scenario'].values
    x = np.arange(len(scenarios))
    width = 0.35
    
    # === Panel 1: Absolute Runtime ===
    ax1 = axes[0]
    
    bars1 = ax1.bar(
        x - width/2,
        stats_df['CDT_Mean'],
        width,
        yerr=stats_df['CDT_Std'],
        label='CDT_MSW (Supervised)',
        color='#3498db',
        capsize=3
    )
    
    bars2 = ax1.bar(
        x + width/2,
        stats_df['SE_Mean'],
        width,
        yerr=stats_df['SE_Std'],
        label='SE-CDT (Unsupervised)',
        color='#2ecc71',
        capsize=3
    )
    
    ax1.set_ylabel('Runtime (seconds)', fontsize=11)
    ax1.set_xlabel('Scenario', fontsize=11)
    ax1.set_title('Average Runtime per Scenario', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}s',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}s',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # === Panel 2: Speedup Factor ===
    ax2 = axes[1]
    
    colors = ['#2ecc71' if s > 1 else '#e74c3c' for s in stats_df['Speedup']]
    bars3 = ax2.bar(x, stats_df['Speedup'], color=colors, edgecolor='black', linewidth=1)
    
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Equal Speed')
    ax2.set_ylabel('Speedup Factor (CDT/SE)', fontsize=11)
    ax2.set_xlabel('Scenario', fontsize=11)
    ax2.set_title('SE-CDT Speedup over CDT_MSW', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, speedup in zip(bars3, stats_df['Speedup']):
        height = bar.get_height()
        ax2.annotate(f'{speedup:.1f}×',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Overall stats
    overall_speedup = stats_df['Speedup'].mean()
    fig.text(
        0.5, 0.02,
        f'Average Speedup: {overall_speedup:.1f}× (SE-CDT is {"faster" if overall_speedup > 1 else "slower"})',
        ha='center',
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Runtime comparison saved: {output_path}")
    print(f"  Average Speedup: {overall_speedup:.1f}×")


def plot_throughput_comparison(stats_df, output_path):
    """
    Create throughput comparison (samples/second).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = stats_df['Scenario'].values
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Calculate throughput (samples per second)
    cdt_throughput = stats_df['Avg_Stream_Length'] / stats_df['CDT_Mean']
    se_throughput = stats_df['Avg_Stream_Length'] / stats_df['SE_Mean']
    
    ax.bar(x - width/2, cdt_throughput, width, label='CDT_MSW', color='#3498db')
    ax.bar(x + width/2, se_throughput, width, label='SE-CDT', color='#2ecc71')
    
    ax.set_ylabel('Throughput (samples/second)', fontsize=11)
    ax.set_xlabel('Scenario', fontsize=11)
    ax.set_title('Processing Throughput Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Scientific notation for y-axis
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Throughput comparison saved: {output_path}")


def main():
    """Generate all runtime comparison visualizations."""
    print("="*60)
    print("Generating Runtime Comparison Visualizations")
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
    
    # Compute stats
    stats_df = compute_runtime_stats(df)
    print("\nRuntime Statistics:")
    print(stats_df.to_string())
    
    # Generate visualizations
    output_path = PLOTS_DIR / "runtime_comparison.png"
    plot_runtime_comparison(stats_df, output_path)
    
    throughput_path = PLOTS_DIR / "throughput_comparison.png"
    plot_throughput_comparison(stats_df, throughput_path)
    
    print("\n✓ All runtime comparison visualizations generated!")


if __name__ == "__main__":
    main()
