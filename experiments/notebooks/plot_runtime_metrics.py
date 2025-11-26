"""
Runtime and Resource Usage Visualization for Drift Detectors
==============================================================

This script creates comprehensive plots for runtime and computational performance metrics.
Add this cell to your notebook after results are computed.

Usage in Jupyter:
    %run plot_runtime_metrics.py

Or copy the code below into a new cell.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ============================================================================
# PLOT 1: Runtime Comparison Bar Chart
# ============================================================================

def plot_runtime_comparison(results_df):
    """
    Compare average runtime across methods
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate mean runtime per method across all datasets
    runtime_summary = results_df.groupby('Method').agg({
        'Runtime_s': ['mean', 'std']
    }).reset_index()

    runtime_summary.columns = ['Method', 'Mean_Runtime', 'Std_Runtime']
    runtime_summary = runtime_summary.sort_values('Mean_Runtime')

    # Create bar plot
    x = np.arange(len(runtime_summary))
    bars = ax.bar(x, runtime_summary['Mean_Runtime'],
                   yerr=runtime_summary['Std_Runtime'],
                   capsize=5, alpha=0.7, edgecolor='black')

    # Color bars by runtime (green = fast, red = slow)
    max_runtime = runtime_summary['Mean_Runtime'].max()
    for i, bar in enumerate(bars):
        normalized = runtime_summary.iloc[i]['Mean_Runtime'] / max_runtime
        bar.set_color(plt.cm.RdYlGn_r(normalized))

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Runtime Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(runtime_summary['Method'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (mean_val, std_val) in enumerate(zip(runtime_summary['Mean_Runtime'],
                                                  runtime_summary['Std_Runtime'])):
        ax.text(i, mean_val + std_val + 0.1, f'{mean_val:.2f}s',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('runtime_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: runtime_comparison.png")
    return fig


# ============================================================================
# PLOT 2: Throughput Comparison
# ============================================================================

def plot_throughput_comparison(results_df):
    """
    Compare throughput (samples/second) across methods
    """
    if 'Throughput' not in results_df.columns:
        print("⚠ Throughput column not found, skipping this plot")
        return None

    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate mean throughput per method
    throughput_summary = results_df.groupby('Method').agg({
        'Throughput': ['mean', 'std']
    }).reset_index()

    throughput_summary.columns = ['Method', 'Mean_Throughput', 'Std_Throughput']
    throughput_summary = throughput_summary.sort_values('Mean_Throughput', ascending=False)

    # Create bar plot
    x = np.arange(len(throughput_summary))
    bars = ax.bar(x, throughput_summary['Mean_Throughput'],
                   yerr=throughput_summary['Std_Throughput'],
                   capsize=5, alpha=0.7, edgecolor='black')

    # Color bars (green = fast, red = slow)
    max_throughput = throughput_summary['Mean_Throughput'].max()
    for i, bar in enumerate(bars):
        normalized = throughput_summary.iloc[i]['Mean_Throughput'] / max_throughput
        bar.set_color(plt.cm.RdYlGn(normalized))

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (samples/second)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(throughput_summary['Method'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (mean_val, std_val) in enumerate(zip(throughput_summary['Mean_Throughput'],
                                                  throughput_summary['Std_Throughput'])):
        ax.text(i, mean_val + std_val + 100, f'{mean_val:.0f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('throughput_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: throughput_comparison.png")
    return fig


# ============================================================================
# PLOT 3: Runtime vs F1 Score Scatter (Speed-Accuracy Trade-off)
# ============================================================================

def plot_runtime_vs_f1(results_df):
    """
    Scatter plot showing trade-off between speed and accuracy
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate mean F1 and Runtime per method
    summary = results_df.groupby('Method').agg({
        'F1': 'mean',
        'Runtime_s': 'mean'
    }).reset_index()

    # Create scatter plot
    scatter = ax.scatter(summary['Runtime_s'], summary['F1'],
                        s=200, alpha=0.6, edgecolors='black', linewidth=1.5)

    # Add method labels
    for idx, row in summary.iterrows():
        ax.annotate(row['Method'],
                   (row['Runtime_s'], row['F1']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)

    # Highlight Pareto frontier (best trade-off)
    # Sort by runtime
    summary_sorted = summary.sort_values('Runtime_s')
    pareto_front = []
    max_f1_so_far = -1

    for idx, row in summary_sorted.iterrows():
        if row['F1'] > max_f1_so_far:
            pareto_front.append(idx)
            max_f1_so_far = row['F1']

    if len(pareto_front) > 0:
        pareto_df = summary.loc[pareto_front].sort_values('Runtime_s')
        ax.plot(pareto_df['Runtime_s'], pareto_df['F1'],
               'r--', linewidth=2, alpha=0.5, label='Pareto Frontier')

    ax.set_xlabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add quadrant lines (median)
    median_runtime = summary['Runtime_s'].median()
    median_f1 = summary['F1'].median()
    ax.axvline(median_runtime, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(median_f1, color='gray', linestyle=':', alpha=0.5)

    # Add quadrant labels
    ax.text(ax.get_xlim()[1] * 0.95, ax.get_ylim()[1] * 0.95,
           'Slow but\nAccurate', ha='right', va='top',
           fontsize=10, alpha=0.5, style='italic')
    ax.text(ax.get_xlim()[0] * 1.05, ax.get_ylim()[1] * 0.95,
           'Fast &\nAccurate', ha='left', va='top',
           fontsize=10, alpha=0.5, style='italic', color='green', weight='bold')
    ax.text(ax.get_xlim()[0] * 1.05, ax.get_ylim()[0] * 1.05,
           'Fast but\nInaccurate', ha='left', va='bottom',
           fontsize=10, alpha=0.5, style='italic')

    plt.tight_layout()
    plt.savefig('runtime_vs_f1.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: runtime_vs_f1.png")
    return fig


# ============================================================================
# PLOT 4: Runtime Distribution Box Plot
# ============================================================================

def plot_runtime_distribution(results_df):
    """
    Box plot showing runtime distribution across datasets for each method
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for box plot
    methods = sorted(results_df['Method'].unique())
    data_to_plot = [results_df[results_df['Method'] == method]['Runtime_s'].values
                    for method in methods]

    # Create box plot
    bp = ax.boxplot(data_to_plot, labels=methods, patch_artist=True,
                    showmeans=True, meanline=True,
                    medianprops={'color': 'red', 'linewidth': 2},
                    meanprops={'color': 'blue', 'linewidth': 2, 'linestyle': '--'})

    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Runtime Distribution Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('runtime_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: runtime_distribution.png")
    return fig


# ============================================================================
# PLOT 5: Heatmap - Runtime by Method and Dataset
# ============================================================================

def plot_runtime_heatmap(results_df):
    """
    Heatmap showing runtime for each method on each dataset
    """
    # Create pivot table
    pivot = results_df.pivot_table(values='Runtime_s',
                                   index='Method',
                                   columns='Dataset',
                                   aggfunc='mean')

    fig, ax = plt.subplots(figsize=(14, 10))

    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'Runtime (seconds)'},
                linewidths=0.5, ax=ax)

    ax.set_title('Runtime Heatmap: Method × Dataset', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Method', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('runtime_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: runtime_heatmap.png")
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if running in notebook
    try:
        # Try to access results_df from notebook
        if 'results_df' not in globals():
            print("⚠ Error: results_df not found!")
            print("  Make sure you run this after computing benchmark results.")
        else:
            print("=" * 80)
            print("GENERATING RUNTIME & RESOURCE USAGE PLOTS")
            print("=" * 80)
            print()

            # Generate all plots
            figs = []

            print("[1/5] Runtime Comparison Bar Chart...")
            figs.append(plot_runtime_comparison(results_df))

            print("[2/5] Throughput Comparison...")
            figs.append(plot_throughput_comparison(results_df))

            print("[3/5] Speed vs Accuracy Trade-off...")
            figs.append(plot_runtime_vs_f1(results_df))

            print("[4/5] Runtime Distribution Box Plot...")
            figs.append(plot_runtime_distribution(results_df))

            print("[5/5] Runtime Heatmap...")
            figs.append(plot_runtime_heatmap(results_df))

            print()
            print("=" * 80)
            print("✓ ALL PLOTS GENERATED SUCCESSFULLY!")
            print("=" * 80)
            print()
            print("Saved files:")
            print("  1. runtime_comparison.png - Bar chart of average runtime")
            print("  2. throughput_comparison.png - Throughput (samples/sec)")
            print("  3. runtime_vs_f1.png - Speed-accuracy trade-off scatter")
            print("  4. runtime_distribution.png - Runtime distribution box plot")
            print("  5. runtime_heatmap.png - Method × Dataset heatmap")
            print()
            print("To display plots in notebook, run: plt.show()")

    except Exception as e:
        print(f"⚠ Error: {e}")
        print("  Make sure results_df is available in your notebook environment.")


# ============================================================================
# COPY THIS TO NOTEBOOK
# ============================================================================
"""
USAGE IN JUPYTER NOTEBOOK:
==========================

After running your benchmark (Cell 6), add a new cell with:

```python
# ============================================================================
# RUNTIME & RESOURCE USAGE VISUALIZATION
# ============================================================================

%run plot_runtime_metrics.py
plt.show()  # Display all plots
```

Or copy the functions above and call them individually:

```python
fig1 = plot_runtime_comparison(results_df)
fig2 = plot_throughput_comparison(results_df)
fig3 = plot_runtime_vs_f1(results_df)
fig4 = plot_runtime_distribution(results_df)
fig5 = plot_runtime_heatmap(results_df)
plt.show()
```
"""
