"""
Visualization module for publication-quality figures.

Contains functions for generating:
- F1/Precision/Recall/MTTD heatmaps (with bold best values)
- Method comparison bar charts (with CI error bars)
- Detection timelines
- Runtime comparison
- Speed-accuracy trade-off plots
- Critical difference diagram (Nemenyi post-hoc test)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats


def setup_plot_style():
    """Set up publication-quality plot style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
    })
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})


def categorize_dataset(name):
    """Categorize dataset by drift type."""
    name_lower = name.lower()
    if 'gradual' in name_lower or 'circles' in name_lower:
        return 'B_Gradual'
    elif 'rbf' in name_lower:
        return 'C_Incremental'
    elif 'electricity' in name_lower or 'covertype' in name_lower:
        return 'D_Real-World'
    elif 'none' in name_lower:
        return 'E_Stationary'
    else:
        return 'A_Sudden'


def save_figure(fig, name, output_dir):
    """Save figure in PNG and PDF formats."""
    for fmt in ['png', 'pdf']:
        filepath = output_dir / f"{name}.{fmt}"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', format=fmt)
    print(f"  Saved: {name}.png/.pdf")


def annotate_heatmap_with_best(ax, data, fmt='.3f', higher_is_better=True):
    """Annotate heatmap with bold text for best values per column."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            col_vals = data.iloc[:, j]

            if higher_is_better:
                is_best = val == col_vals.max()
            else:
                is_best = val == col_vals.min()

            # Format text
            if pd.isna(val):
                text = "-"
                weight = 'normal'
            else:
                text = f"{val:{fmt.replace('.', '')}}"
                weight = 'bold' if is_best else 'normal'

            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   fontsize=10, fontweight=weight,
                   color='black' if val > 0.5 or pd.isna(val) else 'white')


def plot_critical_difference_diagram(results_df, metric='F1', output_dir=None, alpha=0.05):
    """
    Generate Critical Difference diagram using Nemenyi post-hoc test.

    This is the standard visualization for comparing multiple methods
    across multiple datasets (Demsar, 2006).

    Args:
        results_df: DataFrame with 'Method', 'Dataset', and metric columns
        metric: Metric to compare (default: 'F1')
        output_dir: Directory to save figure
        alpha: Significance level (default: 0.05)

    Returns:
        fig: matplotlib figure
    """
    # Pivot to get method x dataset matrix
    pivot = results_df.pivot_table(values=metric, index='Dataset', columns='Method', aggfunc='mean')

    n_datasets = len(pivot)
    n_methods = len(pivot.columns)

    # Compute ranks (higher is better for F1, so negate)
    ranks = pivot.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean()

    # Friedman test
    stat, p_value = stats.friedmanchisquare(*[pivot[col].values for col in pivot.columns])

    # Critical difference (Nemenyi)
    # CD = q_alpha * sqrt(k(k+1) / (6*n))
    # q_alpha values for alpha=0.05 (from Demsar 2006, Table 5)
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
        12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
        17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544
    }
    q_alpha = q_alpha_table.get(n_methods, 2.8)  # Default approximation
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))

    # Sort methods by average rank
    sorted_methods = avg_ranks.sort_values()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(4, n_methods * 0.4)))

    # Plot parameters
    y_min, y_max = 1, n_methods
    x_positions = np.linspace(0.1, 0.9, n_methods)

    # Draw the axis
    ax.hlines(0.5, 0, 1, colors='black', linewidth=1)

    # Draw tick marks and labels
    for i, (method, rank) in enumerate(sorted_methods.items()):
        x_pos = (rank - 1) / (n_methods - 1) * 0.8 + 0.1

        # Tick mark
        ax.vlines(x_pos, 0.45, 0.55, colors='black', linewidth=1)

        # Rank number above
        ax.text(x_pos, 0.6, f'{rank:.2f}', ha='center', va='bottom', fontsize=10)

        # Method name below (alternating sides for readability)
        if i % 2 == 0:
            ax.text(x_pos, 0.35, method, ha='center', va='top', fontsize=9, rotation=45)
        else:
            ax.text(x_pos, 0.2, method, ha='center', va='top', fontsize=9, rotation=45)

    # Draw CD bar
    cd_normalized = cd / (n_methods - 1) * 0.8
    ax.hlines(0.75, 0.1, 0.1 + cd_normalized, colors='red', linewidth=2)
    ax.text(0.1 + cd_normalized / 2, 0.8, f'CD = {cd:.2f}', ha='center', va='bottom',
           fontsize=10, color='red', fontweight='bold')

    # Draw cliques (groups that are not significantly different)
    # Methods within CD of each other are connected
    sorted_ranks = sorted_methods.values
    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if sorted_ranks[j] - sorted_ranks[i] < cd:
                x1 = (sorted_ranks[i] - 1) / (n_methods - 1) * 0.8 + 0.1
                x2 = (sorted_ranks[j] - 1) / (n_methods - 1) * 0.8 + 0.1
                y = 0.9 + (j - i) * 0.03
                ax.hlines(y, x1, x2, colors='gray', linewidth=2, alpha=0.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    ax.set_title(f'Critical Difference Diagram ({metric})\n'
                f'Friedman p={p_value:.4f}, CD={cd:.2f} (α={alpha})',
                fontsize=12, fontweight='bold')

    plt.tight_layout()

    if output_dir:
        save_figure(fig, f"critical_difference_{metric.lower()}", Path(output_dir))

    return fig


def generate_all_figures(all_results, output_dir="./publication_figures"):
    """
    Generate all publication-quality figures.

    Args:
        all_results: List of result dictionaries from benchmark
        output_dir: Directory to save figures (default: ./publication_figures)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    setup_plot_style()

    if len(all_results) == 0:
        print("ERROR: No results found. Please run the benchmark first.")
        return

    # Create results DataFrame
    results_df = pd.DataFrame([{
        'Dataset': r['dataset'],
        'Method': r['method'],
        'N_Features': r.get('n_features', 0),
        'N_Drifts': len(r.get('drift_positions', [])),
        'Intensity': r.get('intensity', 0),
        'TP': r.get('tp', 0),
        'FP': r.get('fp', 0),
        'FN': r.get('fn', 0),
        'Precision': r.get('precision', 0.0),
        'Recall': r.get('recall', 0.0),
        'F1': r.get('f1_score', 0.0),
        'MTTD': r.get('mttd', np.nan) if r.get('mttd') != float('inf') else np.nan,
        'Detection_Rate': r.get('detection_rate', 0.0),
        'N_Detections': r.get('n_detections', 0),
        'Runtime_s': r.get('runtime_s', 0.0)
    } for r in all_results])

    drift_results = results_df[results_df['N_Drifts'] > 0].copy()

    if len(drift_results) == 0:
        print("ERROR: No drift datasets found in results.")
        return

    print("=" * 70)
    print("GENERATING THESIS FIGURES (9 total)")
    print("=" * 70)

    # ========================================================================
    # FIGURE 1: F1-Score Heatmap
    # ========================================================================
    print("\n[1/9] F1-Score Heatmap...")

    f1_pivot = drift_results.pivot_table(values='F1', index='Method', columns='Dataset', aggfunc='mean')
    f1_pivot['_avg'] = f1_pivot.mean(axis=1)
    f1_pivot = f1_pivot.sort_values('_avg', ascending=False).drop('_avg', axis=1)

    # Sort datasets by category
    dataset_cats = {col: categorize_dataset(col) for col in f1_pivot.columns}
    sorted_cols = sorted(f1_pivot.columns, key=lambda x: (dataset_cats[x], x))
    f1_pivot = f1_pivot[sorted_cols]

    fig, ax = plt.subplots(figsize=(max(12, len(sorted_cols) * 1.5), max(6, len(f1_pivot) * 0.7)))
    sns.heatmap(f1_pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'F1-Score', 'shrink': 0.8},
                linewidths=0.5, linecolor='white', annot_kws={'fontsize': 11, 'weight': 'bold'}, ax=ax)
    ax.set_title('F1-Score by Method and Dataset', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detection Method', fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig, "figure_1_f1_heatmap", output_dir)
    plt.close()

    # ========================================================================
    # FIGURE 2: Precision Heatmap
    # ========================================================================
    print("\n[2/9] Precision Heatmap...")

    prec_pivot = drift_results.pivot_table(values='Precision', index='Method', columns='Dataset', aggfunc='mean')
    prec_pivot['_avg'] = prec_pivot.mean(axis=1)
    prec_pivot = prec_pivot.sort_values('_avg', ascending=False).drop('_avg', axis=1)
    prec_pivot = prec_pivot[sorted_cols]

    fig, ax = plt.subplots(figsize=(max(12, len(sorted_cols) * 1.5), max(6, len(prec_pivot) * 0.7)))
    sns.heatmap(prec_pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Precision', 'shrink': 0.8},
                linewidths=0.5, linecolor='white', annot_kws={'fontsize': 11, 'weight': 'bold'}, ax=ax)
    ax.set_title('Precision by Method and Dataset', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detection Method', fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig, "figure_2_precision_heatmap", output_dir)
    plt.close()

    # ========================================================================
    # FIGURE 3: Recall Heatmap
    # ========================================================================
    print("\n[3/9] Recall Heatmap...")

    recall_pivot = drift_results.pivot_table(values='Recall', index='Method', columns='Dataset', aggfunc='mean')
    recall_pivot['_avg'] = recall_pivot.mean(axis=1)
    recall_pivot = recall_pivot.sort_values('_avg', ascending=False).drop('_avg', axis=1)
    recall_pivot = recall_pivot[sorted_cols]

    fig, ax = plt.subplots(figsize=(max(12, len(sorted_cols) * 1.5), max(6, len(recall_pivot) * 0.7)))
    sns.heatmap(recall_pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label': 'Recall', 'shrink': 0.8},
                linewidths=0.5, linecolor='white', annot_kws={'fontsize': 11, 'weight': 'bold'}, ax=ax)
    ax.set_title('Recall by Method and Dataset', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detection Method', fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig, "figure_3_recall_heatmap", output_dir)
    plt.close()

    # ========================================================================
    # FIGURE 4: MTTD Heatmap
    # ========================================================================
    print("\n[4/9] MTTD Heatmap...")

    mttd_pivot = drift_results.pivot_table(values='MTTD', index='Method', columns='Dataset', aggfunc='mean')
    mttd_pivot['_avg'] = mttd_pivot.mean(axis=1)
    mttd_pivot = mttd_pivot.sort_values('_avg', ascending=True).drop('_avg', axis=1)
    mttd_pivot = mttd_pivot[sorted_cols]

    fig, ax = plt.subplots(figsize=(max(12, len(sorted_cols) * 1.5), max(6, len(mttd_pivot) * 0.7)))
    sns.heatmap(mttd_pivot, annot=True, fmt='.0f', cmap='RdYlGn_r',
                cbar_kws={'label': 'MTTD (samples)', 'shrink': 0.8},
                linewidths=0.5, linecolor='white', annot_kws={'fontsize': 11, 'weight': 'bold'}, ax=ax)
    ax.set_title('Mean Time To Detection (MTTD) by Method and Dataset', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Detection Method', fontsize=12, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig, "figure_4_mttd_heatmap", output_dir)
    plt.close()

    # ========================================================================
    # FIGURE 5: Method Comparison Bar Chart (with 95% CI)
    # ========================================================================
    print("\n[5/9] Method Comparison Bar Chart...")

    # Compute mean and 95% CI for each method
    method_stats = drift_results.groupby('Method').agg({
        'F1': ['mean', 'std', 'count'],
        'Precision': ['mean', 'std', 'count'],
        'Recall': ['mean', 'std', 'count']
    }).round(4)

    # Calculate 95% CI: mean ± t * (std / sqrt(n))
    def calc_ci(mean, std, n, confidence=0.95):
        if n <= 1:
            return 0
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        return t_val * (std / np.sqrt(n))

    method_summary = pd.DataFrame({
        'F1': method_stats[('F1', 'mean')],
        'F1_ci': [calc_ci(method_stats[('F1', 'mean')].iloc[i],
                         method_stats[('F1', 'std')].iloc[i],
                         method_stats[('F1', 'count')].iloc[i])
                 for i in range(len(method_stats))],
        'Precision': method_stats[('Precision', 'mean')],
        'Precision_ci': [calc_ci(method_stats[('Precision', 'mean')].iloc[i],
                                method_stats[('Precision', 'std')].iloc[i],
                                method_stats[('Precision', 'count')].iloc[i])
                        for i in range(len(method_stats))],
        'Recall': method_stats[('Recall', 'mean')],
        'Recall_ci': [calc_ci(method_stats[('Recall', 'mean')].iloc[i],
                             method_stats[('Recall', 'std')].iloc[i],
                             method_stats[('Recall', 'count')].iloc[i])
                     for i in range(len(method_stats))],
    }, index=method_stats.index)

    method_summary = method_summary.sort_values('F1', ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    colors = ['steelblue', 'forestgreen', 'coral']
    titles = ['F1-Score', 'Precision', 'Recall']
    metrics = ['F1', 'Precision', 'Recall']

    for ax, metric, color, title in zip(axes, metrics, colors, titles):
        y_pos = range(len(method_summary))
        bars = ax.barh(y_pos, method_summary[metric],
                      xerr=method_summary[f'{metric}_ci'],
                      color=color, edgecolor='black', alpha=0.8, capsize=4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(method_summary.index)
        ax.set_xlabel(f'{title} (95% CI)', fontsize=11, fontweight='bold')
        ax.set_title(f'{title} by Method', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (v, ci) in enumerate(zip(method_summary[metric], method_summary[f'{metric}_ci'])):
            ax.text(min(v + ci + 0.02, 1.05), i, f'{v:.3f}', va='center', fontsize=9)

    axes[1].set_ylabel('')
    axes[2].set_ylabel('')

    plt.suptitle('Method Performance Comparison (with 95% Confidence Intervals)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "figure_5_method_comparison", output_dir)
    plt.close()

    # ========================================================================
    # FIGURE 6: Detection Timeline
    # ========================================================================
    print("\n[6/9] Detection Timelines...")

    datasets = drift_results['Dataset'].unique()

    for dataset_name in datasets:
        dataset_results_list = [r for r in all_results if r['dataset'] == dataset_name]
        if not dataset_results_list:
            continue

        true_drifts = dataset_results_list[0].get('drift_positions', [])
        n_drifts = len(true_drifts)

        fig, ax = plt.subplots(figsize=(14, max(4, len(dataset_results_list) * 0.5)))

        for i, drift_pos in enumerate(true_drifts):
            ax.axvline(drift_pos, color='red', linestyle='--', linewidth=2,
                      alpha=0.7, label='True Drift' if i == 0 else '')

        for idx, result in enumerate(dataset_results_list):
            detections = result.get('detections', [])
            method = result['method']
            f1 = result.get('f1_score', 0)

            if detections:
                ax.scatter(detections, [idx]*len(detections), s=80, alpha=0.7,
                          label=f"{method} (F1={f1:.2f})")

        ax.set_yticks(range(len(dataset_results_list)))
        ax.set_yticklabels([r['method'] for r in dataset_results_list])
        ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Method', fontsize=11, fontweight='bold')
        ax.set_title(f'Detection Timeline - {dataset_name} ({n_drifts} drifts)',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_figure(fig, f"figure_6_timeline_{dataset_name}", output_dir)
        plt.close()

    # ========================================================================
    # FIGURE 7: Runtime Comparison
    # ========================================================================
    print("\n[7/9] Runtime Comparison...")

    runtime_summary = results_df.groupby('Method').agg({
        'Runtime_s': ['mean', 'std']
    }).round(4)
    runtime_summary.columns = ['Runtime_mean', 'Runtime_std']
    runtime_summary = runtime_summary.sort_values('Runtime_mean')

    fig, ax = plt.subplots(figsize=(12, 6))

    y_pos = range(len(runtime_summary))
    bars = ax.barh(y_pos, runtime_summary['Runtime_mean'],
                   xerr=runtime_summary['Runtime_std'],
                   color='steelblue', edgecolor='black', alpha=0.8, capsize=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(runtime_summary.index)
    ax.set_xlabel('Runtime (seconds)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Method', fontsize=11, fontweight='bold')
    ax.set_title('Average Runtime by Method', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for i, (mean, std) in enumerate(zip(runtime_summary['Runtime_mean'], runtime_summary['Runtime_std'])):
        ax.text(mean + std + 0.01, i, f'{mean:.3f}s', va='center', fontsize=9)

    plt.tight_layout()
    save_figure(fig, "figure_7_runtime_comparison", output_dir)
    plt.close()

    # ========================================================================
    # FIGURE 8: Speed-Accuracy Trade-off
    # ========================================================================
    print("\n[8/9] Speed-Accuracy Trade-off...")

    tradeoff = drift_results.groupby('Method').agg({
        'F1': 'mean', 'Runtime_s': 'mean'
    }).round(4)

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(tradeoff['Runtime_s'], tradeoff['F1'],
                        s=200, c=range(len(tradeoff)), cmap='tab10',
                        edgecolors='black', linewidths=1.5, alpha=0.8)

    for method, row in tradeoff.iterrows():
        ax.annotate(method, (row['Runtime_s'], row['F1']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    pareto_points = []
    max_f1 = -1
    for _, row in tradeoff.sort_values('Runtime_s').iterrows():
        if row['F1'] > max_f1:
            pareto_points.append(row)
            max_f1 = row['F1']

    if len(pareto_points) > 1:
        pareto_df = pd.DataFrame(pareto_points)
        ax.plot(pareto_df['Runtime_s'], pareto_df['F1'],
               'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')

    ax.set_xlabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Speed-Accuracy Trade-off', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')

    median_runtime = tradeoff['Runtime_s'].median()
    median_f1 = tradeoff['F1'].median()
    ax.axvline(median_runtime, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(median_f1, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_figure(fig, "figure_8_speed_accuracy_tradeoff", output_dir)
    plt.close()

    # ========================================================================
    # FIGURE 9: Critical Difference Diagram
    # ========================================================================
    print("\n[9/9] Critical Difference Diagram...")

    try:
        plot_critical_difference_diagram(drift_results, metric='F1', output_dir=output_dir)
    except Exception as e:
        print(f"  Warning: Could not generate CD diagram: {e}")
        print("  (Requires at least 3 methods and 3 datasets)")

    print("\n" + "=" * 70)
    print(f"All figures saved to: {output_dir.absolute()}")
    print("=" * 70)

