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
import warnings

def setup_plot_style():
    """Set up publication-quality plot style for thesis."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'legend.framealpha': 0.9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })
    sns.set_palette('colorblind')  # Accessible color palette


def categorize_dataset(name):
    """Categorize dataset by drift type."""
    name_lower = name.lower()
    if 'gradual' in name_lower or 'circles' in name_lower:
        return 'B_Gradual'
    elif 'rbfblips' in name_lower:
        return 'A_Sudden'
    elif 'rbf' in name_lower:
        return 'C_Incremental'
    elif 'electricity' in name_lower or 'covertype' in name_lower:
        return 'D_Real-World'
    elif 'none' in name_lower:
        return 'E_Stationary'
    else:
        return 'A_Sudden'


def save_figure(fig, name, output_dir):
    """Save figure in PNG format only (for thesis)."""
    filepath = output_dir / f"{name}.png"
    fig.savefig(filepath, dpi=300, bbox_inches='tight', format='png',
                facecolor='white', edgecolor='none')
    print(f"  Saved: {name}.png")


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
    # ===== Validation =====
    required_cols = ['Method', 'Dataset', metric]
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    if results_df[metric].isna().any():
        warnings.warn("NaN values found, excluding affected rows")
        results_df = results_df.dropna(subset=[metric])

    # ===== Data Preparation =====
    pivot = results_df.pivot_table(
        values=metric, 
        index='Dataset', 
        columns='Method', 
        aggfunc='mean'
    )
    
    # Drop any columns/rows with NaN after pivot
    pivot = pivot.dropna(axis=1, how='any')

    n_datasets = len(pivot)
    n_methods = len(pivot.columns)

    if n_methods < 2:
        raise ValueError("Need at least 2 methods to compare")
    if n_datasets < 2:
        raise ValueError("Need at least 2 datasets for Friedman test")

    # ===== Compute Ranks =====
    # Higher metric value = better = lower rank (rank 1 is best)
    ranks = pivot.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean()

    # ===== Friedman Test =====
    stat, p_value = stats.friedmanchisquare(
        *[pivot[col].values for col in pivot.columns]
    )

    # ===== Critical Difference (Nemenyi) =====
    q_alpha_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
        12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
        17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544
    }
    
    if n_methods > 20:
        # Approximation for large k
        q_alpha = 2.569 + 0.1 * (n_methods - 4)  # Rough approximation
        warnings.warn(f"Using approximated q_alpha for {n_methods} methods")
    else:
        q_alpha = q_alpha_table.get(n_methods, 2.8)
    
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))

    # ===== Sort Methods by Average Rank =====
    sorted_methods = avg_ranks.sort_values()
    sorted_names = sorted_methods.index.tolist()
    sorted_ranks = sorted_methods.values

    # ===== Find Cliques =====
    def find_cliques(ranks, cd):
        """Find groups of methods that are not significantly different."""
        n = len(ranks)
        cliques = []
        
        for i in range(n):
            j = i + 1
            while j < n and ranks[j] - ranks[i] < cd:
                j += 1
            if j - i > 1:
                cliques.append((i, j - 1))
        
        # Remove cliques that are subsets of others
        final_cliques = []
        for c in cliques:
            is_subset = False
            for other in cliques:
                if other != c and other[0] <= c[0] and c[1] <= other[1]:
                    is_subset = True
                    break
            if not is_subset:
                final_cliques.append(c)
        
        return final_cliques

    cliques = find_cliques(sorted_ranks, cd)

    # ===== Create Figure =====
    fig, ax = plt.subplots(figsize=(12, max(5, n_methods * 0.5)))

    # Axis line
    ax.hlines(0.5, 0.05, 0.95, colors='black', linewidth=1.5)

    # ===== Draw Tick Marks and Labels =====
    rank_to_x = lambda r: (r - 1) / (n_methods - 1) * 0.85 + 0.075

    for i, (method, rank) in enumerate(sorted_methods.items()):
        x_pos = rank_to_x(rank)
        
        # Tick mark
        ax.vlines(x_pos, 0.47, 0.53, colors='black', linewidth=1.5)
        
        # Rank number above axis
        ax.text(x_pos, 0.56, f'{rank:.2f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
        
        # Method name below axis (rotated)
        ax.text(x_pos, 0.44, method, ha='right', va='top', fontsize=9, 
                rotation=45, rotation_mode='anchor')

    # ===== Draw CD Bar =====
    cd_x_length = cd / (n_methods - 1) * 0.85
    cd_start = 0.075
    ax.hlines(0.72, cd_start, cd_start + cd_x_length, colors='red', linewidth=3)
    ax.vlines(cd_start, 0.70, 0.74, colors='red', linewidth=2)
    ax.vlines(cd_start + cd_x_length, 0.70, 0.74, colors='red', linewidth=2)
    ax.text(cd_start + cd_x_length / 2, 0.76, f'CD = {cd:.2f}', 
            ha='center', va='bottom', fontsize=11, color='red', fontweight='bold')

    # ===== Draw Cliques =====
    clique_y_start = 0.82
    clique_y_step = 0.04
    
    for idx, (start, end) in enumerate(cliques):
        x1 = rank_to_x(sorted_ranks[start])
        x2 = rank_to_x(sorted_ranks[end])
        y = clique_y_start + idx * clique_y_step
        
        ax.hlines(y, x1, x2, colors='dimgray', linewidth=3, alpha=0.7)

    # ===== Final Adjustments =====
    ax.set_xlim(0, 1)
    ax.set_ylim(0, clique_y_start + len(cliques) * clique_y_step + 0.1)
    ax.axis('off')
    
    # Title with test results
    significance = "Significant" if p_value < alpha else "Not Significant"
    title = (f'Critical Difference Diagram ({metric})\n'
             f'Friedman p={p_value:.4f} ({significance}), CD={cd:.2f} (α={alpha})')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    # ===== Save Figure =====
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path / f"critical_difference_{metric.lower()}.png", 
                    dpi=150, bbox_inches='tight', facecolor='white')
        # PDF generation removed as per workspace consolidation requirements


    return fig, {
        'friedman_stat': stat,
        'p_value': p_value,
        'cd': cd,
        'avg_ranks': avg_ranks.to_dict(),
        'significant': p_value < alpha
    }


def generate_all_figures(all_results, output_dir="./publication_figures"):
    """
    Generate publication-quality figures organized by drift type.

    Output Structure:
        - Figure 1: Overall Method Ranking (bar chart with 95% CI)
        - Figure 2: Sudden Drift Performance (heatmap)
        - Figure 3: Gradual/Incremental Drift Performance (heatmap)
        - Figure 4: Real-world & Stationary Results (heatmap)
        - Figure 5: Speed-Accuracy Trade-off (scatter plot)
        - Figure 6: Runtime Comparison (bar chart)
        - Figure 7: Detection Timeline (combined for representative datasets)
        - Figure 8: Critical Difference Diagram

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
        'DriftType': categorize_dataset(r['dataset']),
        'N_Features': r.get('n_features', 0),
        'N_Drifts': len(r.get('drift_positions', [])),
        'TP': r.get('tp', 0),
        'FP': r.get('fp', 0),
        'FN': r.get('fn', 0),
        'Precision': r.get('precision', 0.0),
        'Recall': r.get('recall', 0.0),
        'F1': r.get('f1_score', 0.0),
        'MTTD': r.get('mttd', np.nan) if r.get('mttd') != float('inf') else np.nan,
        'Detection_Rate': r.get('detection_rate', 0.0),
        'Runtime_s': r.get('runtime_s', 0.0)
    } for r in all_results])

    # Split by drift type
    sudden_results = results_df[results_df['DriftType'] == 'A_Sudden'].copy()
    gradual_results = results_df[results_df['DriftType'].isin(['B_Gradual', 'C_Incremental'])].copy()
    realworld_results = results_df[results_df['DriftType'].isin(['D_Real-World', 'E_Stationary'])].copy()
    drift_results = results_df[results_df['N_Drifts'] > 0].copy()

    print("=" * 70)
    print("GENERATING THESIS FIGURES (Organized by Drift Type)")
    print("=" * 70)
    print(f"  Sudden drift datasets: {sudden_results['Dataset'].nunique()}")
    print(f"  Gradual/Incremental datasets: {gradual_results['Dataset'].nunique()}")
    print(f"  Real-world/Stationary datasets: {realworld_results['Dataset'].nunique()}")

    # ========================================================================
    # FIGURE 1: Overall Method Ranking (Bar Chart with 95% CI)
    # ========================================================================
    print("\n[1/8] Overall Method Ranking...")

    method_stats = drift_results.groupby('Method').agg({
        'F1': ['mean', 'std', 'count'],
        'Precision': ['mean'],
        'Recall': ['mean']
    }).round(4)

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
        'Recall': method_stats[('Recall', 'mean')],
    }, index=method_stats.index)
    method_summary = method_summary.sort_values('F1', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(method_summary))
    colors = plt.cm.RdYlGn(method_summary['F1'] / method_summary['F1'].max())
    
    bars = ax.barh(y_pos, method_summary['F1'],
                   xerr=method_summary['F1_ci'],
                   color=colors, edgecolor='black', alpha=0.9, capsize=4)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_summary.index, fontsize=10)
    ax.set_xlabel('F1-Score (with 95% CI)', fontsize=11, fontweight='bold')
    ax.set_title('Overall Method Ranking', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    
    for i, (v, ci) in enumerate(zip(method_summary['F1'], method_summary['F1_ci'])):
        ax.text(min(v + ci + 0.02, 0.95), i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, "figure_1_overall_ranking", output_dir)
    plt.close()

    # ========================================================================
    # FIGURE 2: Sudden Drift Performance (Heatmap)
    # ========================================================================
    print("\n[2/8] Sudden Drift Performance...")

    if len(sudden_results) > 0 and sudden_results['Dataset'].nunique() > 0:
        f1_sudden = sudden_results.pivot_table(values='F1', index='Method', columns='Dataset', aggfunc='mean')
        f1_sudden['Mean'] = f1_sudden.mean(axis=1)
        f1_sudden = f1_sudden.sort_values('Mean', ascending=False)
        
        # Separate Mean column for display
        mean_col = f1_sudden[['Mean']]
        f1_sudden_datasets = f1_sudden.drop('Mean', axis=1)

        fig, ax = plt.subplots(figsize=(max(8, len(f1_sudden_datasets.columns) * 1.2), 
                                         max(5, len(f1_sudden_datasets) * 0.6)))
        
        sns.heatmap(f1_sudden_datasets, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                    cbar_kws={'label': 'F1-Score', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white', 
                    annot_kws={'fontsize': 10}, ax=ax)
        
        ax.set_title('Sudden Drift: F1-Score by Method and Dataset', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=11)
        ax.set_ylabel('Method', fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        save_figure(fig, "figure_2_sudden_drift", output_dir)
        plt.close()
    else:
        print("  Skipped: No sudden drift datasets")

    # ========================================================================
    # FIGURE 3: Gradual/Incremental Drift Performance (Heatmap)
    # ========================================================================
    print("\n[3/8] Gradual/Incremental Drift Performance...")

    if len(gradual_results) > 0 and gradual_results['Dataset'].nunique() > 0:
        f1_gradual = gradual_results.pivot_table(values='F1', index='Method', columns='Dataset', aggfunc='mean')
        f1_gradual['Mean'] = f1_gradual.mean(axis=1)
        f1_gradual = f1_gradual.sort_values('Mean', ascending=False)
        f1_gradual_datasets = f1_gradual.drop('Mean', axis=1)

        fig, ax = plt.subplots(figsize=(max(6, len(f1_gradual_datasets.columns) * 1.5), 
                                         max(5, len(f1_gradual_datasets) * 0.6)))
        
        sns.heatmap(f1_gradual_datasets, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
                    cbar_kws={'label': 'F1-Score', 'shrink': 0.8},
                    linewidths=0.5, linecolor='white', 
                    annot_kws={'fontsize': 10}, ax=ax)
        
        ax.set_title('Gradual/Incremental Drift: F1-Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=11)
        ax.set_ylabel('Method', fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        save_figure(fig, "figure_3_gradual_incremental_drift", output_dir)
        plt.close()
    else:
        print("  Skipped: No gradual/incremental drift datasets")

    # ========================================================================
    # FIGURE 4: Stationary Analysis (False Positives)
    # ========================================================================
    print("\n[4/8] Stationary Analysis (False Positives)...")

    # Filter stationary datasets (no drift)
    stationary_results = results_df[results_df['DriftType'] == 'E_Stationary'].copy()

    if len(stationary_results) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fp_stats = stationary_results.groupby('Method').agg({
            'FP': 'sum'
        }).sort_values('FP', ascending=True)
        
        y_pos = range(len(fp_stats))
        # Color coding: Green (0 FP), Orange (<5 FP), Red (High FP)
        colors = ['green' if fp == 0 else 'orange' if fp < 5 else 'red' for fp in fp_stats['FP']]
        
        bars = ax.barh(y_pos, fp_stats['FP'], color=colors, edgecolor='black', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(fp_stats.index)
        ax.set_xlabel('Total False Positives', fontsize=11, fontweight='bold')
        ax.set_title('Stationary Datasets: False Positive Analysis\n(Lower is Better)', fontsize=12, fontweight='bold')
        
        # Add value labels
        for i, fp in enumerate(fp_stats['FP']):
            ax.text(fp + 0.1, i, str(int(fp)), va='center', fontsize=10, fontweight='bold')
            
        plt.tight_layout()
        save_figure(fig, "figure_4_stationary_fp", output_dir)
        plt.close()
    else:
        print("  Skipped: No stationary datasets")

    # ========================================================================
    # FIGURE 5: Speed-Accuracy Trade-off
    # ========================================================================
    print("\n[5/8] Speed-Accuracy Trade-off...")

    tradeoff = drift_results.groupby('Method').agg({
        'F1': 'mean', 'Runtime_s': 'mean'
    }).round(4)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by method type
    method_colors = {
        'D3': 'gray', 'DAWIDD': 'gray', 'MMD': 'gray', 'KS': 'gray',
        'ShapeDD': 'blue', 'ShapeDD_SNR_Adaptive': 'blue',
        'MMD_OW': 'green', 'ShapeDD_OW_MMD': 'green'
    }
    
    colors = [method_colors.get(m, 'purple') for m in tradeoff.index]
    
    scatter = ax.scatter(tradeoff['Runtime_s'], tradeoff['F1'],
                        s=200, c=colors, edgecolors='black', linewidths=1.5, alpha=0.8)

    for method, row in tradeoff.iterrows():
        ax.annotate(method, (row['Runtime_s'], row['F1']),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, fontweight='bold')

    # Pareto frontier
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

    ax.set_xlabel('Runtime (seconds)', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax.set_title('Speed-Accuracy Trade-off\n(Upper-left is ideal)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')

    # Add quadrant lines
    median_runtime = tradeoff['Runtime_s'].median()
    median_f1 = tradeoff['F1'].median()
    ax.axvline(median_runtime, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(median_f1, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    save_figure(fig, "figure_5_speed_accuracy", output_dir)
    plt.close()

    # ========================================================================
    # FIGURE 6: Runtime Comparison
    # ========================================================================
    print("\n[6/8] Runtime Comparison...")

    runtime_summary = results_df.groupby('Method').agg({
        'Runtime_s': ['mean', 'std']
    }).round(4)
    runtime_summary.columns = ['Runtime_mean', 'Runtime_std']
    runtime_summary = runtime_summary.sort_values('Runtime_mean')

    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos = range(len(runtime_summary))
    bars = ax.barh(y_pos, runtime_summary['Runtime_mean'],
                   xerr=runtime_summary['Runtime_std'],
                   color='steelblue', edgecolor='black', alpha=0.8, capsize=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(runtime_summary.index)
    ax.set_xlabel('Runtime (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Average Runtime by Method', fontsize=12, fontweight='bold')

    for i, (mean, std) in enumerate(zip(runtime_summary['Runtime_mean'], runtime_summary['Runtime_std'])):
        ax.text(mean + std + 0.01, i, f'{mean:.2f}s', va='center', fontsize=9)

    plt.tight_layout()
    save_figure(fig, "figure_6_runtime", output_dir)
    plt.close()

    # ========================================================================
    # FIGURE 7: Detection Timeline (Representative example)
    # ========================================================================
    print("\n[7/8] Detection Timeline (Representative)...")

    # Pick one representative dataset from sudden drift
    sudden_datasets = sudden_results['Dataset'].unique()
    if len(sudden_datasets) > 0:
        representative = sudden_datasets[0]  # First sudden drift dataset
        
        dataset_results_list = [r for r in all_results if r['dataset'] == representative]
        if dataset_results_list:
            true_drifts = dataset_results_list[0].get('drift_positions', [])
            
            fig, ax = plt.subplots(figsize=(12, max(4, len(dataset_results_list) * 0.5)))

            # Plot true drifts
            for i, drift_pos in enumerate(true_drifts):
                ax.axvline(drift_pos, color='red', linestyle='--', linewidth=1.5,
                          alpha=0.8, label='True Drift' if i == 0 else '')

            # Plot detections
            colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_results_list)))
            for idx, (result, color) in enumerate(zip(dataset_results_list, colors)):
                detections = result.get('detections', [])
                method = result['method']
                f1 = result.get('f1_score', 0)

                if detections:
                    ax.scatter(detections, [idx]*len(detections), s=60, 
                              color=color, alpha=0.8, marker='o',
                              label=f"{method} (F1={f1:.2f})")

            ax.set_yticks(range(len(dataset_results_list)))
            ax.set_yticklabels([r['method'] for r in dataset_results_list], fontsize=9)
            ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
            ax.set_title(f'Detection Timeline: {representative}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8, ncol=2)
            ax.grid(alpha=0.3, axis='x')

            plt.tight_layout()
            save_figure(fig, "figure_7_timeline", output_dir)
            plt.close()

    # ========================================================================
    # FIGURE 8: Critical Difference Diagram
    # ========================================================================
    print("\n[8/8] Critical Difference Diagram...")

    try:
        plot_critical_difference_diagram(drift_results, metric='F1', output_dir=output_dir)
    except Exception as e:
        print(f"  Warning: Could not generate CD diagram: {e}")

    print("\n" + "=" * 70)
    print(f"All figures saved to: {output_dir.absolute()}")
    print("=" * 70)
    print("\nFigure Summary:")
    print("  1. Overall Method Ranking (bar chart)")
    print("  2. Sudden Drift Performance (heatmap)")
    print("  3. Gradual/Incremental Drift Performance (heatmap)")
    print("  4. Stationary Analysis (False Positives)")
    print("  5. Speed-Accuracy Trade-off (scatter)")
    print("  6. Runtime Comparison (bar chart)")
    print("  7. Detection Timeline (example)")
    print("  8. Critical Difference Diagram")
