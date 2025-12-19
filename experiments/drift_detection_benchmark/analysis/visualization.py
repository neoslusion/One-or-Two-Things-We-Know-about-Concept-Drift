"""
Visualization module for publication-quality figures.

Contains functions for generating:
- F1/Precision/Recall/MTTD heatmaps
- Method comparison bar charts
- Detection timelines
- Runtime comparison
- Speed-accuracy trade-off plots
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


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
    print("GENERATING THESIS FIGURES")
    print("=" * 70)

    # ========================================================================
    # FIGURE 1: F1-Score Heatmap
    # ========================================================================
    print("\n[1/8] F1-Score Heatmap...")

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
    print("\n[2/8] Precision Heatmap...")

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
    print("\n[3/8] Recall Heatmap...")

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
    print("\n[4/8] MTTD Heatmap...")

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
    # FIGURE 5: Method Comparison Bar Chart
    # ========================================================================
    print("\n[5/8] Method Comparison Bar Chart...")

    method_summary = drift_results.groupby('Method').agg({
        'F1': 'mean', 'Precision': 'mean', 'Recall': 'mean'
    }).round(3).sort_values('F1', ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ['steelblue', 'forestgreen', 'coral']
    titles = ['F1-Score', 'Precision', 'Recall']

    for ax, col, color, title in zip(axes, ['F1', 'Precision', 'Recall'], colors, titles):
        method_summary[col].plot(kind='barh', ax=ax, color=color, edgecolor='black', alpha=0.8)
        ax.set_xlabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f'{title} by Method', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(method_summary[col]):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)

    axes[1].set_ylabel('')
    axes[2].set_ylabel('')

    plt.suptitle('Method Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, "figure_5_method_comparison", output_dir)
    plt.close()

    # ========================================================================
    # FIGURE 6: Detection Timeline
    # ========================================================================
    print("\n[6/8] Detection Timelines...")

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
    print("\n[7/8] Runtime Comparison...")

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
    print("\n[8/8] Speed-Accuracy Trade-off...")

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

    print("\n" + "=" * 70)
    print(f"All figures saved to: {output_dir.absolute()}")
    print("=" * 70)

