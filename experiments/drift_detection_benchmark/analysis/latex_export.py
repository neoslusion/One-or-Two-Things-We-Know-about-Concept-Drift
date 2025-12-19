"""
LaTeX table export module for thesis/publication.

Contains functions for generating publication-ready LaTeX tables:
- Comprehensive performance summary
- F1-Score by dataset
- Runtime statistics
"""

from pathlib import Path
import pandas as pd
import numpy as np


def export_all_tables(all_results, stream_size, output_dir="./publication_figures"):
    """
    Export all LaTeX tables for thesis.

    Args:
        all_results: List of result dictionaries from benchmark
        stream_size: Size of each data stream
        output_dir: Directory to save tables (default: ./publication_figures)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

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

    drift_results_df = results_df[results_df['N_Drifts'] > 0].copy()

    if len(drift_results_df) == 0:
        print("ERROR: No drift datasets found in results.")
        return

    print("=" * 80)
    print("LATEX TABLE EXPORT")
    print("=" * 80)

    # ========================================================================
    # TABLE I: Comprehensive Performance Summary
    # ========================================================================
    print("\n[Table I] Comprehensive Performance Summary")

    method_stats = drift_results_df.groupby('Method').agg({
        'F1': ['mean', 'std'],
        'Precision': 'mean',
        'Recall': 'mean',
        'MTTD': 'mean',
        'TP': 'sum',
        'FP': 'sum',
        'FN': 'sum'
    }).round(4)

    pub_table = pd.DataFrame({
        'Method': method_stats.index,
        'F1': method_stats[('F1', 'mean')],
        'F1_std': method_stats[('F1', 'std')],
        'Precision': method_stats[('Precision', 'mean')],
        'Recall': method_stats[('Recall', 'mean')],
        'MTTD': method_stats[('MTTD', 'mean')].fillna(0).astype(int),
        'TP': method_stats[('TP', 'sum')].astype(int),
        'FP': method_stats[('FP', 'sum')].astype(int),
        'FN': method_stats[('FN', 'sum')].astype(int)
    })

    pub_table = pub_table.sort_values('F1', ascending=False).reset_index(drop=True)
    pub_table['F1_formatted'] = pub_table.apply(
        lambda row: f"${row['F1']:.3f} \\pm {row['F1_std']:.3f}$", axis=1
    )

    latex_table = pub_table[['Method', 'F1_formatted', 'Precision', 'Recall', 'MTTD', 'TP', 'FP', 'FN']].copy()
    latex_table.columns = ['Method', 'F1 ($\\mu \\pm \\sigma$)', 'Precision', 'Recall', 'MTTD', 'TP', 'FP', 'FN']

    for col in ['Precision', 'Recall']:
        latex_table[col] = latex_table[col].apply(lambda x: f"{x:.3f}")

    latex_output = latex_table.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'c' * (len(latex_table.columns) - 1),
        caption='Comprehensive drift detection performance. F1 is reported as mean $\\pm$ standard deviation across all datasets. MTTD = Mean Time To Detection (samples). TP/FP/FN = cumulative counts.',
        label='tab:comprehensive_performance',
        position='htbp'
    )

    latex_file = output_dir / "table_I_comprehensive_performance.tex"
    with open(latex_file, 'w') as f:
        f.write(latex_output)
    print(f"  Saved: {latex_file}")

    print("\nTable I Preview:")
    print(latex_table.to_string(index=False))

    # ========================================================================
    # TABLE II: Performance by Dataset
    # ========================================================================
    print("\n" + "-" * 80)
    print("[Table II] Performance by Dataset")

    f1_by_dataset = drift_results_df.pivot_table(
        values='F1', index='Method', columns='Dataset', aggfunc='mean'
    ).round(3)

    f1_by_dataset['Mean'] = f1_by_dataset.mean(axis=1).round(3)
    f1_by_dataset = f1_by_dataset.sort_values('Mean', ascending=False)

    latex_dataset = f1_by_dataset.to_latex(
        escape=False,
        column_format='l' + 'c' * len(f1_by_dataset.columns),
        caption='F1-Score by method and dataset. Best scores per dataset are highlighted.',
        label='tab:f1_by_dataset',
        position='htbp'
    )

    latex_file2 = output_dir / "table_II_f1_by_dataset.tex"
    with open(latex_file2, 'w') as f:
        f.write(latex_dataset)
    print(f"  Saved: {latex_file2}")

    print("\nTable II Preview:")
    print(f1_by_dataset.to_string())

    # ========================================================================
    # TABLE III: Runtime Statistics
    # ========================================================================
    print("\n" + "-" * 80)
    print("[Table III] Runtime Statistics")

    runtime_stats = results_df.groupby('Method').agg({
        'Runtime_s': ['mean', 'std', 'min', 'max']
    }).round(4)
    runtime_stats.columns = ['Mean (s)', 'Std (s)', 'Min (s)', 'Max (s)']

    # Handle NaN/inf values in throughput calculation
    throughput = stream_size / runtime_stats['Mean (s)']
    throughput = throughput.replace([np.inf, -np.inf], np.nan)
    runtime_stats['Throughput (samples/s)'] = throughput.fillna(0).round(0).astype(int)

    runtime_stats = runtime_stats.sort_values('Mean (s)')

    latex_runtime = runtime_stats.to_latex(
        escape=False,
        column_format='l' + 'c' * len(runtime_stats.columns),
        caption='Runtime statistics by detection method. Throughput = samples processed per second.',
        label='tab:runtime_stats',
        position='htbp'
    )

    latex_file3 = output_dir / "table_III_runtime_stats.tex"
    with open(latex_file3, 'w') as f:
        f.write(latex_runtime)
    print(f"  Saved: {latex_file3}")

    print("\nTable III Preview:")
    print(runtime_stats.to_string())

    print("\n" + "=" * 80)
    print(f"All LaTeX tables saved to: {output_dir.absolute()}")
    print("=" * 80)

