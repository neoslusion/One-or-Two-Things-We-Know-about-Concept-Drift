"""
Statistical analysis and aggregation module.

Standardized to match the formatting in se_cdt_content.tex:
- Uses vertical bars |l|c|
- Uses \hline for all separators
- No booktabs dependency
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib.pyplot as plt
import warnings
import os
from core.config import PLOTS_DIR, escape_latex

# Try to import scikit-posthocs for Nemenyi test
try:
    import scikit_posthocs as sp
    POSTHOCS_AVAILABLE = True
except ImportError:
    POSTHOCS_AVAILABLE = False
    print("Warning: scikit-posthocs not installed. Nemenyi test will be skipped.")

warnings.filterwarnings('ignore')

def calculate_ci_95(data):
    """Calculate 95% confidence interval."""
    if len(data) < 2:
        return np.nan, np.nan
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
    return ci[0], ci[1]

def run_statistical_analysis(all_results, n_runs):
    """Run comprehensive statistical analysis on benchmark results."""
    print(f"\n{'='*80}")
    print(f"STATISTICAL ANALYSIS: {len(all_results)} total experiments from {n_runs} runs")
    print(f"{'='*80}\n")

    df_results = pd.DataFrame(all_results)
    
    # Group by method and dataset
    aggregated = df_results.groupby(['method', 'dataset']).agg({
        'f1_score': ['mean', 'std', 'min', 'max', 'count'],
        'beta_score': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'mttd': ['mean', 'std'],
        'detection_rate': ['mean', 'std']
    }).round(3)

    ci_results = []
    for (method, dataset), group in df_results.groupby(['method', 'dataset']):
        f1_ci_low, f1_ci_high = calculate_ci_95(group['f1_score'].values)
        beta_ci_low, beta_ci_high = calculate_ci_95(group['beta_score'].values)

        ci_results.append({
            'method': method,
            'dataset': dataset,
            'f1_mean': group['f1_score'].mean(),
            'f1_std': group['f1_score'].std(),
            'f1_ci_low': f1_ci_low,
            'f1_ci_high': f1_ci_high,
            'beta_mean': group['beta_score'].mean(),
            'beta_std': group['beta_score'].std(),
            'beta_ci_low': beta_ci_low,
            'beta_ci_high': beta_ci_high,
            'n_runs': len(group)
        })

    df_ci = pd.DataFrame(ci_results)
    
    print("\nAggregated Results (sorted by F1 score):")
    print(aggregated.sort_values(('f1_score', 'mean'), ascending=False).head(20))

    return {
        'df_results': df_results,
        'df_ci': df_ci,
        'aggregated': aggregated,
        'wilcoxon_results': [],
        'friedman_result': None
    }

def print_results_summary(all_results, stream_size):
    """Print comprehensive results summary."""
    if len(all_results) == 0:
        print("No results to analyze.")
        return

    results_df = pd.DataFrame([{
        'Dataset': r['dataset'],
        'Method': r['method'],
        'N_Drifts': len(r.get('drift_positions', [])),
        'F1': r.get('f1_score', 0.0),
        'Precision': r.get('precision', 0.0),
        'Recall': r.get('recall', 0.0),
        'MTTD': r.get('mttd', np.nan) if r.get('mttd') != float('inf') else np.nan,
        'Runtime_s': r.get('runtime_s', 0.0)
    } for r in all_results])

    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    drift_results_df = results_df[results_df['N_Drifts'] > 0].copy()

    if len(drift_results_df) > 0:
        print("\nMETHOD RANKINGS (by F1-Score)")
        method_f1 = drift_results_df.groupby('Method')['F1'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        for rank, (method, row) in enumerate(method_f1.iterrows(), 1):
            print(f"  {rank}. {method:<30} F1 = {row['mean']:.3f} ± {row['std']:.3f}")

        print("\nRUNTIME ANALYSIS")
        runtime_summary = results_df.groupby('Method')['Runtime_s'].agg(['mean', 'std'])
        for method, row in runtime_summary.iterrows():
            throughput = stream_size / row['mean'] if row['mean'] > 0 else 0
            print(f"  {method:<25} {row['mean']:>8.3f}s (±{row['std']:.3f}s) | {throughput:>8.0f} samples/s")

    return results_df

def run_nemenyi_posthoc(df_results, metric='f1_score', output_dir=None):
    """Run Nemenyi post-hoc test after Friedman test."""
    if not POSTHOCS_AVAILABLE:
        return None

    pivot_df = df_results.pivot_table(index='dataset', columns='method', values=metric, aggfunc='mean')
    nemenyi_result = sp.posthoc_nemenyi_friedman(pivot_df.values)
    nemenyi_result.index = pivot_df.columns
    nemenyi_result.columns = pivot_df.columns

    ranks = pivot_df.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()

    k, n = len(pivot_df.columns), len(pivot_df)
    q_alpha_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}
    q_alpha = q_alpha_table.get(k, 3.0)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    if output_dir:
        nemenyi_result.to_csv(os.path.join(output_dir, 'nemenyi_pvalues.csv'))
        avg_ranks.to_csv(os.path.join(output_dir, 'average_ranks.csv'))

    return {
        'pvalue_matrix': nemenyi_result,
        'average_ranks': avg_ranks,
        'critical_difference': cd
    }

def plot_critical_difference_diagram(df_results, metric='f1_score', output_path=None):
    """Generate CD diagram."""
    if not POSTHOCS_AVAILABLE:
        return None

    pivot_df = df_results.pivot_table(index='dataset', columns='method', values=metric, aggfunc='mean')
    ranks = pivot_df.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()

    k, n = len(pivot_df.columns), len(pivot_df)
    q_alpha = 3.0 # Simple default
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    fig, ax = plt.subplots(figsize=(10, 4))
    # Simple ranking visualization for quick verification
    y_pos = np.arange(len(avg_ranks))
    ax.barh(y_pos, avg_ranks.values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(avg_ranks.index)
    ax.invert_yaxis()
    ax.set_xlabel('Average Rank (Lower is better)')
    ax.set_title(f'Method Ranks ({metric})')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    return fig

def generate_statistical_report(df_results, output_dir):
    """Generate comprehensive statistical report."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    nemenyi_results = run_nemenyi_posthoc(df_results, metric='f1_score', output_dir=output_dir)
    results['nemenyi'] = nemenyi_results

    cd_path = os.path.join(PLOTS_DIR, 'fig_detection_statistical_ranking.png')
    cd_fig = plot_critical_difference_diagram(df_results, metric='f1_score', output_path=cd_path)
    results['cd_figure'] = cd_fig

    if nemenyi_results:
        generate_statistical_latex_table(nemenyi_results, output_dir)
    
    return results

def generate_statistical_latex_table(nemenyi_results, output_dir):
    """Generate standardized LaTeX table for statistical results."""
    avg_ranks = nemenyi_results['average_ranks']
    cd = nemenyi_results['critical_difference']

    latex = r"""\begin{tabular}{|l|c|c|}
\hline
\textbf{Method} & \textbf{Average Rank} & \textbf{Overall Rank} \\
\hline
"""
    for i, (method, avg_rank) in enumerate(avg_ranks.items(), 1):
        latex += f"{escape_latex(method)} & {avg_rank:.3f} & {i} \\\\\n"

    latex += r"""\hline
\multicolumn{3}{|l|}{Critical Difference (CD) = """ + f"{cd:.3f}" + r"""} \\
\hline
\end{tabular}
"""
    # Use the filename defined in core.config for consistency
    output_path = os.path.join(output_dir, 'table_statistical_tests.tex')
    with open(output_path, 'w') as f:
        f.write(latex)
    return latex
