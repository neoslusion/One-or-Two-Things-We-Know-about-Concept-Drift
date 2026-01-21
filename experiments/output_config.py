"""
Unified Output Configuration for All Benchmark Scripts.

This module centralizes all output paths and naming conventions to ensure
consistency across benchmark_proper.py, main.py, and evaluate_prequential.py.

Directory Structure:
--------------------
report/latex/
├── tables/                         # LaTeX tables for thesis
│   ├── table_se_cdt_aggregate.tex
│   ├── table_detection_methods.tex
│   ├── table_prequential_summary.tex
│   └── ...
└── figures/                        # Publication-quality figures
    ├── fig_detection_comparison.pdf
    ├── fig_prequential_accuracy.pdf
    └── ...

experiments/results/                # Raw results data
├── benchmark_proper/
│   ├── benchmark_proper_results.pkl
│   └── benchmark_proper_log.txt
├── detection_benchmark/
│   ├── detection_results.pkl
│   └── detection_log.txt
└── prequential_evaluation/
    ├── prequential_results.pkl
    └── prequential_log.txt
"""

from pathlib import Path

# ============================================================================
# BASE DIRECTORIES
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "report" / "latex"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# LaTeX output directories
LATEX_TABLES_DIR = REPORT_DIR / "tables"
LATEX_FIGURES_DIR = REPORT_DIR / "figures"

# Raw results directories (organized by benchmark type)
BENCHMARK_PROPER_DIR = RESULTS_DIR / "benchmark_proper"
DETECTION_BENCHMARK_DIR = RESULTS_DIR / "detection_benchmark"
PREQUENTIAL_EVAL_DIR = RESULTS_DIR / "prequential_evaluation"

# Create all directories
for directory in [
    LATEX_TABLES_DIR,
    LATEX_FIGURES_DIR,
    BENCHMARK_PROPER_DIR,
    DETECTION_BENCHMARK_DIR,
    PREQUENTIAL_EVAL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# BENCHMARK_PROPER.PY OUTPUTS (SE-CDT vs CDT_MSW Comparison)
# ============================================================================

BENCHMARK_PROPER_OUTPUTS = {
    # Raw data
    "results_pkl": BENCHMARK_PROPER_DIR / "benchmark_proper_results.pkl",
    "log_file": BENCHMARK_PROPER_DIR / "benchmark_proper_log.txt",
    
    # LaTeX tables (for se_cdt_content.tex)
    "aggregate_table": LATEX_TABLES_DIR / "table_se_cdt_aggregate.tex",
    "supervised_comparison": LATEX_TABLES_DIR / "table_supervised_comparison.tex",
    "detailed_by_type": LATEX_TABLES_DIR / "table_se_cdt_by_type.tex",
    
    # Legacy name compatibility (for existing tex files)
    "comparison_aggregate": LATEX_TABLES_DIR / "table_comparison_aggregate.tex",
}

# ============================================================================
# MAIN.PY OUTPUTS (Comprehensive Detection Benchmark)
# ============================================================================

DETECTION_BENCHMARK_OUTPUTS = {
    # Raw data
    "results_pkl": DETECTION_BENCHMARK_DIR / "detection_results.pkl",
    "log_file": DETECTION_BENCHMARK_DIR / "detection_log.txt",
    
    # LaTeX tables
    "methods_comparison": LATEX_TABLES_DIR / "table_detection_methods.tex",
    "f1_scores_table": LATEX_TABLES_DIR / "table_detection_f1_scores.tex",
    "runtime_table": LATEX_TABLES_DIR / "table_detection_runtime.tex",
    "statistical_tests": LATEX_TABLES_DIR / "table_statistical_tests.tex",
    
    # Legacy table names (for compatibility with existing tex files)
    "comprehensive_performance": LATEX_TABLES_DIR / "table_I_comprehensive_performance.tex",
    "f1_by_dataset": LATEX_TABLES_DIR / "table_II_f1_by_dataset.tex",
    "runtime_stats": LATEX_TABLES_DIR / "table_III_runtime_stats.tex",
}

# ============================================================================
# EVALUATE_PREQUENTIAL.PY OUTPUTS (Adaptive Learning System)
# ============================================================================

PREQUENTIAL_OUTPUTS = {
    # Raw data
    "results_pkl": PREQUENTIAL_EVAL_DIR / "prequential_results.pkl",
    "log_file": PREQUENTIAL_EVAL_DIR / "prequential_log.txt",
    
    # LaTeX tables
    "adaptation_summary": LATEX_TABLES_DIR / "table_prequential_summary.tex",
    "metrics_by_drift": LATEX_TABLES_DIR / "table_adaptation_by_drift.tex",
    
    # Figures stored as PNG in results directory (for debugging only, not for LaTeX)
    "sudden_accuracy": PREQUENTIAL_EVAL_DIR / "fig_prequential_sudden.png",
    "gradual_accuracy": PREQUENTIAL_EVAL_DIR / "fig_prequential_gradual.png",
    "incremental_accuracy": PREQUENTIAL_EVAL_DIR / "fig_prequential_incremental.png",
    "recurrent_accuracy": PREQUENTIAL_EVAL_DIR / "fig_prequential_recurrent.png",
    "mixed_accuracy": PREQUENTIAL_EVAL_DIR / "fig_prequential_mixed.png",
}

# ============================================================================
# LATEX TABLE FORMATTING CONVENTIONS
# ============================================================================

LATEX_TABLE_CONFIG = {
    # Escape underscores in method names for LaTeX
    "escape_underscores": True,
    
    # Number formatting
    "float_precision": 3,
    "percentage_precision": 1,
    
    # Table style
    "use_booktabs": True,  # Use \toprule, \midrule, \bottomrule
    "bold_best_values": True,
    
    # Arrow indicators for metrics
    "arrows": {
        "higher_better": "$\\uparrow$",  # F1, Precision, Recall, EDR
        "lower_better": "$\\downarrow$",  # MDR, FP, Runtime
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def escape_latex(text: str) -> str:
    """
    Escape special LaTeX characters in text.
    
    Args:
        text: Input string
        
    Returns:
        LaTeX-safe string with escaped characters
    """
    if not isinstance(text, str):
        return text
    
    # Escape underscores (most common issue)
    text = text.replace("_", "\\_")
    
    # Escape other special characters
    text = text.replace("%", "\\%")
    text = text.replace("&", "\\&")
    text = text.replace("#", "\\#")
    
    return text


def format_metric(value: float, metric_type: str = "float") -> str:
    """
    Format metric values for LaTeX tables.
    
    Args:
        value: Numeric value
        metric_type: Type of metric ('float', 'percentage', 'integer')
        
    Returns:
        Formatted string
    """
    if metric_type == "percentage":
        return f"{value * 100:.{LATEX_TABLE_CONFIG['percentage_precision']}f}\\%"
    elif metric_type == "integer":
        return str(int(value))
    else:
        return f"{value:.{LATEX_TABLE_CONFIG['float_precision']}f}"


def get_output_path(benchmark: str, output_key: str) -> Path:
    """
    Get standardized output path for a specific benchmark and output type.
    
    Args:
        benchmark: One of 'benchmark_proper', 'detection', 'prequential'
        output_key: Key from the corresponding OUTPUTS dictionary
        
    Returns:
        Path object
        
    Example:
        >>> get_output_path('benchmark_proper', 'aggregate_table')
        Path('report/latex/tables/table_se_cdt_aggregate.tex')
    """
    output_maps = {
        "benchmark_proper": BENCHMARK_PROPER_OUTPUTS,
        "detection": DETECTION_BENCHMARK_OUTPUTS,
        "prequential": PREQUENTIAL_OUTPUTS,
    }
    
    if benchmark not in output_maps:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    output_map = output_maps[benchmark]
    
    if output_key not in output_map:
        raise ValueError(f"Unknown output key '{output_key}' for benchmark '{benchmark}'")
    
    return output_map[output_key]


# ============================================================================
# COMPATIBILITY CONSTANTS (for gradual migration)
# ============================================================================

# Old paths that may still be referenced (deprecated)
DEPRECATED_PATHS = {
    "old_publication_figures": PROJECT_ROOT / "experiments" / "drift_detection_benchmark" / "publication_figures",
    "old_prequential_results": PROJECT_ROOT / "prequential_results",
}
