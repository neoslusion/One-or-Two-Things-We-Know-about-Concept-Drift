"""
Centralized Configuration for the Concept Drift Research Workspace.

This module defines all output paths, LaTeX formatting conventions, and 
global settings to ensure consistency across the entire project.

Directory Structure:
--------------------
results/
├── logs/                   # Execution logs
├── plots/                  # Generated figures (.png, .pdf)
├── tables/                 # LaTeX tables (.tex)
└── raw/                    # Raw data (.pkl, .json, .txt)
"""

from pathlib import Path
import os

# ============================================================================
# BASE DIRECTORIES
# ============================================================================

# Project root is the parent of the folder containing this file (core/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Results directory (Fixed top-level folder)
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
RAW_DIR = RESULTS_DIR / "raw"

# Create all directories if they don't exist
for directory in [RESULTS_DIR, LOGS_DIR, PLOTS_DIR, TABLES_DIR, RAW_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# OUTPUT PATH MAPPINGS
# ============================================================================

# Drift Detection Benchmark (Main suite)
DETECTION_BENCHMARK_OUTPUTS = {
    "results_pkl": RAW_DIR / "detection_results.pkl",
    "log_file": LOGS_DIR / "detection_log.txt",
    "methods_comparison": TABLES_DIR / "table_detection_methods.tex",
    "f1_scores_table": TABLES_DIR / "table_detection_f1_scores.tex",
    "runtime_table": TABLES_DIR / "table_detection_runtime.tex",
    "statistical_tests": TABLES_DIR / "table_statistical_tests.tex",
    
    # Visualization aliases used by main.py
    "f1_comparison": PLOTS_DIR / "fig_detection_f1_heatmap.png",
    "f1_heatmap": PLOTS_DIR / "fig_detection_f1_heatmap.png",
    "ranking_plot": PLOTS_DIR / "fig_detection_ranking.png",
}

# SE-CDT vs CDT_MSW Comparison (Benchmark Proper)
BENCHMARK_PROPER_OUTPUTS = {
    "results_pkl": RAW_DIR / "benchmark_proper_results.pkl",
    "log_file": LOGS_DIR / "benchmark_proper_log.txt",
    "aggregate_table": TABLES_DIR / "table_se_cdt_aggregate.tex",
    "supervised_comparison": TABLES_DIR / "table_supervised_comparison.tex",
    "detailed_by_type": TABLES_DIR / "table_se_cdt_by_type.tex",
}

# Prequential Accuracy Evaluation (Adaptive Learning System)
PREQUENTIAL_OUTPUTS = {
    "results_pkl": RAW_DIR / "prequential_results.pkl",
    "log_file": LOGS_DIR / "prequential_log.txt",
    "adaptation_summary": TABLES_DIR / "table_prequential_summary.tex",
    "metrics_by_drift": TABLES_DIR / "table_adaptation_by_drift.tex",
    
    # Specific drift type plots
    "sudden_accuracy": PLOTS_DIR / "fig_prequential_sudden.png",
    "gradual_accuracy": PLOTS_DIR / "fig_prequential_gradual.png",
    "incremental_accuracy": PLOTS_DIR / "fig_prequential_incremental.png",
    "recurrent_accuracy": PLOTS_DIR / "fig_prequential_recurrent.png",
    "mixed_accuracy": PLOTS_DIR / "fig_prequential_mixed.png",
}

# ============================================================================
# LATEX TABLE FORMATTING (Standardized to match se_cdt_content.tex)
# ============================================================================

LATEX_TABLE_CONFIG = {
    "use_booktabs": False,         # Use standard \hline and | separators
    "escape_underscores": True,
    "float_precision": 3,
    "percentage_precision": 1,
    "bold_best_values": True,
    "arrows": {
        "higher_better": "$\\uparrow$",
        "lower_better": "$\\downarrow$",
    },
}

def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text."""
    if not isinstance(text, str):
        return text
    text = text.replace("_", "\\_")
    text = text.replace("%", "\\%")
    text = text.replace("&", "\\&")
    text = text.replace("#", "\\#")
    return text

def format_metric(value: float, metric_type: str = "float") -> str:
    """Format metric values for LaTeX tables."""
    if metric_type == "percentage":
        return f"{value * 100:.{LATEX_TABLE_CONFIG['percentage_precision']}f}\\%"
    elif metric_type == "integer":
        return str(int(value))
    else:
        return f"{value:.{LATEX_TABLE_CONFIG['float_precision']}f}"

def generate_standard_table(headers, data, align=None):
    """
    Generate a standardized LaTeX table string.
    
    Args:
        headers: List of column headers
        data: List of lists (rows)
        align: Column alignment string (e.g., '|l|c|c|')
    """
    if align is None:
        align = "|" + "|".join(["c"] * len(headers)) + "|"
    
    lines = []
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\hline")
    
    # Header row
    header_line = " & ".join([f"\\textbf{{{escape_latex(h)}}}" for h in headers])
    lines.append(header_line + " \\\\")
    lines.append("\\hline")
    
    # Data rows
    for row in data:
        formatted_row = " & ".join([str(val) for val in row])
        lines.append(formatted_row + " \\\\")
        
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    
    return "\n".join(lines)
