"""
Analysis module for benchmark results.
"""

from .statistics import run_statistical_analysis
from .visualization import generate_all_figures
from .latex_export import export_all_tables

__all__ = [
    "run_statistical_analysis",
    "generate_all_figures",
    "export_all_tables",
]

