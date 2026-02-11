# Change: Consolidate Workspace Structure

## Why
The project currently has a fragmented and messy directory structure with redundant output locations and scattered scripts, making it difficult to maintain and run benchmarks consistently.

## What Changes
- **Structural Cleanup**: Move core algorithms to `core/`, dataset logic to `data/`, and consolidate experiments into `experiments/benchmark/` and `experiments/monitoring/`.
- **Centralized Results**: Move all experimental outputs (logs, plots, tables, raw data) to a single top-level `results/` directory.
- **Standardized LaTeX Outputs**: Standardize LaTeX table generation to match the format in `se_cdt_content.tex` (using standard `tabular` with `\hline` and vertical bars, instead of `booktabs`). Disable PDF generation for tables and plots to focus on LaTeX assets.
- **Root Entry Point**: Implement a `main.py` dispatcher at the root to run all benchmarks and monitoring tasks.
- **Path Standardization**: Update all internal imports to absolute package paths and update LaTeX source to reference the new consolidated result paths.
- **Redundancy Removal**: Delete obsolete or duplicate folders like `experiments/backup` and `experiments/experiments`.

## Impact
- Affected specs: `workspace-consolidation` (new)
- Affected code: Almost all Python files (imports), shell scripts, and LaTeX source files.
