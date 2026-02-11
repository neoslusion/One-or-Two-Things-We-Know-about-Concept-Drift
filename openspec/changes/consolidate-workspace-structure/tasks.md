# Tasks: Workspace Consolidation

## Phase 1: Structural Setup
- [x] Create new top-level directories (`core/`, `data/`, `results/`, `scripts/`)
- [x] Initialize `results/` subdirectories (`logs/`, `plots/`, `tables/`, `raw/`)
- [x] Move utility scripts from root to `scripts/` (leaving entry points)

## Phase 2: Code Migration
- [x] Move algorithms from `experiments/backup/` to `core/detectors/`
- [x] Move dataset logic from `experiments/drift_detection_benchmark/datasets/` to `data/`
- [x] Move monitoring system to `experiments/monitoring/`
- [x] Move benchmark logic to `experiments/benchmark/`
- [x] Consolidate all visualization scripts into `experiments/visualizations/` and rename them to follow a `plot_<description>.py` convention
- [x] Standardize plotting functions to accept a data source and an output path from the central config

## Phase 3: Integration & Refactoring
- [x] Create root-level `main.py` dispatcher
- [x] Update `core/config.py` with unified output paths and standardized LaTeX formatting (matching `se_cdt_content.tex`)
- [x] Disable PDF generation in all benchmarking and table export functions
- [x] Refactor all imports to use absolute package paths
- [x] Update `run_benchmark.sh` and other shell scripts to reflect new structure

## Phase 4: Report & Verification
- [x] Update LaTeX source paths for figures and tables
- [x] Verify full benchmark execution
- [x] Verify LaTeX compilation
- [x] Clean up redundant folders (e.g., `experiments/backup`, `experiments/experiments`)
