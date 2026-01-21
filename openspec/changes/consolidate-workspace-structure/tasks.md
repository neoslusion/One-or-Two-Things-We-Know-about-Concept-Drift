# Tasks: Workspace Consolidation

## Phase 1: Structural Setup
- [ ] Create new top-level directories (`core/`, `data/`, `results/`, `scripts/`)
- [ ] Initialize `results/` subdirectories (`logs/`, `plots/`, `tables/`, `raw/`)
- [ ] Move utility scripts from root to `scripts/` (leaving entry points)

## Phase 2: Code Migration
- [ ] Move algorithms from `experiments/backup/` to `core/detectors/`
- [ ] Move dataset logic from `experiments/drift_detection_benchmark/datasets/` to `data/`
- [ ] Move monitoring system to `experiments/monitoring/`
- [ ] Move benchmark logic to `experiments/benchmark/`
- [ ] Consolidate all visualization scripts into `experiments/visualizations/` and rename them to follow a `plot_<description>.py` convention
- [ ] Standardize plotting functions to accept a data source and an output path from the central config

## Phase 3: Integration & Refactoring
- [ ] Create root-level `main.py` dispatcher
- [ ] Update `core/config.py` with unified output paths and standardized LaTeX formatting (matching `se_cdt_content.tex`)
- [ ] Disable PDF generation in all benchmarking and table export functions
- [ ] Refactor all imports to use absolute package paths
- [ ] Update `run_benchmark.sh` and other shell scripts to reflect new structure

## Phase 4: Report & Verification
- [ ] Update LaTeX source paths for figures and tables
- [ ] Verify full benchmark execution
- [ ] Verify LaTeX compilation
- [ ] Clean up redundant folders (e.g., `experiments/backup`, `experiments/experiments`)
