# Design: Workspace Consolidation

This document describes the target architecture and migration plan for consolidating the workspace.

## Target File Structure

```
/
├── core/                       # Core algorithms & mathematical foundations
│   ├── detectors/              # Individual detector implementations (MMD, ShapeDD, etc.)
│   └── utils/                  # Core math/signal processing utilities
├── data/                       # Dataset management
│   ├── generators/             # Synthetic stream generators
│   └── catalog.py              # Central dataset configuration
├── experiments/                # Research experiments
│   ├── benchmark/              # Unified benchmarking framework
│   ├── monitoring/             # Real-time monitoring system (Kafka-based)
│   ├── shared/                 # Shared evaluation metrics and logic
│   └── visualizations/         # Consolidated plotting logic
├── results/                    # FIXED output folder
│   ├── logs/                   # Execution logs
│   ├── plots/                  # Generated figures (.png, .pdf)
│   ├── tables/                 # LaTeX tables (.tex)
│   └── raw/                    # Pickled results (.pkl, .json)
├── report/                     # Thesis/paper source
├── scripts/                    # Infrastructure and build scripts
├── main.py                     # Entry point dispatcher
└── requirements.txt            # Dependencies
```

## Implementation Details

### 1. Centralized Output Configuration
The existing `experiments/output_config.py` will be moved to `core/config.py` and expanded. 
- It will enforce a standardized LaTeX table format: `\begin{tabular}{|l|...|}`, `\hline`, and `\textbf{Header}`.
- It will disable PDF generation for figures and tables by default, outputting `.tex` and `.png` (or only `.tex` for tables) to `results/`.

### 2. Entry Point Dispatcher (`main.py`)
A new root-level `main.py` will use `argparse` to allow running specific sub-tasks:
- `python main.py benchmark`: Runs the comprehensive detection benchmark.
- `python main.py compare`: Runs the SE-CDT vs CDT_MSW comparison.
- `python main.py monitoring`: Runs the Kafka monitoring simulation.
- `python main.py plot`: Re-generates all figures from saved results.

### 3. Path Resolution
All imports will be converted to absolute package imports (e.g., `from core.detectors.mmd import ...`). The root directory will be the primary `PYTHONPATH`.

### 4. Migration Strategy
1. Create new directory structure.
2. Move files according to the new mapping.
3. Update imports in all Python files.
4. Update paths in shell scripts and Dockerfiles.
5. Update LaTeX `\includegraphics` and `\input` paths.
6. Verify by running the full benchmark suite.
