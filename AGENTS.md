<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# AGENTS.md - Coding Agent Instructions

This document provides guidance for AI coding agents working on this repository.

## Project Overview

**Domain:** Academic research on Concept Drift Detection in unsupervised data streams.
**Language:** Python 3.12+
**Purpose:** Benchmarking and comparing drift detection algorithms for a survey paper published in Frontiers in Artificial Intelligence (2024).

## Directory Structure (Refactored)

```
/
├── core/
│   ├── detectors/              # Centralized algorithm implementations
│   └── config.py               # Unified output configuration and LaTeX formatting
├── data/
│   ├── generators/             # Synthetic stream generators
│   ├── catalog.py              # Central dataset configuration
│   └── __init__.py             # Data package entry point
├── experiments/
│   ├── benchmark/              # Unified benchmarking framework
│   ├── monitoring/             # Real-time monitoring system (Kafka-based)
│   ├── shared/                 # Shared evaluation metrics and logic
│   └── visualizations/         # Consolidated plotting logic
├── results/                    # FIXED top-level output directory
│   ├── logs/                   # Execution logs
│   ├── plots/                  # Generated figures (.png)
│   ├── tables/                 # LaTeX tables (.tex)
│   └── raw/                    # Raw result data (.pkl, .json)
├── report/                     # LaTeX thesis/paper source
├── scripts/                    # System-level utility scripts
├── main.py                     # Entry point dispatcher
├── requirements.txt            # Python dependencies
└── run_benchmark.sh            # Benchmark runner script (calls main.py)
```

## Build/Run Commands

### Environment Setup
```bash
# Create and activate virtual environment
./scripts/setup_environment.sh
# OR manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Benchmark (Unified Dispatcher)
```bash
# Use the main dispatcher at root
python main.py benchmark [--quick]
python main.py compare
python main.py monitoring
python main.py plot

# Or use the legacy-compatible runner
./run_benchmark.sh
```

### Running Individual Tests
```bash
# Test a specific detector (from project root)
python -c "from core.detectors.shape_dd import shape; import numpy as np; X = np.random.randn(500, 5); result = shape(X, l1=50, l2=150, n_perm=100); print('Shape result:', result.shape)"

# Test dataset generation
python -c "from data.generators import generate_drift_stream; X, y, drifts, info = generate_drift_stream({'type': 'gaussian_shift', 'n_drift_events': 3}, total_size=1000); print(f'Generated: {X.shape}, drifts at {drifts}')"
```

### Building LaTeX Documents
```bash
./scripts/build_thesis.sh        # Build thesis
./scripts/build_presentation.sh  # Build slides
./scripts/docker_build.sh        # Docker-based LaTeX build
```

## Code Style Guidelines

### Import Organization
Use absolute package imports starting from the root:
```python
import numpy as np
from core.config import DETECTION_BENCHMARK_OUTPUTS
from core.detectors.se_cdt import SE_CDT
from data.catalog import get_enabled_datasets
```

### LaTeX Formatting
All tables exported to `results/tables/` MUST use the standardized format:
- `\begin{tabular}{|l|c|...|}` with vertical separators.
- `\hline` for all horizontal lines.
- Bold headers using `\textbf{Header}`.
- NO usage of the `booktabs` package.

### Output Locations
NEVER use hardcoded relative paths for outputs. ALWAYS import path mappings from `core.config`:
```python
from core.config import BENCHMARK_PROPER_OUTPUTS
output_path = BENCHMARK_PROPER_OUTPUTS["aggregate_table"]
```
