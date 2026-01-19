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
**Language:** Python 3.7+
**Purpose:** Benchmarking and comparing drift detection algorithms for a survey paper published in Frontiers in Artificial Intelligence (2024).

## Directory Structure

```
/
├── experiments/
│   ├── drift_detection_benchmark/    # Main benchmarking framework
│   │   ├── main.py                   # Entry point for benchmarks
│   │   ├── config.py                 # Hyperparameters and settings
│   │   ├── datasets/                 # Dataset generators and catalog
│   │   ├── evaluation/               # Detector evaluation logic
│   │   ├── analysis/                 # Statistical analysis & LaTeX export
│   │   ├── visualizations/           # Figure generation
│   │   ├── utils/                    # Utilities (windowing, logging)
│   │   └── publication_figures/      # Output directory
│   ├── drift_monitoring_system/      # Real-time Kafka-based monitoring
│   └── backup/                       # Standalone detector implementations
├── report/latex/                     # LaTeX thesis/paper source
├── requirements.txt                  # Python dependencies
├── setup_environment.sh              # Virtual environment setup
├── run_benchmark.sh                  # Benchmark runner script
└── build_thesis.sh                   # LaTeX compilation
```

## Build/Run Commands

### Environment Setup
```bash
# Create and activate virtual environment
./setup_environment.sh
# OR manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Benchmark
```bash
# Full benchmark (30 runs, all methods, all datasets)
./run_benchmark.sh

# Or run directly:
source .venv/bin/activate
python -m experiments.drift_detection_benchmark.main

# Or from the benchmark directory:
cd experiments/drift_detection_benchmark
python main.py
```

### Running Individual Tests

This project uses benchmarks rather than unit tests. To test individual components:

```bash
# Test a specific detector (from project root with PYTHONPATH set)
export PYTHONPATH="experiments/backup:."
python -c "
from experiments.backup.shape_dd import shape
import numpy as np
X = np.random.randn(500, 5)
result = shape(X, l1=50, l2=150, n_perm=100)
print('Shape result:', result.shape)
"

# Test dataset generation
python -c "
from experiments.drift_detection_benchmark.datasets import generate_drift_stream
X, y, drifts, info = generate_drift_stream({'type': 'gaussian_shift', 'n_drift_events': 3}, total_size=1000)
print(f'Generated: {X.shape}, drifts at {drifts}')
"
```

### Building LaTeX Documents
```bash
./build_thesis.sh        # Build thesis
./build_presentation.sh  # Build slides
./docker_build.sh        # Docker-based LaTeX build
```

## Code Style Guidelines

### Import Organization

Order imports in this sequence with blank lines between groups:
1. Standard library (`gc`, `sys`, `time`, `warnings`, `pathlib`)
2. Third-party packages (`numpy`, `scipy`, `sklearn`, `matplotlib`, `pandas`)
3. Local/package imports (relative `.` imports within package)

```python
import gc
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.metrics.pairwise import pairwise_kernels

from .config import CHUNK_SIZE, OVERLAP
from .datasets import generate_drift_stream
```

### Module-Level Docstrings

Every module should have a docstring describing its purpose:
```python
"""
Module description and purpose.

Contains:
    function_name - Brief description
    AnotherFunction - Brief description
"""
```

### Function Documentation

Use comprehensive docstrings with Parameters/Returns sections:
```python
def function_name(X, param1, param2=default):
    """
    Brief description of function purpose.
    
    Longer explanation if needed, including algorithm details,
    references to papers, and usage notes.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data stream
    param1 : int
        Description of param1
    param2 : str, default='value'
        Description of param2
    
    Returns:
    --------
    result : array-like, shape (n_samples, 3)
        [:, 0] - First column meaning
        [:, 1] - Second column meaning
        [:, 2] - Third column meaning
    
    References:
    -----------
    Author et al. (Year). "Paper Title." Journal Name.
    """
```

### Type Hints

Use type hints for function signatures, especially for public APIs:
```python
def calculate_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    ...

def generate_gaussian_shift_stream(
    total_size: int,
    n_drift_events: int,
    seed: int = 42,
    n_features: int = 10,
    shift_magnitude: float = 2.0
) -> tuple:
    ...
```

### Naming Conventions

- **Functions/methods:** `snake_case` (e.g., `generate_drift_stream`, `calculate_beta_score`)
- **Variables:** `snake_case` (e.g., `drift_positions`, `true_drifts`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `STREAM_SIZE`, `N_RUNS`, `CHUNK_SIZE`)
- **Classes:** `PascalCase` (rarely used in this codebase)
- **Private functions:** Leading underscore (e.g., `_helper_function`)

### Configuration Management

All tunable parameters are centralized in `config.py` files:
```python
# experiments/drift_detection_benchmark/config.py
STREAM_SIZE = 10000
N_RUNS = 30
RANDOM_SEEDS = [42 + i * 137 for i in range(N_RUNS)]
CHUNK_SIZE = 150
OVERLAP = 100
```

### Error Handling

- Suppress warnings globally where appropriate for clean output:
  ```python
  import warnings
  warnings.filterwarnings('ignore')
  ```
- Use try/except for per-window detector evaluation:
  ```python
  try:
      result = detector(window)
  except Exception as e:
      if verbose:
          print(f"Window {idx} failed: {e}")
  ```
- Validate inputs explicitly and raise `ValueError` for unknown configurations:
  ```python
  if dataset_type not in KNOWN_TYPES:
      raise ValueError(f"Unknown dataset_type: {dataset_type}")
  ```

### NumPy Array Patterns

- Use descriptive shapes in docstrings: `shape (n_samples, n_features)`
- Prefer vectorized operations over loops
- Use `np.einsum` for complex tensor operations:
  ```python
  stat = np.einsum('ij,ij->i', np.dot(W, K), W)
  ```

### Results Dictionary Pattern

Return comprehensive dictionaries with metadata:
```python
return {
    'method': method_name,
    'detections': detections,
    'stream_size': len(X),
    'runtime_s': end_time - start_time,
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score,
    # ... additional metrics
}
```

## Reproducibility Requirements

- **Always use fixed random seeds** for experiments
- Seeds are generated with prime spacing to avoid correlation:
  ```python
  RANDOM_SEEDS = [42 + i * 137 for i in range(N_RUNS)]
  ```
- Use `np.random.seed(seed)` at the start of generator functions
- Benchmark runs use N_RUNS = 30 for statistical validity

## Key Implementation Patterns

### Dual Import Pattern (for module/script execution)

```python
if __name__ == "__main__" and __package__ is None:
    # Running as script: python main.py
    from experiments.drift_detection_benchmark.config import SETTING
else:
    # Running as module: python -m experiments.drift_detection_benchmark.main
    from .config import SETTING
```

### Sliding Window Evaluation

All drift detectors use unified sliding window evaluation:
```python
windows, centers = create_sliding_windows(X, chunk_size, overlap)
for window, center_idx in zip(windows, centers):
    if detector(window) and (center_idx - last_detection >= COOLDOWN):
        detections.append(center_idx)
        last_detection = center_idx
```

### Adding backup modules to PYTHONPATH

Detector implementations in `experiments/backup/` require path manipulation:
```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backup')))
from shape_dd import shape, shape_mmdagg
```

## Dependencies

Core scientific stack (from requirements.txt):
- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- pandas >= 1.3.0
- statsmodels >= 0.13.0
- river >= 0.15.0 (for online learning and drift detection)

## Output Locations

- Benchmark results: `experiments/drift_detection_benchmark/publication_figures/`
- Generated figures: PNG files in publication_figures/
- LaTeX tables: `.tex` files in publication_figures/
- Compiled thesis: `report/latex/main.pdf`

## OW-MMD Integration (Bharti et al., ICML 2023)

### Overview

OW-MMD (Optimally-Weighted Maximum Mean Discrepancy) uses variance-optimal weights that upweight samples in sparse regions (distribution boundaries) and downweight samples in dense regions. This provides better sample efficiency, especially for small windows (n < 100).

**Key Benefits:**
- +3-5% better F1 score at 50-sample windows (moderate drift)
- Equivalent detection with 33% fewer samples
- Trade-off: 20x slower computation than standard MMD

### Key Files

| File | Description |
|------|-------------|
| `experiments/backup/ow_mmd.py` | Core OW-MMD implementation (cleaned up, well-documented) |
| `experiments/backup/shape_dd.py` | Original ShapeDD for reference |
| `experiments/backup/mmd.py` | Original MMD with permutation test |
| `experiments/drift_detection_benchmark/config.py` | Method configuration (WINDOW_METHODS list) |
| `experiments/drift_detection_benchmark/evaluation/window_detectors.py` | Detector evaluation logic |

### Main Functions in ow_mmd.py

```python
# Recommended for drift detection:
from ow_mmd import shape_ow_mmd, mmd_ow_permutation

# Option 1: Full ShapeDD with OW-MMD (recommended)
results = shape_ow_mmd(data_stream, l1=50, l2=150, n_perm=500)
drift_points = np.where(results[:, 2] < 0.05)[0]  # p-value < 0.05

# Option 2: Standalone OW-MMD test on a window
mmd_val, p_val = mmd_ow_permutation(window, n_perm=500)
is_drift = p_val < 0.05
```

### Available Detector Methods

The following OW-MMD methods are available in the benchmark framework:

| Method | Description | Speed | Use Case |
|--------|-------------|-------|----------|
| `MMD_OW` | OW-MMD with fixed threshold | Fast | Quick screening |
| `MMD_OW_Perm` | OW-MMD with permutation test | Slow | Fair comparison with MMD |
| `ShapeDD_OW_MMD` | Heuristic pattern detection + OW-MMD | Fast | Real-time monitoring |
| `ShapeDD_OW` | Proper ShapeDD algorithm with OW-MMD | Slow | Research/publication |

### Running OW-MMD Benchmarks

```bash
# Quick validation (5 runs, ~15 min)
cd experiments/drift_detection_benchmark
python benchmark_window_comparison.py

# Full benchmark (30 runs, ~2-3 hours)
# Edit benchmark_window_comparison.py: set QUICK_MODE = False
python benchmark_window_comparison.py
```

### Verified Results

OW-MMD provides sample efficiency improvements at small window sizes:

| Scenario | ShapeDD+MMD | ShapeDD+OW-MMD | Improvement |
|----------|-------------|----------------|-------------|
| 50 samples, 0.5σ drift | 85.9% | 91.2% | +5.3% |
| 50 samples, 0.75σ drift | 87.9% | 91.0% | +3.1% |
| 75 samples, 0.3σ drift | 88.3% | 82.3% | -6.0% (subtle drift) |

**Key insight:** OW-MMD helps with moderate drift (≥0.5σ) but hurts with very subtle drift (0.3σ) because the variance-reduction weighting emphasizes boundary regions that overlap with bulk at subtle drift levels.

### Theory: Why OW-MMD Works

From Bharti et al. (2023):

1. **Standard MMD** uses uniform weights → high variance in high-density regions
2. **OW-MMD** weights samples inversely to local kernel density:
   ```python
   k_sums = np.sum(K_off, axis=1)  # Local density
   inv_weights = 1.0 / np.sqrt(k_sums)  # Inverse weighting
   W = np.outer(inv_weights, inv_weights)
   ```
3. This upweights samples at distribution boundaries → better drift sensitivity

### Adding New OW-MMD Variants

To add a new OW-MMD variant to the benchmark:

1. **Implement the detector** in `experiments/backup/ow_mmd.py`
2. **Add to imports** in `experiments/drift_detection_benchmark/evaluation/window_detectors.py`:
   ```python
   from ow_mmd import your_new_function
   ```
3. **Add method case** in `evaluate_drift_detector()`:
   ```python
   elif method_name == 'Your_Method_Name':
       # Your detection logic
       trigger = ...
   ```
4. **Add to config** in `experiments/drift_detection_benchmark/config.py`:
   ```python
   WINDOW_METHODS = [
       ...
       'Your_Method_Name',
   ]
   ```

### Real-World Dataset: Electricity Semi-Synthetic

For evaluating OW-MMD on real-world data with known drift positions:

```python
from experiments.drift_detection_benchmark.datasets import generate_drift_stream

config = {'type': 'electricity_semisynthetic', 'n_drift_events': 10}
X, y, true_drifts, info = generate_drift_stream(config, total_size=10000)
# X: Real Elec2 features (7 features + 1 synthetic)
# true_drifts: Known drift positions (for evaluation)
```

This dataset uses real electricity market features with controlled synthetic drifts, enabling proper ground-truth evaluation.

### Thesis Writing Notes

**For the methodology section:**
- OW-MMD's variance-optimal weighting improves drift detection in resource-constrained scenarios
- Benefits most with small windows (≤50 samples) and moderate drift (≥0.5σ)
- The ability to achieve equivalent detection with 33% fewer samples enables earlier drift detection

**For the results section:**
- Report F1 scores at different window sizes
- Show trade-off: computational cost vs. sample efficiency
- Note limitation: not universally better (hurts at very subtle drift)

**Citation:**
```bibtex
@inproceedings{bharti2023optimally,
  title={Optimally-weighted Estimators of the Maximum Mean Discrepancy for Likelihood-Free Inference},
  author={Bharti, Ayush and Naslidnyk, Masha and Key, Oscar and Kaski, Samuel and Briol, Fran{\c{c}}ois-Xavier},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```
