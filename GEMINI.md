# One or Two Things We Know about Concept Drift

## Project Overview
This repository contains the experimental code, real-time monitoring system, and LaTeX source for a survey and thesis on **Concept Drift**. The project focuses on detecting, locating, and explaining concept drift in unsupervised data streams. It includes a benchmarking framework for comparing various drift detection algorithms (ShapeDD, D3, MMD, etc.) and a prototype for real-time drift monitoring using Kafka.

## Directory Structure

*   **`experiments/`**: The core research component. Contains the benchmarking framework, dataset generators, and implementations of various drift detection algorithms.
    *   `drift_detection_benchmark/`: Main benchmarking suite with `main.py` entry point.
    *   `backup/`: Implementations of individual detectors (e.g., `shape_dd.py`, `mmd.py`, `d3.py`).
    *   `notebooks/`: Jupyter notebooks for analysis (`ConceptDrift_Pipeline.ipynb`, etc.).
*   **`drift-monitoring/`**: A real-time application for monitoring data streams for drift.
    *   `producer.py` / `consumer_stream.py`: Components for the streaming pipeline.
    *   `config.py`: Configuration for Kafka brokers, window sizes, and detection thresholds.
    *   `docker-compose.yml`: Container orchestration for the monitoring stack.
*   **`report/`**: LaTeX source code for the associated thesis/survey paper.
*   **`setup_environment.sh`**: Script to initialize the Python virtual environment.
*   **`run_benchmark.sh`**: Script to execute the drift detection benchmark.
*   **`build_thesis.sh`**: Script (and Docker support) to compile the LaTeX report.

## Key Components & Configuration

### Benchmarking (`experiments/drift_detection_benchmark/config.py`)
*   **Methods:** Supports window-based methods (D3, DAWIDD, MMD, KS, ShapeDD, etc.) and streaming methods.
*   **Reliability:** configured for statistical validation with `N_RUNS = 30` using distinct random seeds.
*   **Parameters:** Configurable window sizes (`CHUNK_SIZE`, `SHAPE_L1/L2`), permutation counts, and drift event generation.

### Monitoring (`drift-monitoring/config.py`)
*   **Infrastructure:** Designed to work with Kafka (`localhost:19092`).
*   **Logic:** Implements a sliding window approach for drift detection (`BUFFER_SIZE`, `CHUNK_SIZE`).
*   **Adaptation:** Includes parameters for model retraining (`ADAPTATION_DELAY`, `ADAPTATION_WINDOW`).

## Usage

### 1. Environment Setup
Initialize the Python environment (requires Python 3.7+):
```bash
./setup_environment.sh
```
*Note: This script creates a `.venv` and attempts to install dependencies. If `requirements.txt` is missing, ensure standard scientific packages (`numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`) are installed.*

### 2. Running Benchmarks
Execute the full drift detection benchmark:
```bash
./run_benchmark.sh
```
Results are saved to `experiments/drift_detection_benchmark/publication_figures/`.

### 3. Building the Thesis
Compile the LaTeX documentation:
```bash
./build_thesis.sh
# OR using Docker
./docker_build.sh
```

### 4. Real-time Monitoring
Deploy the monitoring stack (likely requires Docker/Kafka):
```bash
cd drift-monitoring
./deploy.sh
# or
docker-compose up
```

## Development Conventions
*   **Configuration:** All tunable parameters are isolated in `config.py` files within their respective modules.
*   **Reproducibility:** Experiments are designed with fixed seeds (`RANDOM_SEEDS`) to ensure reproducible results.
*   **Modular Design:** Detectors are implemented as standalone modules (mostly in `experiments/backup/`) and integrated via the benchmark framework.
