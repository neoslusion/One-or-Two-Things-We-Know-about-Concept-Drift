# GPU Optimization and Benchmark Consistency

## Summary
Optimize the drift detection benchmark and monitoring system to support GPU acceleration and ensure consistency in data generation and evaluation metrics.

## Rationale
The previous CPU-based implementation was too slow for 30-run validation. Additionally, inconsistencies existed between the monitoring system (`evaluate_prequential.py`) and the main benchmark regarding data generators and thresholds.

## Changes
1.  **GPU Support:** Updated `mmd_variants.py`, `shape_dd.py`, and `mmd.py` to auto-detect CUDA and use PyTorch for heavy kernel computations.
2.  **Consistency:** Refactored `evaluate_prequential.py` to use shared `detection_metrics` and standardized `datasets.generators`.
3.  **Validation:** Created `run_full_validation.sh` for robust 30-run statistical validation.
