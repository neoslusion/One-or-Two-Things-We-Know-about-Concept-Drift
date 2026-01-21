# Revert GPU Optimization for Hardware Consistency

## Summary
Revert the GPU optimization for drift detection methods (`ShapeDD`, `MMD`) and enforce standard CPU execution across all methods.

## Rationale
Testing revealed that for small sliding windows ($N=150$), GPU execution introduces significant overhead (PCIe transfer + kernel launch), making it 30x slower than CPU (`0.01s` vs `0.35s`). To ensure consistent and optimal performance, all methods should run on the same hardware (CPU).

## Changes
1.  **Disable GPU Auto-Detection:** Force `HAS_TORCH = False` or remove the GPU logic in `mmd_variants.py`.
2.  **Revert Kernels:** Restore `sklearn.metrics.pairwise.rbf_kernel` usage in `shape_dd.py` and `mmd.py`.
3.  **Validation:** Verify that all benchmarks run efficiently on CPU.
