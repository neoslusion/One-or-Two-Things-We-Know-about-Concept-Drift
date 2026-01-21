# Tasks: Final Cleanup & Table Consistency

## 1. Remove GPU Residuals
- [ ] 1.1 `experiments/backup/mmd_variants.py`: Remove all `HAS_TORCH`, `DEVICE`, `torch` imports, and `_rbf_kernel_torch` code.
- [ ] 1.2 `experiments/backup/shape_dd.py`: Remove commented out GPU branches and imports.
- [ ] 1.3 `experiments/backup/mmd.py`: Remove GPU branches.

## 2. Fix Table Generation
- [ ] 2.1 Audit `experiments/drift_detection_benchmark/analysis/latex_export.py` to match thesis table format (e.g., column names, precision).
- [ ] 2.2 Ensure metrics exported are: Precision, Recall (EDR), F1, Delay, FP.
- [ ] 2.3 Verify `experiments/drift_detection_benchmark/publication_figures/table_comparison_aggregate.tex` matches paper style.

## 3. Validation
- [ ] 3.1 Run `run_gpu_test.py` (CPU) one last time to ensure no `NameError: torch` occurs.
- [ ] 3.2 Check generated LaTeX table snippet.
