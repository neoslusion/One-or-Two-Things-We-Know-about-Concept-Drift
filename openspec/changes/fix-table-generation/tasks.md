# Tasks: Fix Table Generation

## 1. Remove GPU Residuals (Cleanup)
- [ ] 1.1 `experiments/backup/mmd_variants.py`: Remove all `HAS_TORCH`, `DEVICE`, `torch` imports, and `_rbf_kernel_torch` code.
- [ ] 1.2 `experiments/backup/shape_dd.py`: Remove commented out GPU branches and imports.
- [ ] 1.3 `experiments/backup/mmd.py`: Remove GPU branches.

## 2. Fix Table Generation (Primary)
- [ ] 2.1 Audit `experiments/drift_detection_benchmark/analysis/latex_export.py` to match thesis table styling.
- [ ] 2.2 Update `generate_latex_table` function:
    - Use `\toprule`, `\midrule`, `\bottomrule` (booktabs style).
    - Ensure consistent decimal formatting (3 places).
    - Ensure bolding of best results is robust.
- [ ] 2.3 Apply this format to ALL generated tables:
    - `table_comparison_aggregate`
    - `se_cdt_results_table`
    - `table_runtime_stats` (if applicable)
- [ ] 2.4 Verify column headers match thesis convention (Method, Precision, Recall, F1, Delay, FP).

## 3. Validation
- [ ] 3.1 Run `run_gpu_test.py` (CPU) one last time to ensure no `NameError: torch` occurs.
- [ ] 3.2 Run benchmark analysis to generate sample tables.
- [ ] 3.3 Check generated LaTeX content against `report/latex/tables/`.
