# Fix Table Generation for Consistency

## Summary
Update the benchmark analysis scripts to generate LaTeX tables that strictly follow the thesis/paper format (metrics, precision, layout).

## Rationale
Current generated tables differ slightly from the manual tables in the thesis (`report/latex/tables/`). Automation ensures future results (e.g., from new CPU runs) immediately produce publication-ready artifacts without manual editing.

## Changes
1.  **Metric Standardization:** Ensure export scripts output: Precision, Recall (EDR), F1, Delay, FP.
2.  **Formatting:** Match column headers, decimal precision (3 decimal places), and layout.
3.  **Filenames:** Ensure scripts overwrite the correct `.tex` files used by the thesis.
