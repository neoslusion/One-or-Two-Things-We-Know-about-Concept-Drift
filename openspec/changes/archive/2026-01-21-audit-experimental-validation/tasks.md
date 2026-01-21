# Tasks: Deep Experimental Validation

## 1. Preparation
- [x] 1.1 Create the audit report template at `report/experimental_validation_audit.md`.
- [x] 1.2 Extract list of all citations and baselines from `04_experiments_evaluation.tex`.

## 2. Literature Cross-Reference
- [x] 2.1 Verify Base Paper (ShapeDD) claims and equations.
- [x] 2.2 Verify Baseline Methods (DDM, EDDM, ADWIN, HDDM, etc.) description and setup.
- [x] 2.3 Verify Datasets (SEA, STAGGER, Elec2) characteristics and origins.
- [x] 2.4 Verify Evaluation Metrics against standard survey papers (Gama et al.).

## 3. Experimental Setup Audit
- [x] 3.1 Audit Data Pipeline in `experiments/drift_detection_benchmark/datasets/`.
- [x] 3.2 Audit Training/Evaluation code in `experiments/benchmark_proper.py` and `window_detectors.py`.
- [x] 3.3 Verify Random Seeds and Statistical Tests in `analysis` scripts.

## 4. Tables & Figures Audit
- [x] 4.1 Audit `table_comparison_aggregate.tex` and `se_cdt_results_table.tex`.
- [x] 4.2 Audit `vis_mixed_*.png` and other generated figures.
- [x] 4.3 Check consistency between tables, figures, and text.

## 5. Reporting
- [x] 5.1 Compile findings into `report/experimental_validation_audit.md`.
- [x] 5.2 Identify Critical, Important, and Recommended actions.
- [x] 5.3 (Optional) Apply quick fixes to LaTeX files if errors are obvious.
