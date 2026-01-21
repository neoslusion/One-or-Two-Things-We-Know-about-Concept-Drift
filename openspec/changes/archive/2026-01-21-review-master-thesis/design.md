# Review Process Design

## Audit Methodology
The audit will be performed by analyzing the LaTeX source files and the codebase. A report will be generated in `report/thesis_review_audit.md`.

### 1. Research Foundation Audit
- Check `00_introduction.tex` and `01_related_works.tex`.
- Verify clear problem statement, motivation, and research gaps.
- Check if hypotheses/questions are explicitly stated.

### 2. Methodology Audit
- Check `03_proposed_model.tex`.
- Verify logical flow of the proposed method.
- Check if ADW-MMD and SE-CDT are explained clearly and justified.

### 3. Data & Experiments Audit
- Check `04_experiments_evaluation.tex` and `experiments/` code.
- Verify data sources, preprocessing steps, and evaluation metrics.
- Check for data leakage risks (train/test split).

### 4. Results & Analysis Audit
- Check `04_experiments_evaluation.tex`.
- Verify if results support claims.
- Check statistical analysis rigor.
- Check if limitations are discussed.

### 5. Writing & Technical Audit
- Scan all files for consistency, clarity, and citations.
- Check equation correctness against code (`experiments/backup/mmd_variants.py`).
