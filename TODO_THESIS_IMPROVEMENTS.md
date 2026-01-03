# Thesis Improvement TODO List

This document outlines remaining tasks and improvements for the thesis based on a comprehensive review.

---

## HIGH PRIORITY (Should fix before defense)

### 1. ~~D3 Implementation Verification~~ ✅ DONE
- **Resolution:** D3 is working correctly! The F1=0.000 is NOT a bug.
- **Root Cause:** D3 can only detect **real/covariate drift** (P(X) changes), not **virtual/concept drift** (only P(Y|X) changes).
- **Evidence:** D3 achieves F1=0.998 on STAGGER (real drift) but F1=0.000 on SEA/Hyperplane (virtual drift only).
- **Thesis Update:** Added detailed explanation in Chapter 4 Section "D3: Phân tích sự khác biệt giữa Real Drift và Virtual Drift" with table comparing drift types.
- **Academic Value:** This is a methodological insight confirming discriminative detector limitations (Sethi & Kantardzic, 2017).

### 2. ~~Regenerate Publication Figures~~ ✅ DONE
- **Status:** User ran `./run_benchmark.sh` - figures/tables are up-to-date
- **Verification:** Chapter 4 uses `\input{}` to include auto-generated files directly

### 3. ~~Resolve Chapter 4 Structure~~ ✅ DONE
- **Resolution:** Removed old single-drift experiment sections (4.2-4.5)
- **Current state:** Chapter 4 now only contains the benchmark experiment results
- **Files modified:**
  - `report/latex/chapters/04_experiments_evaluation.tex` - removed ~380 lines
  - `report/latex/chapters/05_conclusion_future_work.tex` - updated conclusions

---

## MEDIUM PRIORITY (Strengthens the thesis)

### 4. Add ShapeDD_OW_MMD Runtime Explanation
- **Issue:** ShapeDD_OW_MMD takes 6.8s (vs 0.08s for KS), but no explanation why
- **Action Required:**
  - [ ] Add subsection in Chapter 3 explaining OW-MMD computational complexity
  - [ ] Explain that OW-MMD requires solving optimization problem for each window
  - [ ] Mention O(n²) complexity for kernel matrix computation
- **Location:** Chapter 3, Section 3.2 (after OW-MMD description)

### 5. Add More Real-World Dataset Results
- **Issue:** Only 1 real-world dataset (electricity_sorted) out of 7
- **Action Required:**
  - [ ] Consider adding: Weather, Covertype, or Airline datasets
  - [ ] Or acknowledge this limitation explicitly in "Hạn chế" section
  - [ ] Update abstract if new datasets are added
- **Estimated Effort:** 1-2 days if adding new datasets

### 6. Statistical Power Discussion
- **Issue:** Nemenyi test with only 8 methods × 7 datasets may have limited power
- **Action Required:**
  - [ ] Add note about statistical power limitations
  - [ ] Mention CD=3.969 means only large rank differences are significant
  - [ ] Consider using Wilcoxon signed-rank test for pairwise comparisons
- **Location:** Chapter 4, Section 4.8 (Literature Comparison)

### 7. Verify All Figure/Table References
- **Issue:** Some figures may not exist or paths may be wrong
- **Action Required:**
  - [ ] Run LaTeX compilation and check for missing figure warnings
  - [ ] Verify each figure path in Chapter 4 is correct
  - [ ] Check that all `\input{}` commands point to existing files
- **Command:** `cd report/latex && pdflatex main.tex`

---

## LOW PRIORITY (Nice to have)

### 8. Update Literature Review with 2024-2025 Papers
- **Action Required:**
  - [ ] Search for recent concept drift papers (2024-2025)
  - [ ] Add to Chapter 2 if highly relevant
  - [ ] Update `references.bib` with new citations
- **Suggested Search:** "concept drift detection 2024 benchmark"

### 9. Add Reproducibility Instructions
- **Action Required:**
  - [ ] Create `experiments/drift_detection_benchmark/README.md`
  - [ ] Document required packages and versions
  - [ ] Add step-by-step instructions to reproduce results
  - [ ] Include expected output examples

### 10. Code Documentation
- **Action Required:**
  - [ ] Add docstrings to key functions in `window_detectors.py`
  - [ ] Document the benchmark configuration in `config.py`
  - [ ] Add comments explaining the evaluation metrics

### 11. Add Confidence Intervals to More Results
- **Action Required:**
  - [ ] Add 95% CI to F1 scores in manual tables (not auto-generated)
  - [ ] Consider bootstrap confidence intervals for MTTD
- **Location:** Tables in Chapter 4 that are manually written

---

## OPTIONAL EXTENSIONS (Future work)

### 12. Implement Gradual Drift Experiments
- Extend evaluation to gradual, incremental, and recurrent drifts
- Framework already supports this via CDT_MSW
- See `drift-monitoring/adaptation_strategies.py`

### 13. Add Online Learning Comparison
- Compare frozen model + adaptation vs continuous online learning
- Use River framework for online learning baseline

### 14. Parameter Sensitivity Analysis
- Analyze sensitivity of L1, L2, n_perm parameters
- Create heatmaps showing F1 vs parameter values

---

## Quick Checklist Before Defense

- [ ] All F1 scores in text match auto-generated tables
- [ ] Dataset count is consistent (7 datasets everywhere)
- [ ] Method count is consistent (8 methods in Experiment 2)
- [ ] LaTeX compiles without errors
- [ ] All figures render correctly
- [ ] References are complete and properly formatted
- [ ] Abstract matches experimental results
- [ ] Conclusions align with actual findings

---

## Files Modified in Recent Session

1. `report/latex/abstract.tex` - Fixed dataset count
2. `report/latex/abstract_vietnamese.tex` - Fixed dataset count
3. `report/latex/chapters/03_proposed_model.tex` - Fixed method count
4. `report/latex/chapters/04_experiments_evaluation.tex` - Major fixes to statistics
5. `report/latex/chapters/05_conclusion_future_work.tex` - Fixed F1 scores

---

*Last updated: 2026-01-03*
*Generated by thesis review session*
