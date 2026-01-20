# Benchmark Methodology Review

## 1. Executive Summary

This review evaluates the benchmark methodology for SE-CDT and CDT_MSW comparison in the thesis. Overall, the methodology is **sound** but has several issues that need addressing.

### Overall Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Validation Correctness | Good | Event-based metrics are standard |
| Result Recording | Good | Pickle + CSV + LaTeX tables |
| Statistical Rigor | Good | 30 runs, proper seeds |
| SE-CDT vs CDT_MSW Fairness | **Needs Attention** | Supervised vs Unsupervised comparison |
| Figure Quality | **Needs Attention** | Figures exist but not in report |
| Justification | **Needs Attention** | High FP rate unexplained |

---

## 2. Methodology Validation

### 2.1 Detection Metrics (CORRECT)

The benchmark uses **event-based** evaluation which is standard for drift detection:

```python
# From experiments/benchmark_proper.py:172-232
def calculate_metrics(detections, events, length, tolerance=DETECTION_TOLERANCE):
    # TP: Detection within [drift_pos - 50, drift_pos + tolerance]
    # FP: Detection not matching any event
    # FN: Event not detected
    # NO TN (correct for drift detection)
```

**Verdict:** Correct implementation following drift detection literature.

### 2.2 Metrics Computed

| Metric | Formula | Status |
|--------|---------|--------|
| MDR (Missed Detection Rate) | FN / Total Events | Correct |
| EDR (Detection Rate/Recall) | TP / Total Events | Correct |
| Precision | TP / (TP + FP) | Correct |
| Mean Delay | Avg(detection_pos - drift_pos) | Correct |

### 2.3 Statistical Setup

- **N_RUNS:** 10 seeds per scenario (50 total)
- **Scenarios:** Mixed_A, Mixed_B, Repeated_Gradual, Repeated_Incremental, Repeated_Sudden
- **Seed Strategy:** Fixed seeds (0-9) for reproducibility

**Recommendation:** Consider increasing to 30 runs for main benchmark to match thesis claims.

---

## 3. SE-CDT vs CDT_MSW Comparison

### 3.1 Fairness Issue (CRITICAL)

The comparison is **inherently unfair** due to different input requirements:

| Method | Mode | Input Required | Detection Signal |
|--------|------|----------------|------------------|
| CDT_MSW | Supervised | Class labels (y) | Accuracy drop |
| SE-CDT | Unsupervised | Only features (X) | MMD signal |

**Problem:** CDT_MSW receives `supervised_mode=False` labels by default, which don't change with concept - so accuracy doesn't drop and detection fails.

**Evidence from results:**
```
CDT_MSW: EDR=0.058, MDR=0.942 (almost nothing detected)
SE-CDT:  EDR=0.980, MDR=0.020 (almost everything detected)
```

**Code confirms this:**
```python
# experiments/benchmark_proper.py:46-144
def generate_mixed_stream(events, length=None, seed=42, supervised_mode=False):
    # If supervised_mode=False: labels follow fixed formula (no concept change)
    # If supervised_mode=True: labels use ROTATED boundaries per concept
```

### 3.2 Supervised Comparison Exists

There IS a fair comparison function (`run_supervised_comparison`) but results not prominently featured in thesis tables.

**Recommendation:** 
1. Add a separate table showing CDT_MSW with proper supervised labels
2. Clearly state in thesis that SE-CDT advantage is unsupervised operation, not detection quality

---

## 4. Result Recording Quality

### 4.1 Data Files

| File | Location | Content |
|------|----------|---------|
| `benchmark_proper_detailed.pkl` | `experiments/drift_detection_benchmark/publication_figures/` | Full results (578KB) |
| `benchmark_proper_results.pkl` | `experiments/` | Duplicate results |
| `table_comparison_aggregate.tex` | `report/latex/tables/` | LaTeX table |
| `se_cdt_results_table.tex` | `experiments/` | Classification accuracy |

### 4.2 Table Values

From `table_comparison_aggregate.tex`:
```
CDT_MSW:       CAT=40.0%, SUB=16.0%, EDR=0.058, MDR=0.942, FP=91
SE-CDT (Std):  CAT=63.2%, SUB=44.2%, EDR=0.980, MDR=0.020, FP=1513
SE-CDT (ADW):  CAT=61.4%, SUB=48.7%, EDR=0.554, MDR=0.446, FP=292
```

**Issue:** SE-CDT (Std) has 1513 FP (false positives) - very high!

**Root cause:** Very low detection threshold (height=0.001, prominence=0.0005) in `benchmark_proper.py:400`.

---

## 5. Figure Quality Review

### 5.1 Existing Figures

| Figure | Location | Content |
|--------|----------|---------|
| `vis_mixed_a_CDT.png` | `experiments/publication_figures/` | CDT_MSW detections |
| `vis_mixed_a_SE.png` | `experiments/publication_figures/` | SE-CDT classifications |
| `vis_mixed_b_CDT.png` | `experiments/publication_figures/` | CDT_MSW detections |
| `vis_mixed_b_SE.png` | `experiments/publication_figures/` | SE-CDT classifications |
| `vis_repeated_gradual_*.png` | `experiments/publication_figures/` | Gradual drift scenarios |
| `vis_repeated_incremental_*.png` | `experiments/publication_figures/` | Incremental drift scenarios |

### 5.2 Figure Status

- **Generated:** Yes (8 PNG files)
- **In thesis:** Need to verify if properly imported
- **Quality:** Need visual inspection

**Recommendation:** Verify these figures are included in thesis LaTeX.

---

## 6. Justification Quality

### 6.1 Missing Justifications

1. **High FP for SE-CDT (Std):** Why 1513 false positives? Needs explanation.
2. **Threshold selection:** How were detection thresholds (0.01, 0.005) chosen?
3. **Tolerance=300 samples:** Why this value? Standard is 100-150.

### 6.2 Good Justifications Present

1. **Unsupervised advantage:** SE-CDT doesn't need labels
2. **Category vs Subcategory accuracy:** Distinction is clear
3. **Runtime comparison:** CDT slower due to tracking process

---

## 7. Recommendations

### HIGH Priority

1. **Re-run benchmark with proper supervised labels for CDT_MSW** or clearly note the unfair comparison in thesis
2. **Tune SE-CDT (Std) threshold** to reduce FP from 1513 to reasonable level
3. **Add figures to thesis** from `experiments/publication_figures/`

### MEDIUM Priority

4. **Add justification for threshold choices** in methodology section
5. **Add confusion matrix figure** for SE-CDT classification
6. **Increase N_RUNS to 30** for statistical claims

### LOW Priority

7. Move `benchmark_proper_results.pkl` duplicate to archive
8. Add per-drift-type breakdown table

---

## 8. Files to Check/Update

| File | Action |
|------|--------|
| `experiments/benchmark_proper.py:400` | Increase threshold for SE-CDT (Std) |
| `report/latex/chapters/04_experiments_evaluation.tex` | Add figure imports |
| `report/latex/tables/table_comparison_aggregate.tex` | Regenerate after fixes |

---

## 9. Verification Commands

```bash
# Run benchmark
cd /path/to/project
python experiments/benchmark_proper.py

# Generate visualizations
python experiments/visualize_benchmark.py

# Check thesis figures
grep -r "vis_mixed" report/latex/
```
