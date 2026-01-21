# Experimental Validation Audit Report

**Date:** 2026-01-21
**Reviewer:** Automated Agentic System
**Scope:** Deep Audit of Experimental Validity, Literature Cross-Reference, and Artifact Integrity

## 1. Literature Cross-Reference Validation

### 1.1 Base Paper (ShapeDD)
| Check | Status | Findings | Impact |
| :--- | :--- | :--- | :--- |
| Description Accuracy | ✅ Valid | Thesis correctly describes ShapeDD as MMD + convolution pattern detection. | Positive |
| Equation Match | ✅ Valid | Thesis formulas (Triangle Shape Property, MMD) match standard literature and code implementation in `shape_dd.py`. | Positive |
| Performance Claim | ✅ Valid | Claims of high detection accuracy for sudden drift are supported by reproduction results (>99% F1). | Positive |

### 1.2 Baseline Methods
| Baseline | Accuracy | Implementation | Hyperparameters | Fairness |
| :--- | :--- | :--- | :--- | :--- |
| D3 | ✅ Valid | Uses AUC-based drift detection as described in Gözüaçık et al. (2019). | Default | Fair |
| DAWIDD | ✅ Valid | Distance-based method implemented. | Default | Fair |
| MMD (Std) | ✅ Valid | Standard kernel two-sample test with permutation. | Perm=2500 | Fair |
| KS-Test | ✅ Valid | Standard Kolmogorov-Smirnov test. | Alpha=0.05 | Fair |
| **ADWIN** | ⚠️ Missing | Listed in `STREAMING_METHODS` config but commented out. Not actually run in main benchmark. | N/A | **Gap** |
| **DDM/EDDM** | ⚠️ Missing | Commented out in `config.py`. Only window methods are active. | N/A | **Gap** |

### 1.3 Datasets
| Dataset | Source Verified? | Characteristics Match? | Suitability |
| :--- | :--- | :--- | :--- |
| SEA | ✅ Yes | Matches Street et al. (2001) definition (variants 0-3). | Suitable for sudden drift |
| STAGGER | ✅ Yes | Matches Schlimmer (1987) rules. | Suitable for concept drift |
| Hyperplane | ✅ Yes | Matches Hulten et al. (2001) rotating hyperplane. | Suitable for incremental drift |
| Elec2 | ✅ Yes | Matches Harries (1999). Used as semi-real benchmark. | Standard in field |
| Sine1/2 | ✅ Yes | Matches Gama et al. (2004). | Standard synthetic |

## 2. Experimental Setup Deep Audit

### 2.1 Data Pipeline
- [x] **Preprocessing:** No normalization seen in `generators.py` for synthetic data (standard for synthetic). Elec2 uses raw values.
- [x] **Train/Test Split:** Unsupervised methods don't use train/test split. Evaluation is prequential/windowed. This is appropriate.
- [x] **Data Leakage Check:** Generators produce fresh data streams. Random seeds (`seed + seg_idx * 100`) ensure independent segments.

### 2.2 Training & Evaluation
- [x] **Protocol:** Sliding window evaluation (`evaluate_drift_detector`) with fixed `CHUNK_SIZE=150` and `OVERLAP=100`. Standard for window-based comparison.
- [x] **Seeds:** `RANDOM_SEEDS` list ensures 30 independent runs (`config.py`).
- [x] **Hyperparameter Tuning:** Fixed parameters (`SHAPE_L1=50`, `SHAPE_L2=150`) used across all datasets. No dataset-specific tuning observed (good for generalization claims).
- [x] **Statistical Tests:** Friedman and Nemenyi post-hoc tests are implemented correctly in `analysis/statistics.py`.

## 3. Tables & Figures Audit

### 3.1 Tables
| Table | Completeness | Correctness | Consistency |
| :--- | :--- | :--- | :--- |
| `table_comparison_aggregate.tex` | ✅ Complete | Shows CAT/SUB/EDR/MDR/FP. Metrics are standard. | Consistent with text. |
| `se_cdt_results_table.tex` | ⚠️ Issue | "Nhầm lẫn với Gradual?" in notes suggests uncertainty. Low accuracy for Gradual/Recurrent (0%). | Honest reporting, but results are poor. |

### 3.2 Figures
| Figure | Clarity | Correctness | Necessity |
| :--- | :--- | :--- | :--- |
| `critical_difference_f1.png` | ✅ High | Standard CD diagram. | Essential for statistical significance. |
| `benchmark_comparison.png` | ✅ High | Comparative bar charts. | Good summary. |

## 4. Prioritized Action Items

### 🚨 Critical
1. **Enable ADWIN/DDM Baselines:** The thesis text mentions comparing with ADWIN, DDM, EDDM, but they are commented out in `config.py` and not run. **Action:** Uncomment streaming methods or remove claims of comparison from text.

### ⚠️ Important
1. **Clarify ADW-MMD Naming:** Code uses `mmd_adw` but some text might still refer to `OW-MMD`. Ensure strict consistency (Task 2.1 in previous change addressed this, but double check).
2. **Explain Poor PCD Classification:** SE-CDT fails on Gradual (46%) and Recurrent (0%). Thesis should explicitly discuss *why* (e.g., "ADW variance reduction filters out gradual changes as noise").

### 💡 Recommended
1. **Add Sensitivity Analysis:** Plot F1 vs Window Size to show robustness.
2. **Normalize Real Data:** Ensure Elec2 features are scaled if using RBF kernel (scale sensitivity).
