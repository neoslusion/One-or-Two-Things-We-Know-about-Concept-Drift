# Change: Technical Audit - Method Naming Accuracy in Thesis

## Why

This change addresses scientific integrity concerns identified during a thorough technical audit of the thesis document. The audit found several cases where method names used in the thesis do not accurately reflect the relationship between the implementation and the original papers being referenced. This is a **critical issue** for academic publication as it can lead to:

1. **Misrepresentation of contributions**: Claiming implementation of a published method when only a variant or inspired approach is used
2. **Reproducibility issues**: Other researchers cannot replicate results if the actual method differs from what is documented
3. **Citation integrity**: Incorrect attribution of techniques to original authors
4. **Examination risk**: Thesis committee members may identify these discrepancies during defense

## What Changes

### Critical Issues Found (Priority: HIGH)

1. **ADW-MMD naming and description** - **REQUIRES CLARIFICATION**
   - **Status**: Currently named "ADW-MMD" (Adaptive Density-Weighted MMD)
   - **Paper Reference**: Bharti et al. (2023) "Optimally-Weighted MMD" (OW-MMD)
   - **Issue**: Thesis correctly states "lấy cảm hứng từ" (inspired by) but later uses terminology suggesting full implementation
   - **Required Fix**: Explicitly clarify throughout that this is a **variant/adaptation**, not a faithful implementation. The original OW-MMD uses specific optimal weights derived analytically; the thesis implementation uses a heuristic approximation.

2. **ShapeDD_ADW_MMD vs actual implementation** - **CRITICAL MISMATCH**
   - **Thesis claim**: Combines "ShapeDD pattern detection" with ADW-MMD
   - **Code reality**: `shapedd_adw_mmd()` in `mmd_variants.py:240-337` uses **simplified heuristic pattern detection** (checks triangle shape with 3 simple conditions) - NOT the proper ShapeDD algorithm (convolution + zero-crossing + permutation test)
   - **Required Fix**: Rename to "ADW-MMD with Heuristic Pattern Detection" or "ShapeDD-inspired ADW-MMD Detector" and clearly document the differences from original ShapeDD

3. **MMD-Agg implementation** - **IMPLEMENTATION VARIANT**
   - **Thesis claim**: References Schrab et al. (2023) MMD-Agg method
   - **Code reality**: `shape_mmdagg()` in `shape_dd.py:69-206` uses **Gaussian approximation** for p-values with Bonferroni correction
   - **Paper original**: Uses **wild bootstrap** for non-asymptotic Type I error control
   - **Required Fix**: Add note stating "simplified variant using Gaussian approximation instead of wild bootstrap"

4. **SE-CDT / SHAPED_CDT naming inconsistency** - **CONSISTENCY ISSUE**
   - **Chapter 3**: Called "SE-CDT (ShapeDD-Enhanced CDT)"
   - **Chapter 4**: Called "SHAPED_CDT"
   - **Required Fix**: Standardize on one name throughout (recommend: "SHAPED_CDT" as it's clearer)

5. **CDT_MSW implementation** - **NEEDS DISCLAIMER**
   - **Thesis**: References Guo et al. (2022) CDT_MSW method
   - **Issue**: Chapter 4 mentions "implementation độc lập của CDT_MSW dựa trên mô tả trong paper"
   - **Required Fix**: Add clear disclaimer that this is an "independent reimplementation based on paper description, not the authors' original code" in methodology section

### Moderate Issues (Priority: MEDIUM)

6. **ShapeDD attribution** - **NEEDS CITATION CONSISTENCY**
   - Currently cited as: shapeDD2024 referencing Hinder et al. (2024) Frontiers survey
   - The survey presents ShapeDD but the theoretical foundation (Triangle Shape Property) appears first there
   - **Required Fix**: Verify this is the correct primary citation for ShapeDD algorithm

## Impact

- **Affected chapters**: Chapters 2, 3, 4 (Theoretical Foundation, Proposed Model, Experiments)
- **Affected tables**: Table I (Comprehensive Performance), Table II (F1 by Dataset)
- **Affected figures**: Algorithm pseudocode blocks (Algorithm 3.1, 3.2)
- **Affected code documentation**: `mmd_variants.py`, `shape_dd.py` docstrings

## Summary Table: Paper vs Implementation vs Recommended Name

| Current Name | Original Paper | Implementation Status | Recommended Name |
|-------------|----------------|----------------------|------------------|
| ADW-MMD | OW-MMD (Bharti 2023) | Inspired/Variant | ADW-MMD (với ghi chú "biến thể lấy cảm hứng từ OW-MMD") |
| ShapeDD_ADW_MMD | ShapeDD (Hinder 2024) | Heuristic only | "ADW-MMD + Heuristic Pattern Detection" hoặc "ShapeDD-inspired ADW-MMD" |
| MMD-Agg | MMDAgg (Schrab 2023) | Simplified variant | ShapeDD_MMDAgg (với ghi chú "simplified variant") |
| SE-CDT / SHAPED_CDT | Novel contribution | Correct | SHAPED_CDT (thống nhất) |
| CDT_MSW | CDT_MSW (Guo 2022) | Independent reimpl | CDT_MSW (với ghi chú "independent implementation") |

## Risk Assessment

### If Not Fixed Before Defense:
- **HIGH RISK**: Examiners familiar with Bharti et al. or Schrab et al. papers will notice discrepancies
- **HIGH RISK**: Claim of "ShapeDD pattern detection" when using heuristics is technically inaccurate
- **MEDIUM RISK**: Inconsistent naming (SE-CDT vs SHAPED_CDT) appears unprofessional
- **LOW RISK**: CDT_MSW reimplementation is acceptable if disclosed properly
