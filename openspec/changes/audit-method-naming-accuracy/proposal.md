# Change: Audit Method Naming Accuracy in Thesis

## Why

Luận văn thạc sĩ về Concept Drift Detection hiện có **sai lệch nghiêm trọng giữa tên phương pháp được ghi trong văn bản và cách hoạt động thực tế trong code**. Điều này vi phạm nguyên tắc trung thực khoa học và có thể dẫn đến:
1. Bị phản biện khi bảo vệ luận văn
2. Gây hiểu lầm cho người đọc về bản chất kỹ thuật
3. Trích dẫn không chính xác các paper gốc

## What Changes

### 1. **ADW-MMD vs OW-MMD** (CRITICAL - Phải sửa)
- **Vấn đề**: Luận văn gọi là "ADW-MMD (Adaptive Density-Weighted MMD)" nhưng implementation thực tế là biến thể đơn giản hóa của OW-MMD (Bharti et al., 2023)
- **Sửa đổi**: Đổi tên thành "Density-Weighted MMD (DW-MMD)" hoặc "OW-MMD-inspired heuristic"

### 2. **ShapeDD_ADW_MMD** (HIGH - Cần sửa)
- **Vấn đề**: Không phải là ShapeDD gốc + ADW-MMD, mà là heuristic pattern detection
- **Sửa đổi**: Đổi tên thành "ShapeDD-Heuristic" hoặc "Triangle-Pattern Detector"

### 3. **SE-CDT vs CDT_MSW** (MEDIUM)
- **Vấn đề**: SE-CDT được mô tả như thay thế CDT_MSW nhưng dùng tín hiệu hoàn toàn khác
- **Sửa đổi**: Ghi rõ là "CDT_MSW-inspired unsupervised approach"

### 4. **MMD-Agg Implementation** (MEDIUM)
- **Vấn đề**: Implementation dùng Bonferroni correction thay vì wild bootstrap như paper gốc
- **Sửa đổi**: Ghi rõ là "Simplified MMD-Agg with Bonferroni correction"

## Impact

- **Affected specs**: thesis-consistency (new)
- **Affected code**: Không cần thay đổi code, chỉ sửa documentation và thesis text
- **Affected files**:
  - `report/latex/chapters/02_theoretical_foundation.tex`
  - `report/latex/chapters/03_proposed_model.tex`
  - `report/latex/chapters/04_experiments_evaluation.tex`
  - `report/latex/references.bib`
  - `AGENTS.md` (documentation)

## Risk Assessment

| Severity | Issue | Risk if not fixed |
|----------|-------|-------------------|
| **CRITICAL** | ADW-MMD naming | Bị challenge khi bảo vệ: "Implementation không match paper" |
| **HIGH** | ShapeDD_ADW_MMD | Reviewer hỏi: "Đây có phải ShapeDD không?" |
| **MEDIUM** | SE-CDT naming | Confusion về supervised vs unsupervised |
| **LOW** | MMD-Agg simplification | Cần disclaimer về implementation |

## Acceptance Criteria

1. Mọi tên phương pháp trong luận văn phải match với implementation
2. Mọi biến thể phải được ghi rõ là "X-based" hoặc "X-inspired"
3. Mọi simplification so với paper gốc phải được document
4. Trích dẫn phải chính xác với những gì thực sự được implement
