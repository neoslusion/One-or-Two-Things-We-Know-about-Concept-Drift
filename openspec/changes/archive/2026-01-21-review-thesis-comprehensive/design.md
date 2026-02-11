# Báo Cáo Review Toàn Diện Luận Văn Thạc Sĩ

## Thông tin chung

- **Đề tài:** Phát hiện và Thích ứng với Concept Drift trong Dữ liệu Luồng Thời gian Thực
- **Người review:** Giảng viên/Giáo sư ngành KHMT-KTMT (vai trò)
- **Ngày review:** 21/01/2026
- **Phương pháp:** Review 3 lượt (Flow & Logic → Kỹ thuật → Câu chữ)

---

# LƯỢT 1: FLOW TRÌNH BÀY & LOGIC NỘI DUNG

## 1.1 Đánh giá Cấu trúc Tổng thể

### Thứ tự các chương

| Chương | Tiêu đề | Đánh giá |
|--------|---------|----------|
| 0 | Giới thiệu đề tài | ✅ Hợp lý - Đặt vấn đề rõ ràng |
| 1 | Công trình liên quan | ✅ Hợp lý - Tổng quan literature |
| 2 | Cơ sở lý thuyết | ⚠️ Nên xem xét gộp với Chương 1 |
| 3 | Mô hình đề xuất | ✅ Hợp lý - Core contribution |
| 4 | Thực nghiệm và đánh giá | ✅ Hợp lý |
| 5 | Kết luận và Hướng phát triển | ✅ Hợp lý |

**Nhận xét:**
- Việc tách Chương 1 (Công trình liên quan) và Chương 2 (Cơ sở lý thuyết) là hợp lý cho luận văn thạc sĩ.
- Chương 1 đóng vai trò survey, Chương 2 đi sâu vào ShapeDD làm nền tảng cho đề xuất ở Chương 3.

## 1.2 Các Vấn Đề Forward Reference

### VẤN ĐỀ 1: Kết quả định lượng trong Giới thiệu
**Vị trí:** `00_introduction.tex`, dòng 55-59

**Nội dung gây vấn đề:**
> "...tích hợp các kỹ thuật tiên tiến gần đây như ADW-MMD...và MMD-Agg vào framework ShapeDD. Mục tiêu là giải quyết các hạn chế về chi phí tính toán và độ nhạy của phương pháp gốc."

**Vấn đề:** Đề cập đến ADW-MMD và MMD-Agg như các kỹ thuật "tiên tiến" nhưng chưa giải thích chúng là gì. Người đọc phải đợi đến Chương 3 mới hiểu.

**Đề xuất sửa:** Thêm một câu ngắn giải thích: "...ADW-MMD (một phương pháp tối ưu trọng số kernel để giảm phương sai, sẽ được trình bày chi tiết tại Mục 3.2)..."

**Mức độ:** ⚠️ Nhẹ - Có thể chấp nhận được cho phần giới thiệu

---

### VẤN ĐỀ 2: Tham chiếu kết quả "7 lần nhanh hơn" trong Chương 3
**Vị trí:** `03_proposed_model.tex`, dòng 95-96

**Nội dung:**
> "...việc giảm số lần permutation từ 2500 xuống 50 có thể mang lại tốc độ tăng đáng kể."

**Vấn đề:** Đây là claim định lượng nhưng kết quả cụ thể (7x) chỉ xuất hiện ở Chương 4. Tuy nhiên, cách diễn đạt "có thể mang lại" là phù hợp vì nó là dự đoán/phân tích lý thuyết.

**Đánh giá:** ✅ Chấp nhận được - Đã dùng ngôn ngữ phỏng đoán

---

### VẤN ĐỀ 3: Tham chiếu Mục~\ref{sec:cdt-msw-theory} trong Giới thiệu
**Vị trí:** `00_introduction.tex`, dòng 58

**Nội dung:**
> "...CDT_MSW (Concept Drift Type identification based on Multi-Sliding Windows --- phương pháp phân loại loại drift dựa trên phân tích cửa sổ trượt đa kích thước~\cite{guo2022cdtmsw}, sẽ được trình bày chi tiết ở Mục~\ref{sec:cdt-msw-theory})"

**Đánh giá:** ✅ Hợp lý - Có citation và forward reference rõ ràng

---

### VẤN ĐỀ 4: Định nghĩa khái niệm trước khi sử dụng
**Kiểm tra:** Các khái niệm quan trọng

| Khái niệm | Lần đầu sử dụng | Được định nghĩa tại | Đánh giá |
|-----------|-----------------|---------------------|----------|
| Concept Drift | Chương 0, dòng 9 | Chương 0, dòng 9-10 | ✅ |
| Virtual Drift | Chương 0, dòng 77 | Chương 2, dòng 25 | ⚠️ Cần thêm giải thích ngắn ở Chương 0 |
| MMD | Chương 1, dòng 262 | Chương 2, dòng 194-217 | ✅ Hợp lý |
| ShapeDD | Chương 0, dòng 50 | Chương 2, dòng 71-91 | ✅ Có citation |
| ADW-MMD | Chương 0, dòng 55 | Chương 3, dòng 25-46 | ⚠️ Cần giải thích ngắn |
| SHAPED_CDT vs SE-CDT | Chương 3, dòng 154 | - | ❌ Hai tên khác nhau cho cùng phương pháp |

## 1.3 Vấn Đề Nhất Quán Thuật Ngữ (Chi tiết)

### VẤN ĐỀ 5: SHAPED_CDT vs SE-CDT
**Mức độ nghiêm trọng:** ❌ CAO

**Chi tiết:**
- Chương 3 định nghĩa: **SHAPED_CDT** (ShapeDD-Enhanced Concept Drift Type)
- Chương 4, dòng 206: "SE-CDT được đổi tên thành **SHAPED_CDT**..."
- Bảng `se_cdt_results_table.tex`: Sử dụng "SE-CDT"
- Bảng `table_comparison_aggregate.tex`: Sử dụng "SE-CDT (Std)", "SE-CDT (ADW)"

**Vấn đề:** Báo cáo sử dụng cả hai tên không nhất quán, gây nhầm lẫn.

**Đề xuất sửa:**
1. Chọn MỘT tên duy nhất (khuyến nghị: **SHAPED_CDT** vì có ý nghĩa rõ hơn)
2. Cập nhật tất cả bảng và đoạn văn để sử dụng tên này
3. Nếu muốn giữ cả hai, cần có câu giải thích rõ ràng: "SHAPED_CDT (viết tắt: SE-CDT)..."

---

## 1.4 Đánh Giá Vai Trò Các Chương

| Chương | Vai trò mong đợi | Thực tế | Chồng chéo? |
|--------|------------------|---------|-------------|
| 1 | Survey các phương pháp | Đúng | Có ít với Chương 2 về MMD |
| 2 | Nền tảng lý thuyết ShapeDD | Đúng | - |
| 3 | Đề xuất cải tiến | Đúng | - |
| 4 | Thực nghiệm | Đúng | - |
| 5 | Kết luận | Đúng | - |

**Nhận xét:** Chương 1 có phần nói về ShapeDD (Bảng 1.2, dòng 262) và Chương 2 cũng nói về ShapeDD. Đây là overlap nhỏ nhưng chấp nhận được vì Chương 1 chỉ overview còn Chương 2 đi sâu vào lý thuyết.

---

# LƯỢT 2: TÍNH NHẤT QUÁN KỸ THUẬT

## 2.1 Kiểm Tra Công Thức Toán Học

### Công thức MMD (Chương 2)

**Công thức 2.5-2.8:** ✅ Đúng với literature
```
MMD(P, Q) = sup_{f ∈ F} |E[f(X)] - E[f(Y)]|
MMD²(P, Q) = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
```

### Công thức ADW-MMD (Chương 3)

**Công thức 3.1-3.2:**
```
w_i ∝ 1/√(Σ_j k(x_i, x_j))
MMD²_ADW = Σ w_i w_j k(x,x) + Σ v_p v_q k(y,y) - 2 Σ (1/nm) k(x,y)
```

**Vấn đề tiềm ẩn:** Công thức 3.2 nói cross-term dùng uniform weights `1/nm`, nhưng Algorithm 3.1 dòng 66 lại tính:
```
Σ w_i w_p k(x_i, y_p)
```

**Kiểm tra với code:** Cần xác nhận implementation thực tế trong `experiments/backup/mmd_variants.py` hoặc `ow_mmd.py`.

**Đề xuất:** Làm rõ trong văn bản - cross-term dùng uniform hay weighted?

---

## 2.2 Kiểm Tra Nhất Quán Số Liệu Giữa Các Bảng

### Discrepancy 1: CAT Accuracy của SE-CDT

| Nguồn | Giá trị |
|-------|---------|
| `se_cdt_results_table.tex` | Không có CAT overall |
| `table_comparison_aggregate.tex` | SE-CDT (Std) CAT = 63.2% |
| Chương 4, dòng 192 | "Độ chính xác nhóm (CAT) 78.0%" |
| Chương 4, dòng 288 | "SHAPED_CDT đạt độ chính xác nhóm (CAT) cao nhất (78.0%)" |

**Vấn đề:** Có hai con số khác nhau: **63.2%** (bảng) vs **78.0%** (văn bản)

**Nguyên nhân có thể:**
1. Hai benchmark khác nhau (khác dataset, khác configuration)
2. 63.2% là từ aggregate benchmark, 78.0% là từ fair comparison

**Đề xuất sửa:** Làm rõ trong văn bản context của mỗi con số. Ví dụ:
- "Trên benchmark tổng hợp, SE-CDT đạt CAT = 63.2%"
- "Trong so sánh công bằng với CDT_MSW, SE-CDT đạt CAT = 78.0%"

---

### Discrepancy 2: EDR/MDR metrics

| Bảng | CDT_MSW EDR | SE-CDT EDR |
|------|-------------|------------|
| `table_comparison_aggregate.tex` | 0.058 | 0.980 |
| Chương 4, Bảng 4.5 | 0.39 (Sudden) | 0.96 (Sudden) |

**Nhận xét:** Số liệu khác nhau vì khác dataset/configuration. Đây là hợp lý nhưng cần note rõ hơn trong văn bản.

---

## 2.3 Kiểm Tra Tên Phương Pháp vs Implementation

### ShapeDD_ADW_MMD

| Mô tả trong luận văn | File code | Tên hàm | Match? |
|---------------------|-----------|---------|--------|
| ShapeDD + ADW-MMD | `mmd_variants.py` | `shapedd_adw_mmd_full()` | ✅ |
| ADW-MMD standalone | `ow_mmd.py` | `mmd_ow_permutation()` | ⚠️ Tên file khác |

**Vấn đề:** Tên file `ow_mmd.py` sử dụng "OW-MMD" (Optimally-Weighted) trong khi luận văn dùng "ADW-MMD" (Adaptive Density-Weighted).

**Giải thích:** Theo paper Bharti et al. 2023, tên gốc là "Optimally-Weighted MMD". Luận văn đặt lại tên thành "ADW-MMD" để nhấn mạnh cơ chế density-weighting.

**Đề xuất:** Thêm ghi chú trong luận văn: "ADW-MMD, hay còn gọi là OW-MMD trong literature gốc [citation]..."

---

## 2.4 Thiếu Sót Trong Phần Thực Nghiệm

### THIẾU: Demo Hệ Thống Kafka POC

**Vị trí:** Chương 3, Mục 3.5 mô tả kiến trúc Kafka chi tiết
**Vấn đề:** Chương 4 KHÔNG có kết quả thực nghiệm nào về:
- Latency của hệ thống streaming
- Throughput dưới tải cao
- Screenshot/hình ảnh Kafka monitoring dashboard
- Metrics từ hệ thống real-time

**Trích dẫn từ Chương 3:**
> "Lưu ý: Phần này trình bày thiết kế kiến trúc; đánh giá hiệu năng hệ thống (latency, throughput dưới tải cao) là hướng phát triển trong tương lai."

**Đánh giá:** ⚠️ Đây là gap lớn vì:
1. Mục tiêu đề tài (Chương 0, dòng 66-68) nói rõ "Triển khai hệ thống trên môi trường dữ liệu streaming real-time sử dụng Apache Kafka"
2. Kiến trúc được mô tả chi tiết nhưng không có validation

**Đề xuất:**
1. **Tối thiểu:** Chạy demo Kafka với 1 stream nhỏ, capture screenshot dashboard
2. **Đầy đủ:** Đo latency end-to-end, throughput, resource usage
3. **Nếu không kịp:** Điều chỉnh mục tiêu trong Chương 0 - bỏ "triển khai", thay bằng "thiết kế kiến trúc"

---

# LƯỢT 3: CÂU CHỮ, VĂN PHONG & TRÍCH DẪN

## 3.1 Đánh Giá Văn Phong

### Đoạn văn phong "journal-like" cần viết lại

**Vị trí 1:** `02_theoretical_foundation.tex`, dòng 75-78
```
Trong học máy cổ điển, dữ liệu thường được giả định là \textit{độc lập và phân phối 
đồng nhất} (IID - Independent and identically distributed data) theo một phân phối 
tĩnh $P_X$. Tuy nhiên, trong các ứng dụng thực tế...
```

**Nhận xét:** Đoạn này dịch khá sát từ paper tiếng Anh, nhưng vẫn ở mức chấp nhận được cho luận văn thạc sĩ.

---

**Vị trí 2:** `02_theoretical_foundation.tex`, dòng 77 - LỖI ENCODING
```
hiện tượng này được gọi là \textbf{tr??i d???t kh??i ni???m (concept drift)}
```

**Vấn đề:** ❌ Lỗi encoding Unicode nghiêm trọng

**Đề xuất:** Sửa thành: "hiện tượng này được gọi là **trôi dạt khái niệm** (concept drift)"

---

**Vị trí 3:** `05_conclusion_future_work.tex`, dòng 23 - LỖI ENCODING
```
tri???n khai m?? h??nh ????ng b??ng (Frozen Model Deployment)
```

**Vấn đề:** ❌ Lỗi encoding

**Đề xuất:** Sửa thành: "triển khai mô hình đóng băng (Frozen Model Deployment)"

---

## 3.2 Đánh Giá Mức Độ Phù Hợp Với Luận Văn Thạc Sĩ

| Tiêu chí | Đánh giá | Ghi chú |
|----------|----------|---------|
| Độ sâu lý thuyết | ✅ Phù hợp | Trình bày định lý, chứng minh ở mức cần thiết |
| Thực nghiệm | ✅ Phù hợp | 30 runs, 10 datasets, kiểm định thống kê |
| Ngôn ngữ | ⚠️ Cần cải thiện | Một số chỗ quá formal/dịch máy |
| Đóng góp | ✅ Rõ ràng | 3 đóng góp được liệt kê cụ thể |

## 3.3 Kiểm Tra Trích Dẫn

### Trích dẫn chính xác ngữ cảnh

| Citation | Nội dung claim | Đúng ngữ cảnh? |
|----------|---------------|----------------|
| Bharti 2023 | OW-MMD giảm phương sai | ✅ |
| Schrab 2023 | MMD-Agg multi-kernel | ✅ |
| Guo 2022 | CDT_MSW phân loại drift | ✅ |
| ShapeDD 2024 | Triangle shape property | ✅ |
| Jourdan 2023 | Process monitoring | ✅ |

### Trích dẫn thiếu

**Vấn đề:** `03_proposed_model.tex`, dòng 34 nhắc đến "Yan et al." nhưng không thấy citation format
```
lấy cảm hứng từ...và phương pháp reweighting trong domain adaptation của Yan et al.~\cite{yan2017mind}
```

**Đề xuất:** Kiểm tra citation này có trong bibliography không.

---

# TỔNG HỢP VẤN ĐỀ THEO MỨC ĐỘ ƯU TIÊN

## Ưu tiên CAO (Phải sửa trước bảo vệ)

1. **Lỗi encoding Unicode** - 2 vị trí trong Chương 2 và 5
2. **Nhất quán thuật ngữ SHAPED_CDT vs SE-CDT** - Toàn bộ báo cáo
3. **Discrepancy số liệu CAT Accuracy** (63.2% vs 78.0%) - Cần clarify

## Ưu tiên TRUNG BÌNH (Nên sửa)

4. **Thiếu demo Kafka POC** - Cần ít nhất screenshot hoặc điều chỉnh scope
5. **Làm rõ cross-term trong ADW-MMD** (uniform vs weighted)
6. **Thêm ghi chú về tên OW-MMD vs ADW-MMD**

## Ưu tiên THẤP (Có thể bỏ qua)

7. Forward reference ADW-MMD trong Chương 0 (chấp nhận được)
8. Overlap nhỏ giữa Chương 1 và 2 về ShapeDD

---

# KẾ HOẠCH HÀNH ĐỘNG

## Task 1: Sửa lỗi encoding (15 phút)
- [ ] `02_theoretical_foundation.tex` dòng 77
- [ ] `05_conclusion_future_work.tex` dòng 23

## Task 2: Chuẩn hóa thuật ngữ (1 giờ)
- [ ] Quyết định: SHAPED_CDT hoặc SE-CDT
- [ ] Find-replace toàn bộ báo cáo
- [ ] Cập nhật các bảng LaTeX

## Task 3: Clarify số liệu (30 phút)
- [ ] Thêm context cho 63.2% và 78.0%
- [ ] Giải thích sự khác biệt do benchmark khác nhau

## Task 4: Demo Kafka (2-4 giờ)
- [ ] Option A: Chạy demo nhỏ, capture screenshot
- [ ] Option B: Điều chỉnh mục tiêu trong Chương 0

## Task 5: Review công thức (30 phút)
- [ ] Kiểm tra code để xác nhận cross-term implementation
- [ ] Cập nhật công thức nếu cần

---

# LƯỢT 4: BENCHMARK METHODOLOGY REVIEW

## 4.1 Validation Methodology Assessment

### Detection Metrics (✅ CORRECT)

Benchmark sử dụng **event-based** evaluation - chuẩn mực cho drift detection:
- TP: Phát hiện trong khoảng [drift_pos - 50, drift_pos + tolerance]
- FP: Phát hiện không khớp với event nào
- FN: Event không được phát hiện
- **KHÔNG có TN** (đúng chuẩn drift detection)

### Statistical Setup (✅ CORRECT)

| Parameter | Value | Assessment |
|-----------|-------|------------|
| N_RUNS | 10 seeds × 5 scenarios = 50 | Cần tăng lên 30 runs để khớp với claims |
| Seed Strategy | Fixed (0-9) | ✅ Reproducible |
| Tolerance | 300 samples | ⚠️ Cao hơn chuẩn (100-150) |

## 4.2 SE-CDT vs CDT_MSW Comparison (⚠️ FAIRNESS ISSUE)

### Critical Finding

So sánh **không công bằng** do khác input requirements:

| Method | Mode | Input | Detection Signal |
|--------|------|-------|------------------|
| CDT_MSW | Supervised | Labels (y) | Accuracy drop |
| SE-CDT | Unsupervised | Features (X) | MMD signal |

**Vấn đề:** CDT_MSW nhận labels với `supervised_mode=False` (labels không đổi theo concept) → accuracy không giảm → detection thất bại.

**Bằng chứng từ kết quả:**
```
CDT_MSW: EDR=0.058, MDR=0.942 (gần như không detect được gì)
SE-CDT:  EDR=0.980, MDR=0.020 (detect gần như tất cả)
```

### Recommendation

1. **Chạy lại benchmark với `supervised_mode=True`** cho CDT_MSW
2. Hoặc **nêu rõ trong thesis** rằng so sánh không fair vì SE-CDT có lợi thế unsupervised

## 4.3 Result Recording Quality (✅ GOOD)

### Data Files Recorded

| File | Location | Status |
|------|----------|--------|
| `benchmark_proper_detailed.pkl` | `publication_figures/` | ✅ 578KB |
| `table_comparison_aggregate.tex` | `report/latex/tables/` | ✅ Generated |
| `se_cdt_results_table.tex` | `experiments/` | ✅ Generated |

### Issue: High False Positive Rate

```
SE-CDT (Std): FP=1513 ← Very high!
```

**Root cause:** Threshold quá thấp (height=0.001, prominence=0.0005) trong `benchmark_proper.py:400`.

**Recommendation:** Tăng threshold hoặc giải thích lý do trong thesis.

## 4.4 Figure Quality (⚠️ FIGURES NOT IN THESIS)

### Generated Figures (✅ Exist)

| Figure | Location |
|--------|----------|
| `vis_mixed_a_CDT.png` | `experiments/publication_figures/` |
| `vis_mixed_a_SE.png` | `experiments/publication_figures/` |
| `vis_mixed_b_*.png` | `experiments/publication_figures/` |
| `vis_repeated_gradual_*.png` | `experiments/publication_figures/` |
| `vis_repeated_incremental_*.png` | `experiments/publication_figures/` |

### Issue

Figures đã được generate nhưng **chưa được import vào thesis LaTeX**.

**Recommendation:** Add `\includegraphics` commands trong Chapter 4.

## 4.5 Justification Quality (⚠️ GAPS)

### Missing Justifications

1. **High FP for SE-CDT (Std):** Tại sao 1513 FP? Cần giải thích.
2. **Threshold selection:** Tại sao chọn thresholds 0.01, 0.005?
3. **Tolerance=300:** Tại sao cao hơn chuẩn (100-150)?

### Good Justifications Present

1. ✅ Unsupervised advantage của SE-CDT
2. ✅ Category vs Subcategory accuracy distinction
3. ✅ Runtime comparison

---

## 4.6 Benchmark Review Action Items

### HIGH Priority

| Task | Description |
|------|-------------|
| Fix fairness | Re-run CDT_MSW với `supervised_mode=True` hoặc note unfair comparison |
| Add figures | Import 8 PNG figures vào Chapter 4 |
| Tune threshold | Tăng SE-CDT threshold để giảm FP |

### MEDIUM Priority

| Task | Description |
|------|-------------|
| Add justification | Giải thích threshold choices trong methodology |
| Increase N_RUNS | Tăng từ 10 lên 30 runs nếu có thời gian |
| Add confusion matrix | Visualize SE-CDT classification accuracy |

---

# KẾT LUẬN REVIEW

## Điểm mạnh của luận văn:

1. **Cấu trúc logic rõ ràng** - Flow từ vấn đề → giải pháp → thực nghiệm hợp lý
2. **Nền tảng lý thuyết vững chắc** - Định lý Triangle Shape được giải thích tốt
3. **Thực nghiệm có hệ thống** - 30 runs, nhiều dataset, kiểm định thống kê
4. **Đóng góp rõ ràng** - 3 contributions được xác định cụ thể
5. **So sánh công bằng** - Có cả supervised vs unsupervised comparison

## Điểm cần cải thiện:

1. **Nhất quán thuật ngữ** - SHAPED_CDT/SE-CDT cần thống nhất
2. **Lỗi kỹ thuật nhỏ** - Encoding, discrepancy số liệu
3. **Gap thực nghiệm Kafka** - Cần demo hoặc điều chỉnh scope
4. **Văn phong** - Một số chỗ còn "journal-like"

## Đánh giá tổng thể:

> Báo cáo đạt **chất lượng tốt cho luận văn thạc sĩ**. Các vấn đề phát hiện chủ yếu là cosmetic và có thể sửa trong 1-2 ngày. Vấn đề lớn nhất là thiếu validation cho phần Kafka - cần quyết định: demo thực tế hay điều chỉnh scope.
>
> **Khuyến nghị:** Cho phép bảo vệ sau khi sửa các lỗi ưu tiên CAO.
