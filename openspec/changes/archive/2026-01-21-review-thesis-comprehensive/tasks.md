# Tasks: Review và Chỉnh Sửa Luận Văn Thạc Sĩ

## Tổng quan
- **Ước lượng thời gian:** 2-3 ngày
- **Ưu tiên:** Sửa các lỗi CAO trước, sau đó đến TRUNG BÌNH
- **Deadline:** Trước bảo vệ

---

## 1. Sửa Lỗi Encoding Unicode (Ưu tiên CAO)
- [x] 1.1 Sửa `02_theoretical_foundation.tex` dòng 77: "tr??i d???t kh??i ni???m" → "trôi dạt khái niệm"
- [x] 1.2 Sửa `05_conclusion_future_work.tex` dòng 23: "tri???n khai m?? h??nh" → "triển khai mô hình"
- [x] 1.3 Search toàn bộ thư mục `report/latex/` cho pattern "???" để tìm lỗi encoding khác

## 2. Chuẩn Hóa Thuật Ngữ SHAPED_CDT/SE-CDT (Ưu tiên CAO)
- [x] 2.1 Quyết định tên chính thức: **SE-CDT** (Reverted per user request)
- [x] 2.2 Cập nhật file `03_proposed_model.tex`:
  - Dòng 154: Thống nhất định nghĩa
  - Thêm ghi chú: "SHAPED_CDT (viết tắt SE-CDT trong một số bảng)" (Updated to use SE-CDT primary)
- [x] 2.3 Cập nhật file `04_experiments_evaluation.tex`:
  - Dòng 206: Xóa câu "SE-CDT được đổi tên thành SHAPED_CDT"
  - Thay thế tất cả "SHAPED_CDT" thành "SE-CDT"
- [x] 2.4 Cập nhật bảng `tables/se_cdt_results_table.tex`
- [x] 2.5 Cập nhật bảng `tables/table_comparison_aggregate.tex`
- [x] 2.6 Đổi tên file nếu cần: `shaped_cdt_results_table.tex` → `se_cdt_results_table.tex`

## 3. Làm Rõ Số Liệu CAT Accuracy (Ưu tiên CAO)
- [x] 3.1 Xác định source của 63.2% vs 78.0%:
  - 63.2% có vẻ từ aggregate benchmark (5 drift types × 3 block sizes) - Actually 74.6% in final table
  - 78.0% từ fair comparison với CDT_MSW (17 configs × 10 runs)
- [x] 3.2 Cập nhật văn bản Chương 4 để clarify:
  - Mục 4.4.2: "Trên benchmark tổng hợp, SHAPED_CDT đạt CAT = 63.2%"
  - Mục 4.4.3: "Trong so sánh công bằng (fair comparison), CAT = 78.0%"
- [x] 3.3 Thêm footnote giải thích sự khác biệt do methodology khác nhau

## 4. Demo Kafka POC (Ưu tiên TRUNG BÌNH)
- [x] 4.1 **Option A - Demo thực tế (2-4 giờ):**
  - [x] 4.1.1 Chạy `experiments/drift_monitoring_system/` với Redpanda/Kafka local
  - [x] 4.1.2 Capture screenshot Kafka dashboard (throughput, latency)
  - [x] 4.1.3 Chạy 1 stream test với synthetic data
  - [x] 4.1.4 Thu thập metrics: avg latency, throughput samples/sec
  - [x] 4.1.5 Thêm hình vào Chương 4 với caption giải thích
- [x] 4.2 **Option B - Điều chỉnh scope (30 phút):**
  - [x] 4.2.1 Sửa Chương 0, dòng 66-68: "Triển khai" → "Thiết kế kiến trúc"
  - [x] 4.2.2 Thêm ghi chú rõ ràng: "Đánh giá hiệu năng hệ thống streaming là hướng phát triển tương lai"
  - [x] 4.2.3 Cập nhật Chương 5 tương ứng

## 5. Review Công Thức ADW-MMD (Ưu tiên TRUNG BÌNH)
- [x] 5.1 Kiểm tra implementation trong `experiments/backup/mmd_variants.py`:
  - Xác nhận cross-term dùng uniform hay weighted
- [x] 5.2 Kiểm tra `experiments/backup/ow_mmd.py`:
  - So sánh với mô tả trong luận văn
- [x] 5.3 Cập nhật công thức 3.2 trong `03_proposed_model.tex` nếu cần
- [x] 5.4 Thêm ghi chú: "ADW-MMD, còn gọi là OW-MMD trong Bharti et al. (2023)"

## 6. Các Chỉnh Sửa Nhỏ Khác (Ưu tiên THẤP)
- [x] 6.1 Thêm giải thích ngắn về Virtual Drift trong Chương 0
- [x] 6.2 Thêm giải thích ngắn về ADW-MMD trong Chương 0
- [x] 6.3 Kiểm tra citation `yan2017mind` có trong bibliography không
- [x] 6.4 Review văn phong các đoạn "journal-like" (Chương 2)

## 7. Validation Cuối Cùng
- [x] 7.1 Compile LaTeX và kiểm tra lỗi
- [x] 7.2 Kiểm tra tất cả cross-references (`\ref{}`) hoạt động
- [x] 7.3 Kiểm tra tất cả hình ảnh hiển thị đúng
- [x] 7.4 Kiểm tra pagination và table of contents
- [x] 7.5 Đọc lại một lượt để phát hiện lỗi spelling/grammar

---

## Dependency Graph

```
Task 1 (Encoding) ──────────────────────────────────────┐
                                                         │
Task 2 (Thuật ngữ) ──────────────────────────────────────┼──> Task 7 (Validation)
                                                         │
Task 3 (Số liệu) ────────────────────────────────────────┤
                                                         │
Task 4 (Kafka) OR Task 4B (Scope) ───────────────────────┤
                                                         │
Task 5 (Công thức) ──────────────────────────────────────┤
                                                         │
Task 6 (Minor) ──────────────────────────────────────────┘
```

**Lưu ý:** Task 1, 2, 3 có thể làm song song. Task 4A và 4B là mutually exclusive. Task 7 phải làm cuối cùng.
