# Cẩm nang Bảo vệ — Rà soát Báo cáo / Mã nguồn / Phương pháp + Bộ câu hỏi phản biện

**Đề tài:** *SE-CDT — Phát hiện, phân loại và thích ứng concept drift không giám sát* (Lê Phúc Đức, ĐH Bách Khoa — ĐHQG-HCM)

Tài liệu này là kết quả rà soát chéo giữa: mã nguồn (`core/detectors/se_cdt.py`, `mmd_variants.py`, `experiments/monitoring/adaptation_strategies.py`), toàn bộ 6 chương trong `report/latex/chapters/`, và các bảng trong `results/tables/`.

> **Kết luận tổng quát:** Phương pháp, mã nguồn và báo cáo **rất nhất quán và trung thực**. Mọi hạn chế đã thừa nhận đều **đúng sự thật và được mô tả chính xác**. Tuy nhiên có một số điểm yếu mà `THESIS_GUIDE.md` chuẩn bị chưa kỹ (Phần 2), cần luyện trả lời thật trơn. Lưu ý: trong bản tiếng Anh của guide trước đây có một con số sai (DAWIDD "15.18s / ~22×") — **đã được sửa thành 6.79s / ≈9.7×** đúng theo `table_III_runtime_stats.tex`. Bản tiếng Việt của guide không nhắc tới DAWIDD nên không cần sửa.

Cách dùng: tài liệu này **bổ sung** cho `THESIS_GUIDE_VN.md`, không thay thế. Guide đã lo phần kiến thức nền; tài liệu này lo phần **các câu hỏi khó / câu hỏi bẫy** và **phán quyết về tính đúng đắn** của từng nhận định.

---

## Phần 1 — Kiểm toán: các ưu/nhược điểm đã thừa nhận có ĐÚNG và VỮNG không?

Thang đánh giá: ✅ đúng & lập luận vững · ⚠️ đúng nhưng lập luận còn lỗ hổng · ❌ sai/nói quá.

| # | Nhận định đã thừa nhận | Vị trí | Phán quyết | Ghi chú |
|---|---|---|---|---|
| 1 | IDW-MMD kém nhạy với gradual/center drift; nên trace dùng **Standard MMD**, IDW chỉ dùng để kiểm định | `03:160-163, 235-259` | ✅ | Mã nguồn xác nhận đúng: `shapedd_idw_mmd_proper` tính trace σ(t) bằng Standard unbiased MMD (`mmd_variants.py:447-478`), chỉ gọi `wmmd_gamma` (IDW) cho từng ứng viên (`:520-534`). Tự nhất quán. |
| 2 | Gradual/Incremental là hai tiểu loại khó nhất (27.2% / 30.3%); "hạn chế quan trọng nhất, không nên che giấu" | `04`, `05`, bảng | ✅ | Đúng và trung thực. Hai nguyên nhân: (a) phân bố Variance Ratio của Gradual và Incremental chồng lấn nên ~47% Incremental bị gán Gradual; (b) chuỗi drift chậm lặp lại bão hoà nên bị Concept Memory hút sang Recurrent (~57% Gradual). Cả hai vẫn nằm đúng nhóm PCD nên CAT (80.1%) không đổi. Đây là giới hạn cố hữu của phân loại không giám sát, không phải bug. |
| 3 | Ngưỡng cây quyết định là heuristic, không suy ra từ lý thuyết | `03:499-505`, `05:40-54` | ✅ | Mã nguồn xác nhận (hardcode 0.15/2.0/0.30 + LTS/MS/SDS). Self-calibration chỉ **nới** WR/SNR/CV và chỉ sau ≥20 cửa sổ no-drift; LTS/MS/SDS giữ cố định — đúng như báo cáo nói. |
| 4 | Chỉ quan sát được P(X); real drift thuần P(Y\|X) ngoài phạm vi (giả định joint-drift) | `05:57-59`, `00:42` | ✅ | Vững. Nhóm control (SEA/Hyperplane/LED) đạt F1 0.118/0.179/0.148 — được khung đúng là "phải thấp", chứng minh không hallucination. |
| 5 | Đánh giá chủ yếu trên synthetic | `05:61-62` | ✅ | Đúng; chỉ Electricity là semi-real. Trung thực. |
| 6 | Kafka là prototype, không phải production | `03:596`, `04:471` | ✅ | Đã loại khỏi phạm vi đúng cách. |
| 7 | Phân loại đo ở **Oracle mode** (cho sẵn vị trí drift ground-truth) | `04:264-266` | ✅ | Trung thực và sạch về phương pháp — nhưng xem câu H6: nó làm con số 80.1/55.3 lạc quan hơn so với end-to-end thật. |
| 8 | Gamma null là xấp xỉ; Type-I error tăng lên 0.075 (IDW) / 0.100 (composite) trên AR(1) | `04:228-235` | ✅ | Khớp `wmmd_gamma` (B=20 moment-matched Gamma) và audit note về `wmmd_asymptotic` sai trước đây. Rất thẳng thắn. |
| 9 | Đánh đổi Recall↔FP (Std: EDR 92.6%, FP 4846 / IDW: EDR 56.7%, FP 657) | `04:315-328`, `05:32-33` | ✅ | Thật và có số. Đây là mâu thuẫn cốt lõi trung thực nhất của cả hệ thống (xem H1). |
| 10 | "≈7× nhanh hơn ShapeDD" / "F1=0.531, hòa thống kê với DAWIDD" | bảng, `04:117-142,254` | ✅ | Bảng III: 0.70s vs 5.03s = 7.17×. Bảng I: F1 0.531 = DAWIDD. CD=2.806 ≫ chênh 0.142 → "hòa" được lập luận đúng. Khung "ngang DAWIDD nhưng FP thấp hơn + nhanh hơn + có classification" là trung thực (không bịa "thắng"). |

> **Kết luận Phần 1:** mọi ưu/nhược điểm đã thừa nhận đều đúng sự thật và — trừ các lỗ hổng nêu ở Phần 2 — được lập luận vững. **Sự trung thực học thuật là lợi thế phòng thủ lớn nhất của luận văn — hãy dẫn dắt hội đồng bằng chính điều đó.**

---

## Phần 2 — Những điểm guide/báo cáo CHUẨN BỊ CHƯA KỸ (cần luyện hoặc sửa)

Đây là những hướng mà hội đồng sắc sảo sẽ thực sự dùng để truy.

**A. ✅ (đã sửa) Lỗi số trong `THESIS_GUIDE.md` (Q21).** Guide nói "DAWIDD … 15.18s/stream … ~22× nhanh hơn". Bảng `table_III_runtime_stats.tex` thật ghi **DAWIDD = 6.79s**, nên con số đúng là **6.79/0.70 ≈ 9.7×**, không phải 22×. Đã sửa trong guide tiếng Anh. Khi bảo vệ, nếu nói tốc độ so với DAWIDD, hãy nói **~9.7×** (lấy từ bảng), đừng nói 22×.

**B. ⚠️ Phân loại tiểu loại chỉ ~55%, nhưng Type-Specific adaptation thắng +20pp — tại sao? (bẫy sắc nhất).** Các kịch bản adaptation (Stepping/Sudden/Mixed) đều thiên về Sudden, nơi classifier mạnh (Sudden 82.4%). Bạn **chưa** thử adaptation trên luồng nặng Gradual/Incremental (nơi classifier rớt còn 27–30% ở mức tiểu loại). Phần kết luận cũng liệt kê "bổ sung kịch bản Gradual/Incremental" là future work (`05:65`). Vậy lợi ích adaptation chỉ được chứng minh *trong vùng mà classification hoạt động tốt*. Luyện câu trả lời H2 — đừng khẳng định type-specific adaptation đã được kiểm chứng cho mọi loại drift.

**C. ⚠️ Hệ thống "không giám sát" nhưng adaptation lại cần nhãn.** `adapt_sudden_drift/gradual/incremental` đều gọi `.fit(X, y)` (`adaptation_strategies.py:24-126`). Detection+classification không giám sát; **adaptation tiêu thụ y**. Thực nghiệm prequential cấp y ngay lập tức, hơi mâu thuẫn với động lực "nhãn đến trễ". Vẫn bảo vệ được (detection là early-warning tức thời, adaptation chạy khi nhãn trễ đã về) nhưng phải nói rõ. Xem H3.

**D. ⚠️ Hai "concept memory" khác nhau, cùng con số 0.15, khác metric.** Concept memory của classifier SE-CDT dùng **Standard MMD**, τ=0.15 (`se_cdt.py:251`, ≈2× sàn nhiễu MMD). Còn model cache của adaptation manager dùng **khoảng cách KS trung bình**, ngưỡng 0.15 (`adaptation_strategies.py:149,193`). KS distance 0.15 *không* bằng MMD 0.15 — lý do hiệu chỉnh ngưỡng không chuyển được sang. Người đọc mã có thể bắt lỗi này. Hãy sẵn sàng nói đây là hai ngưỡng độc lập, chỉnh riêng (và nên nêu là một "nit" đã biết). Xem H7.

**E. ⚠️ Đóng góp riêng của IDW vào detection F1 gần như bằng 0.** Trong Bảng I, `IDW_MMD`, `ShapeDD_IDW`, `SE_CDT` giống hệt nhau từng chữ số (0.432/0.737/0.531/36/9.6), và phần hơn Standard MMD rất nhỏ (0.525→0.531, FP 10.5→9.6). Nên IDW *tự thân* gần như không đẩy được F1; điểm thắng thật là (i) giảm FP/runtime so với ShapeDD và (ii) calibration Type-I. Đừng nói IDW làm tăng F1 — hãy nói nó cho calibration + FP tốt hơn ở cùng F1. Xem H4.

**F. Một số lỗ hổng toàn diện nhỏ** (không chí mạng nhưng nên lường trước): chưa có ablation tách riêng đóng góp của IDW vs Gamma vs peak-detection; chỉ `n=14` dataset, `d∈{5,10}`; nhánh GPU có nhưng tắt (`HAS_TORCH=False`, `mmd_variants.py:8`) — nếu bị hỏi, nói tắt để đảm bảo đo runtime đồng nhất phần cứng; `kernel_dd.py` có "TODO: real bad, fix me" nhưng detector đó **không** nằm trong benchmark, nên vô hại (xác nhận bạn không trích dẫn nó).

---

## Phần 3 — Báo cáo có đủ toàn diện để bảo vệ không?

**Có, ở mức luận văn Thạc sĩ.** Cấu trúc đầy đủ: giới thiệu → cơ sở lý thuyết (MMD/ShapeDD) → nghiên cứu liên quan (4 họ detector + dòng dõi ShapeDD/CDT-MSW/OW-MMD) → mô hình đề xuất (IDW-MMD+Gamma, classifier SE-CDT, adaptation, Kafka) → thực nghiệm (3 chế độ, 14 dataset×30 seed, Friedman/Nemenyi, H0 calibration, runtime, adaptation, demo Kafka) → kết luận có nêu rõ hạn chế + hướng phát triển. Ba quy ước đánh giá được lập bảng (`tab:eval-conventions`) để chặn nhầm lẫn giữa các chế độ. Phần thống kê (CD, Iman-Davenport, Wilson CI) đầy đủ và đúng.

**Nên gia cố trước buổi bảo vệ:** (a) một slide ablation (IDW vs Gamma vs shape-filter) để trả lời H4; (b) một slide giải thích rõ vì sao 9.6 FP/run (detection, IDW) và 4846 FP (classification, Std) không mâu thuẫn — cùng detector, khác cấu hình/δ/cooldown — vì hai con số FP lớn này nhìn ra ngoài ngữ cảnh dễ tưởng đá nhau.

---

## Phần 4 — `THESIS_GUIDE.md` đã đủ tốt để ôn chưa?

**Rất tốt và bất thường về độ kỹ** (bản EN ~18.5k từ, 35 Q&A mẫu, implementation map, ví dụ tính số). Quá đủ cho các câu hỏi *tiêu chuẩn*. Lưu ý: lỗi **A** (22× / 15.18s) đã sửa ở bản EN; bản VN ngắn hơn (593 dòng, 15 câu) và không có lỗi này. Mảng câu hỏi *khó* thì guide còn mỏng ở 4 bẫy Phần 2 (B–E) — đã được bổ sung trong bộ Q&A bên dưới.

---

## Phần 5 — Bộ câu hỏi phản biện xếp hạng (dễ → khó), phủ toàn bộ báo cáo

Guide đã lo tốt Q1–Q35 (động lực, MMD, IDW, thống kê, Kafka). Dưới đây là **thang câu hỏi ngày bảo vệ** (i) chạm mọi chương và (ii) đưa lên trước các *bẫy khó* mà guide chưa kỹ. Dùng song song với guide.

### DỄ (khởi động — hội đồng nào cũng mở đầu kiểu này) — Ch.0, Ch.2

- **E1. Concept drift là gì, trong một câu?** Phân phối liên hợp P(X,Y) mà mô hình đã học khi huấn luyện không còn khớp với phân phối lúc vận hành, nên độ chính xác suy giảm âm thầm. Hình thức: P_train(X,Y) ≠ P_online,t(X,Y).
- **E2. Kể 5 hình thái drift và cách nhóm.** Sudden, Blip, Recurrent (→ TCD, tạm thời) và Gradual, Incremental (→ PCD, tiến triển). Phân nhóm TCD/PCD quyết định chiến lược thích ứng (reset vs fine-tune).
- **E3. Tại sao không giám sát / tại sao theo dõi P(X)?** Trong sản xuất, nhãn đến trễ (gian lận xác nhận sau vài tuần) hoặc tốn kém. Theo dõi P(X) cho cảnh báo sớm ngay khi đầu vào đổi, dưới giả định joint-drift rằng P(Y|X) thường đổi theo.
- **E4. MMD đo cái gì?** Khoảng cách giữa hai phân phối trong một RKHS; bằng 0 khi và chỉ khi P=Q với kernel universal. Có ước lượng empirical dạng đóng qua tổng kernel RBF — không cần ước lượng mật độ.
- **E5. Triangle Shape Property là gì?** Một sudden drift tại t₀ khiến tín hiệu MMD cửa sổ trượt σ(t) dâng thành đỉnh tam giác tại t₀; hình dạng tồn tại với mọi metric, chỉ độ cao phụ thuộc ‖P−Q‖. Đây là nền hình học cho cả phát hiện (tìm đỉnh) lẫn phân loại SE-CDT (đọc hình dạng đỉnh).
- **E6. Ba đóng góp của luận văn?** (1) ShapeDD-IDW: tăng tốc phát hiện ShapeDD ~7× bằng IDW-MMD + Gamma null thay permutation test; (2) SE-CDT: classifier loại drift không giám sát đọc hình dạng σ(t); (3) khung thích ứng theo loại drift, nối vào prototype Kafka.

### TRUNG BÌNH — Ch.3 phương pháp, Ch.4 kết quả

- **M1. Tại sao IDW = 1/√d mà không phải 1/d?** 1/d khuếch đại các điểm biên cô lập, làm ước lượng bị nhiễu chi phối; 1/√d nới nhẹ và ổn định hơn. ε=0.5 chặn mẫu số.
- **M2. Tại sao cross-term giữ đều (1/nm) còn within-window dùng IDW?** Cross-term là "mốc neo hình học" đo độ chồng lấp tuyệt đối P–Q. Gán trọng số cho nó sẽ tạo tín hiệu drift giả mỗi khi cấu trúc biên hai cửa sổ khác nhau chỉ do lấy mẫu. (Mã: `mmd_variants.py:195,242`.)
- **M3. IDW-MMD có bias dương cả khi P=Q — có phá hỏng kiểm định không?** Không: ta không đặt ngưỡng trên thống kê thô mà tính p-value so với null Gamma ước lượng empirical (B=20 hoán vị chỉ số), mean của nó đã bao gồm bias. Ta kiểm định "cực trị so với null đã đo", không phải "T>0".
- **M4. Tại sao dùng Gamma thay vì null Gaussian?** Dưới H₀, MMD²ᵤ hội tụ về tổng χ² có trọng (Gretton 2012 Thm.12), không phải Gaussian quanh 0. `wmmd_asymptotic` cũ dùng nhầm Gaussian H₁ làm null → Type-I ≈ 0 (quá bảo thủ). Gamma moment-matched (Gretton 2009) sửa được; audit note trong `wmmd_gamma` ghi rõ bug và cách sửa.
- **M5. Tại sao có hai ngưỡng 0.12 (Growth) và 0.15 (WR Sudden)?** Trả lời hai câu khác nhau: WR_THRESHOLD=0.12 chia TCD vs PCD theo FWHM đỉnh; τ_WR=0.15 sau đó hỏi, trong TCD, có phải Sudden không. (`se_cdt.py:417` vs `:614,678`.)
- **M6. Tại sao dùng cây quyết định thủ công thay vì classifier học?** Không có nhãn lúc vận hành để huấn luyện; tính diễn giải được (vận hành viên thấy đặc trưng nào quyết định); ngưỡng hình học khái quát tốt hơn classifier overfit theo thống kê một luồng. Đã thừa nhận là heuristic (`05:40`).
- **M7. Concept Memory bắt recurrent thế nào?** Ring buffer 8 snapshot sau drift (150 điểm + γ); drift mới được so bằng Standard-MMD với bandwidth trung bình; nếu khoảng cách nhỏ nhất < τ_match=0.15 (≈2× sàn nhiễu) thì gán Recurrent và tái dùng mô hình cache. (`se_cdt.py:229-259`.)
- **M8. Tại sao 30 seed, 14 dataset?** Cân bằng lực thống kê với chi phí: 14×8×30 = 3360 lần chạy; 30 là chuẩn ngành (Demšar) và đủ cho Friedman ở K=8. Dataset phủ sudden/gradual/blip/recurrent/stationary + Electricity semi-real.
- **M9. CD test nói tất cả hòa nhau — vậy đóng góp ở đâu?** F1 hòa với DAWIDD, nhưng FP thấp hơn (9.6 vs 10.4), nhanh hơn ShapeDD ~7× (và nhanh hơn DAWIDD ~9.7×, 6.79s), cộng thêm SE-CDT có phân loại loại drift mà DAWIDD không có. Luận điểm là "ngang độ chính xác, nhanh hơn nhiều, kèm phân loại", không phải "thắng DAWIDD về F1".
- **M10. Tại sao type-specific adaptation chỉ hòa Periodic Retrain trên Mixed?** Periodic huấn luyện lại mỗi 500 mẫu bất kể có drift — lãng phí nhưng hiệu quả trên luồng drift dày. Type-specific đạt độ chính xác ngang (85.73 vs 86.09) với 1.1 vs 8.3 FP/run: cùng phục hồi, ít huấn luyện lại thừa ~8×.

### KHÓ (các bẫy — luyện cho thuộc) — xuyên chương

- **H1. FP headline là 9.6/run, nhưng bảng classification ghi 4846 FP. Rốt cuộc là bao nhiêu?** Hai thực nghiệm khác nhau, cùng detector. 9.6 FP/run là chế độ *detection* (cấu hình IDW, δ=75, cooldown=150, trung bình mỗi run). 4846 là *tổng* FP qua tất cả run của classification benchmark với SE-CDT **(Std)** — Standard MMD, không cooldown, δ=250 lỏng hơn, cộng dồn chứ không trung bình. Không so trực tiếp được; `tab:eval-conventions` định nghĩa ba quy ước. Cấu hình Std được chọn cho classification vì ở đó ưu tiên không bỏ sót sự kiện (EDR 92.6%) hơn là giảm FP.
- **H2. Classifier tiểu loại chỉ ~55% đúng, vậy mà type-specific adaptation thắng baseline ~20pp. Sao một classifier như vậy lại giúp nhiều thế?** Hai lý do, nói thẳng: (1) các kịch bản adaptation (Stepping/Sudden/Mixed) thiên về Sudden, nơi classifier mạnh (Sudden 82.4%, CAT 80.1%) — định tuyến đúng phần lớn *trong vùng này*; (2) ngay cả định tuyến không hoàn hảo vẫn hơn No-Adaptation vì *bất kỳ* reset nào trên một sudden drift thật cũng có ích. Em **chưa** thử adaptation trên luồng nặng Gradual/Incremental, nơi classification rớt còn 27–30% ở mức tiểu loại — đó là future work (`05:65`). Vậy em chỉ khẳng định lợi ích adaptation *cho vùng sudden/recurrent*, không phải phổ quát.
- **H3. Gọi là "không giám sát" nhưng adaptation lại `.fit(X, y)`. Mâu thuẫn không?** Detection và classification hoàn toàn không giám sát (chỉ P(X)). Adaptation cần nhãn để huấn luyện lại — nhưng nó chạy *sau* drift, khi nhãn trễ đã về. Kiến trúc tách early-warning (không giám sát, tức thời) khỏi cập nhật mô hình (có giám sát, trễ). Thiết kế Kafka publish cả MMD trace lẫn prequential accuracy nên hai tín hiệu tách rời. Em không, và không nên, khẳng định *toàn bộ pipeline* không cần nhãn end-to-end.
- **H4. Tách riêng đóng góp của IDW. Trong Bảng I, IDW_MMD, ShapeDD_IDW, SE_CDT giống hệt, và chỉ hơn Standard MMD 0.525→0.531. IDW có thực sự làm gì không?** Với *F1* detection thì gần như không — điều này thẳng thắn và hiện rõ trên bảng. Giá trị của IDW nằm chỗ khác: (i) giảm FP so với ShapeDD (13.1→9.6) và (ii) kết hợp Gamma null cho một kiểm định *được calibrate* (Type-I ≈ α) thay vì ngưỡng MMD chỉnh tay. Ba dòng giống nhau là do thiết kế — SE-CDT *chính là* ShapeDD-IDW cộng một đầu classification không đổi phần detection. Em trình bày là "F1 ngang, calibration và FP tốt hơn", và thừa nhận còn thiếu ablation IDW-riêng vs Gamma-riêng.
- **H5. Gradual 27.2% / Incremental 30.3% là hai lớp thấp nhất — classifier có còn hoạt động trên chúng không?** Có, nhưng yếu, và em nói thẳng. Hai cơ chế nhầm lẫn: (1) phân bố Variance Ratio của Gradual và Incremental chồng lấn (cả hai đều làm phương sai cửa sổ phình lên ở mức trung gian), nên ~47% Incremental bị gán Gradual và ngược lại; (2) với chuỗi drift chậm lặp lại, phân phối bão hoà sau vài sự kiện nên cửa sổ sau drift trùng snapshot cũ và bị Concept Memory gán Recurrent (~57% Gradual). Cả hai vẫn nằm đúng *nhóm lớn* PCD (CAT 80.1% không đổi), và PCD→fine-tune là chiến lược thích ứng đúng dù sao, nên chi phí vận hành của nhầm lẫn Gradual/Incremental là nhỏ. Hướng sửa nằm ở future work: cửa sổ đa thang đo, một đầu phân loại học trên σ(t), hoặc tách bước so khớp Concept Memory khỏi bước phân loại hình dạng.
- **H6. Con số 80.1%/55.3% là "Oracle mode" — cho sẵn vị trí drift thật. End-to-end thật là bao nhiêu?** Thấp hơn, và em nói rõ điều đó (`04:264`). Oracle mode cố ý tách kỹ năng *phân loại* khỏi kỹ năng *phát hiện* — đây là thông lệ chuẩn. End-to-end, các phát hiện sót/lệch vị trí (EDR 92.6% Std) sẽ kéo độ chính xác tiểu loại xuống nữa. Em báo Oracle để đo được trần năng lực của riêng classifier; em không trình bày nó như con số triển khai thực.
- **H7. Cả concept memory lẫn model cache đều dùng ngưỡng 0.15 — có phải cùng một cơ chế không?** Không — và chúng dùng *metric khác nhau*, đây là điểm tinh tế. Concept memory của classifier SE-CDT dùng khoảng cách **Standard-MMD** (τ=0.15 ≈ 2× sàn nhiễu MMD, `se_cdt.py:251`). Model cache của adaptation manager dùng **khoảng cách KS trung bình** trên các đặc trưng (0.15, `adaptation_strategies.py:149,193`). Trùng số không phải là chung calibration — KS 0.15 ≠ MMD 0.15. Chúng độc lập, chỉnh riêng; em nêu việc trùng literal là một "nit" cần dọn, không phải một liên kết có cơ sở.
- **H8. Gamma null giả định mẫu i.i.d.; luồng thì tương quan thời gian. Kiểm định sai cỡ nào?** Với *kiểm định đơn lẻ*: trên Gaussian i.i.d., IDW Type-I ≈ 0.04–0.05 (đúng); trên AR(1) ρ=0.6 IDW lên ~0.075. Nhưng *composite cả pipeline* (lọc đỉnh + nhiều cửa sổ phụ thuộc) thì anti-conservative rõ: 0.115 (iid d=5), 0.080 (iid d=10), **0.140 (AR1) — ~2.8× danh nghĩa**. Em báo thẳng ở `tab:h0_calibration` và diễn giải H0 như một *kiểm tra calibration thực nghiệm, không phải bảo đảm lý thuyết* cho toàn pipeline. Muốn kiểm soát chính xác thì permute (chậm 7×) hoặc block/dependent bootstrap — hướng tương lai đã nêu.
- **H9. Bonferroni trên các đỉnh — vừa quá bảo thủ vừa không đủ (cửa sổ không độc lập)?** Cả hai phê bình đều đúng phần nào. Bonferroni (`adjusted_alpha = alpha/len(peaks)`, `mmd_variants.py:518`) kiểm soát FWER bảo thủ trong mỗi trace; nhưng cửa sổ trượt chồng lấp nên các ứng viên không độc lập, Type-I tổng hợp vẫn trôi trên α (vì thế 0.100 trên AR(1)). Em chọn Bonferroni vì đơn giản và vì một báo động giả đã tốn kém (FWER, không phải FDR, mới là mục tiêu đúng). BH/Holm hoặc hiệu chỉnh có xét phụ thuộc là future work.
- **H10. Em kế thừa triangle của ShapeDD, taxonomy của CDT-MSW, ý tưởng weighting của Bharti. Cái gì thực sự là của em?** Phần *mới* là classifier không giám sát của SE-CDT: đọc *loại* drift từ hình dạng tín hiệu MMD mà không cần nhãn, bằng cách thay tín hiệu accuracy có nhãn (Growth process của CDT-MSW) bằng FWHM-trên-MMD và 9 đặc trưng hình học/thời gian. ShapeDD-IDW là cải tiến kỹ thuật (Gamma null + IDW + dùng chung bandwidth, D1 fix tại `mmd_variants.py:447`); heuristic IDW là bản đơn giản hóa ý tưởng Bharti, được dẫn nguồn đúng (`03:132-134`). Em cẩn thận gọi detection là "cải tiến" và classification là "đóng góp".
- **H11. Nếu giả định joint-drift sai trong một triển khai thực, hệ thống mù. Khi đó làm gì?** Khi đó drift thuần P(Y|X) vô hình với mọi monitor không giám sát dựa P(X) — hạn chế chung của cả họ phương pháp (`05:57`). Giảm thiểu: chạy SE-CDT song song với giám sát accuracy theo nhãn trễ (kiểu DDM); thiết kế Kafka đã publish model.accuracy đúng cho hybrid này. Em loại trường hợp hiếm P(Y|X)-thuần ra khỏi phạm vi và nói rõ.
- **H12. Tại sao tin được 14 dataset synthetic, d≤10, một tập semi-real?** Synthetic là bắt buộc để có ground-truth vị trí drift (không thể tính F1/MTTD trên drift thực không nhãn). *Thứ hạng* tương đối giữa các phương pháp ổn định (Friedman trên rank theo từng dataset, dùng generator chuẩn River/MOA giống các paper ShapeDD/DAWIDD). Em không khẳng định F1 tuyệt đối chuyển sang production — benchmark thực tế có gán nhãn drift là hạng mục future work hàng đầu.

### Câu chốt nên luyện thuộc
> *"SE-CDT đạt độ chính xác phát hiện ngang DAWIDD, nhanh hơn ~7×–9.7× và ít báo động giả hơn, đồng thời bổ sung phân loại loại drift không giám sát. Classifier mạnh ở sudden/recurrent và yếu thật ở incremental — đó là lý do em báo cáo công khai và định tuyến adaptation một cách thận trọng. Đóng góp của luận văn là bộ phân loại loại drift không cần nhãn và pipeline drift-aware end-to-end, không phải một kỷ lục F1 mới."*

---

## Phụ lục — bản đồ nhanh số liệu ↔ nguồn (để tra khi bị hỏi)

| Số liệu | Giá trị | Nguồn |
|---|---|---|
| Detection F1 / Precision / Recall / FP | 0.531 / 0.432 / 0.737 / 9.6 | `table_I_comprehensive_performance.tex` |
| Runtime ShapeDD-IDW vs ShapeDD vs DAWIDD | 0.70s / 5.03s / 6.79s (7.17× / 1× / 0.74×) | `table_III_runtime_stats.tex` |
| CAT / SUB micro / SUB macro | 80.1% / 55.3% / 54.4% | `table_se_cdt_performance_by_type.tex` |
| Per-subtype (Sudden/Blip/Recurrent/Gradual/Incremental) | 82.4 / 60.8 / 71.5 / 27.2 / 30.3 % | cùng bảng |
| SE-CDT Std vs IDW (EDR / FP tổng) | 92.6%/4846 vs 56.7%/657 | `table_se_cdt_aggregate.tex` |
| Type-I error (iid d5/d10/AR1) — IDW / composite | 0.040/0.050/0.075 / 0.115/0.080/0.140 | `table_h0_calibration.tex` |
| Adaptation (Stepping/Sudden/Mixed, Type-Specific) | 74.27 / 73.71 / 85.73 % | `04:423-425` |
| Nemenyi CD (K=8, N=14, α=0.05) | 2.806; DAWIDD rank 3.679 vs đề xuất 3.821 | `table_statistical_tests.tex` |
