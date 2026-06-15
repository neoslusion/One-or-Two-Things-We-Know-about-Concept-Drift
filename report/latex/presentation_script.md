# Kịch bản thuyết trình bảo vệ luận văn — 20 phút (không tính backup)

> **Cách dùng:** Mỗi slide có mốc thời gian *(tích luỹ)* và lời thoại để nói trôi chảy.
> Phần *(in nghiêng trong ngoặc)* là chỉ dẫn sân khấu — đừng đọc thành tiếng.
> Thuật ngữ kỹ thuật để nguyên tiếng Anh cho tự nhiên (concept drift, MMD, drift, false alarm…).

## Mốc thời gian tổng (bám theo để không cháy giờ)

| Phần | Slide | Xong ở mốc |
|---|---|---|
| Mở đầu + Giới thiệu | Title → Contributions | **~4:30** |
| Background (MMD) | MMD → Triangle | **~5:50** |
| Tầng 1: Detection | IDW → Gamma → In-Action | **~8:20** |
| Tầng 2: Classification | CDT-MSW → Hybrid | **~11:40** |
| Tầng 3: Adaptation | Type-Aware | **~12:30** |
| Đánh giá thực nghiệm | Datasets → Result 3 | **~17:40** |
| Triển khai Kafka | Kafka | **~18:30** |
| Kết luận + Cảm ơn | Conclusion → Thank you | **~20:00** |

> **Nếu trễ giờ:** lướt nhanh 2 slide *Per-Dataset F1* và gộp *Runtime* — đó là chỗ "nén" được nhiều nhất.
>
> **Lưu ý:** phần mở đầu (Motivation + Distribution) đã được làm dày hơn ~45s vì đây là phần "ăn tiền". Để bù: ở *Contributions* chỉ điểm nhanh 3 ý (~20s), và *Per-Dataset F1 + Runtime* nói thật gọn. Các mốc từ Background trở đi vẫn giữ nguyên — phần bù sẽ kéo lại đúng 20 phút.

---

## 0:00 — Slide tiêu đề *(30 giây)*

Kính chào quý thầy cô trong hội đồng. Em tên là Lê Phúc Đức. Hôm nay em xin trình bày luận văn thạc sĩ với đề tài **"Nghiên cứu và phát triển hệ thống tự động phát hiện concept drift và cập nhật mô hình học máy thích ứng"**, dưới sự hướng dẫn của thầy Thoại Nam và thầy Nguyễn Quang Hùng. Em xin phép bắt đầu.

## 0:30 — Outline / Agenda *(20 giây)*

Bài trình bày của em gồm: giới thiệu bài toán; nền tảng về tín hiệu MMD; sau đó là **ba tầng đóng góp chính** — phát hiện, phân loại, và thích ứng; cuối cùng là đánh giá thực nghiệm và một nguyên mẫu triển khai trên Apache Kafka.

*(Các slide "Agenda" highlight màu giữa các phần chỉ cần lướt qua 2–3 giây: "Mình sang phần…")*

---

## 0:50 — Motivation & Background *(~70 giây — phần "ăn tiền": nói chậm, rõ, có nhịp nghỉ)*

*(Mở bằng một câu hỏi để kéo sự chú ý — đừng vội vào công thức.)*

Thưa hội đồng, trước khi đi vào kỹ thuật, em xin bắt đầu bằng một câu hỏi rất đơn giản: *điều gì xảy ra với một mô hình học máy **sau** khi nó đã được đưa vào vận hành?*

Em xin ví von thế này: huấn luyện một mô hình cũng giống như đào tạo một nhân viên mới — ta cho họ xem thật nhiều ví dụ trong **quá khứ**, rồi bảo *"cứ theo đó mà quyết định"*. Nhưng có một vấn đề: thế giới ngoài kia **không đứng yên** chờ mô hình của ta. Dữ liệu của hôm nay, từ từ, không còn giống dữ liệu lúc ta huấn luyện nữa. Hiện tượng đó gọi là **concept drift** — và nó chính là nguyên nhân *âm thầm* khiến mọi mô hình "già đi" và sai dần theo thời gian.

*(Giờ mới đưa công thức — nhẹ nhàng.)* Phát biểu một cách hình thức: drift xảy ra khi phân phối đồng thời *P(X, Y)* — tức **cách mà dữ liệu *X* và nhãn *Y* cùng xuất hiện** — ở lúc vận hành đã **khác** so với lúc huấn luyện.

*(Đây là ý quan trọng nhất của cả slide — nhấn mạnh, chỉ tay vào công thức.)* Và chìa khoá nằm ở đây: ta có thể **phân rã** phân phối đó thành hai phần — *P(X, Y) = P(X) × P(Y | X)*. Khi tách ra như vậy, ta thấy ngay drift chỉ có thể đến từ **một trong hai nguồn**: hoặc *P(X)* — phân phối **đầu vào** — dịch chuyển; hoặc *P(Y | X)* — chính **quy luật** ánh xạ từ đầu vào sang nhãn — thay đổi. Hai nguồn này nghe thì gần giống nhau, nhưng hệ quả lại **rất khác nhau** — và đó chính là nền tảng cho toàn bộ luận văn của em.

## 2:00 — Concept Drift: Distribution vs Pattern *(~55 giây — một ví dụ xuyên suốt, kể như kể chuyện)*

Để thấy thật rõ hai nguồn drift đó, em xin lấy **một ví dụ quen thuộc và theo nó tới cùng: phát hiện gian lận thẻ tín dụng.** Ở đây, *X* là đặc trưng của một giao dịch — số tiền, thời điểm, địa điểm, loại cửa hàng; còn *Y* là nhãn: *gian lận* hay *hợp lệ*.

*(Trường hợp 1 — nói tay chỉ vào "P(X)".)* **Thứ nhất là virtual drift — chỉ *P(X)* đổi.** Cứ tới mùa Tết, mọi người chi tiêu nhiều hơn hẳn, các giao dịch giá trị lớn tăng vọt. Phân phối **đầu vào** *X* đã dịch chuyển — *nhưng* định nghĩa thế nào là gian lận thì vẫn **y nguyên**. Ranh giới quyết định không đổi; chỉ là mô hình gặp nhiều dữ liệu ở vùng nó ít thấy lúc huấn luyện, nên bắt đầu **báo động giả nhiều hơn** — *đúng như mũi tên "False Alarms ↑" trên slide trước.*

*(Trường hợp 2 — nhấn mạnh "đây mới là loại nguy hiểm".)* **Thứ hai là real drift — *P(Y | X)* đổi.** Kẻ gian nghĩ ra chiêu mới: thay vì một giao dịch lớn, chúng chia nhỏ thành nhiều giao dịch lắt nhắt. Bây giờ một giao dịch **trông y hệt** một giao dịch hợp lệ ngày xưa lại là gian lận — tức là **quy luật** từ *X* sang *Y* đã đổi, ranh giới quyết định dịch chuyển. Đây mới là loại **nguy hiểm**, vì nó trực tiếp khiến mô hình ra quyết định sai — *chính là "Accuracy ↓".*

*(Câu chốt — đây là bản lề chuyển sang toàn bộ luận văn.)* Và phân biệt được hai loại này chính là **chìa khoá**: nó giải thích vì sao luận văn không dừng ở việc *phát hiện* có drift, mà còn phải **phân loại** drift — bởi mỗi loại đòi hỏi một cách xử lý hoàn toàn khác nhau.

## 2:55 — Objectives & System Architecture *(45 giây)*

Hệ thống của em là một **pipeline ba tầng**. *(chỉ vào hình)* Tầng 1 — **Detection** — trả lời câu hỏi *"khi nào"* có drift: nhanh, và không cần nhãn. Tầng 2 — **Classification** — trả lời *"drift loại gì"*. Tầng 3 — **Adaptation** — quyết định *"cập nhật mô hình ra sao"*.

Tương ứng là ba mục tiêu: phát hiện nhanh và đáng tin với rất ít báo động giả; phân loại drift hoàn toàn không cần nhãn; và đánh giá end-to-end trên Apache Kafka.

## 3:40 — Related Works: The Gap *(30 giây)*

Em đặt các công trình hiện có lên hai trục: *có cần nhãn hay không*, và *có phân loại được drift hay không*. Nhóm như **DDM, EDDM** theo dõi suy giảm độ chính xác — nhanh, nhưng cần nhãn. **CDT-MSW** phân loại rất tốt nhưng phụ thuộc hoàn toàn vào nhãn. Các phương pháp dựa trên MMD như **ShapeDD** thì không cần nhãn, nhưng chỉ phát hiện chứ không phân loại.

*(chỉ vào góc dưới phải)* Khoảng trống — góc chưa ai lấp — chính là **phân loại drift mà không cần nhãn**. Đó đúng là vị trí của **SE-CDT** trong luận văn này.

## 4:10 — Contributions *(20 giây — điểm nhanh 3 đóng góp, đừng kể dài)*

Tóm lại ba đóng góp. **Một**, ở tầng phát hiện: một phiên bản ShapeDD cải tiến với **trọng số IDW** và **p-value xấp xỉ Gamma**, nhanh hơn khoảng 7 lần. **Hai**, ở tầng phân loại: **SE-CDT**, bộ phân loại drift hoàn toàn không nhãn, đọc *hình dạng* của tín hiệu MMD. **Ba**, ở tầng thích ứng: chính sách cập nhật theo loại drift, kiểm chứng end-to-end trên Kafka.

---

## 4:00 — MMD: The Unsupervised Drift Signal *(50 giây)*

*(Sang phần nền tảng.)* Tại sao lại không nhãn? Vì trong các luồng IIoT, nhãn đến **trễ** hoặc **rất đắt**. Nên thay vì chờ nhãn, ta giám sát trực tiếp phân phối đầu vào *P(X)*.

Công cụ để làm việc đó là **Maximum Mean Discrepancy — MMD**. Ý tưởng: ánh xạ các điểm dữ liệu vào một không gian hàm RKHS, rồi so sánh *"tâm khối"* — mean embedding — của hai cửa sổ dữ liệu. MMD bình phương bằng độ tự tương đồng trong P, cộng tự tương đồng trong Q, **trừ** tương đồng chéo giữa hai bên. Nếu hai phân phối giống nhau, MMD xấp xỉ 0; nếu khác nhau, MMD lớn — đó là một ứng viên drift.

## 4:50 — From MMD Trace to ShapeDD's Triangle *(60 giây)*

Ta cho một **cửa sổ trượt** kích thước *2·l₁* chạy dọc luồng, tách thành nửa tham chiếu và nửa kiểm tra, rồi tính MMD ở mỗi bước. Kết quả là một chuỗi liên tục *σ(t)* — em gọi là **MMD trace**. *(chỉ hình)* Khi dữ liệu ổn định, trace dao động quanh một **nền nhiễu** thấp; khi cửa sổ quét qua điểm drift, trace dâng lên thành một **đỉnh**.

Phát hiện then chốt của ShapeDD: một drift đột ngột tạo ra một đỉnh **hình tam giác** — vì khi cửa sổ trượt qua điểm drift, mức độ "nhiễm" tăng tuyến tính. Nên ta đi tìm *hình tam giác* đó, chứ không chỉ so ngưỡng đơn thuần. Chiều cao đỉnh chính là **độ lớn drift**.

Nhưng ShapeDD gốc có một **nút thắt**: kiểm định ý nghĩa bằng permutation test phải dựng lại ma trận kernel hàng nghìn lần — rất chậm. Và đây chính là chỗ hai đóng góp của em bước vào.

---

## 5:50 — Inverse Density-Weighted MMD (IDW-MMD) *(50 giây)*

MMD chuẩn có một hạn chế: các **cụm dày đặc** ở vùng trung tâm chi phối thống kê, làm những drift sớm ở **vùng biên** bị lu mờ.

Giải pháp của em là **trọng số nghịch mật độ — IDW**: mỗi điểm được gán trọng số tỉ lệ nghịch với mật độ cục bộ của nó — *wᵢ tỉ lệ với 1 trên căn dᵢ*, với *dᵢ* là tổng kernel quanh điểm đó. Điểm ở lõi dày thì trọng số nhỏ; điểm thưa ở biên thì trọng số lớn. *(chỉ hình)* Kết quả: **nhạy hơn** nhiều với thay đổi sớm ở biên, mà **không tăng** báo động giả trên phần dữ liệu chính.

## 6:40 — Statistical Calibration (Gamma-Null) *(50 giây)*

Nút thắt thứ hai là permutation test — ShapeDD gốc chạy tới 2500 hoán vị để ước lượng p-value, mỗi lần lại dựng lại kernel. Thay vào đó em dùng **xấp xỉ Gamma**: dưới giả thuyết *không có drift*, thống kê MMD xấp xỉ một phân phối **Gamma**. Em chỉ cần ước lượng hai moment đầu — trung bình và phương sai — từ một số ít hoán vị, khoảng 20 lần, và **tái sử dụng** ma trận kernel.

Kết quả khớp về độ chính xác nhưng nhanh hơn đầu-cuối khoảng **7 đến 10 lần**.

## 7:30 — ShapeDD-IDW in Action *(50 giây)*

Slide này gói toàn bộ tầng phát hiện vào **một ví dụ**. *(chỉ từng panel)* **(a)** Luồng dữ liệu có một drift thật. **(b)** Ta tính MMD trace, làm mượt, rồi giữ những điểm vượt ngưỡng *trung bình cộng một độ lệch chuẩn* — đó là các đỉnh ứng viên. Để ý: một **blip** ngắn cũng tạo bướu nhỏ, nhưng nó nằm *dưới* ngưỡng nên bị bỏ qua — tức là **không** báo động giả. **(c)** Tại đỉnh ứng viên, p-value Gamma cùng hiệu chỉnh Bonferroni xác nhận; cả một cụm báo động liên tiếp được **gộp lại thành đúng một sự kiện drift**, nằm trong dung sai cho phép.

Tóm lại tầng 1: một phát hiện đã hiệu chỉnh, không cần nhãn, cho mỗi drift thật.

---

## 8:20 — CDT-MSW (baseline) *(40 giây)*

*(Sang tầng phân loại.)* Phương pháp tham chiếu là **CDT-MSW** của Guo và cộng sự: chia drift thành nhóm *tạm thời — TCD* và nhóm *tiến triển — PCD*, dựa trên phương sai của đường suy giảm độ chính xác. Nhưng nó cần **nhãn ở mọi bước** để tính độ chính xác — bất khả thi với luồng IIoT, nơi nhãn đến trễ hoặc không bao giờ đến.

Ý tưởng của em: **thay đường độ chính xác bằng *hình dạng* của tín hiệu MMD**. Cùng một phân biệt TCD/PCD, nhưng giờ làm được mà *không cần nhãn* lúc chạy.

## 9:00 — SE-CDT: 9 Attributes *(50 giây)*

SE-CDT đọc hình dạng của *σ(t)* quanh điểm phát hiện bằng **9 thuộc tính**. Sáu thuộc tính **hình học** mô tả cấu trúc đỉnh: số đỉnh, tỉ lệ độ rộng, hệ số biến thiên, tỉ số tín hiệu trên nhiễu, và hai thuộc tính chuyên cho blip. Ba thuộc tính **thời gian** mô tả hành vi theo thời gian: độ mạnh xu hướng tuyến tính, điểm bước nhảy, và độ đơn điệu.

*(chỉ hình)* Cộng thêm một tín hiệu thứ mười tính từ **dữ liệu thô** — Variance Ratio — chuyên để phân biệt các drift chậm. Hình bên minh hoạ rõ: đỉnh nhọn, nền nhiễu, độ rộng nửa đỉnh — tất cả đều đo được một cách định lượng.

## 9:50 — SE-CDT: Variance Ratio (VR) *(40 giây)*

Vì sao cần Variance Ratio? Vì **Gradual** và **Incremental** đều tạo bướu rộng, phẳng, *giống nhau* — hình dạng đơn thuần không tách được. Mẹo: đọc **phương sai của cửa sổ dữ liệu thô**. Gradual là một *pha trộn xác suất* giữa hai khái niệm, nên phương sai **vọt lên**; Incremental là một phân phối *dịch chuyển liên tục*, nên phương sai **phẳng**. VR là tỉ số phương sai lớn nhất trên phương sai nền: lớn hơn 1.3 là Gradual, nhỏ hơn 1.1 là Incremental.

## 10:30 — SE-CDT: Decision Logic & Signatures *(40 giây)*

Toàn bộ quy tắc gói trong một **cây quyết định đơn giản** *(chỉ hình)*: nhìn hình dạng đỉnh; nếu là đỉnh **hẹp** — nhóm TCD — thì phân biệt Sudden, Blip, Recurrent theo số đỉnh và bộ nhớ khái niệm; nếu là **bướu rộng** — nhóm PCD — thì dùng VR để tách Gradual với Incremental. Hình bên phải là *"chữ ký"* đặc trưng của từng loại drift trên tín hiệu MMD — nhìn là nhận ra ngay.

## 11:10 — Hybrid Pipeline *(30 giây)*

Một chi tiết thiết kế quan trọng: vì sao dùng **hai biến thể MMD**? IDW-MMD rất *nhọn*, ổn định sai số loại I — lý tưởng để **phát hiện**; nhưng trọng số mật độ lại làm phẳng các chuyển tiếp từ từ. Còn MMD *chuẩn* giữ nguyên hình học tự nhiên của trace — cần cho **phân loại**. Nên em chạy IDW-MMD *liên tục* để phát hiện, rồi chỉ tính MMD chuẩn *tại điểm phát hiện* để lấy hình dạng đem phân loại.

---

## 11:40 — Layer 3: Type-Aware Adaptation *(50 giây)*

Tầng ba: **thích ứng theo loại**. Khi đã biết loại drift, ta chọn cách cập nhật *rẻ nhất mà đúng*: Sudden thì **reset** toàn bộ; Incremental thì **cập nhật liên tục**; Gradual thì **cửa sổ có trọng số** — tức warm-start; Recurrent thì **tái dùng** mô hình từ bộ nhớ; còn Blip thì **bỏ qua**, không làm gì cả.

Mục tiêu là tránh kiểu *"một cỡ cho tất cả"* — vừa tránh quên kiến thức cũ, vừa giảm chi phí tính toán. Trong triển khai Kafka, tín hiệu này kích hoạt các microservice tải-nóng hoặc tinh chỉnh mô hình.

---

## 12:30 — Datasets & Experimental Setup *(40 giây)*

*(Sang phần đánh giá.)* Em đánh giá trên **14 bộ dữ liệu**, nhóm theo loại: các drift *đột ngột trên P(X)* như Gaussian shift và bốn mức Random Uniform — từ nhẹ tới cực nặng; *blip* với RBF; một bộ *bán thực* Electricity với 10 điểm drift; các bộ STAGGER cho *recurrent* và *stationary*.

Đặc biệt, có **ba bộ "nhóm chứng"** — SEA, Hyperplane, LED — chỉ đổi *P(Y|X)* chứ không đổi *P(X)*. Mục đích: kiểm chứng rằng bộ phát hiện không-nhãn của em *không* "bắn nhầm" vào loại drift mà về nguyên tắc nó không nên thấy.

## 13:10 — Classification Setup *(30 giây)*

Riêng phần phân loại, em đánh giá ở chế độ **oracle**: bộ phân loại được cho trước *vị trí* drift, và chỉ chấm điểm ở *bước phân loại* — để tách kỹ năng phân loại khỏi kỹ năng phát hiện. Sáu kịch bản × 30 hạt giống × 10 sự kiện, tổng **1800 sự kiện**. Hai chỉ số: **CAT** — phân biệt TCD/PCD, chính là quyết định dẫn dắt thích ứng; và **SUB** — đúng cả 5 loại con.

## 13:40 — Result 1a: Detection (CD diagram) *(40 giây)*

Kết quả phát hiện. *(chỉ hình)* Đây là biểu đồ **Critical Difference** theo kiểm định Nemenyi trên cả 14 bộ. Các phương pháp nối nhau bằng thanh đậm là *không khác biệt có ý nghĩa thống kê*. SE-CDT đạt hạng **3.89**, **hoà về mặt thống kê** với DAWIDD — state-of-the-art — ở hạng 3.46. Điểm cải thiện then chốt: cùng mức F1, nhưng **tỉ lệ báo động giả thấp hơn**.

## 14:20 — Result 1b: Comprehensive *(30 giây)*

Bảng tổng hợp: F1 mạnh và ổn định trên các loại drift đa dạng, với số báo động giả giữ **dưới 10**. Trên các drift *P(X)* tổng hợp, F1 từ 0.69 đến 0.74; còn ba bộ nhóm chứng *P(Y|X)* cho F1 thấp — *đúng như thiết kế*, vì không có thay đổi *P(X)* nào để phát hiện cả.

## 14:50 — Per-Dataset F1 (1/2 + 2/2) *(20 giây — LƯỚT NHANH)*

Hai slide này là chi tiết F1 từng bộ, em xin phép **lướt nhanh**. Thông điệp chính: SE-CDT bám sát hoặc vượt các baseline trên hầu hết các bộ, và đặc biệt **ổn định khi tăng dần mức độ** drift từ nhẹ tới cực nặng.

## 15:10 — Result 1c: Type-I Error *(30 giây)*

Về sai số loại I trên dữ liệu *tĩnh*: bộ kiểm định **IDW-MMD đơn lẻ** giữ rất sát mức danh nghĩa 0.05, trong khoảng 0.025–0.075. Trace tổng hợp của SE-CDT thì hơi "phóng khoáng" hơn, 0.08–0.14, do phụ thuộc vào việc chọn đỉnh và cửa sổ trượt. Em báo cáo **trung thực** đây là một giới hạn của pipeline.

## 15:40 — Result 1d: Runtime *(30 giây — gộp 2 slide)*

Về tốc độ: nhờ xấp xỉ Gamma, SE-CDT chạy khoảng **0.6 giây cho 10 nghìn mẫu** — nhanh hơn ShapeDD gốc khoảng **7 lần**, đưa một phương pháp kernel về cùng nhóm tốc độ với các kiểm định nhẹ như KS hay D3. Thông lượng trên **16 nghìn mẫu mỗi giây** — độ trễ gần như không đáng kể trong pipeline streaming.

## 16:10 — Result 2: Unsupervised Classification *(50 giây — ĐIỂM NHẤN)*

Đây là kết quả em tâm đắc nhất — **phân loại không nhãn**. CAT đạt **80.1%**, tức phân biệt đúng TCD/PCD — quyết định dẫn dắt thích ứng — ở mức recall **92.6%**, hoàn toàn *không nhãn*. Theo từng loại: Sudden 82.4%, Recurrent 71.5% nhờ **bộ nhớ khái niệm**, Blip 60.8%.

Em cũng **thẳng thắn về điểm yếu**: SUB tổng thể là 55.3% — hai loại chậm Gradual và Incremental chồng lấn trên Variance Ratio, đó là điểm yếu hiện tại. CDT-MSW *có nhãn* thì SUB cao hơn — nhưng nó **cần nhãn**, còn em thì không.

## 17:00 — Result 3: Model Adaptation *(40 giây)*

Cuối cùng là thích ứng. *(chỉ hình)* So bốn chiến lược trên luồng Stepping và Sudden: chính sách **theo-loại** của em đạt 71.4% và 70.8% — cao hơn rõ rệt so với retrain định kỳ, retrain đơn giản, và không thích ứng. Mức tăng là **cộng 17 điểm phần trăm** so với không thích ứng, với *p nhỏ hơn 10 mũ trừ 15* — tức không phải ngẫu nhiên.

---

## 17:40 — End-to-End Deployment on Apache Kafka *(50 giây)*

Để chứng minh hệ thống chạy *thật*, em triển khai một nguyên mẫu **end-to-end trên Apache Kafka**. *(chỉ từng panel)* Hình cho thấy: luồng dữ liệu có drift tại mẫu 1500; độ chính xác mô hình tụt xuống; và SE-CDT phát hiện với p-value cỡ *10 mũ trừ 6* — rất tự tin. CPU và bộ nhớ giữ **ổn định** suốt quá trình, nên chi phí của SE-CDT là không đáng kể trong một pipeline thực.

---

## 18:30 — Conclusion *(60 giây)*

Tóm lại toàn luận văn: ở **phát hiện**, ShapeDD-IDW với p-value Gamma cho một kiểm định đã hiệu chỉnh, không nhãn, và nhanh hơn. Ở **phân loại**, SE-CDT hoạt động không nhãn, đạt CAT 80.1% và recall 92.6%. Ở **thích ứng**, định tuyến theo-loại mang lại mức tăng đáng kể. Và toàn hệ thống đã được kiểm chứng trên một nguyên mẫu Kafka.

Về **hạn chế**, em xin thẳng thắn: hai loại drift chậm vẫn còn khó phân biệt; ngưỡng bộ nhớ khái niệm hiện chọn theo kinh nghiệm; và phần thích ứng mới thử nghiệm nhiều trên các luồng đột ngột. Đây cũng chính là các hướng phát triển tương lai của luận văn.

## 19:30 — Thank You / Q&A *(30 giây)*

Em xin chân thành cảm ơn quý thầy cô đã lắng nghe. Em rất mong nhận được góp ý từ hội đồng, và em xin sẵn sàng trả lời các câu hỏi ạ.

---

## Ghi chú luyện tập

- **Câu chuyển vàng** giữa các phần (học thuộc): *"Như vậy ta đã có cách **phát hiện** drift — nhưng phát hiện thôi chưa đủ, ta cần biết **loại** drift để xử lý đúng cách…"* (cầu nối Tầng 1 → Tầng 2); *"Biết được loại drift rồi, câu hỏi cuối là: cập nhật mô hình **ra sao**?"* (Tầng 2 → Tầng 3).
- **3 con số phải nhớ tuyệt đối:** CAT **80.1%** / recall **92.6%**; nhanh hơn **~7×**; thích ứng **+17 điểm**.
- **Khi bị hỏi về SUB 55.3%:** đừng phòng thủ — nói ngay *"Đây là điểm yếu em đã nhận diện rõ; nó nằm gọn trong nhóm PCD nên không ảnh hưởng quyết định thích ứng, và em có slide backup phân tích chi tiết."*
- **Tốc độ nói:** ~130–140 từ/phút. Nếu thấy mình đọc nhanh hơn, hãy chủ động dừng 1 nhịp ở các con số kết quả.
- Mỗi lần qua slide "Agenda" highlight, hít một hơi — đó là nhịp nghỉ tự nhiên của bạn.

Chúc bạn bảo vệ thật tốt! 💪
