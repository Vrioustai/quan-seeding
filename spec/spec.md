# SPEC Sản Phẩm — TrustBite 🕵️‍♂️
### Thám Tử AI Phân Tích Review Quán Ăn

**Track:** Food & Local Delivery (Zone 3)
**Ngày:** 04/06/2026 — Batch 02 · Day 06

---

## 1. Bằng Chứng

### Nỗi đau xuất phát từ đâu?

**Trải nghiệm trực tiếp:**
Khi tìm quán ăn trên Google Maps tại Hà Nội, chúng tôi nhận thấy nhiều quán đạt 4.7–5.0 sao với hàng trăm đánh giá nhưng thực tế chất lượng không tương xứng. Các review thường ngắn, chỉ gồm 1–2 câu ca ngợi chung chung ("Quán ngon lắm, sẽ quay lại!", "Đồ ăn tuyệt vời!!!"), đến từ tài khoản mới lập và không có lịch sử review nào khác.

**Nguồn bên ngoài:**
- Theo báo cáo của BrightLocal (2023), khoảng 30–40% đánh giá trực tuyến trên các nền tảng nhà hàng có dấu hiệu không xác thực. *(Nguồn: brightlocal.com/research/fake-reviews-statistics)*
- Trên các hội nhóm Facebook về ẩm thực Hà Nội (Hà Nội Ăn Gì, Gợi Ý Quán Ăn HN với hàng trăm nghìn thành viên), nhiều bài đăng phàn nàn về "quán trên mạng khác hẳn thực tế" và "mất tiền vì đọc review ảo".
- Dịch vụ seeding review được rao bán công khai trên Facebook và Telegram với giá từ 2.000–5.000 VNĐ/review, thậm chí có gói 100 review chỉ 300.000 VNĐ. *(Quan sát trực tiếp từ các group Facebook)*

**Vấn đề cốt lõi:** Người dùng không có công cụ nào để phân biệt review thật với review seeding, dẫn đến quyết định chọn quán dựa trên thông tin sai lệch.

---

## 2. Lát Cắt để Build

> **Một người dùng** đang phân vân chọn quán ăn tối, **muốn kiểm tra** liệu một quán được bạn bè giới thiệu có đang dùng dịch vụ seeding không, để **AI đưa ra xác suất từng review là ảo**, và **nhận lại** tỷ lệ seeding tổng thể kèm lời khuyên từ LLM Agent có đổi quán hay không.

Đây là lát cắt duy nhất được build và demo — không mở rộng sang tính năng đặt bàn hay gợi ý menu.

---

## 3. AI Product Canvas

### Value — Giá trị

| | |
|---|---|
| **Dành cho ai** | Người dùng Hà Nội tìm quán ăn qua Google Maps, đặc biệt những ai thường bị "lừa" bởi review 5 sao ảo |
| **Đau ở đâu** | Không phân biệt được đâu là review thật — mất thời gian, tiền bạc vì chọn nhầm quán dựa trên đánh giá được mua |
| **AI giải được gì** | Mô hình ML (Random Forest + SBERT + XGBoost + TF-IDF) phân tích ngôn ngữ, hành vi reviewer, và metadata để gắn cờ review nghi vấn — việc mà con người không thể làm thủ công với hàng trăm review |

### Trust — Niềm tin

| Tình huống | Xử lý |
|---|---|
| AI gắn cờ sai một review thật | Người dùng thấy chi tiết từng review bị gắn cờ, có thể tự đánh giá lại bằng cách đọc nội dung và xem xác suất cụ thể |
| Tỷ lệ seeding không phản ánh trải nghiệm thực | Hệ thống hiển thị rõ đây là *dự đoán của mô hình* (không phải sự thật tuyệt đối), kèm ngưỡng cảnh báo người dùng tự điều chỉnh |
| Quán mới không có trong database | Tính năng crawl Apify realtime cho phép phân tích quán bất kỳ khi nhập tên |

### Feasibility — Tính Khả Thi

| Hạng mục | Đánh giá |
|---|---|
| **Chi phí/lượt gọi** | Apify crawl ~50 reviews ≈ $0.05–0.10; Gemini Flash miễn phí ở tier free |
| **Độ trễ** | Inference ML < 1 giây; crawl Apify 30–60 giây; cache 24h giảm thiểu crawl lại |
| **Dữ liệu** | Dataset 1.000+ reviews Hà Nội đã crawl sẵn; model đã train với nhãn bán tự động |
| **Rủi ro lớn nhất** | False positive cao — gắn cờ nhầm review thật có thể ảnh hưởng uy tín quán. Mitigate: hiển thị xác suất thay vì nhị phân tuyệt đối |
| **Ngưỡng dừng** | Nếu accuracy < 70% trên validation set hoặc Apify thay đổi API, dừng tính năng crawl realtime |

### Tín Hiệu Học

Hiện tại prototype chưa thu thập feedback vòng lặp. Hướng phát triển tiếp theo: khi người dùng đánh dấu "review này thật" hoặc "review này ảo", tín hiệu được lưu lại để cải thiện ngưỡng phân loại và bổ sung vào tập huấn luyện.

---

## 4. Tăng Năng Lực hay Tự Động Hóa?

**Quyết định: Tăng năng lực (Augment)**

AI đưa ra xác suất và tỷ lệ tổng hợp, nhưng **người dùng vẫn tự quyết định** có tin vào quán hay không. Hệ thống không tự động chặn quán hay gửi cảnh báo — nó chuẩn bị thông tin để người dùng đưa ra lựa chọn sáng suốt hơn.

**Lý do chọn mức Augment:**
Hậu quả của việc sai (gắn cờ nhầm một quán ngon) là đáng kể — ảnh hưởng đến quyết định kinh doanh và trải nghiệm thực tế của người dùng. Vì vậy con người cần giữ quyền quyết định cuối cùng. Chỉ khi độ chính xác mô hình vượt 90%+ mới nên cân nhắc tự động hóa cảnh báo.

---

## 5. Bốn Đường Đi Của Trải Nghiệm

| Tình huống | Người dùng thấy gì | Sản phẩm xử lý thế nào |
|---|---|---|
| **Đường thuận** — AI đúng, tỷ lệ seeding cao | Banner cảnh báo đỏ, lời khuyên đổi quán từ LLM, danh sách quán thay thế có tỷ lệ seeding thấp hơn | Hiển thị ngay sau khi nhấn "Quét Ngay", tối đa 1–2 giây |
| **Khi AI không chắc** — tỷ lệ seeding gần ngưỡng (18–22%) | Lời khuyên LLM dè dặt hơn: "Cần thêm dữ liệu, hãy đọc kỹ các review được gắn cờ" | LLM prompt được thiết kế để phân biệt 3 mức rủi ro: thấp/trung bình/cao |
| **Khi AI sai** — người dùng không đồng ý | Người dùng có thể điều chỉnh ngưỡng cảnh báo từ 5%–50% trong sidebar và xem lại kết quả ngay lập tức | Slider ngưỡng real-time, không cần tải lại trang |
| **Khi người dùng sửa** | Hiện tại không thu thập — đây là giả định | Roadmap: lưu feedback vào database để cải thiện model |

---

## 6. Những Kiểu Lỗi Đáng Lo Nhất

### Lỗi 1: False Positive Cao — Gắn cờ nhầm review thật

- **Xuất hiện khi:** Review thật viết theo phong cách ngắn gọn, nhiều cảm thán, hoặc người dùng mới tạo tài khoản lần đầu đánh giá
- **Ai chịu thiệt:** Quán ăn bị đánh giá thấp oan; người dùng bỏ qua quán ngon
- **Prototype xử lý:** Hiển thị xác suất cụ thể (0–100%) thay vì chỉ nhãn "Fake/Real"; người dùng điều chỉnh ngưỡng; LLM không nói chắc chắn mà nói "có dấu hiệu nghi vấn"

### Lỗi 2: Quán Mới Không Có Trong Database

- **Xuất hiện khi:** Người dùng nhập tên quán chưa được crawl
- **Ai chịu thiệt:** Người dùng không có thông tin để quyết định
- **Prototype xử lý:** Tính năng crawl Apify realtime — nhập tên quán, hệ thống crawl 50 review mới nhất trong 30–60 giây và phân tích ngay. Cache 24h tránh crawl lại cùng ngày.

### Lỗi 3: LLM Agent Hallucinate — Đưa ra lời khuyên sai

- **Xuất hiện khi:** LLM không nhận đủ context, tự sáng tạo thông tin về quán
- **Ai chịu thiệt:** Người dùng tin vào thông tin sai, đổi quán không cần thiết
- **Prototype xử lý:** Prompt được thiết kế chặt chẽ với chỉ 2 input (tên quán + tỷ lệ seeding); không yêu cầu LLM biết thông tin ngoài; có fallback mock response khi không có API key

---

## 7. Kế Hoạch Kiểm Thử và Bằng Chứng Demo

### Hai đầu vào cho demo

**Happy case (đường thuận):**
- Chọn quán có tỷ lệ seeding cao (> 20%) trong dataset có sẵn
- Expected: Banner cảnh báo đỏ, LLM đưa ra lời khuyên đổi quán, hiển thị top 5 review bị gắn cờ với xác suất cao nhất

**Edge case (AI không chắc / phục hồi):**
- Điều chỉnh ngưỡng slider từ 20% → 5% để thấy cùng một quán nhưng kết quả thay đổi
- Hoặc: nhập tên quán mới qua Apify crawler → show flow end-to-end từ crawl đến phân tích

### Bằng chứng đã chuẩn bị

| Bằng chứng | Trạng thái |
|---|---|
| Dataset reviews Hà Nội (1.000+ reviews từ Google Maps) | ✅ Có sẵn: `reviews_ha_noi_output.csv` |
| Model đã train: Random Forest + SBERT | ✅ Có sẵn: `models/random_forest_seeding.pkl` |
| Model đã train: XGBoost + TF-IDF | ✅ Có sẵn: `models/xgboost_seeding.pkl` |
| Precomputed SBERT embeddings | ✅ Chạy `precompute_embeddings.py` một lần |
| App chạy được: `streamlit run app.py` | ✅ Đang chạy tại localhost:8501 |
| Apify API key cấu hình | ✅ `.env` đã set `APIFY_TOKEN` |
| Gemini API key cấu hình | ✅ `.env` đã set `GEMINI_API_KEY` |
| Fallback mock khi không có LLM key | ✅ Implemented trong `app.py` |

---

## 8. Phân Công

| Thành viên | Phụ trách |
|---|---|
| **[Tên thành viên 1]** | Thu thập dữ liệu, crawl Google Maps, xây dựng dataset `reviews_ha_noi_output.csv`; viết `crawler.py` tích hợp Apify |
| **[Tên thành viên 2]** | Huấn luyện mô hình ML: Random Forest + SBERT (`seeding_review_classifier.ipynb`); tối ưu feature engineering |
| **[Tên thành viên 3]** | Xây dựng giao diện Streamlit (`app.py`): UI/UX, sidebar, chart, custom CSS; tích hợp LLM Agent Gemini/Anthropic |
| **[Tên thành viên 4]** | Huấn luyện mô hình XGBoost + TF-IDF; đánh giá ensemble; viết kịch bản demo |
| **[Tất cả thành viên]** | Viết SPEC, kiểm thử end-to-end, chuẩn bị slides demo |

> *Cập nhật tên thành viên thực tế và mã học viên vào bảng trên.*

---

## Tóm Tắt Kỹ Thuật

### Stack công nghệ

```
Frontend:     Streamlit (Python) — dark mode premium UI
ML Models:    Random Forest + SBERT (keepitreal/vietnamese-sbert)
              XGBoost + TF-IDF (char n-gram + 19 behavioral features)
Ensemble:     Trung bình xác suất của 2 mô hình
LLM Agent:   Google Gemini 2.5 Flash / Anthropic Claude (fallback mock)
Data Source: Google Maps via Apify (compass/crawler-google-places)
Cache:        JSON file, daily TTL (tránh crawl lại cùng ngày)
```

### Kiến trúc flow

```
Người dùng chọn quán
        ↓
[Có trong DB?] ──Yes──→ Load từ CSV + Precomputed embeddings
        ↓ No
[Crawl Apify] → Cache JSON → Load DataFrame
        ↓
[ML Pipeline]
  ├── SBERT encode text → Random Forest → prob_rf
  └── TF-IDF + 19 features → XGBoost → prob_xgb
        ↓
[Ensemble] = (prob_rf + prob_xgb) / 2
        ↓
[Tính tỷ lệ seeding] → [Gọi LLM Agent] → Hiển thị kết quả
```

### 19 Features hành vi cho XGBoost

| Nhóm | Features |
|---|---|
| **Ngôn ngữ** | text_len, word_count, exclamation_count, emoji_count, caps_ratio, unique_char_ratio, avg_word_len, vocab_richness |
| **Nội dung** | has_price, has_recommend, has_negative |
| **Reviewer** | reviewer_review_count, reviewer_photo_count, low_review_count, low_photo_count, new_reviewer |
| **Pattern** | perfect_from_new, rating, rating_5 |

---

## Augment hay Automate?

**Câu trả lời ngắn gọn cho demo:** **Augment.**

AI chuẩn bị thông tin (xác suất, tỷ lệ, danh sách review nghi vấn, lời khuyên LLM), nhưng người dùng giữ toàn quyền quyết định cuối cùng. Hệ thống không tự động chặn hay đánh giá quán — nó làm người dùng sáng suốt hơn, không thay thế phán đoán của họ.

---

## Failure Mode Chính?

**False positive** — gắn cờ nhầm review thật là ảo. Xử lý bằng cách hiển thị xác suất thay vì nhị phân, cho người dùng điều chỉnh ngưỡng, và không đưa ra kết luận tuyệt đối.

---

*Batch 02 · Day 06 — VinUni AI Thực Chiến · 04/06/2026*
