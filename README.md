# TrustBite - Thám Tử AI Phân Tích Review Quán Ăn 🕵️‍♂️

Ứng dụng Streamlit phát hiện review ảo (seeding) tại các quán ăn Hà Nội sử dụng **Ensemble ML** (SBERT + Random Forest, TF-IDF + XGBoost) kết hợp **LLM Agent** (Claude / Gemini).

## 🚀 Tính Năng

### Phân tích ML
- **Ensemble hai mô hình** chạy song song:
  - 🧠 **SBERT + Random Forest** — vector ngữ nghĩa tiếng Việt (`keepitreal/vietnamese-sbert`) + rating/reviewer count
  - ⚡ **TF-IDF + XGBoost** — char n-gram + 19 đặc trưng hành vi reviewer (emoji, caps ratio, perfect_from_new…)
  - 🔀 **Ensemble** — trung bình xác suất của cả 2 mô hình
- Chọn mô hình phân loại ngay trên sidebar (ensemble / xgb / rf)
- Ngưỡng cảnh báo seeding tuỳ chỉnh (5–50%)

### LLM Agent
- Gọi **Claude** (Anthropic) → **Gemini** → mock response theo thứ tự fallback
- Prompt 2 bước: xác nhận địa điểm là quán ăn trước, sau đó mới phân tích
- Đề xuất quán thay thế có fake_ratio thấp nhất khi phát hiện seeding cao

### Crawl Realtime
- Nhập tên quán bất kỳ ở Hà Nội → crawl review qua **Apify** (`compass/crawler-google-places`)
- Cache JSON 24h — lần sau chọn lại trả kết quả ngay
- **Validate input trước khi crawl** — chặn non-restaurant (khách sạn, bệnh viện, ngân hàng, spa…) với 2 lớp kiểm tra:
  - Structural: độ dài, có chữ cái hợp lệ
  - Keyword blocklist: từ chối các loại địa điểm không phải ẩm thực

### Giao diện
- Dark mode premium, responsive
- Top 5 review bị gắn cờ nghi vấn kèm xác suất fake
- Biểu đồ phân bổ review thật / ảo
- Nút tải **Demo Report** (product canvas, AI flows, failure modes)

## ⚙️ Cài Đặt

### 1. Cài thư viện
```bash
pip install streamlit joblib pandas numpy scikit-learn sentence-transformers google-generativeai anthropic
```

### 2. Cấu hình API Keys

Tạo file `.env` (KHÔNG commit):
```env
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_anthropic_key   # tuỳ chọn, ưu tiên hơn Gemini
APIFY_TOKEN=your_apify_token           # cần cho crawl realtime
```

Hoặc dùng `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_gemini_key"
```

### 3. Chạy ứng dụng
```bash
streamlit run app.py
```

## 📁 Cấu Trúc Project

```
├── app.py                        # Main Streamlit app
├── crawler.py                    # Apify crawl + 24h cache
├── debug_check.py                # Script kiểm tra dữ liệu
├── demo-report.html              # Hackathon product canvas & demo report
├── precompute_embeddings.py      # Pre-compute SBERT embeddings cho dataset
├── seeding_review_classifier.ipynb  # Training notebook
├── reviews_ha_noi_output.csv     # Dataset reviews Hà Nội
├── models/
│   ├── random_forest_seeding.pkl
│   ├── xgboost_seeding.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── metadata_scaler.pkl
│   └── sbert_embeddings.npy      # Pre-computed embeddings (fast startup)
├── cache/                        # JSON cache crawl (24h TTL)
└── .streamlit/
    └── config.toml
```

## 🛡️ Lưu Ý Bảo Mật

- **Không** commit `.env` hoặc `secrets.toml` lên GitHub
- API keys được đọc từ `.env` → `st.secrets` → biến môi trường
