"""
crawler.py — Apify crawl + daily cache cho TrustBite
Dựa trên CrawlData.py, tích hợp vào Streamlit app với:
  - Option A: Cache kết quả theo ngày (tránh crawl lại cùng quán)
  - Option C: Dùng Apify compass/crawler-google-places
  - Mock reviewer_total_reviews & reviewer_is_local_guide khi Apify trả về None
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from apify_client import ApifyClient

# Thư mục cache nằm cùng cấp với app.py
CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Xóa cache files cũ hơn CACHE_TTL_DAYS ngày khi module được import
CACHE_TTL_DAYS = 7

def _cleanup_old_cache(ttl_days: int = CACHE_TTL_DAYS) -> None:
    """Xóa tất cả file cache cũ hơn ttl_days ngày."""
    cutoff = date.today() - timedelta(days=ttl_days)
    for f in CACHE_DIR.glob("*.json"):
        # Tên file có dạng <slug>_YYYY-MM-DD.json
        parts = f.stem.rsplit("_", 3)  # tách phần ngày ở cuối
        if len(parts) >= 4:
            date_str = "_".join(parts[-3:])  # YYYY-MM-DD
        else:
            date_str = parts[-1]
        try:
            file_date = date.fromisoformat(date_str)
            if file_date < cutoff:
                f.unlink(missing_ok=True)
        except ValueError:
            pass  # Tên file không đúng format — bỏ qua

_cleanup_old_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(restaurant_name: str) -> Path:
    """Trả về đường dẫn file cache cho quán + ngày hôm nay."""
    slug = "".join(c if c.isalnum() else "_" for c in restaurant_name.lower())[:60]
    return CACHE_DIR / f"{slug}_{date.today().isoformat()}.json"


def _mock_missing(reviews: list) -> list:
    """
    Điền giá trị hợp lý cho reviewer_total_reviews, reviewer_photo_count,
    và reviewer_is_local_guide khi Apify không trả về (None/null).

    Phân phối reviewer_total_reviews dựa trên thực tế Google Maps:
      - ~35% reviewer chỉ viết 1–5 review (có thể seeder hoặc người dùng mới)
      - ~35% viết 6–30 review (người dùng bình thường)
      - ~30% viết 30+ review (người dùng tích cực / Local Guide)

    reviewer_photo_count: tương quan thuận với review_count (~30–50% số review).
    reviewer_is_local_guide: ~20% reviewer là Local Guide (số liệu Google công bố).

    Seed dựa trên nội dung thực tế của batch để mỗi quán có phân phối riêng biệt.
    """
    # Seed từ hash nội dung batch — mỗi quán khác nhau cho kết quả khác nhau
    seed_val = hash(tuple(r.get("reviewer_id") or r.get("review_id") or "" for r in reviews)) % (2**32)
    rng = np.random.default_rng(seed=seed_val)

    choices_reviews = [1, 2, 3, 5, 10, 15, 25, 50, 100, 200]
    probs_reviews   = [0.08, 0.10, 0.10, 0.10, 0.15, 0.12, 0.12, 0.10, 0.08, 0.05]

    for r in reviews:
        if r.get("reviewer_total_reviews") is None:
            r["reviewer_total_reviews"] = int(
                rng.choice(choices_reviews, p=probs_reviews)
            )
        if r.get("reviewer_photo_count") is None:
            # photo_count ~ 30–50% của review_count, tối thiểu 0
            review_cnt = r["reviewer_total_reviews"]
            r["reviewer_photo_count"] = int(max(0, round(review_cnt * rng.uniform(0.0, 0.5))))
        if r.get("reviewer_is_local_guide") is None:
            r["reviewer_is_local_guide"] = bool(rng.random() < 0.20)

    return reviews


def _build_run_input(restaurant_name: str, max_reviews: int) -> dict:
    """Tạo payload cho Apify actor — clone từ CrawlData.py của dự án gốc."""
    return {
        "searchStringsArray": [restaurant_name],
        "locationQuery": "Ha Noi, Vietnam",
        "maxCrawledPlacesPerSearch": 1,
        "language": "vi",
        "categoryFilterWords": [],
        "searchMatching": "all",
        "website": "allPlaces",
        "skipClosedPlaces": False,
        "scrapePlaceDetailPage": False,
        "scrapeTableReservationProvider": False,
        "scrapeOrderOnline": False,
        "includeWebResults": False,
        "scrapeDirectories": False,
        "maxQuestions": 0,
        "scrapeContacts": False,
        "scrapeSocialMediaProfiles": {
            "facebooks": False, "instagrams": False, "youtubes": False,
            "tiktoks": False, "twitters": False,
        },
        "maximumLeadsEnrichmentRecords": 0,
        "leadsEnrichmentDepartments": ["sales", "marketing"],
        "verifyLeadsEnrichmentEmails": False,
        # --- Review config ---
        "maxReviews": max_reviews,
        "reviewsSort": "newest",
        "reviewsFilterString": "",
        "reviewsOrigin": "all",
        "scrapeReviewsPersonalData": True,   # Bắt buộc để lấy authorNumberOfReviews
        # --- Image ---
        "maxImages": 0,
        "scrapeImageAuthors": False,
        # --- Địa lý ---
        "countryCode": "vn",
        "city": "Ha Noi",
        "state": "Ha Noi",
        "county": "",
        "postalCode": "",
        "startUrls": [],
        "placeIds": [],
        "allPlacesNoSearchAction": "",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crawl_restaurant(restaurant_name: str, max_reviews: int = 50) -> pd.DataFrame:
    """
    Crawl review cho một quán ăn bằng Apify.

    - Nếu đã crawl hôm nay → đọc từ cache, trả về ngay (Option A).
    - Nếu chưa → gọi Apify compass/crawler-google-places (Option C),
      mock các trường thiếu, lưu cache rồi trả về.

    Returns:
        DataFrame với các cột:
        restaurant_name, review_id, review_text, review_rating,
        review_timestamp, reviewer_id, reviewer_total_reviews,
        reviewer_is_local_guide
    Raises:
        ValueError nếu APIFY_TOKEN chưa được set trong .env
        RuntimeError nếu Apify không trả về review nào
    """
    cache_file = _cache_path(restaurant_name)

    # ---- Option A: Đọc cache ----
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        return pd.DataFrame(cached)

    # ---- Option C: Crawl via Apify ----
    apify_token = os.environ.get("APIFY_TOKEN", "").strip()
    if not apify_token:
        raise ValueError(
            "APIFY_TOKEN chưa được cấu hình. "
            "Hãy thêm APIFY_TOKEN=<token> vào file .env"
        )

    client = ApifyClient(apify_token)
    run_input = _build_run_input(restaurant_name, max_reviews)

    run = client.actor("compass/crawler-google-places").call(run_input=run_input)

    data_mapped = []
    for item in client.dataset(run.default_dataset_id).iterate_items():
        reviews_list = item.get("reviews", [])
        source_name = item.get("title", restaurant_name)

        if reviews_list:
            for rev in reviews_list:
                data_mapped.append({
                    "restaurant_name":         source_name,
                    "review_id":               rev.get("reviewId"),
                    "review_text":             rev.get("text") or "",
                    "review_rating":           rev.get("stars") or 3,
                    "review_timestamp":        rev.get("publishedAtDate"),
                    "reviewer_id":             rev.get("authorId"),
                    "reviewer_total_reviews":  rev.get("authorNumberOfReviews"),    # có thể None → mock
                    "reviewer_photo_count":    rev.get("authorNumberOfPhotos"),     # có thể None → mock
                    "reviewer_is_local_guide": rev.get("authorIsLocalGuide"),       # có thể None → mock
                })
        else:
            # Fallback: item chính là 1 review (flat structure)
            if item.get("reviewId"):
                data_mapped.append({
                    "restaurant_name":         source_name,
                    "review_id":               item.get("reviewId"),
                    "review_text":             item.get("text") or "",
                    "review_rating":           item.get("stars") or 3,
                    "review_timestamp":        item.get("publishedAtDate"),
                    "reviewer_id":             item.get("authorId"),
                    "reviewer_total_reviews":  item.get("authorNumberOfReviews"),
                    "reviewer_photo_count":    item.get("authorNumberOfPhotos"),
                    "reviewer_is_local_guide": item.get("authorIsLocalGuide"),
                })

    if not data_mapped:
        raise RuntimeError(
            f"Apify không tìm thấy review nào cho quán: \"{restaurant_name}\". "
            "Thử kiểm tra lại tên quán hoặc token Apify."
        )

    # Mock các trường None
    data_mapped = _mock_missing(data_mapped)

    # Lưu cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data_mapped, f, ensure_ascii=False, default=str)

    return pd.DataFrame(data_mapped)


def is_cached_today(restaurant_name: str) -> bool:
    """Kiểm tra xem quán này đã được crawl hôm nay chưa."""
    return _cache_path(restaurant_name).exists()
