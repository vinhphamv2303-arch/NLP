# Báo cáo Thực hành: Giới thiệu về Transformers và các bài toán

## 1. Các bước triển khai
* Ôn tập kiến thức cơ bản về kiến trúc Transformer.
* Làm quen với thư viện Hugging Face Transformers.
## 2. Cách chạy code và ghi log
* **Code:** `./notebooks/lab6_transformers.ipynb`.
* **Đánh giá:**

  * **BERT (Encoder-only):** Chuyên để **"Đọc hiểu"**. Dùng cơ chế Attention **2 chiều** nên nắm bắt ngữ cảnh cực tốt (phù hợp bài toán điền từ, phân loại).
  * **GPT (Decoder-only):** Chuyên để **"Viết lách"**. Dùng cơ chế Attention **1 chiều** (tự hồi quy) nên sinh văn bản mượt mà, tự nhiên.