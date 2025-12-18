# Báo cáo Thực hành: Phân tích cú pháp phụ thuộc (Dependency Parsing)
## 1. Các bước triển khai

* 1.1. Sử dụng thư viện spaCy để thực hiện phân tích cú pháp phụ thuộc
cho một câu.

* 1.2. Trực quan hóa cây phụ thuộc để hiểu rõ cấu trúc câu.

* 1.3. Truy cập và duyệt (traverse) cây phụ thuộc theo chương trình.

* 1.4. Trích xuất thông tin có ý nghĩa từ các mối quan hệ phụ thuộc (ví
dụ: tìm chủ ngữ, tân ngữ, bổ ngữ).


## 2. Cách chạy code và ghi log
* Cài đặt thư vện
```pip
pip install -U spacy
```

```pip
python -m spacy download en_core_web_md
```

**Code:** `./notebooks/lab7_dependency_parsing.ipynb`.