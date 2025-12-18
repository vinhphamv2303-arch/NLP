# Báo cáo Thực hành: Tách từ và biểu diễn văn bản bằng vector thưa

## 1. Các bước triển khai

Bài tập được tổ chức theo mô hình **lập trình hướng đối tượng (OOP)**, tách biệt rõ ràng giữa **Interface (giao diện)**, **Implementation (triển khai)** và **Evaluation (kiểm thử)**.

---

### 1.1. Lab 1: Text Tokenization

**Mục tiêu:** Xây dựng bộ tách từ (Tokenizer) để chuyển văn bản thô thành danh sách các token.

#### Bước 1: Định nghĩa Interface

* File: `src/core/interfaces.py`
* Xây dựng lớp trừu tượng `Tokenizer`

#### Bước 2: Triển khai Simple Tokenizer

* File: `src/preprocessing/simple_tokenizer.py`

* Các bước xử lý:

  * Chuyển toàn bộ văn bản về chữ thường.
  * Tách từ dựa trên khoảng trắng bằng `split()`.
  * Xử lý tách dấu câu ở cuối từ (ví dụ: `"word." → "word", "."`).

#### Bước 3: Triển khai Regex Tokenizer

* File: `src/preprocessing/regex_tokenizer.py`

* Sử dụng thư viện `re` của Python.

* Xây dựng biểu thức chính quy (Regex) để xử lý:

  * Từ có gạch nối (hyphen): `Al-Zaman`
  * Từ viết tắt, từ có dấu nháy: `isn't`
  * Số liệu, phần trăm: `10%`, `3.14`
  * Các dấu câu liên tiếp: `...`, `?!`

* Regex: 

  ```regex
  r'\.{2,}|[!?]{2,}|--+|[+-]?\d+[.,\d]*%?|\w+(?:[-\']\w+)*|[.!?:;,]'
  ```

---

### 1.2. Lab 2: Count Vectorization

**Mục tiêu:** Biểu diễn văn bản dưới dạng vector số học theo mô hình **Bag-of-Words**.

#### Bước 1: Định nghĩa Interface Vectorizer

* File: `src/core/interfaces.py`
* Xây dựng lớp trừu tượng `Vectorizer` với các phương thức:

  * `fit(corpus)`
  * `transform(corpus)`
  * `fit_transform(corpus)`

#### Bước 2: Triển khai CountVectorizer

* File: `src/representations/count_vectorizer.py`

* Kế thừa từ lớp `Vectorizer`.

* Sử dụng `Tokenizer` (Lab 1) để tách từ.

* Cơ chế hoạt động:

  * **Hàm `fit`:**

    * Duyệt toàn bộ corpus.
    * Xây dựng bộ từ điển `vocabulary_`.
    * Mỗi từ duy nhất được gán một chỉ số index (0, 1, 2, ...).

  * **Hàm `transform`:**

    * Duyệt từng văn bản.
    * Tạo vector đếm số lần xuất hiện của từng từ tương ứng với index trong từ điển.

---

## 2. Cách chạy code và ghi log

### Thực hiện Lab 1

* **Lệnh chạy:**

  ```bash
  python main.py
  ```

* **Kết quả:**

  * Log được lưu tại: `./output/lab1/tokenization_log.txt`
  * Nội dung:

    * Kết quả tokenization các câu mẫu.
    * 500 ký tự đầu của dataset `UD_English-EWT` sau khi tách từ.

---

### Thực hiện Lab 2

* **Lệnh chạy:**

  ```bash
  python test/lab2_test.py
  ```

* **Kết quả:**

  * Log được lưu tại: `./output/lab2/vectorization_log.txt`
  * Nội dung:
    
    * Vector hoá một số câu ngắn.

---

## 3. Giải thích các kết quả thu được

### 3.1. Lab 1: So sánh SimpleTokenizer và RegexTokenizer

Dựa trên log thực tế từ dataset `UD_English-EWT`:

* **Input:**

  > "Al-Zaman : American forces killed..."

* **SimpleTokenizer:**

  ```text
  ['al-zaman', '', ':', 'american', ...]
  ```

  * Nhận xét:

    * Hoạt động tốt với từ đơn.
    * Phát sinh token rỗng `''` khi gặp khoảng trắng hoặc dấu câu liền nhau.
    * Giữ được từ ghép `al-zaman` do không xử lý dấu gạch nối.

* **RegexTokenizer:**

  ```text
  ['al-zaman', ':', 'american', ...]
  ```

  * Nhận xét:

    * Không sinh token rỗng.
    * Dấu câu được tách riêng đúng chuẩn.

---

### 3.2. Lab 2: Count Vectorization

* **Corpus:**

  1. "I love NLP."
  2. "I love programming."
  3. "NLP is a subfield of AI."

* **Vocabulary:**

  * Học được 10 từ duy nhất (bao gồm dấu chấm).
  * Ví dụ:

    ```python
    {'i': 0, 'love': 1, 'nlp': 2, ...}
    ```

* **Document-Term Matrix:**

  * Câu 1 có vector:

    ```text
    [1, 1, 1, 1, 0, ...]
    ```
  * Thể hiện số lần xuất hiện của từng từ trong câu.
  * Các giá trị `0` biểu thị từ không xuất hiện.

* Kết luận:

  * Kết quả phản ánh chính xác mô hình **Bag-of-Words**.
    
    * Ưu điểm: Đơn giản
    * Nhược điểm: Vector thưa có nhiều giá trị 0, kích thước lớn theo số lượng từ của từ điển.