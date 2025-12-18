# Báo cáo Thực hành: Giải quyết bài toán phân loại văn bản bằng pipeline sử dụng các kỹ thuật tiền xử lý, mã hóa văn bản và model đã học

---

## 1. Các bước triển khai

### 1.1. Xây dựng Class phân loại cơ bản
* Tạo class TextClassifier đóng gói cả phần Vectorizer và Model (Logistic Regression).
* Huấn luyện trên data nhỏ.

---

### 1.2. Phân tích cảm xúc trên dữ liệu lớn với PySpark (Advanced Task)

Bài toán Sentiment Analysis trong môi trường Big Data bằng cách xử lý dữ liệu phân tán với Apache Spark.

**Spark Pipeline:** 

1. `Tokenizer`: Tách văn bản thành danh sách token.
2. `StopWordsRemover`: Loại bỏ các từ không mang nhiều thông tin ngữ nghĩa.
3. `HashingTF`: Biến đổi văn bản thành vector số nguyên bằng kỹ thuật hashing, với số chiều cố định (10.000 features) nhằm tiết kiệm bộ nhớ.
4. `IDF`: Tính trọng số Inverse Document Frequency để giảm ảnh hưởng của các từ xuất hiện quá thường xuyên.
5. `LogisticRegression`: Mô hình phân loại tuyến tính.

**Chuẩn hóa nhãn:**

* Nhãn cảm xúc ban đầu có dạng `-1 / 1`.
* Được chuyển đổi về `0 / 1` bằng công thức `(label + 1) / 2` để tương thích với Spark MLlib.

---

### 1.3. Cải thiện và tối ưu hóa mô hình (Improvement Task)

Hai chiến lược cải thiện được thử nghiệm nhằm đánh giá tác động của độ phức tạp mô hình so với chất lượng dữ liệu đầu vào:

**Chiến lược 1 – Mô hình phức tạp:**

* Sử dụng `Word2Vec` để học biểu diễn ngữ nghĩa.
* Kết hợp với `Gradient-Boosted Trees (GBT)` – mô hình phi tuyến tính.

**Chiến lược 2 – Tinh chỉnh đặc trưng (Giải pháp cuối):**

* Thay thế `HashingTF` bằng `CountVectorizer` để kiểm soát từ vựng rõ ràng hơn.
* Áp dụng tham số `minDF = 2.0` nhằm loại bỏ các từ xuất hiện quá ít (nhiễu).
* So sánh hiệu quả giữa `LogisticRegression` (có tinh chỉnh `regParam`) và `Naive Bayes`.

---

## 2. Cách chạy code và ghi log

Hệ thống được thiết kế để tự động ghi log chi tiết vào thư mục `output/lab4/`.

1. **Chạy TextClassifier Implementation:**

* Code: `python test/lab4_test.py`
* Log: `output/lab4/lab4_results.log`

2. **Chạy Advanced (PySpark Baseline):**

* Code: `python test/lab4_spark_sentiment_analysis.py`
* Log: `output/lab4/lab4_spark_sentiment.log`

3. **Chạy Cải thiện (Final Tuning):**

* Code: `python test/lab4_improvement.py`
* Log: `output/lab4/lab4_improvement.log`

---

## 3. Giải thích các kết quả thu được

### 3.1. Kết quả khi của mô hình text_classifier trên dữ liệu nhỏ.
* Tập dữ liệu mẫu (Toy Dataset) gồm 10 câu (5 Positive, 5 Negative) có chỉ số rất thấp.
* Vấn đề từ vựng mới (Out-of-Vocabulary - OOV):

  * Trong tập Test xuất hiện hai từ khóa quan trọng mang nghĩa tiêu cực là "awful" và "dull".

  * Tuy nhiên, do cách chia dữ liệu ngẫu nhiên (random_state=10), các câu chứa từ này chỉ nằm ở tập Test mà không hề xuất hiện trong tập Train.

  * Hệ quả: Đối với mô hình, "awful" và "dull" là những từ lạ (unknown), không có trọng số để đóng góp vào quyết định phân loại.

* Nhiễu do từ dừng (Stopwords Noise):

  * Tokenizer hiện tại (re.findall) giữ lại các từ dừng như "The", "was".

  * Trong tập Train (nhãn Positive), có thể xuất hiện các cấu trúc câu bắt đầu bằng "The..." (ví dụ: "The acting was superb"). Mô hình học được sai lầm rằng: cứ thấy "The" và "was" thì khả năng cao là Positive.

  * Khi gặp câu test "The script was..." (thiếu từ khóa "awful" để nhận diện), mô hình dựa vào các từ dừng này và đoán sai thành Positive.

* Kết quả cho thấy hạn chế chí mạng của phương pháp Đếm từ (CountVectorizer) khi áp dụng trên dữ liệu nhỏ: Mô hình hoàn toàn bất lực trước các từ chưa từng gặp (OOV).

* Điều này dẫn dắt đến sự cần thiết của việc: (1) Sử dụng tập dữ liệu lớn hơn, (2) Loại bỏ Stopwords để giảm nhiễu, và (3) Sử dụng các kỹ thuật làm mượt (Smoothing) hoặc Word Embedding (Lab nâng cao) để xử lý ngữ nghĩa tốt hơn.

### 3.1. Kết quả mô hình cơ sở (Baseline – PySpark)

* **Cấu hình:** HashingTF (10.000 features) + IDF + Logistic Regression.
* **Độ chính xác (Accuracy):** ~72.95%.

**Nhận xét:**

* Kết quả ở mức chấp nhận được đối với bài toán phân loại nhị phân.
* Tuy nhiên, `HashingTF` có thể gây ra hiện tượng va chạm (hash collision), khi nhiều từ khác nhau bị ánh xạ vào cùng một chỉ số vector, dẫn đến nhiễu thông tin.

---

### 3.2. Kết quả sau khi cải thiện

Hai hướng tiếp cận trái ngược cho kết quả khác biệt rõ rệt:

| Chiến lược           | Phương pháp                                     | Accuracy | Đánh giá       |
| -------------------- | ----------------------------------------------- | -------- | -------------- |
| Mô hình phức tạp     | Word2Vec + Gradient-Boosted Trees               | ~67.00%  | Giảm hiệu suất |
| Tinh chỉnh đặc trưng | CountVectorizer (minDF=2) + Logistic Regression | ~76.56%  | Tăng đáng kể   |

---

### 3.3. Phân tích nguyên nhân và bài học rút ra

**Nguyên nhân thất bại của mô hình phức tạp:**

* **Data Starvation:** Word2Vec yêu cầu lượng dữ liệu rất lớn để học được ngữ cảnh ổn định. Tập dữ liệu khoảng 5.000 câu là không đủ, dẫn đến vector embedding kém chất lượng.
* **Mất tín hiệu quan trọng:** Các từ khóa tài chính mang tính quyết định (như *short, long, call, put*) bị làm mờ khi học embedding, trong khi mô hình GBT dễ bị overfitting trên dữ liệu nhiễu.

**Nguyên nhân thành công của chiến lược đơn giản:**

* **Giảm nhiễu hiệu quả:** Tham số `minDF` loại bỏ các từ hiếm, thường là lỗi chính tả hoặc token không mang ý nghĩa.
* **Phù hợp với dữ liệu thưa:** Logistic Regression và Naive Bayes hoạt động ổn định hơn trên dữ liệu văn bản sparse, đặc biệt với kích thước dữ liệu nhỏ và trung bình.

**Kết luận:** Việc cải thiện chất lượng đặc trưng đầu vào (Feature Engineering) mang lại hiệu quả cao hơn đáng kể so với việc gia tăng độ phức tạp của thuật toán.

---

## 4. Khó khăn gặp phải và cách giải quyết

1. **Định dạng nhãn không đồng nhất:**

* Nhãn ban đầu `-1 / 1` không tương thích với Spark MLlib.
* Giải pháp: Chuẩn hóa về `0 / 1` bằng phép biến đổi cột trong Spark DataFrame.

3. **Hiệu năng kém với Word2Vec:**

* Accuracy giảm mạnh khi áp dụng embedding trên tập dữ liệu nhỏ.
* Giải pháp: Thay đổi chiến lược, tập trung vào Bag-of-Words kết hợp lọc nhiễu và regularization.

