# Báo cáo Thực hành: Mã hóa văn bản bằng vector dày đặc

## 1. Các bước triển khai

### 1.1. Phần 1: Giảm Chiều và Trực quan hóa Vector

* Giảm chiều dữ liệu (Dimensionality Reduction)

  * Các vector embedding ban đầu có số chiều lớn (300 chiều), không thể biểu diễn trực tiếp trên đồ thị không gian.
  * Sử dụng thuật toán **PCA (Principal Component Analysis)** để giảm số chiều từ 300 xuống còn **3 chiều (3D)**.

* Trực quan hóa dữ liệu (Visualization)

  * Sử dụng thư viện `matplotlib` để xây dựng biểu đồ phân tán (**Scatter Plot**) trong không gian 3 chiều.
  * Mỗi điểm dữ liệu tương ứng với một từ, kèm theo nhãn giúp quan sát vị trí và mối quan hệ tương đối giữa các từ trong không gian embedding sau khi giảm chiều.

* Tìm kiếm độ tương đồng (Similarity Search)

  * Thực hiện tìm kiếm **Top-K** từ gần nhất với một từ khóa cho trước dựa trên độ đo **Cosine Similarity**.
  * So sánh kết quả tìm kiếm trong hai không gian khác nhau: không gian gốc (300 chiều) và không gian đã giảm chiều (3D).
---

## 2. Cách chạy code và ghi log

### 1.1. Phần 1: Giảm Chiều và Trực quan hóa Vector

File code `./notebooks/lab3_embedding.ipynb`

---

## 3. Giải thích các kết quả thu được

### 3.1. Độ tương đồng trên mô hình Pre-trained (Không gian đầy đủ 300 chiều)

Khi thực hiện tìm kiếm Top-K từ tương đồng với từ khóa **"people"** trong không gian vector gốc (300 chiều), kết quả thể hiện rõ đặc trưng của từng mô hình:

* **Word2Vec (Google News):**

  * *Kết quả:* Trả về các từ đồng nghĩa hoặc có thể thay thế trực tiếp như *persons, human, individuals*.
  * *Nhận xét:* Word2Vec học tốt quan hệ ngữ nghĩa dựa trên ngữ cảnh cục bộ, đặc biệt phù hợp với dữ liệu tin tức.

* **GloVe:**

  * *Kết quả:* Xuất hiện các từ như *others, many, some*.
  * *Nhận xét:* Do dựa trên thống kê đồng xuất hiện toàn cục, GloVe phản ánh mối quan hệ đồng thời xuất hiện trong văn bản hơn là quan hệ đồng nghĩa thuần túy.

* **fastText:**

  * *Kết quả:* Trả về nhiều biến thể chính tả như *peopel, poeple*.
  * *Nhận xét:* fastText thể hiện ưu thế trong việc xử lý từ ngoài từ điển (OOV) nhờ cơ chế học trên các n-gram ký tự.

### 3.2. Trực quan hóa và ảnh hưởng của giảm chiều

* **Quan sát trực quan:**
  Biểu đồ 3D cho thấy các từ có ngữ nghĩa gần nhau thường hình thành các cụm nhỏ (clusters), ví dụ như nhóm tên quốc gia hoặc nhóm tháng trong năm. Tuy nhiên, do giới hạn của không gian 3 chiều, một số cụm có xu hướng chồng lấn.

* **Hiện tượng sai lệch khi tìm kiếm trên không gian 3D:**
  Khi tính độ tương đồng trên các vector đã giảm xuống 3 chiều, một số từ không liên quan xuất hiện với độ tương đồng Cosine rất cao (xấp xỉ 1.0).

* **Giải thích:**
  Đây là hậu quả của hiện tượng **mất mát thông tin (information loss)** khi PCA nén dữ liệu quá mạnh. Nhiều vector vốn cách xa nhau trong không gian 300 chiều bị chiếu xuống các hướng gần nhau trong không gian 3D, dẫn đến sai lệch về mặt ngữ nghĩa.

* **Kết luận:**
  Không gian embedding sau khi giảm chiều chỉ nên sử dụng cho mục đích trực quan hóa, không phù hợp cho các phép tính độ tương đồng chính xác.
